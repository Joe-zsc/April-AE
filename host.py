import numpy as np
from util import *
from actions import *
from NLP_Module.Encoder import encoder
class HOST:
    def __init__(self, ip="", env_data: dict = None):
        self.ip = ip
        self.host_state = StateEncoder(ip=self.ip)
        self.action = Action()
        self.info = self.host_state.host_info
        self.action_history = self.action.history_set
        self.env_data = env_data  
        assert env_data["ip"] == ip, env_data

    def reset(self):
        self.action.reset()
        return self.host_state.reset()

    def perform_action(self, action_mask):
        if Action.test_action(action_mask):
            if type(action_mask) == str:
                for id in range(len(Action.legal_actions)):
                    if Action.legal_actions[id].name == action_mask:
                        action_mask = id
                        break
            next_o, r, done, result = self.host_state.step(
                self.action, action_mask, self.env_data
            )
            self.action_history = self.action.history_set
            self.info = self.host_state.host_info
            # UTIL.write_csv(self.info)
            return next_o, r, done, result


class StateEncoder:

    # 状态空间划分
    state_vector_key = [
        "access",
        "os",
        "port",
        "service",
        "web_fingerprint",
    ]
    state_vector = dict.fromkeys(state_vector_key, 0)

    access_dim = 2
    state_vector["access"] = access_dim
    os_dim = 100
    state_vector["os"] = os_dim
    port_dim = 100
    state_vector["port"] = port_dim
    service_dim = 100
    state_vector["service"] = service_dim
    web_fingerprint_dim = 100
    state_vector["web_fingerprint"] = web_fingerprint_dim

    access = 2
    OS_vector_idx = access_dim
    port_vector_idx = access_dim + os_dim
    services_vector_idx = access_dim + os_dim + port_dim
    web_fingerprint_idx = access_dim + os_dim + port_dim + service_dim
    final_idx = access_dim + os_dim + port_dim + service_dim + web_fingerprint_dim

    state_space = access_dim + os_dim + port_dim + service_dim + web_fingerprint_dim

    def __init__(self, ip):
        self.ip = ip
        """
        state related info
        """
        self.os = None  # string
        self.access = None
        self.port = None  # list of string
        self.services = None  # list of string
        self.web_fingerprint = None  # str
        self.host_vector = self.initialize()
        """
        reforcement learning related info
        """
        self.done = 0
        self.reward = 0
        self.steps = 0
        """
        host info
        """
        self.host_info = Host_info(ip=ip)

        self.port_vector = self.host_vector[
            self.port_vector_idx : self.services_vector_idx
        ]
        self.serv_vector = self.host_vector[
            self.services_vector_idx : self.web_fingerprint_idx
        ]
        self.os_vector = self.host_vector[self.OS_vector_idx : self.port_vector_idx]
        self.web_vector = self.host_vector[self.web_fingerprint_idx : self.final_idx]
        self.act_vector = self.host_vector[self.final_idx :]

    def observ(self):
        return self.host_vector

    def reset(self):

        self.done = 0
        self.reward = 0

        self.access = None
        self.port = None
        self.services = None
        self.os = None
        self.web_fingerprint = None
        self.steps = 0
        # self.host_info = dict.fromkeys(self.info, None)
        self.host_vector = self.initialize()

        return self.host_vector

    def goal_reached(self):
        done = 0
        if self.access == "compromised":
            done = 1
        return done

    def step(self, Action: Action, action_mask: int, env_data: dict):

        done = 0
        reward = 0
        a: Action_Class = Action.legal_actions[action_mask]
        action_constraint = Action.action_constraint(a)
        Action.last_action_id = a.id
        if action_constraint:
            cost = action_constraint["cost"]
            result = action_constraint["message"]
            reward = reward - cost
        else:

            Action.history_set.add(a.id)
            cost = a.act_cost
            if a == Action.PORT_SCAN:
                action = PortScan(target_info=self.host_info, env_data=env_data)
                action.act()
                self.host_info.port = action.port_list
                self.port = action.port_list
                if action.port_list:
                    self.access = "reachable"
                    self.update_vector(port=True, access=True)
                    reward = a.success_reward
                result = action.port_list

            elif a == Action.OS_SCAN:
                action = OSScan(target_info=self.host_info, env_data=env_data)
                action.act()
                self.host_info.os = action.os
                self.os = action.os
                if action.os:
                    self.update_vector(os=True)
                    reward = a.success_reward
                result = action.os

            elif a == Action.SERVICE_SCAN:
                action = ServicesScan(target_info=self.host_info, env_data=env_data)
                action.act()
                self.host_info.services = action.services_list
                self.services = action.services_list
                if action.services_list:
                    self.update_vector(service=True)
                    reward = a.success_reward
                result = action.services_list

            elif a == Action.WEB_SCAN:

                action = WebScan(target_info=self.host_info, env_data=env_data)
                action.act()
                self.host_info.web_fingerprint = action.fliter_info
                self.web_fingerprint = action.fliter_info
                result = action.fliter_info
                if result:
                    Action.webscan_counts += 1
                    self.update_vector(web_fingerprint=True)
                    reward = (
                        a.success_reward if Action.webscan_counts == 1 else 0
                    )  
            elif a in Action.All_EXP:
                action = Exploit(target_info=self.host_info, env_data=env_data, exp=a)
                result, target_info = action.act()
                Action.exp_counts += 1
                if result:
                    self.host_info = target_info
                    self.access = "compromised"
                    self.update_vector(access=True)
                    reward = a.success_reward  
                else:
                    cost += Action.action_failed["cost"]
        reward = int(reward - cost)
        done = self.goal_reached()
        next_state = self.host_vector
        if isinstance(result, list):
            result = ",".join(result)
        self.steps += 1
        return next_state, reward, done, result

    def change_os_vector(self):
        os_vector = np.zeros(self.os_dim, dtype=np.float32)
        all_possible_os = []
        if self.os.find("or") != -1:
            all_possible_os = self.os.split("or")
        else:
            all_possible_os.append(self.os)
        for i in range(len(all_possible_os)):
            os = all_possible_os[i]
            vec = encoder.encode_SBERT(
                sentences=os, reduction_dim=self.os_dim
            ).flatten()
            os_vector += vec
        vector = os_vector / len(all_possible_os)
        return vector

    def change_port_vector(self):
        vector = np.zeros(self.port_dim, dtype=np.float32)
        all_ports = ",".join(self.port)
        vector = encoder.encode_SBERT(
            sentences=all_ports, reduction_dim=self.port_dim
        ).flatten()
        return vector


    def change_services_vector(self):
        assert len(self.port) > 0
        assert len(self.services) == len(self.port)
        vector = np.zeros(self.service_dim, dtype=np.float32)
        all_services = ",".join(self.services)
        vector = encoder.encode_SBERT(
            sentences=all_services, reduction_dim=self.service_dim
        ).flatten()
        return vector

    def change_access_vector(self):
        vector = np.zeros(2, dtype=np.float32)
        if self.access == "reachable":
            vector[1] = 1
        elif self.access == "compromised":
            vector[0] = 1
        return vector

    def change_web_fingerprint_vector(self):
        wp_vector = np.zeros(self.web_fingerprint_dim, dtype=np.float32)
        for wp in self.web_fingerprint:
            # vector = get_vector(
            #     wp, dim=self.sentence_vector_dim).detach().numpy().flatten()
            vector = encoder.encode_SBERT(
                sentences=[wp], reduction_dim=self.web_fingerprint_dim
            ).flatten()
            wp_vector += vector
        wp_vector = wp_vector / len(self.web_fingerprint)
        return wp_vector

    def update_vector(
        self, access=False, os=False, port=False, service=False, web_fingerprint=False
    ):
        if access:
            vector = self.change_access_vector()
            self.host_vector[: self.OS_vector_idx] = vector
        if os:
            vector = self.change_os_vector()
            self.host_vector[self.OS_vector_idx : self.port_vector_idx] = vector
        if port:
            vector = self.change_port_vector()
            self.host_vector[self.port_vector_idx : self.services_vector_idx] = vector
        if service:
            vector = self.change_services_vector()
            self.host_vector[self.services_vector_idx : self.web_fingerprint_idx] = (
                vector
            )

        if web_fingerprint:
            vector = self.change_web_fingerprint_vector()
            self.host_vector[self.web_fingerprint_idx : self.final_idx] = vector
        return self.host_vector

    def initialize(self):
        vector = np.zeros(self.state_space, dtype=np.float32)
        return vector
