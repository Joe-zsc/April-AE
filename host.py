import sys, os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
import numpy as np

from util import *
# from action import *
from actions.Action import *
from actions.Exploit import Exploit
from actions.OSScan import OSScan
from actions.PortScan import PortScan
from actions.ServiceScan import ServicesScan
from actions.WebScan import WebScan
from NLP_Module.sentence_vector import get_vector
from NLP_Module.SBERT_sentence_vector import sentence_embeddings

from collections import deque

class HOST:
    def __init__(self, ip='',env_data:dict=None):
        self.ip = ip
        self.host_state = Host_state(ip=self.ip)
        self.action = Action()
        self.info = self.host_state.host_info
        self.action_history = self.action.history_set
        self.env_data=env_data #环境数据，用于执行动作丛中获得反馈
        assert env_data['ip']==ip,env_data
        

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
                self.action, action_mask,self.env_data)
            self.action_history = self.action.history_set
            self.info = self.host_state.host_info
            # UTIL.write_csv(self.info)
            return next_o, r, done, result


class Host_state:


    word_vector_dim = int(Configure.get('Embedding', 'word_vector_dim'))
    sentence_vector_dim = int(Configure.get('Embedding', 'sentence_vector_dim'))
    support_port = PortScan.support_ports
    port_num = len(support_port)
    port_vector_eye = np.eye(port_num, dtype=np.float32)
    services_num = port_num
    port_index = []
    # 状态空间划分
    state_vector_key = [
        "access", "os", "port", "service", "web_fingerprint", "action_history"
    ]
    state_vector = dict.fromkeys(state_vector_key, 0)

    access_dim = 2
    state_vector["access"] = access_dim
    os_dim = word_vector_dim
    state_vector["os"] = os_dim
    port_dim = port_num
    state_vector["port"] = port_dim
    service_dim = word_vector_dim * services_num
    state_vector["service"] = service_dim
    web_fingerprint_dim = sentence_vector_dim
    state_vector["web_fingerprint"] = web_fingerprint_dim
    

    access = 2
    OS_vector_idx = access_dim
    port_vector_idx = access_dim + os_dim
    services_vector_idx = access_dim + os_dim + port_dim
    web_fingerprint_idx = access_dim + os_dim + port_dim + service_dim
    action_history_idx = access_dim + os_dim + port_dim + service_dim + web_fingerprint_dim
    


    state_space = access_dim + os_dim + port_dim + \
        service_dim+ web_fingerprint_dim

    def __init__(self, ip):
        self.ip = ip
        '''
        state related info
        '''
        self.os = None  # string
        # string:unknown,reachable,compromised, 00:unknow , 11:compromised , 01: uncompromised
        self.access = None
        self.port = None  # list of string
        self.services = None  # list of string
        self.web_fingerprint = None  # str
        self.host_vector = self.initialize()
        '''
        reforcement learning related info
        '''
        self.done = 0
        self.reward = 0
        self.steps=0
        '''
        host info
        '''
        self.host_info = Host_info(ip=ip)

        self.port_vector = self.host_vector[self.port_vector_idx:self.
                                            services_vector_idx]
        self.serv_vector = self.host_vector[self.services_vector_idx:self.
                                            web_fingerprint_idx]
        self.os_vector = self.host_vector[self.OS_vector_idx:self.
                                          port_vector_idx]
        self.web_vector = self.host_vector[self.web_fingerprint_idx:self.
                                           action_history_idx]
        self.act_vector = self.host_vector[self.action_history_idx:]

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
        self.steps=0
        # self.host_info = dict.fromkeys(self.info, None)
        self.host_vector = self.initialize()
        
        
        return self.host_vector

    def goal_reached(self):
        done = 0
        if self.access == "compromised":
            done = 1
        return done

    def step(self, Action: Action, action_mask: int,env_data:dict):

        # The action_idx here is the index of the action set
        done = 0
        reward = 0
        # action_exec_vector = self.change_action_history_vector(action_mask)
        a: Action_Class = Action.legal_actions[action_mask]  # 真实的动作编号
        action_constraint = Action.action_constraint( a)
        Action.last_action_id=a.id
        if action_constraint:
            cost = action_constraint['cost']
            result = action_constraint['message']
            reward = (reward - cost) 
        else:
            
            Action.history_set.add(a.id)
            cost = a.act_cost
            if a == Action.PORT_SCAN:
                action = PortScan(target_info=self.host_info,env_data=env_data)
                action.act()
                self.host_info.port = action.port_list
                self.port = action.port_list
                if action.port_list:
                    self.access = "reachable"
                    self.update_vector(port=True, access=True)
                    reward = a.success_reward
                result = action.port_list

            elif a == Action.OS_SCAN:
                action = OSScan(target_info=self.host_info,env_data=env_data)
                action.act()
                self.host_info.os = action.os
                self.os = action.os
                if action.os:
                    self.update_vector(os=True)
                    reward = a.success_reward
                result = action.os

            elif a == Action.SERVICE_SCAN:
                action = ServicesScan(target_info=self.host_info,env_data=env_data)
                action.act()
                self.host_info.services = action.services_list
                self.services = action.services_list
                if action.services_list:
                    self.update_vector(service=True)
                    reward = a.success_reward
                result = action.services_list

            elif a == Action.PORT_SERVICE_SCAN:

                action = ServicesScan(target_info=self.host_info,
                                      port_scan=True,env_data=env_data)
                action.act()
                self.host_info.port = action.port
                self.port = action.port
                if action.port:
                    self.access = "reachable"
                    self.update_vector(port=True, access=True)
                self.host_info.services = action.services_list
                self.services = action.services_list
                if action.services_list:
                    self.update_vector(service=True)
                    reward = a.success_reward
                result = action.services_list

            elif a == Action.WEB_SCAN:

                action = WebScan(target_info=self.host_info,env_data=env_data)
                action.act()
                self.host_info.web_fingerprint = action.fliter_info
                self.web_fingerprint = action.fliter_info
                result = action.fliter_info
                if result:
                    Action.webscan_counts += 1
                    self.update_vector(web_fingerprint=True)
                    reward = a.success_reward if Action.webscan_counts == 1 else 0  #*math.exp(-Action.webscan_counts)

            elif a in Action.All_EXP:
                action = Exploit(target_info=self.host_info,env_data=env_data,
                                 exp=a)
                result, target_info = action.act()
                Action.exp_counts += 1
                if result:
                    self.host_info = target_info
                    self.access = "compromised"
                    self.update_vector(access=True)
                    reward = a.success_reward  #*math.exp(-Action.exp_counts)
                else:
                    cost += Action.action_failed['cost']
        reward = int(reward - cost)
        # cost = cost * action_exec_vector[action_idx]
        done = self.goal_reached()
        next_state = self.host_vector
        if isinstance(result, list):
            result = ','.join(result)
        # reward = reward - cost
        self.steps+=1
        return next_state, reward, done, result

    def change_os_vector(self):
        os_vector = np.zeros(self.word_vector_dim, dtype=np.float32)
        all_possible_os = []
        if self.os.find("or") != -1:
            all_possible_os = self.os.split("or")
        else:
            all_possible_os.append(self.os)
        for i in range(len(all_possible_os)):
            os = all_possible_os[i]
            vec = get_vector(
                os, dim=self.word_vector_dim).detach().numpy().flatten()
            os_vector += vec
        vector = os_vector / len(all_possible_os)
        return vector

    def change_port_vector(self):
        vector = np.zeros(self.port_num, dtype=np.float32)
        self.port_index = []
        for p in self.port:
            if p in self.support_port:
                idx = self.support_port.index(p)
                self.port_index.append(idx)
                vector += self.port_vector_eye[idx]
        return vector.reshape(-1, 1).squeeze()

    def change_services_vector(self):
        assert len(self.port_index) > 0
        assert len(self.services) == len(self.port)
        vector = np.zeros([self.port_num, self.word_vector_dim],
                          dtype=np.float32)

        for s in range(len(self.services)):
            # v = self.get_WordVector(self.services[s])
            v = get_vector(self.services[s],
                           dim=self.word_vector_dim).detach().numpy()
            vector[self.port_index[s]] = v
        service_vector = vector.flatten()
        return service_vector

    def change_access_vector(self):
        vector = np.zeros(2, dtype=np.float32)
        if self.access == 'reachable':
            vector[1] = 1
        elif self.access == 'compromised':
            vector[0] = 1
        return vector

    def change_web_fingerprint_vector(self):
        wp_vector = np.zeros(self.sentence_vector_dim, dtype=np.float32)
        for wp in self.web_fingerprint:
            # vector = get_vector(
            #     wp, dim=self.sentence_vector_dim).detach().numpy().flatten()
            vector=sentence_embeddings([wp]).flatten()
            wp_vector += vector
        wp_vector = wp_vector / len(self.web_fingerprint)
        return wp_vector


    def update_vector(self,
                      access=False,
                      os=False,
                      port=False,
                      service=False,
                      web_fingerprint=False):
        if access:
            vector = self.change_access_vector()
            self.host_vector[:self.OS_vector_idx] = vector
        if os:
            vector = self.change_os_vector()
            self.host_vector[self.OS_vector_idx:self.port_vector_idx] = vector
        if port:
            vector = self.change_port_vector()
            self.host_vector[self.port_vector_idx:self.
                             services_vector_idx] = vector
        if service:
            vector = self.change_services_vector()
            self.host_vector[self.services_vector_idx:self.
                             web_fingerprint_idx] = vector

        if web_fingerprint:
            vector = self.change_web_fingerprint_vector()
            self.host_vector[self.web_fingerprint_idx:self.
                             action_history_idx] = vector
        return self.host_vector

    def initialize(self):
        vector = np.zeros(self.state_space, dtype=np.float32)
        return vector

    
