import sys, os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path

from util import  Host_info
class ServicesScan(): 

    def __init__(self,
                 target_info: Host_info,
                 env_data:dict=None,
                 port_scan=False):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port = self.target_info.port
        self.services_list = []
        self.env_data=env_data
    def act(self):


        services_list = self.simulate_act()
        
        self.services_list = services_list
        self.target_info.services = services_list
        result = True if services_list else False
        return result,self.target_info
    def simulate_act(self):
        
        if self.env_data["ip"] == self.target_ip:
            return self.env_data["services"]
        return []

    