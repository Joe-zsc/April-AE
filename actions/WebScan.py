
import sys
import os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from util import Host_info

class WebScan():
    def __init__(self, target_info: Host_info,env_data:dict=None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.info = []
        self.fliter_info = []
        self.env_data=env_data
    def act(self):
        
        self.info = self.simulate_act()
        self.fliter_info = self.info

        
        result = True if self.fliter_info else False
        self.target_info.web_fingerprint = self.fliter_info
        return result, self.target_info

    def simulate_act(self):

        if self.env_data["ip"] == self.target_ip:
            if "web_fingerprint" in self.env_data:
                return self.env_data["web_fingerprint"]
        return []



    
