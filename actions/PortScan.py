
from util import UTIL, Configure, Host_info
import re
import sys
import os
import time
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path


class PortScan():

    support_ports_str = "21, 22, 25, 80, 139, 161, 443, 445, 2222, 3000, 3306, 3389, 4369, 4505, 4506, 5432, 5984, 6123, 6379, 7001, 8000, 8009, 8080, 8081, 8090,8111, 8161,8825, 8443, 8888, 8983, 9001, 9090,9100, 20022, 61616".strip().replace(' ','')
    support_ports = support_ports_str.split(',')  # list
    
    def __init__(self, target_info: Host_info, env_data:dict=None):
        self.target_ip = target_info.ip
        self.target_info = target_info
        self.port_list = []
        self.env_data=env_data

    def act(self):
        
        port_list = self.simulate_act()
        
        self.target_info.port = port_list
        self.port_list = port_list
        result = True if port_list else False
        return result, self.target_info

    def simulate_act(self):

        if self.env_data["ip"] == self.target_ip:
            return self.env_data["port"]
        return []

    