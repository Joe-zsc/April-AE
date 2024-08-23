import configparser
import logging
import IPy
import time
import os
from pprint import pprint, pformat
import pandas as pd
import csv
from pathlib import Path


class IP:
    def __init__(self, ip, netmask):
        self.address = ip
        self.net_mask = netmask

    def subnet(self):
        return IPy.IP(self.address + '/' + self.net_mask, make_net=True)

    @classmethod
    def checkIP(cls, ip_address):
        try:
            IPy.IP(ip_address)
            return True
        except Exception as e:
            print(str(ip_address) + "不是ip地址,异常原因：" + str(e))
            return False


class Host_info:
    def __init__(self, ip):
        self.ip: str = ip
        self.os: str = ''
        self.port: list = []
        self.web_fingerprint: str = ''
        self.services: list = []
        self.vul: list = []
        
class Action_Class:
    def __init__(
        self,
        id: int,
        name: str,
        act_cost: int,
        success_reward: int = 0,
        type: str = None,
        config: dict = None,
    ):
        self.id = id
        self.name = name
        self.type = type
        self.act_cost = act_cost
        self.success_reward = success_reward
        self.config = config

class Configure():
    conf = configparser.ConfigParser()
    try:
        conf.read(os.path.join(os.path.dirname(__file__),"config.ini"))
    except Exception as e:
        print("config file not found" + e)

    @classmethod
    def get(cls, label, name):
        return cls.conf.get(label, name)

    @classmethod
    def set(cls, label, name, value):
        cls.conf.set(label, name, str(value))
        cls.conf.write(open("config.ini", "w"))

class UTIL:
    '''
    Running Mode:
    '''

    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                 time.localtime(time.time()))
    project_path = Path(__file__).parent
    Running_title=''
    @classmethod
    def show_banner(cls):
        
        banner = u"""
 
     ___      .______   .______       __   __      
    /   \     |   _  \  |   _  \     |  | |  |     
   /  ^  \    |  |_)  | |  |_)  |    |  | |  |     
  /  /_\  \   |   ___/  |      /     |  | |  |     
 /  _____  \  |  |      |  |\  \----.|  | |  `----.
/__/     \__\ | _|      | _| `._____||__| |_______|
                                                   

"""
        
        print(banner)
        cls.show_credit()
        time.sleep(2)

    # flag_log
    @classmethod
    def show_credit(cls):
        credit = u"""
+ -- --=[ APRIL\t: Autonomous Penetesting based on ReInforcement Learning             ]=-- -- +
+ -- --=[ Author\t: NUDT-HFBOT Team                                   ]=-- -- +
+ -- --=[ Website\t: https://gitee.com/JoeSC/April-AE  ]=-- -- +
    """
        print(credit)





    @classmethod
    def line_break(cls, length=60, symbol='-'):
        line_break = symbol * length
        logging.info(line_break)
        
    def write_csv_DictList(file:Path,data:list):
        variables = list(data[0].keys())
        pd_data = pd.DataFrame([[i[j] for j in variables] for i in data], columns=variables)
        pd_data.to_csv(file,mode='w',index=False)
        return pd_data

class color:

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @classmethod
    def print(cls, s, c=GREEN, end='\n'):
        print(c + s + cls.END, end=end)

    @classmethod
    def color_str(cls, s, c=GREEN):
        s = pformat(s)
        return c + s + cls.END




def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log

    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0

    logging.critical()
    logging.fatal()
    logging.error()
    logging.warning()
    logging.warn()
    logging.info()
    logging.debug()

    """
    logger = logging.getLogger()
    # root_logger = logging.getLogger()
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


