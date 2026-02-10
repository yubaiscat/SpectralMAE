import argparse

import yaml


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # for displays
        self.parser.add_argument('--config', type=str, help='# of input config') 

    def get_config(self, config):
        with open(config, 'r') as stream:
            return yaml.load(stream)

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        with open(self.opt.config, 'r') as stream:
            self.config =  yaml.load(stream,Loader=yaml.FullLoader)
        print(self.config)

        return self.config
