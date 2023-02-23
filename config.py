#Functions specifically related the the management of the config.yaml file.

import yaml

def loadConfig(name="config.yaml"):
    with open(name, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
