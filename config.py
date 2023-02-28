#Functions specifically related the the management of the config.yaml file.

import yaml

"""
Function to load in the config file to control most aspects of the Transformer network.

Arguments:
    name: The name of the file (including path). 
        Defaults to 'config.yaml' found in the project's base URL.
"""
def loadConfig(name="config.yaml"):
    with open(name, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
