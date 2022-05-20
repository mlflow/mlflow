import json
import os
import pkg_resources

def read_configs():
    default_config_file = pkg_resources.resource_filename('mlflow', 'server/default-config.json')
    config_file = os.getenv("_MLFLOW_CONFIG_PATH", default_config_file)
    with open(config_file) as config_file:
        data = json.load(config_file)

    return data