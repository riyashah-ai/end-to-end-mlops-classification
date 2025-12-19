import yaml

def load_config(config_path:str = "configs/train.yaml"):
    with open(config_path , 'r') as f:
        config_params = yaml.safe_load(f)
        print(config_params)
    return config_params

# print(load_config()["training"])