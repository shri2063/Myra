import os
import yaml
def txtread(path):

    with open('myra-app-main/demo.yml', 'r') as f:
        return f.read()
def yamlread(path):
    return yaml.safe_load(txtread(path=path))

print(yamlread(''))