import json

def read_json_config(path):
    return json.load(open(path, "r", encoding="utf-8"))


def write_json_config(config, path):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=4)