import pathlib
import json
import os


def save_settings(sv_dir, args):

    log_dict = vars(args)
    json_dict = json.dumps(log_dict, sort_keys=True, indent=4)

    if not os.path.exists(f"results/{sv_dir}"):
        os.makedirs(f"results/{sv_dir}")
    with open("results/" + sv_dir + "/settings.json", "w") as file:
        json.dump(json_dict, file)
