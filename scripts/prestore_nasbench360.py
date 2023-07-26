import argparse
import os
import pickle
from typing import Dict, AnyStr, Any

# Archive with NATS-tss-v1_0-daa55.pickle can be downloaded from
# https://drive.google.com/file/d/1y_Y3TbIE5rVhJ42hwIq6alUeMqlFr-bv/view?usp=sharing
# To download itself use https://drive.google.com/drive/folders/1LvfKgYhgx9g9ZV5bbvuxBZN2YwP0Ku6z?usp=share_link
NINAPRO = ("NATS-tss-v1_0-daa55.pickle", "nb201_ninapro_full_training.pickle")


def read_pickle(filename: AnyStr) -> Dict[AnyStr, Any]:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data: Dict[AnyStr, Any], filename: AnyStr):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def process_ninapro(filepath, output):
    data = read_pickle(filepath)
    new_database = {}
    for info_ in data["arch2infos"].values():
        info = info_["200"]
        archstr = info["arch_str"]
        info = next(iter(info["all_results"].values()))
        cost_info = {
            "flops": info["flop"],
            "params": info["params"],
            "latency": info["latency"],
            "train_time": info["train_times"][0]
        }
        eval_name = info["eval_names"][0]
        new_database[archstr] = {"ninapro": {
            "train_losses": [info["train_losses"][i] for i in range(info["epochs"])],
            "eval_losses": [info["eval_losses"][f"{eval_name}@{i}"] for i in range(info["epochs"])],
            "train_acc1es": [info["train_acc1es"][i] for i in range(info["epochs"])],
            "eval_acc1es": [info["eval_acc1es"][f"{eval_name}@{i}"] for i in range(info["epochs"])],
            "cost_info": cost_info
        }}
    write_pickle(new_database, output)

def main(input, output):
    # process ninapro
    ninapro_path = os.path.join(input, NINAPRO[0])
    if os.path.isfile(ninapro_path):
        process_ninapro(ninapro_path, os.path.join(output, NINAPRO[1]))
    else:
        print(f"Ninapro is not in {input}")


if __name__ == '__main__':
    parcer = argparse.ArgumentParser("Evaluation data convertor for NinaPro and Darcyflow.")
    parcer.add_argument("-i", "--input", type=str, required=True, help="Folder with downloaded and unpacked (.pickle) evaluations")
    parcer.add_argument("-o", "--output", type=str, default="naslib/data", help="Folder to output results")

    args = parcer.parse_args()

    input = args.input
    assert os.path.isdir(input), f"'{input}' is not a directory"
    output = args.output

    os.makedirs(output, exist_ok=True)

    main(input, output)