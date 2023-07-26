import os
import json
import argparse


def main(vals_buildings, test_buildings):
    all_tasks = []
    dirs = [ f.path for f in os.scandir(os.path.dirname(os.path.abspath(__file__))) if f.is_dir() ]
    for d in dirs:
        taskname = os.path.basename(d)
        templates = [ f"{taskname}/{{domain}}/"+os.path.basename(f.path).replace("_rgb.","_{domain}.") for f in os.scandir(os.path.join(d,"rgb")) if f.is_file() ]
        templates = sorted(templates)
        with open(d+".json", "w") as f:
            json.dump(templates, f)

        all_tasks.append(taskname)

    train_tasks = []
    val_tasks = []
    test_tasks = []
    for task in all_tasks:
        if task in test_buildings:
            test_tasks.append(task)
        elif task in vals_buildings:
            val_tasks.append(task)
        else:
            train_tasks.append(task)

    foldername = os.path.dirname(d)
    for s,f in zip([train_tasks, val_tasks, test_tasks], ["train_split.json", "val_split.json", "test_split.json"]):
        with open(os.path.join(foldername, f), "w") as file:
            json.dump(s, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Taskonomy splits generator")
    parser.add_argument("--val", nargs="*", type=str, default=[])
    parser.add_argument("--test", nargs="+", type=str, default=["uvalda", "merom", "stockman"])
    args = parser.parse_args()

    main(args.val, args.test)
