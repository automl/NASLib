import time
import os
import subprocess

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()

def main():
    config_files = []

    # Obtaining config files
    search_space = "nasbench201"
    dataset_dir = f"/Users/lars/Projects/NASLib/naslib/benchmarks/bbo/configs_m1/{search_space}"
    datasets = os.listdir(dataset_dir)
    for dataset in datasets:
        optimizer_dir = os.path.join(dataset_dir, dataset)
        if not os.path.isdir(optimizer_dir):
            continue
        optimizers = os.listdir(optimizer_dir)
        for optimizer in optimizers:
            if not os.path.isdir(os.path.join(optimizer_dir, optimizer)):
                continue
            config_path = os.path.join(optimizer_dir, optimizer, "config_0")
            for seed in os.listdir(config_path):
                config_file = os.path.join(config_path, seed)
                if not os.path.isfile(config_file):
                    continue
                config_files.append(config_file)

    # A Nicer, Single-Call Usage
    for config in progressBar(config_files, prefix = 'Progress:', suffix = 'Complete', length = 50):
        # Use a list of args instead of a string
        my_cmd = ["/Users/lars/Projects/naslib-venv/bin/python", 
            "/Users/lars/Projects/NASLib/naslib/benchmarks/bbo/runner.py", 
            "--config-file", config]
        with open('bbo-exps-output.log', "w") as outfile:
            subprocess.run(my_cmd, stdout=outfile, stderr=outfile)

if __name__ == '__main__':
    main()