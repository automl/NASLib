from collections import namedtuple
import ast
import json
from pathlib import Path

# import nasbench301 as nb
# performance_model = nb.load_ensemble("xgb")
Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

for p in Path("logs/").glob("*.log"):
    print(p)

    accuracies = []
    with open(p) as f:
        for line in f:
            if "normal=" in line and "reduction" not in line:
                pos = line.index("normal=")
                normal = ast.literal_eval(line[pos + 7 :].rstrip())
            elif "reduce=" in line:
                pos = line.index("reduce=")
                reduce = ast.literal_eval(line[pos + 7 :].rstrip())

                genotype_config = Genotype(
                    normal=normal,
                    normal_concat=[2, 3, 4, 5],
                    reduce=reduce,
                    reduce_concat=[2, 3, 4, 5],
                )

                # val_acc = performance_model.predict(config=genotype_config, representation="genotype", with_noise=False)
                # accuracies.append(float(val_acc))

                normal = None
                reduce = None

    valid_acc = []
    with open(p) as f:
        for line in f:
            if "Validation accuracy:" in line:
                pos = line.index("Validation accuracy: ")
                val = ast.literal_eval(line[pos + 21 : pos + 29])
                valid_acc.append(val)

    with open("results/{}.json".format(p.stem), "w") as f:
        json.dump({"test_acc": accuracies, "valid_acc": valid_acc}, f)
