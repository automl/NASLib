from collections import namedtuple
import ast
import json
from pathlib import Path

for p in Path('nb201/').glob("*.log"):
    
    archs = []
    with open(p) as f:
        for line in f:
            if "Anytime results: " in line:
                arch_eval = ast.literal_eval(line[line.index("{'cifar10-valid':"):].rstrip())
                archs.append(arch_eval)

    valid_acc = []
    with open(p) as f:
        for line in f:
            if "Validation accuracy:" in line:
                pos = line.index("Validation accuracy: ")
                val = ast.literal_eval(line[pos + 21:pos + 29])
                valid_acc.append(val)

    with open('nb201/{}.json'.format(p.stem), 'w') as f:
        json.dump({
            'test_acc': archs,
            'valid_acc': valid_acc}, f)

