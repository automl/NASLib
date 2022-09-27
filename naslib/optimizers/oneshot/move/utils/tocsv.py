import os
import glob
import csv

path='slurm/'
extension='out'
os.chdir(path)
files=glob.glob('*.{}'.format(extension))
f = open('../nb201runs.csv', 'w', newline='\n')
writer=csv.writer(f, lineterminator='\n')

row=['file name', 'using absolute', 'instantenous', 'masking interval', 'seed', 'train portion', 'warm start epochs', 'cifar-10 test acc', 'cifar-100 test acc', 'imagenet16-120 test acc', 'cifar10 val acc']
writer.writerow(row)

def substring_after(s, delim):
    return s.partition(delim)[2]

row=[]
for file1 in files:
    f1 = open(file1, 'r')
    lines = f1.readlines()
    for line in lines:        
        if 'optimizer...' in line:
            row.append('\n')
            writer.writerow(row)
            row=[]
            row.append(file1)
            if 'test' in line:
                row.append('No')
            else:
                row.append('Yes')
        if 'seed: ' in line:
            row.append(line.replace('seed: ', ''))
            #row.append(line)
        if 'warm_start_epochs: ' in line and not 'warm_start_epochs: 0' in line:
            row.append(line.replace('warm_start_epochs: ',''))
            #row.append(line)
        if 'masking_interval: ' in line:
            row.append(line.replace('masking_interval: ',''))
            #row.append(line)
        if 'instantenous: ' in line:
            row.append(line.replace('instantenous: ',''))
            #row.append(line)
        if 'train_portion: ' in line and not '1.0' in line:
            row.append(line.replace('train_portion: ',''))
            #row.append(line)
        if 'cifar10: ' in line:
            row.append(line.replace('cifar10: ',''))
            #row.append(line)
        if 'cifar100: ' in line:
            row.append(line.replace('cifar100: ',''))
            #row.append(line)
        if 'ImageNet16-120: ' in line:
            row.append(line.replace('ImageNet16-120: ',''))
            #row.append(line)
        if '(Metric.VAL_ACCURACY):' in line:
            row.append(substring_after(line, 'Metric.VAL_ACCURACY): '))
            #row.append(line)
            #writer.writerow(row)
            #writer.write('\n')
            #row=[]
    f1.close()
f.close()
