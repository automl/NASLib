#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd 
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np


# In[2]:


def get_statistics(filepath):

    train_accuracy = []
    test_accuracy = []
    val_accuracy = []
    
    epoch = 0
    for e in summary_iterator(filepath):
        for v in e.summary.value:
            if v.tag == "Test accuracy (queried from benchmark)":
                test_accuracy.append(v.simple_value)
                #test_accuracy.append((e.step, v.simple_value))
            elif v.tag == "Validation accuracy (top 1)":
                val_accuracy.append(v.simple_value)
            elif v.tag == "Train accuracy (top 1)":
                train_accuracy.append(v.simple_value)

    return {"train_acc": train_accuracy, "val_acc": val_accuracy, "test_accuracy": test_accuracy}


# In[3]:


def add_incumbent_column(df, col):
    new_col = f"incumbent_{col}"
    df[new_col] = df[col]

    max_acc = -1

    for idx in df.index:
        acc = df[col][idx]
        if acc > max_acc:
            max_acc = acc

        df[new_col][idx] = max_acc

    return df
    

def get_run_data_as_df(filepath, parse_filename):
    
    opts = parse_filename(filepath)
    run_data = get_statistics(filepath)
    
    data = {**run_data, **opts}

    n_epochs = max([len(data[item]) for item in ['train_acc', 'val_acc', 'test_accuracy']])

    for item in ['train_acc', 'val_acc', 'test_accuracy']:
        if len(data[item]) == 0:
            data[item] = [-1] * n_epochs

    df = pd.DataFrame(data=data)
    df['epochs'] = df.index

    add_incumbent_column(df, 'test_accuracy')
    add_incumbent_column(df, 'val_acc')

    return df


# In[4]:


def find_filename_with_string(string, files):
    for idx, s in enumerate(files):
        if string in s:
            return s

    return None

def get_event_files(root):
    event_files = []

    for root, dirs, files in os.walk(root, topdown=False):
        file = find_filename_with_string('events.out.tfevents', files)

        if file is not None:
            event_files.append(os.path.join(root, file))

    return event_files


# In[5]:


def get_all_runs_as_csv(root):
    tf_files = get_event_files(root)
    
    def parse_filename(filename):
        file = filename.replace(root, "")
        print(file)
        benchmark, dataset, optimizer, seed, _ = file.split('/')

        return {
            "benchmark": benchmark,
            "dataset": dataset,
            "optimizer": optimizer,
            "seed": seed
        }

    all_dfs = []

    for tf_file in tf_files:
        all_dfs.append(get_run_data_as_df(tf_file, parse_filename))
    
    dataframe = pd.concat(all_dfs)
    return dataframe


# In[6]:

if __name__ == "__main__":
    df = get_all_runs_as_csv("/home/mehtay/research/NASLib/naslib/benchmarks/bbo/run/")
    df.to_csv('runs.csv')