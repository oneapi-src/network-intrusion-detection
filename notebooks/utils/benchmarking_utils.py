# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
"""
Process raw dataset for experiments
"""

import os
from tqdm import tqdm
from contextlib import redirect_stdout
import sys
sys.path.append("../../src")
import argparse
from importlib import reload
import logging
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from subprocess import CalledProcessError, call


def run_training_benchmark(intel=False, iterations = 1):
    if intel:
        subfolder_name = 'intel'
    else:
        subfolder_name = 'stock'
        
    test_params = [
        ["10000",10000],
    ]

    for k in tqdm(range(1,iterations+1), desc='iteration'):
        for foldername, datasize in tqdm(test_params, desc='experiment'):
            os.makedirs(f"./logs/{foldername}/{subfolder_name}", exist_ok=True)

            logfile = f"./logs/{foldername}/{subfolder_name}/performance.log"
            fairness_logfile = f"./logs/{foldername}/{subfolder_name}/fairness.log"

            for j in tqdm(range(1,5), desc='train'):
                reload(logging)
                if intel:
                    command = f"python -m sklearnex src/run_benchmarks.py -i 1 -t 1 -d {datasize} --algo nusvc -c ./data/data.csv -l {fairness_logfile}"
                else:
                    command = f"python src/run_benchmarks.py -t 1 -d {datasize} --algo nusvc -c ./data/data.csv -l {fairness_logfile}"
                
                with open(logfile, 'a') as file:
                    try:
                        call(command, shell=True, stdout=file)
                    except CalledProcessError as e:
                        print(e.output)



def run_inference_benchmark(intel=False, iterations = 1):
    if intel:
        subfolder_name = 'intel'
    else:
        subfolder_name = 'stock'
        
    test_params = [
        ["10000",10000],
    ]

    for k in tqdm(range(1,iterations+1), desc='iteration'):
        for foldername, datasize in tqdm(test_params, desc='experiment'):
            os.makedirs(f"./logs/{foldername}/{subfolder_name}", exist_ok=True)

            logfile = f"./logs/{foldername}/{subfolder_name}/inference.log"
            fairness_logfile = f"./logs/{foldername}/{subfolder_name}/inference_fairness.log"

            for j in tqdm(range(1,5), desc='inference'):
                reload(logging)
                if intel:
                    command = f"python -m sklearnex src/inference.py --i 1 --modelpath models/NUSVC_model_hp.sav -c ./data/data.csv -d 10000 -l {fairness_logfile}"
                else:
                    command = f"python src/inference.py --modelpath models/NUSVC_model_hp.sav -c ./data/data.csv -l {fairness_logfile} -d 10000"
                                
                with open(logfile, 'a') as file:
                    try:
                        call(command, shell=True, stdout=file)
                    except CalledProcessError as e:
                        print(e.output)


def load_results_dict_training():
    results_dict = defaultdict(dict)
    subfolder_names = {'stock':'stock','intel':'intel'}
    foldernames = {'10000':'10000'}
    for experiment_n, foldername in enumerate(foldernames.keys()):
        for increment_n in range(1,4):
            results_dict['Experiment'][experiment_n * 3 + increment_n] = experiment_n + 1
            for subfolder_name in subfolder_names.keys():
                logfile = f"./logs/{foldername}/{subfolder_name}/performance.log"
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                filtered_lines = [line for line in lines if line.find('NUSVC training time with best params is') != -1]

                results_dict['Data Size'][experiment_n * 3 + increment_n] = foldernames[foldername]
                results_dict['Round'][experiment_n * 3 + increment_n] = f'{increment_n}'

                start = (increment_n - 1) * 4
                end = start + 4 

                time = np.mean([float(re.findall("\d+.\d+",filtered_lines[i])[0]) for i in range(start,end)])
                results_dict[subfolder_names[subfolder_name]][experiment_n * 3 + increment_n] = time
            stock_time = results_dict[subfolder_names['stock']][experiment_n * 3 + increment_n]
            stock_intel = results_dict[subfolder_names['intel']][experiment_n * 3 + increment_n]
            results_dict['Intel speedup over stock'][experiment_n * 3 + increment_n] = stock_time / stock_intel
    return results_dict

def load_results_dict_inference():
    results_dict = defaultdict(dict)
    subfolder_names = {'stock':'stock','intel':'intel'}
    foldernames = {'10000':'10000'}
    for experiment_n, foldername in enumerate(foldernames.keys()):
        for increment_n in range(1,4):
            results_dict['Experiment'][experiment_n * 3 + increment_n] = experiment_n + 1
            for subfolder_name in subfolder_names.keys():
                logfile = f"./logs/{foldername}/{subfolder_name}/inference.log"
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                    
                results_dict['Batch Size'][experiment_n * 3 + increment_n] = foldernames[foldername]
                results_dict['Round'][experiment_n * 3 + increment_n] = f'{increment_n}'
                
                start = (increment_n - 1) * 4
                end = start + 4

                filtered_lines = [line for line in lines if line.find('Batch Prediction time is') != -1]

                if subfolder_name == 'stock':
                    time = np.mean([float(re.findall("\d+.\d+",filtered_lines[i])[0]) for i in range(start,end)])
                    results_dict[subfolder_names[subfolder_name]][experiment_n * 3 + increment_n] = time
                else:
                    time = np.mean([float(re.findall("\d+.\d+",filtered_lines[i])[0]) for i in range(start,end)])
                    results_dict[subfolder_names[subfolder_name]][experiment_n * 3 + increment_n] = time
    
            results_dict['Intel speedup over stock'][experiment_n * 3 + increment_n] = results_dict[subfolder_names['stock']][experiment_n * 3 + increment_n] / results_dict[subfolder_names['intel']][experiment_n * 3 + increment_n]
    return results_dict

def print_inference_benchmark_table():
    df = pd.DataFrame(load_results_dict_inference())
    df = df.round(4)
    df['stock'] = df['stock'].apply(lambda x:str(x)+'s')
    df['intel'] = df['intel'].apply(lambda x:str(x)+'s')
    df['% gain:intel'] = df['Intel speedup over stock'].apply(lambda x:str(round(x-1,2))+'%')
    df['Intel speedup over stock'] = df['Intel speedup over stock'].apply(lambda x:str(x)+'x')
    return df

def print_training_benchmark_table():
    df = pd.DataFrame(load_results_dict_training())
    df = df.round(2)
    df['stock'] = df['stock'].apply(lambda x:str(x)+'s')
    df['intel'] = df['intel'].apply(lambda x:str(x)+'s')
    df['% gain'] = df['Intel speedup over stock'].apply(lambda x:str(round(x-1,2))+'&')
    df['Intel speedup over stock'] = df['Intel speedup over stock'].apply(lambda x:str(x)+'x')
    return df

def print_training_benchmark_bargraph():
    df = pd.DataFrame(load_results_dict_training())
    fig, (ax1) = plt.subplots(1,1,figsize=[14,6])
    fig.suptitle('Training Time Speed Up - NuSVC$^{\#\#}$\n\
    Intel Extension for Scikit-learn 2021.6.0 against stock Scikit-learn 1.0.2')
    x = np.arange(3)  # the label locations
    width = 0.35 
    size_list = ['10000']
    ax1.set_ylabel('Relative Speedup to Stock \n (Higher is better)')
    for i,ax in enumerate([ax1]):
        curslice = slice(i*3,(i+1)*3)
        xbg081 = round(df['stock'].iloc()[curslice]/df['stock'].iloc()[curslice],2)
        xbg151 = round(df['stock'].iloc()[curslice]/df['intel'].iloc()[curslice],2)
        rects1 = ax.bar(x - width/2, xbg081, width, label='Stock Scikit-learn 1.0.2', color='b')
        rects2 = ax.bar(x + width/2, xbg151, width, label='Intel Extension for Scikit-learn 2021.6.0', color='deepskyblue')
        ax.bar_label(rects1, labels=[str(i) + 'x' for i in xbg081], padding=3)
        ax.bar_label(rects2, labels=[str(i) + 'x' for i in xbg151], padding=3)
        ax.set_xticks(x, ['Round1','Round2','Round3'])
        ax.set_xlabel(f'Experiment {i+1} \n Data size = {size_list[i]}')
        ax.set_ylim([0, 15])
    ax1.legend()
    annotation_1 = "$^{\#\#}$Training with best hyper parameters {'gamma': 0.0001, 'kernel': 'rbf'}"
    annotation_2 = "**Train and Test dataset split ratio 70%:30%"
    ax1.annotate(f"{annotation_1}\n{annotation_2}", 
        xy = (0, -0.2),
        xycoords='axes fraction',
        ha='left',
        va="center",
        fontsize=10)

def print_inference_benchmark_bargraph():
    df = pd.DataFrame(load_results_dict_inference())
    fig, (ax1) = plt.subplots(1,1,figsize=[14,6])
    fig.suptitle('Inference Time Speed Up - NuSVC\n\
    Intel Extension for Scikit-learn 2021.6.0 against stock Scikit-learn 1.0.2')
    x = np.arange(3)  # the label locations
    width = 0.25 
    size_list = ['10000']
    ax1.set_ylabel('Relative Speedup to Stock \n (Higher is better)')
    for i,ax in enumerate([ax1]):
        curslice = slice(i*3,(i+1)*3)
        xbg081 = round(df['stock'].iloc()[curslice]/df['stock'].iloc()[curslice],2)
        xbg151 = round(df['stock'].iloc()[curslice]/df['intel'].iloc()[curslice],2)
        rects1 = ax.bar(x - width/2, xbg081, width, label='Stock Scikit-learn 1.0.2', color='b')
        rects2 = ax.bar(x + width/2, xbg151, width, label='Intel Extension for Scikit-learn 2021.6.0', color='deepskyblue')
        ax.bar_label(rects1, labels=[str(i) + 'x' for i in xbg081], padding=3)
        ax.bar_label(rects2, labels=[str(i) + 'x' for i in xbg151], padding=3)
        ax.set_xticks(x, ['Round1','Round2','Round3'])
        ax.set_xlabel(f'Experiment {i+1} \n Batch Size = {size_list[i]}')
    ax1.legend()