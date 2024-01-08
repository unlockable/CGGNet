# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:12:24 2024

@author: HSJ
"""

from glob import glob
from tqdm import tqdm
import os
import hashlib

log_files = glob("./logs/*.log")

rnn_training_time = []
rnn_generation_time = []
rnn_post_processing_time = []
rnn_compile_time = []

gru_training_time = []
gru_generation_time = []
gru_post_processing_time = []
gru_compile_time = []

lstm_training_time = []
lstm_generation_time = []
lstm_post_processing_time = []
lstm_compile_time = []

for log_file in log_files :
    f = open(log_file)
    lines = f.readlines()

    for line in lines :
        if "consumed training time" in line :
            training_time = float(line.split(":")[1].replace("\n", "").replace(" ", ""))
        elif "consumed generation_time" in line :
            generation_time = float(line.split(":")[1].replace("\n", "").replace(" ", ""))
        elif "consumed post_processing_time" in line :
            post_processing_time = float(line.split(":")[1].replace("\n", "").replace(" ", ""))
        elif "consumed compile_time" in line :
            compile_time = float(line.split(":")[1].replace("\n", "").replace(" ", ""))
            
    
    if 'rnn' in log_file :
        rnn_training_time.append(training_time)
        rnn_generation_time.append(generation_time)
        rnn_post_processing_time.append(post_processing_time)
        rnn_compile_time.append(compile_time)
    
    elif 'gru' in log_file :
        gru_training_time.append(training_time)
        gru_generation_time.append(generation_time)
        gru_post_processing_time.append(post_processing_time)
        gru_compile_time.append(compile_time)
        
    elif 'lstm' in log_file :
        lstm_training_time.append(training_time)
        lstm_generation_time.append(generation_time)
        lstm_post_processing_time.append(post_processing_time)
        lstm_compile_time.append(compile_time)
        
print("rnn_training_time:", sum(rnn_training_time))
print("rnn_generation_time:",sum(rnn_generation_time))
print("rnn_post_processing_time:",sum(rnn_post_processing_time))
print("rnn_compile_time:",sum(rnn_compile_time))

print("gru_training_time:",sum(gru_training_time))
print("gru_generation_time:",sum(gru_generation_time))
print("gru_post_processing_time:",sum(gru_post_processing_time))
print("gru_compile_time:",sum(gru_compile_time))


print("lstm_training_time:",sum(lstm_training_time))
print("lstm_generation_time:",sum(lstm_generation_time))
print("lstm_post_processing_time:",sum(lstm_post_processing_time))
print("lstm_compile_time:",sum(lstm_compile_time))