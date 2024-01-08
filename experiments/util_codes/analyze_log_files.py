# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:14:35 2024

@author: HSJ
"""

from glob import glob

log_files = glob("./results_data/logs/*.log")

rnn_loss = []
gru_loss = []
lstm_loss = []

rnn_loss_avg = {}
gru_loss_avg = {}
lstm_loss_avg = {}

rnn_com_num = []
gru_com_num = []
lstm_com_num = []


rnn_com_avg = {}
gru_com_avg = {}
lstm_com_avg = {}

total_epoch = 30
current_epoch = total_epoch

for log_file in log_files :
    f = open(log_file)
    lines = f.readlines()
    flag = False
    idx = 0
        
    current_epoch = -1
    compile_num = 0
    
    for line in lines :
        if "ADV EPOCH" in line :
            flag = True            
            current_epoch += 1
        if "the number of compilable_tokens" in line :
            compile_num = int(line.split(":")[1])
                
        if "self.adv_dataloader num" in line and flag == True :                       
            #print(line)
                
            if "rnn" in log_file :
                rnn_com_num.append(compile_num)
            elif "gru" in log_file :
                gru_com_num.append(compile_num)
            else :
                lstm_com_num.append(compile_num)
                                

        if "[MLE-GEN] epoch" in line and flag == True :
            #print(line)
            loss = float(line.split("=")[1].replace("\n", "").replace(" ", ""))
            idx += 1
            if compile_num == 0:
                if "rnn" in log_file :
                    rnn_loss.append(0)
                    rnn_com_num.append(0)
                elif "gru" in log_file :
                    gru_loss.append(0)
                    gru_com_num.append(0)
                else :
                    lstm_loss.append(0)   
                    lstm_com_num.append(0)
                continue
            
            if idx % 2 == 0 :
                if "rnn" in log_file :
                    rnn_loss.append(loss)
                elif "gru" in log_file :
                    gru_loss.append(loss)
                else :
                    lstm_loss.append(loss)
                    
for i in range(total_epoch) :
    rnn_loss_avg[i] = 0
    gru_loss_avg[i] = 0
    lstm_loss_avg[i] = 0
    

for i in range(len(rnn_loss)) :
    rnn_loss_avg[i % total_epoch] += rnn_loss[i]
    gru_loss_avg[i % total_epoch] += gru_loss[i]
    lstm_loss_avg[i % total_epoch] += lstm_loss[i]
            
for i in range(total_epoch) :
    rnn_loss_avg[i] /= 12
    gru_loss_avg[i] /= 12
    lstm_loss_avg[i] /= 12

for i in range(total_epoch) :
    rnn_com_avg[i] = 0
    gru_com_avg[i] = 0
    lstm_com_avg[i] = 0

for i in range(len(rnn_loss)) :
    rnn_com_avg[i % total_epoch] += rnn_com_num[i]
    gru_com_avg[i % total_epoch] += gru_com_num[i]
    lstm_com_avg[i % total_epoch] += lstm_com_num[i]
    
for i in range(total_epoch) :
    rnn_com_avg[i] /= 12
    gru_com_avg[i] /= 12
    lstm_com_avg[i] /= 12
             

print((rnn_com_avg))
print((gru_com_avg))
print((lstm_com_avg))

print(len(rnn_com_num))
print(len(gru_com_num))
print(len(lstm_com_num))

print(len(rnn_loss))
print(len(gru_loss))
print(len(lstm_loss))

print((rnn_loss_avg))
print((gru_loss_avg))
print((lstm_loss_avg))