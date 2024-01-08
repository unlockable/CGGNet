import csv
import os

with open('results_normal_rnn.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        #print(row[1]) # 1 name
        if "reentrancy" in row[9] :
            os.remove(row[0])
        # Process each row
