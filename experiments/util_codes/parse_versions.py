# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 22:01:17 2024

@author: HSJ
"""

from glob import glob

f = open("./solidity_codev3.txt")

lines = f.readlines()

f.close()

new_f = open("./versions.txt", "w")

for line in lines :
    version = line.split(" ")[2].replace(";", "")
    #print(line.split(" ")[2].replace(";", ""))
    new_f.write(version+"\n")

new_f.close()