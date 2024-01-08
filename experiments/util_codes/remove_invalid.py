# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 22:38:33 2023

@author: HSJ
"""

import os

f = open("invalid_data.txt", "r")
invaild_files = f.readlines()

for file in invaild_files :
    file_name = file.split(":")[0]
    print(file_name)
    
