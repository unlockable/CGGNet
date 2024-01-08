# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:12:24 2024

@author: HSJ
"""

from glob import glob
from tqdm import tqdm
import os
import hashlib

def sha256sum(filename):
    h  = hashlib.sha256()
    with open(filename, 'rb') as file:
        for block in iter(lambda: file.read(4096), b''):
            h.update(block)
    return h.hexdigest()


folder_name = "./normal_target"
copy_folder_name = "./unique_files"
folders = glob(folder_name+"/*")
print(folders)
unique_files = set()

versions = {}
for folder in tqdm(folders) :
    files = glob(folder + "/*")
    
    for file in files :
        
        file_hash = sha256sum(file)
        if file_hash not in unique_files:
            unique_files.add(file_hash)
        
        '''
        f = open(file)
        lines = f.readlines()
        
        
        
        for line in lines:
            version = line.split(" ")[2]
            if version == '0' :
                print(file)
                
            if version not in versions :
                versions[version] = 0
            versions[version] += 1
            break
        '''
        


print(f"Number of unique files: {len(unique_files)}")