# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:32:49 2023

@author: HSJ
"""

import os
import shutil
import random

src_folder = "./normal_path_rnn2"
dst_folder = "./normal_path_rnn2_2"

files = [f for f in os.listdir(src_folder) if f.endswith(".sol")]

number_of_files_to_copy = 2689

for _ in range(min(number_of_files_to_copy, len(files))) :
    file_to_copy = random.choice(files)
    src_file_path = os.path.join(src_folder, file_to_copy)
    dst_file_path = os.path.join(dst_folder, file_to_copy)

    print(dst_file_path)
    shutil.copy(src_file_path, dst_file_path)
    os.remove(src_file_path)
    files.remove(file_to_copy)