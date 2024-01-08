import os
import shutil
import hashlib
from glob import glob
from tqdm import tqdm 

def sha256sum(filename):
    h  = hashlib.sha256()
    with open(filename, 'rb') as file:
        for block in iter(lambda: file.read(4096), b''):
            h.update(block)
    return h.hexdigest()

folder_name = "./target"
folders = glob(folder_name+"/*")


destination_directory = './unique_files'
unique_hashes = set()

for folder in tqdm(folders) :
    source_directory = folder

    for filename in os.listdir(source_directory):
        source_filepath = os.path.join(source_directory, filename)
        if os.path.isfile(source_filepath):
            file_hash = sha256sum(source_filepath)
            if file_hash not in unique_hashes:
                unique_hashes.add(file_hash)
                destination_filepath = os.path.join(destination_directory, file_hash)
                shutil.copy2(source_filepath, destination_filepath)

print(f"Copied {len(unique_hashes)} unique files.")
