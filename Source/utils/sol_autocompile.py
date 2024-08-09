import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import subprocess

def get_version(sol_file_path):
    with open(sol_file_path, 'r') as contractfile:
        solidity_code = contractfile.read()

    pragma_match = re.search(r'pragma solidity\s+(.*?);', solidity_code)
    if pragma_match:
        pragma_version = pragma_match.group(1)
    
        version_match = re.search(r'\^?(\d+\.\d+\.\d+)', pragma_version)
        if version_match:
            solidity_version = version_match.group(1)
        else:
            #print("Solidity version not found in pragma statement")
            return -1
    else:
        #print("Pragma statement not found in the Solidity file")
        return -1
    
    return solidity_version


def sol_ast_compile(sol_file_path):
    try:
        contract_version = get_version(sol_file_path)
    
        if contract_version == -1:
            print("versioning error")
            return -1
        
        solc_base_path = './artifacts/'
        cmd = solc_base_path + "solc-" + contract_version + "/solc-" + contract_version + ".exe " + sol_file_path + " --ast-compact-json"+  " -o ./asts/"
        #os.system(cmd)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as ex :
        print("Error ", ex)
        


def sol_compile(sol_file_path):
    try:
        contract_version = get_version(sol_file_path)
        
        if contract_version == -1:
            return 0.001
        
        solc_base_path = './artifacts/'
        
        #cmd = os.path.join(solc_base_path, f"solc-{contract_version}/solc-{contract_version}") + f" {sol_file_path} -o ./bins/{os.path.basename(sol_file_path).replace('.sol', '')} --bin"
        cmd = os.path.join(solc_base_path, f"solc-{contract_version}/solc-{contract_version}") + f" {sol_file_path} --bin"
        #cmd = solc_base_path + "solc-" + contract_version + "/solc-" + contract_version + " " + sol_file_path + " --bin"
        # Execute the command
        #print(cmd)
        result = subprocess.run(cmd.split(" "), capture_output=True)
        #result = subprocess.run(cmd, shell=True, capture_output=True)
        
        # Check the return code
        if result.returncode == 0:
            return 1
        else:
            #print(f"Command execution failed for {sol_file_path} with return code {result.returncode}.")
            #print("Error output:", result.stderr.decode())
            return 0.01
    except Exception as ex:
        print(f"Error compiling {sol_file_path}: ", ex)
        return -1

# Extract the number from the filename using a regular expression
def extract_number(filename):
    match = re.search(r'_ADV_\d+_(\d+).txt$', filename)
    if match:
        return int(match.group(1))
    return 0


def parallel_compile(base_dir):
    contract_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
    #sorted_filenames = sorted(contract_files)
    # Sort the filenames based on the extracted number
    sorted_filenames = sorted(contract_files, key=extract_number)
    #print(sorted_filenames)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        compile_result = list(executor.map(sol_compile, sorted_filenames))

    return compile_result

def main() :
    path = ""
    sol_compile()

if __name__ == '__main__':
    main()
