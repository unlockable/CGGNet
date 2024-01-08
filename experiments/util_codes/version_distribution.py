import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

# Standalone parse_version function
def parse_version(version):
    return tuple(map(int, version.split('.')))

f = open("./versions.txt")

lines = f.readlines()

versions = []
for line in lines :
    versions.append(line.replace("\n",""))
    
# Count the frequency of each version
version_counts = Counter(versions)

# Sort versions and their counts
sorted_versions = sorted(version_counts.items(), key=lambda x: parse_version(x[0]))

# Split the sorted versions and their counts for plotting
sorted_version_numbers, sorted_counts = zip(*sorted_versions)

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(sorted_version_numbers, sorted_counts, color='grey')
plt.xlabel('Version Number', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()  # Adjust layout to prevent clipping of ylabel
plt.show()
