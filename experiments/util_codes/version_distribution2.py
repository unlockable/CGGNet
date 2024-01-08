import matplotlib.pyplot as plt

# Data provided by the user
data = {
    '0.4.19': 36380, '0.5.1': 12385, '0.4.23': 54701, '0.5.7': 14599, '0.4.25': 88010,
    '0.4.18': 58790, '0.4.20': 20053, '0.5.4': 13077, '0.5.2': 19738, '0.4.24': 172725,
    '0.5.8': 12907, '0.4.17': 21247, '0.4.13': 10614, '0.4.22': 15449, '0.4.26': 9306,
    '0.4.12': 2574, '0.5.9': 8876, '0.4.16': 22374, '0.4.21': 36348, '0.5.0': 11120,
    '0.5.5': 5584, '0.5.10': 5689, '0.5.3': 8323, '0.5.6': 7632, '0.4.14': 3131
}

# Custom function to sort version numbers
def sort_version(version):
    major, minor, patch = map(int, version.split('.'))
    return major, minor, patch

# Sort the data by version number using the custom function
sorted_data = dict(sorted(data.items(), key=lambda item: sort_version(item[0])))

# Creating the plot with sorted data
plt.figure(figsize=(12, 6))
plt.bar(sorted_data.keys(), sorted_data.values(), color='Indigo')

# Adding titles and labels
plt.xlabel('Version Number', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)

# Save the plot as a PNG file
plt.savefig('/mnt/data/version_distribution_sorted.png')

# Show the plot
plt.show()
