import pickle
import numpy as np

# Load the Pickle file
pickle_file_path = 'path_to_your_pickle_file.pkl'  # Replace with your actual file path

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Access the data from the dictionary
data = data_dict['data']

# Check for inconsistent lengths
inconsistent_entries = []
expected_length = len(data[0])  # Assuming the first entry has the correct length

for i, entry in enumerate(data):
    if len(entry) != expected_length:
        inconsistent_entries.append((i, len(entry)))

# Print out inconsistent entries
if inconsistent_entries:
    print("Inconsistent entries found:")
    for index, length in inconsistent_entries:
        print(f"Index: {index}, Length: {length}")
else:
    print("All entries are consistent.")

