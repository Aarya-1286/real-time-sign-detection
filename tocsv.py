import pickle
import pandas as pd

# Load the Pickle file
with open("data.pickle", "rb") as f:
    data = pickle.load(f)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("filename.csv", index=False)




# opencv-python==4.7.0.68
# mediapipe==0.9.0.1
# scikit-learn==1.2.0