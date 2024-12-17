import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file
df = pd.read_csv('features_akaze.csv', header=None)

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

# Normalize the data (excluding the first column if it's a label)
df.iloc[:, 1:44] = scaler.fit_transform(df.iloc[:, 1:44])

# Save the normalized data back to a CSV file
df.to_csv('preprocess/norm_features_akaze.csv', header=False, index=False)

print("Normalization complete. Saved to 'norm_features_akaze.csv'.")