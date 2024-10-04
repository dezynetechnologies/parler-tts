import pandas as pd
import os
import librosa
import numpy as np

# Load your CSV data, skipping the header
data = pd.read_csv('en_indian_accent_train_data.csv', skiprows=1)

# Assign column names for better readability
data.columns = ['filename','text','up_votes','down_votes','age','gender','accent','duration']

def read_audio(file_path):
    if os.path.exists(file_path):
        # with open(file_path, 'rb') as audio_file:
        #     return audio_file.read()
        audio_data, sampling_rate = librosa.load(file_path, sr=None)  # sr=None to keep original sampling rate
        print(sampling_rate)
        return {'path': file_path, 'array': np.array(audio_data), 'sampling_rate': sampling_rate}
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

# Function to create the struct-like dictionary for the 'audio' column
def create_audio_struct(row):
    # binary_data = read_audio(row['filename'])
    # filename = row['filename']  # Use the 'filename' column directly
    # return {'array': binary_data, 'path': filename, 'sampling_rate': 24000}
    return read_audio(row['filename'])

# Add a new column 'audio' containing the struct-like data
data['audio'] = data.apply(create_audio_struct, axis=1)

# # Add a new column 'audio' containing the audio data
# data['audio'] = data['filename'].apply(read_audio)

# Save the modified DataFrame to a Parquet file
data.to_parquet('en_indian_accent_train.parquet', index=False)
