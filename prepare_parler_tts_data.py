# import pandas as pd

# # Step 1: Load the CSV file
# file_path = '/media/mirae/NewVolume/Downoalds HDD/archive/cv-valid-test.csv'  # Replace with your CSV file path
# df = pd.read_csv(file_path)

# # Step 2: Filter the DataFrame
# # Specify the column name and the value you're looking for
# column_name = 'accent'  # Replace with the actual column name
# value_to_filter = 'indian'  # Replace with the value you're filtering by

# # Create a new DataFrame with only the rows that match the condition
# filtered_df = df[df[column_name] == value_to_filter]

# # Optional: Reset the index of the new DataFrame
# filtered_df.reset_index(drop=True, inplace=True)

# # Step 3: Display the new DataFrame or save it
# # print(filtered_df)
# # Optionally save to a new CSV
# filtered_df.to_csv('indian_accent_eng_test_data.csv', index=False)  # Save if needed




data_path = "/media/mirae/NewVolume/Projects/ParlerTTS-fork/parler-tts/training_data/indian_accent_eng_train"


import pandas as pd

data = pd.read_csv('indian_accent_eng_train_data.csv', header=None)

old_path = 'cv-valid-train'

# Replace the old path in the first column
data[0] = data[0].str.replace(old_path, data_path)

# Save the modified DataFrame back to CSV
data.to_csv('en_indian_accent_train_data.csv', header=False,   index=False)

