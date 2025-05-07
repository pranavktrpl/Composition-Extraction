import pickle
import os

# Get list of pickle files in current directory
pickle_files = [f for f in os.listdir() if f.endswith('.pkl')]

if pickle_files:
    print(f"Found pickle files: {pickle_files}")
    for pkl_file in pickle_files:
        print(f"\nLoading {pkl_file}...")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                print(f"Contents of {pkl_file}:")
                print(data)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
else:
    print("No pickle files found in current directory")
