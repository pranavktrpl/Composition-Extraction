import pickle

def show_pkl_contents(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print(data)

# Example usage:
show_pkl_contents('piis/train_data.pkl')