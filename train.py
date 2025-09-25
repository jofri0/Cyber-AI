import pandas as pd

def train():
    df = pd.read_csv("training_data.csv")
    print("Loaded training data:")
    print(df.head())
    print("Mock training complete. (You can replace this with real ML pipeline).")

if __name__ == "__main__":
    train()
