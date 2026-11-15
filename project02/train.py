import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

def main():
    df = pd.read_csv("poker-hand-training.csv")
    y = df.iloc[:, -1]

    for cards in (3, 4):
        X = df.iloc[:, :cards * 2]

        model = LogisticRegression(max_iter=3000)
        model.fit(X, y)

        filename = f"poker-model{cards}.sav"
        with open(filename, "wb") as f:
            pickle.dump(model, f)

        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
