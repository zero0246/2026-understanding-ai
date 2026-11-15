## ---------------- FILL HERE ---------------------------------


import pickle
import pandas as pd

with open("poker-model3.sav", "rb") as f:
    model3 = pickle.load(f)

with open("poker-model4.sav", "rb") as f:
    model4 = pickle.load(f)

def predict_call(hand, opposite_score):
    inp = []
    for suit, rank in hand:
        inp.extend([suit, rank])

    if len(hand) == 3:
        selected_model = model3
    elif len(hand) == 4:
        selected_model = model4
    else:
        raise ValueError("hand must contain 3 or 4 cards")

    X_test = pd.DataFrame(
        [inp],
        columns=selected_model.feature_names_in_
    )

    probabilities = selected_model.predict_proba(X_test)[0]

    expected_score = sum(
        score * probability
        for score, probability in zip(
            selected_model.classes_, probabilities
        )
    )

    return bool(expected_score >= opposite_score)


## -------------------------------------------------

import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import random

test = pd.read_csv('poker-hand-testing.csv')

money = [1000] * 10
for i in range(10):
    samp = np.random.randint(0, len(test), size=1000)
    X_test = test.iloc[samp, :-1].to_numpy()
    y_test = test.iloc[samp, -1].to_numpy()

    for test_hand, test_score in zip(X_test, y_test):
        test_input = [ (suit, rank) for suit, rank in zip(test_hand[::2], test_hand[1::2]) ]
        oppo_score = y_test[random.randrange(len(y_test))]

        if not predict_call(test_input[:3], oppo_score):
            money[i] -= 1
            continue

        if not predict_call(test_input[:4], oppo_score):
            money[i] -= 2
            continue

        if test_score > oppo_score:
            money[i] += 3

        elif test_score < oppo_score:
            money[i] -= 3
        # if tie, your money stays the same

print('average of money= {}'.format(round(np.mean(money), 3)))