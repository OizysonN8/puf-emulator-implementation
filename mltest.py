import numpy as np
import pandas as pd
import argparse
import os
import time
import pickle
from typing import Tuple, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# to answer Prof's question on our presentation, the main model here is Logistic Regrssion
# it will learn a mapping from challenge bits to PUF responses
#but i will try to made other models supported as well.. we will see


## used for testing if PyPUF emulator isn't working or ready yet
try:
    from pypuf import simulation as pfsim
    from pypuf import io as pfio
    pypufAvaliable = True
except Exception:
    pypufAvaliable = False



### utility helper functions 

# function used to transform the integer given in the challenge into a binary bit array for the ML
def int2Bitarray(x: int, width: int) -> np.ndarray:
    s = np.binaryRepr(int(x), width=width)
    return np.array([int(ch) for ch in s], dtype=np.int8)

# loads .csv files that contain the challenge response pairs
# returns the challenges as bit array, responses as vector of 0/1 ints
def loadCRPFromCSV(path: str, challengeWidth: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    # run our convenient helper function
    ints = df["Challenge"].astype(np.int64).values
    challenges = np.vstack([
        int2Bitarray(x, challengeWidth) for x in ints
    ]).astype(np.int8)

    # ensure response type is int8 and shove into var
    responses = df["Response"].astype(np.int8).values
    
    return challenges, responses


## This function converts raw challenge bits into a format which makes the ML able to linearly learn arbiter PUF behavior, and imporivng Logistic regression accuracu
def arbiterFeatureTransform(challenges: np.ndarray, k: int = 1) -> np.ndarray:
    ch = challenges
    # convert 0s and 1s to -1s and +1s for PyPuf format
    if ch.max() <= 1:
        ch_pm = (ch * 2) - 1
    else:
        ch_pm = ch

    if pypufAvaliable:
        try:
            transformed = pfsim.ArbiterPUF.transform_atf(ch_pm, k=k)
            if transformed.ndim == 3:
                transformed = transformed[:, 0, :]
            return transformed.astype(np.float32)
        except Exception as e:
            print("PyPUF unavaliable; Continuing forward using raw bits instead.  Error:", e)

    # model will still work if mypuf not avaliable, although not ideal
    return challenges.astype(np.float32)



# Model training :O
def train(xTrain: np.ndarray, yTrain: np.ndarray):
    # can train and return a logistic regression model, plus the elapsed time of trianing
    clf = LogisticRegression(max_iter=500, solver='lbfgs')
    t0 = time.time()
    clf.fit(xTrain, yTrain)
    t = time.time() - t0
    return clf, t
