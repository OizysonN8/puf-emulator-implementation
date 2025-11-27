import numpy as np
import pandas as pd
import argparse
import os
import time
import math
import json
import pickle
from typing import Tuple, List, Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

# import matplotlib.pyplot as plt

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
    s = np.binary_repr(int(x), width=width)
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



# Evaluate Model accuracy and efficacy 
def evalModel(clf, xTest: np.ndarray, yTest: np.ndarray) -> Dict[str, Any]:
    t0 = time.time()
    yPred = clf.predict(xTest)
    predictTime = time.time() - t0

    metrics = {
        "accuracy": float(accuracy_score(yTest, yPred)),
        "balancedAccuracy": float(balanced_accuracy_score(yTest, yPred)),
        "f1": float(f1_score(yTest, yPred, zero_division=0)),
        "confusion": confusion_matrix(yTest, yPred).tolist(),
        "predictTimeSec": predict_time,
    }

    try:
        if len(np.unique(yTest)) == 2:
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(xTest)[:,1]
                metrics["roc_auc"] = float(roc_auc_score(yTest, probs))
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(xTest)
                metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
    except Exception:
        metrics["roc_auc"] = None

    return metrics            


#### Model training :O

# Training time helper function
def trainTime(clf, xTrain: np.ndarray, yTrain: np.ndarray):
    t0 = time.time()

    clf.fit(xTrain, yTrain)
    trainTime = time.time() - t0
    return clf, trainTime


def runLearningCurveExp(
    x: np.ndarray,
    y: np.ndarray,
    transform_fn,
    modelsDict: Dict[str, Any],
    trainFracs: List[float],
    numRepeat: int = 5,
    testSize: float = 0.3,
    randomSeed: int = 0,
):

    rng = np.random.RandomState(randomSeed)

    results = []
    sss = StratifiedShuffleSplit(numSplits=numRepeat, testSize=testSize, random_state=randomSeed)
    splits = list(sss.split(x,y))

    for frac in trainFracs:
        for iSplit, (trainIdx, testIdx) in enumerate(splits):
            nTrainFull = len(trainIdx)
            nSub = max(2, int(round(frac * nTrainFull)))


            if nSub >= nTrainFull:
                subSampleIdx = trainIdx
            else:

                sss2 = StratifiedShuffleSplit(n_splits=1, train_size=numSub, random_state=rng.randint(0, 2**31-1))

                xTemp = x[trainIdx]
                yTemp = y[trainIdx]

                for splitTrainIdx, _ in sss2.split(xTemp, yTemp):
                    subSampleIdx = trainIdx[splitTrainIdx]

            xTrain = x[subSampleIdx]
            yTrain = y[subSampleIdx]
            xTest = x[testIdx]
            yTest = y[testIdx]

            for model_name, model_ctor in modelsDict.items():
                #fresh model
                clf = model_ctor() 
                try: 
                    clf, trainingT = trainTime(clf, xTrain, yTrain)
                except Exception as e:
                    print(f"Training has failed.. failed for model {model_name} at frac {frac} on reapeat {iSplit}: {e}")
                    continue
                metrics = evalModel(clf, xTest, yTest)
                row = {
                    "model": model_name,
                    "train_frac": frac,
                    "repeat": iSplit,
                    "n_train": len(yTrain),
                    "n_test": len(yTest),
                    "train_time_s": trainTime,
                    **metrics,
                }
                results.append((row, clf))


    # display results in the array
    rows = [r for (r, clf) in results]
    return rows, results



