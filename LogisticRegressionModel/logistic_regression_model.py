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


# to answer Prof's question on our presentation, the main model here is Logistic Regrssion
# it will learn a mapping from challenge bits to PUF responses
# but i will try to made other models supported as well.. we will see


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
def loadCRPFromCSV(path: str, challengeWidth: int = 64):
    df = pd.read_csv(path)

    challenges = np.vstack([
        np.array([int(b) for b in ch], dtype=np.int8)
        for ch in df["Challenge"]
    ])

    responses = df["Response"].astype(np.int8).values
    
    return challenges, responses


## This function converts raw challenge bits into a format which makes the ML able to linearly learn arbiter PUF behavior, and imporivng Logistic regression accuracu
def arbiterFeatureTransform(challenges: np.ndarray, k: int = 1) -> np.ndarray:
    ch = challenges
    # convert 0s and 1s to -1s and +1s forPyPuf format
    if ch.size == 0:
        return ch.astype(np.float32)
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

    # model will still work if pypuf not avaliable, although not ideal
    return challenges.astype(np.float32)



# Evaluate Model accuracy and efficacy 
def evalModel(clf, xTest: np.ndarray, yTest: np.ndarray) -> Dict[str, Any]:
    t0 = time.time()
    yPred = clf.predict(xTest)
    predictTime = time.time() - t0

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(yTest, yPred)),
        "balancedAccuracy": float(balanced_accuracy_score(yTest, yPred)),
        "f1": float(f1_score(yTest, yPred, zero_division=0)),
        "confusion": confusion_matrix(yTest, yPred).tolist(),
        "predictTimeSec": float(predictTime),
    }

    try:
        # compute ROC AUC only for binary labels and when scores/probs are available
        if len(np.unique(yTest)) == 2:
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(xTest)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(yTest, probs))
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(xTest)
                metrics["roc_auc"] = float(roc_auc_score(yTest, scores))
            else:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None
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


# simple trainer function
def train(xTrain: np.ndarray, yTrain: np.ndarray):
    # can train and return a logistic regression model, plus the elapsed time of trianing
    clf = LogisticRegression(max_iter=500, solver='lbfgs')
    clf, trainingT = trainTime(clf, xTrain, yTrain)
    return clf, trainingT


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


    if len(y) == 0 or x.shape[0] == 0:
        raise ValueError("Empty dataset passed to runLearningCurveExp")
    

    results = []
    sss = StratifiedShuffleSplit(n_splits=numRepeat, test_size=testSize, random_state=randomSeed)
    splits = list(sss.split(x, y))

    for frac in trainFracs:
        for iSplit, (trainIdx, testIdx) in enumerate(splits):
            nTrainFull = len(trainIdx)
            nSub = max(2, int(round(frac * nTrainFull)))


            if nSub >= nTrainFull:
                subSampleIdx = trainIdx
            else:

                # create a small stratified sampler to pick nSub from the training indices
                xTemp = x[trainIdx]
                yTemp = y[trainIdx]

                smallSeed = int(rng.randint(0, 2**31 - 1))
                sss2 = StratifiedShuffleSplit(n_splits=1, train_size=nSub, random_state=smallSeed)

                found = False
                for splitTrainIdx, _ in sss2.split(xTemp, yTemp):
                    # map indices back to original space
                    subSampleIdx = trainIdx[splitTrainIdx]
                    found = True
                if not found:
                    # in the fallback case we'll use random choice trying to preserve class stratification
                    subSampleIdx = rng.choice(trainIdx, size=nSub, replace=False)
            xTrain = x[subSampleIdx]
            yTrain = y[subSampleIdx]
            xTest = x[testIdx]
            yTest = y[testIdx]

            # apply transform before training/testing
            X_train_feat = transform_fn(xTrain)
            X_test_feat = transform_fn(xTest)

            ## check if dict is empty
            if not modelsDict:
                print("runLearningCurveExp: modelsDict is empty, skipping.")
                return [], []
            
            # train the model on the sampled training set then eval
            for model_name, model_ctor in modelsDict.items():
                # fresh model used here
                try:
                    clf = model_ctor()
                except Exception as e:
                    print(f"runLearningCurveExp: model_ctor for {model_name} failed to construct: {e}")
                    continue

                try: 
                    clf, trainingT = trainTime(clf, X_train_feat, yTrain)
                    trainingT = float(trainingT)
                except Exception as e:
                    print(f"Training has failed.. failed for model {model_name} at frac {frac} on repeat {iSplit}: {e}")
                    continue

                try:
                    metrics = evalModel(clf, X_test_feat, yTest)
                except Exception as e:
                    print(f"Evaluation failed for model {model_name} at frac {frac} on repeat {iSplit}: {e}")
                    continue

                # results row
                row = {
                    "model": model_name,
                    "train_frac": frac,
                    "repeat": iSplit,
                    "n_train": len(yTrain),
                    "n_test": len(yTest),
                    "train_time_s": trainingT,
                    **metrics,
                }
                results.append((row, clf))


    # display results in the array
    rows = [r for (r, clf) in results]
    return rows, results



## helper functions for running and testing the experiments
## one-off
def trainEvalOnce(crpPath: str, testFrac: float = 0.2, backendKval: int = 1):
    print(f"\n=== Running a quick experiment test for: {crpPath} ===")
    X, y = loadCRPFromCSV(crpPath)
    if len(y) < 4:
        raise ValueError("We do not have enough CRP data to run this experiment. Increase dataset size and try again.")
    rawXTrain, rawXTest, yTrain, yTest = train_test_split(
        X, y, test_size=testFrac, stratify=y, random_state=1
    )
    trainX = arbiterFeatureTransform(rawXTrain, k=backendKval)
    testX = arbiterFeatureTransform(rawXTest, k=backendKval)
    clf, trainTimeVal = train(trainX, yTrain)
    metrics = evalModel(clf, testX, yTest)
    metrics['trainingTimeSecDuration'] = float(trainTimeVal)
    metrics['numTrainingSamples'] = int(len(yTrain))
    metrics['numTestSamples'] = int(len(yTest))
    
    print("Result metrics:", json.dumps(metrics, indent=2))
    return clf, metrics

## helper used to train the model on CRPs in trainPath, and eval the model on (post training) on CRPs in testPath
## Essentially measures how well a model trained on one PUF file generalizes to another PUF
def crossEval(trainPath: str, testPath: str, backendKval: int = 1):
    print(f"\n=== Cross-eval process initiated.  Training data will be based on train={trainPath} ; Testing data from test={testPath} ===")
    trainX, yTrain = loadCRPFromCSV(trainPath)
    testX, yTest = loadCRPFromCSV(testPath)

    # feature transforms
    trainXfeat = arbiterFeatureTransform(trainX, k=backendKval)
    testXfeat = arbiterFeatureTransform(testX, k=backendKval)


    clf, trainingTimeVal = train(trainXfeat, yTrain)

    metrics = evalModel(clf, testXfeat, yTest)

    ##data reporting
    metrics['trainingTimeSecDuration'] = float(trainingTimeVal)
    metrics['numTrainingSamples'] = int(len(yTrain))
    metrics['numTestSamples'] = int(len(yTest))

    print("CrossEval metrics:", json.dumps(metrics, indent=2))
    return clf, metrics




## run Learning Curve from main
if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    ## configure experiment settings
    full_trainFracs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    full_numRepeat = 5
    full_testSize = 0.3
    seed = 1

    models = {
        "dummy": lambda: DummyClassifier(strategy='stratified', random_state=seed),
        "logreg": lambda: LogisticRegression(max_iter=500, solver='lbfgs', random_state=seed)
    }

    # loading in crp files helper-
    def find_all_crp_files():
        paths = []
        for root, dirs, files in os.walk("."):
            # Skip results folder entirely
            if "lrResults" in root.lower():
                continue
            if "results" in root.lower():
                continue

            for f in files:
                name = f.lower()
                # Only accept true CRP files hopefully
                if name.endswith("crp.csv"):
                    paths.append(os.path.join(root, f))
        return paths


    filesProperlyLocated = find_all_crp_files()
    print("PUF files detected:")
    for p in filesProperlyLocated:
        print("  -", p)


    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "lrResults")
    os.makedirs(RESULTS_DIR, exist_ok=True)


    for p in filesProperlyLocated:
        print(f"\n=== Running FULL learning-curve for: {p} ===")

        X, y = loadCRPFromCSV(p)


        rows, results = runLearningCurveExp(
            x = X,
            y = y,
            transform_fn = lambda Xp: arbiterFeatureTransform(Xp, k=1),
            modelsDict = models,
            trainFracs = full_trainFracs,
            numRepeat = full_numRepeat,
            testSize = full_testSize,
            randomSeed = seed
        )


        base = os.path.basename(p).replace(".csv", "")
        out_csv = os.path.join(RESULTS_DIR, f"learning_curve_full_{base}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        ## summary creation
        df = pd.DataFrame(rows)

        summary = df.groupby(["model", "train_frac"]).agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_auc=("roc_auc", "mean"),
            std_auc=("roc_auc", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_n_train=("n_train", "mean")
        ).reset_index()

        summary_csv = os.path.join(RESULTS_DIR, f"learning_curve_summary_{base}.csv")
        summary.to_csv(summary_csv, index=False)
        print(f"Saved summary: {summary_csv}")


    specs_file = os.path.join(RESULTS_DIR, "learning_curve_specifications.txt")
    with open("LogisticRegressionModel/lrResults/learning_curve_specifications.txt", "w") as f:
        f.write("Full Learning-Curve Experiment\n")
        f.write("---------------------------------\n")
        f.write(f"trainFracs = {full_trainFracs}\n")
        f.write(f"numRepeat = {full_numRepeat}\n")
        f.write(f"testSize  = {full_testSize}\n")
        f.write(f"randomSeed = {seed}\n")
        f.write("transform = arbiterFeatureTransform(k=1)\n")
        f.write("models = DummyClassifier(stratified), LogisticRegression(lbfgs)\n")

    print("\n=== FULL RUN COMPLETE ===")
    print("All CSVs and summaries saved in /LogisticRegressionModel/lrResults/")



