from mltest import loadCRPFromCSV, arbiterFeatureTransform, train

X, y = loadCRPFromCSV("challenge-response-pairs.csv")

Xtrans = arbiterFeatureTransform(X)

model, t = train(Xtrans, y)

yPred = model.predict(Xtrans)

acc = (yPred == y).mean()
print(f"Training accuracy: {acc:.4f}")
print(f"Training time: {t:2f} seconds")

