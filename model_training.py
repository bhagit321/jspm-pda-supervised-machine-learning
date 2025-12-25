import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 3. Save the model to a file
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")