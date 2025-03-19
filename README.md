# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load & Split Data → Load Iris dataset, create DataFrame, split into train & test sets.
2. Train Model → Initialize SGDClassifier, train on X_train, y_train.
3. Predict & Evaluate → Predict y_test, compute accuracy & confusion matrix.
4. Visualize → Plot confusion matrix using Seaborn heatmap.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Jayakumar B
RegisterNumber: 212223040073 
*/
```
```PY
# Importing Required Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the Iris Dataset and Creating a DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

# Splitting Features and Target Variables
X = df.drop('target', axis=1)
y = df['target']

# Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and Training the Classifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

# Making Predictions
y_pred = sgd_clf.predict(X_test)

# Calculating Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Computing and Displaying the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plotting the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

```
## Output:
df.head()

![image](https://github.com/user-attachments/assets/5f65233e-65b6-46de-aa5a-6077f2204131)

Accuracy:

![image](https://github.com/user-attachments/assets/d8f39ae4-19ef-48c4-9e14-cad712716967)

Confusion Matrix:

![image](https://github.com/user-attachments/assets/6cf03b9e-f547-413a-b179-09b854692b18)

![image](https://github.com/user-attachments/assets/11c1be9a-6f72-4c76-8d47-5d3f7052ac7b)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
