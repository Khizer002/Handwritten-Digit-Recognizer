import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

dataset=pd.read_csv("dataset.csv" )
print(f"Dataset Shape: {dataset.shape}")
pd.set_option("display.max_columns",5)
print(dataset.head(5))

labels=dataset.iloc[:,0].values
pixels=dataset.iloc[:,1:].values
print(f"Total Length: {len(labels)}")
print(f"Total Pixels: {pixels.shape[1]}")

missing_values = dataset.isnull().sum().sum()
print(f"Missing values in dataset: {missing_values}")

unique, counts = np.unique(labels, return_counts=True)
print("Class Distribution:")
for digit, count in zip(unique, counts):
    print(f"Digit {digit}: {count}")

pixels = pixels / 255.0
print(f"Pixels scaled to range [0, 1]")

split_ratio=0.7
split_index=int(len(labels)*split_ratio)

X_train = pixels[:split_index]
y_train = labels[:split_index]
X_test = pixels[split_index:]
y_test = labels[split_index:]

print(f"Total Length of train data of labels: {len(y_train)}")
print(f"Total Length of train data of pixels: {len(X_train)}")
print(f"Total Length of test data of labels: {len(y_test)}")
print(f"Total Length of test data of pixels: {len(X_test)}")

# We are going to apply 3 algos here knn, logictics regression and also the decision tree
# At last we will measure the acuuracy and show the confusion matrix too

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
predict_knn=knn.predict(X_test)

lr=LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)
predict_lr=lr.predict(X_test)

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
predict_dt=dt.predict(X_test)

print("KNN Results:")
print("Accuracy:", accuracy_score(y_test, predict_knn))
print("Precision:", precision_score(y_test, predict_knn, average='weighted'))
print("Recall:", recall_score(y_test, predict_knn, average='weighted'))
print("F1-Score:", f1_score(y_test, predict_knn, average='weighted'))

print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, predict_lr))
print("Precision:", precision_score(y_test, predict_lr, average='weighted'))
print("Recall:", recall_score(y_test, predict_lr, average='weighted'))
print("F1-Score:", f1_score(y_test, predict_lr, average='weighted'))

print("Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, predict_dt))
print("Precision:", precision_score(y_test, predict_dt, average='weighted'))
print("Recall:", recall_score(y_test, predict_dt, average='weighted'))
print("F1-Score:", f1_score(y_test, predict_dt, average='weighted'))

cm_knn = confusion_matrix(y_test, predict_knn)
cm_lr = confusion_matrix(y_test, predict_lr)
cm_dt = confusion_matrix(y_test, predict_dt)


#Now, lets visualize all these:
plt.figure(figsize=(8, 5))
plt.bar(unique, counts)
plt.xlabel('Digit')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

algorithms = ['KNN', 'Logistic Regression', 'Decision Tree']
accuracies = [
    accuracy_score(y_test, predict_knn),
    accuracy_score(y_test, predict_lr),
    accuracy_score(y_test, predict_dt)
]

plt.figure(figsize=(8, 5))
plt.bar(algorithms, accuracies)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cm_knn, cmap='Blues')
plt.colorbar()
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm_knn[i, j], ha='center', va='center', color='white' if cm_knn[i, j] > cm_knn.max()/2 else 'black')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cm_lr, cmap='Greens')
plt.colorbar()
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm_lr[i, j], ha='center', va='center', color='white' if cm_lr[i, j] > cm_lr.max()/2 else 'black')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cm_dt, cmap='Oranges')
plt.colorbar()
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm_dt[i, j], ha='center', va='center', color='white' if cm_dt[i, j] > cm_dt.max()/2 else 'black')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()