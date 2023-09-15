# Mine_Rock_prediction
🚢 Sonar-based Mine vs. Rock Classification using Logistic Regression

📌 Overview:
I developed a machine learning model using Logistic Regression to classify substances detected by sonar waves emitted from ships into two categories: mines and rocks. The dataset consisted of 200 samples and 61 features, making this an exciting data science challenge.



### Importing the Dependencies

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

### Data Collection and Data Processing

```python
# Loading the dataset into a pandas DataFrame
sonar_data = pd.read_csv('/content/sonar data.csv', header=None)

### Getting the Number of Rows and Columns in the Dataset

```python
# Get the number of rows and columns in the dataset
num_rows, num_cols = sonar_data.shape

(208,61)

### Separating Data and Labels

```python
# Separate features (data) and labels
X = sonar_data.drop(columns=60, axis=1)  # Features (data)
Y = sonar_data[60]  # Labels

### **Training and Test data**

### Splitting the Data into Training and Test Sets

```python
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Print the shapes of the datasets
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


model = LogisticRegression()
#training the Logistic Regression model with training data
model.fit(X_train, Y_train)


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data : ', training_data_accuracy)


















