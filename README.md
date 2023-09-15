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

