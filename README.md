# Diabetes Prediction with a Handbuilt K-Nearest Neighbors Classifier

A from-scratch implementation of the K-Nearest Neighbors (KNN) algorithm in Python, applied to the Pima Indians Diabetes dataset. The handbuilt classifier supports **Euclidean** and **Manhattan** distance and is validated against scikit-learn's `KNeighborsClassifier`.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Background: How KNN Works](#background-how-knn-works)
5. [Environment and Libraries](#environment-and-libraries)
6. [Workflow](#workflow)
7. [Implementation](#implementation)
8. [Results](#results)
9. [Interpretation](#interpretation)
10. [Issues Encountered and Fixes](#issues-encountered-and-fixes)
11. [Limitations](#limitations)
12. [Future Work](#future-work)
13. [How to Run](#how-to-run)

---

## Project Overview

KNN is a *lazy learner*: it has no training phase in which parameters are learned. It stores the training data and, at prediction time, finds the *k* closest training points to a new point and takes a majority vote of their labels.

This project:

- builds a KNN classifier from scratch using only NumPy and the Python standard library,
- implements two distance metrics (Euclidean and Manhattan),
- predicts whether a patient has diabetes from eight medical measurements,
- compares the handbuilt model against scikit-learn using accuracy, confusion matrices, and classification reports.

## Objectives

- Understand KNN and distance metrics by implementing them without a machine learning library.
- Compare Euclidean and Manhattan distance on the same data and split.
- Verify the handbuilt implementation by reproducing scikit-learn's results exactly.
- Evaluate beyond accuracy, with attention to how well diabetic patients are detected.

## Dataset

**Pima Indians Diabetes dataset** (`diabetes.csv`): 768 rows, 8 features, and a binary target.

| Column | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Skin fold thickness |
| Insulin | Serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| **Outcome** | **Target: 0 = non-diabetic, 1 = diabetic** |

**Split:** 80% training / 20% testing, stratified on `Outcome`, `random_state=2`.

| Set | Shape |
|---|---|
| Full data X | (768, 8) |
| X_train | (614, 8) |
| X_test | (154, 8) |

The test set contains 100 non-diabetic and 54 diabetic patients.

## Background: How KNN Works

For a new point, the classifier:

1. computes the distance from the new point to every training point,
2. sorts the training points by distance,
3. takes the *k* nearest,
4. returns the most common label among them.

**Euclidean distance**

$$d(x, y) = \sqrt{\sum_{i}(x_i - y_i)^2}$$

**Manhattan distance**

$$d(x, y) = \sum_{i}|x_i - y_i|$$

Euclidean squares each difference, so a single large gap can dominate the distance. Manhattan adds differences linearly and is generally less sensitive to large values and outliers.

## Environment and Libraries

- Python 3.13
- Jupyter Notebook (VS Code)
- `numpy`, `pandas`, `scikit-learn`, and the standard-library `statistics` module

```python
import numpy as np
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
```

## Workflow

1. Load the dataset and inspect it (`head`, `shape`, `describe`).
2. Separate the features (`X`) and the target (`Y`), and convert both to NumPy arrays.
3. Split into training and test sets (80/20, stratified, `random_state=2`).
4. Attach the label as the last column of the training array for the handbuilt model.
5. Build the `KNN_Classifier` class.
6. Predict every test row with k=5, once for each distance metric.
7. Evaluate with accuracy, confusion matrix, and classification report.
8. Repeat with scikit-learn's `KNeighborsClassifier` and compare.

### Data preparation

```python
X = diabetes_data.drop(columns='Outcome', axis=1).to_numpy()
Y = diabetes_data['Outcome'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Handbuilt model: training data = features + label (label is the last column)
train_data = np.insert(X_train, 8, Y_train, axis=1)

print(X.shape, X_train.shape, X_test.shape, train_data.shape)
# (768, 8) (614, 8) (154, 8) (614, 9)
```

`train_data` has 9 columns (8 features plus the label) and is used only by the handbuilt model. `X_train` keeps the 8 clean features and is used by scikit-learn. Keeping the two as separate variables prevents the models from interfering with each other.

## Implementation

```python
import numpy as np
import statistics

class KNN_Classifier():

    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    # distance between a training point (label is the last element) and a test point
    def get_distance_metric(self, training_data_point, test_data_point):

        if self.distance_metric == 'euclidean':
            dist = 0
            for i in range(len(training_data_point) - 1):   # skip the label column
                dist = dist + (training_data_point[i] - test_data_point[i])**2
            return np.sqrt(dist)

        elif self.distance_metric == 'manhattan':
            dist = 0
            for i in range(len(training_data_point) - 1):   # skip the label column
                dist = dist + abs(training_data_point[i] - test_data_point[i])
            return dist

    # the k training points closest to the test point
    def nearest_neighbors(self, X_train, test_data, k):
        distance_list = []

        for training_data in X_train:
            distance = self.get_distance_metric(training_data, test_data)
            distance_list.append((training_data, distance))

        distance_list.sort(key=lambda x: x[1])

        neighbors_list = []
        for j in range(k):
            neighbors_list.append(distance_list[j][0])

        return neighbors_list

    # majority vote among the k nearest neighbors
    def predict(self, X_train, test_data, k):
        neighbors = self.nearest_neighbors(X_train, test_data, k)

        label = []
        for data in neighbors:
            label.append(data[-1])

        return statistics.mode(label)
```

**Design notes**

- The `distance_metric` value is a lowercase string (`'euclidean'` or `'manhattan'`). Any other spelling, such as `'Manhattan'`, matches neither branch and returns `None`.
- The training array carries its label as the **last column**, which is why distances loop over `len(...) - 1` and the vote uses `data[-1]`.
- `statistics.mode` is safe here because k is odd and there are only two classes, so a tied vote cannot happen. Use odd values of k.

### Running the handbuilt model

```python
classifier = KNN_Classifier(distance_metric='manhattan')   # or 'euclidean'

y_pred = []
for i in range(X_test.shape[0]):
    prediction = classifier.predict(train_data, X_test[i], k=5)
    y_pred.append(prediction)

accuracy = accuracy_score(Y_test, y_pred)
print(accuracy * 100)
```

### Running scikit-learn for comparison

```python
sk_knn = KNeighborsClassifier(n_neighbors=5, p=1)   # p=1 Manhattan, p=2 Euclidean
sk_knn.fit(X_train, Y_train)

sk_pred = sk_knn.predict(X_test)
print(accuracy_score(Y_test, sk_pred) * 100)
```

Although the handbuilt model has no `fit` step, scikit-learn does. For KNN, `fit` only stores the data (and optionally builds a search tree). The distance computations happen in `predict`.

## Results

All models use **k = 5** and the same stratified split on **unscaled** features.

### Accuracy

| Distance metric | Handbuilt KNN | scikit-learn KNN |
|---|---|---|
| Euclidean | 72.73% | 72.73% |
| Manhattan | 77.92% | 77.92% |

### Confusion matrices

Rows are actual classes and columns are predicted classes.

| | Euclidean | Manhattan |
|---|---|---|
| Matrix | `[[88 12] [30 24]]` | `[[92 8] [26 28]]` |
| True negatives | 88 | 92 |
| False positives | 12 | 8 |
| False negatives (missed diabetics) | 30 | 26 |
| True positives | 24 | 28 |

The handbuilt and scikit-learn confusion matrices are identical for both metrics.

### Classification report

| Metric | Euclidean | Manhattan |
|---|---|---|
| Precision, class 0 | 0.75 | 0.78 |
| Recall, class 0 | 0.88 | 0.92 |
| F1, class 0 | 0.81 | 0.84 |
| Precision, class 1 | 0.67 | 0.78 |
| Recall, class 1 | 0.44 | 0.52 |
| F1, class 1 | 0.53 | 0.62 |
| Accuracy | 0.73 | 0.78 |
| Macro avg F1 | 0.67 | 0.73 |
| Weighted avg F1 | 0.71 | 0.77 |

## Interpretation

1. **The implementation is validated.** The handbuilt classifier matches scikit-learn in accuracy and in all four cells of the confusion matrix, for both metrics. Because the two metrics rank neighbors differently, an error in either distance function would almost certainly have produced a gap.

2. **Manhattan outperformed Euclidean.** It was about 5 percentage points more accurate (120 vs 112 correct predictions out of 154), and the improvement appeared on both classes: false alarms fell from 12 to 8, and missed diabetics fell from 30 to 26.

3. **A likely explanation is the unscaled features.** Insulin reaches 846, while features such as Pregnancies and DiabetesPedigreeFunction are very small. Euclidean squares differences, so large-range features dominate; Manhattan weights differences linearly. This is a plausible explanation and was not directly tested.

4. **Both models are weak at detecting diabetes.** Recall for the diabetic class is only 0.44 (Euclidean) and 0.52 (Manhattan), so Manhattan still misses about 48% of true diabetic patients. Accuracy looks respectable mainly because non-diabetic patients are classified well (recall 0.88 and 0.92).

5. **Precision exceeds recall for the diabetic class.** When the model predicts "diabetic," it is usually right (0.67 and 0.78), but it flags only about half of the real cases. For a screening application this is the wrong trade-off.

6. **Baseline context.** Always predicting "non-diabetic" would score about 65% on this test set (100 of 154). The models beat that baseline by about 8 points (Euclidean) and 13 points (Manhattan).

## Issues Encountered and Fixes

These problems came up during development. They are documented because they are easy to hit when working in notebooks.

### 1. `TypeError: 'str' object is not callable`

**Cause:** an instance attribute and a method shared the name `get_distance_metric`, so the string attribute hid the method.
**Fix:** store the metric string as `self.distance_metric` and keep `get_distance_metric` as the method name.

### 2. `TypeError: list.append() takes exactly one argument (2 given)`

**Cause:** `distance_list.append(training_data, distance)` passed two arguments.
**Fix:** append a single tuple: `distance_list.append((training_data, distance))`.

### 3. `KeyError: 0` when predicting

**Cause:** `X_test` had become a pandas DataFrame. On a DataFrame, `X_test[0]` looks up a **column** named `0`, not the first row. Re-running the scikit-learn cells, which redefined `X`, `Y`, and the split as DataFrames, silently overwrote the NumPy arrays the handbuilt model needed.
**Fix:** convert with `.to_numpy()` when creating `X` and `Y`, and keep the handbuilt and scikit-learn sections from overwriting each other's variables.

### 4. Accuracy of 0% with ages as predictions

**Cause:** the 8-column `X_train` was passed to the handbuilt model instead of the 9-column `train_data`. The class read the last column (Age) as the label, and also dropped that column from the distance.
**Fix:** always pass `train_data` (features plus label) to `KNN_Classifier.predict`.

### 5. Handbuilt accuracy (69.48%) below scikit-learn (72.73%)

**Cause:** `label = []` was inside the `for data in neighbors:` loop in `predict`. The list was reset on every iteration, so only the 5th neighbor's label survived and no real vote took place.
**Fix:** create `label = []` once, before the loop. After the fix, the handbuilt results matched scikit-learn exactly.

### 6. Metric name capitalization

Passing `'Manhattan'` instead of `'manhattan'` matches neither branch and returns `None`. Use lowercase strings.

### Practical lessons

- In a notebook, the variables hold whatever was **last executed**, regardless of cell order. Use **Restart, then Run All** to confirm the notebook works top to bottom.
- Give each model its own variable names (`classifier`, `sk_knn`) and its own data variables so one cannot overwrite the other.
- Print shapes and types before running a model to catch mistakes early.
- Matching confusion matrices are stronger evidence of correctness than matching accuracy alone.

## Limitations

- **Small test set.** With 154 samples, the margin of error on accuracy is roughly 6 to 7 percentage points, so the 5-point Manhattan advantage is suggestive, not conclusive.
- **Single split.** Only `random_state=2` was used; a different seed could change the ranking.
- **k was not tuned.** All results use k=5.
- **Features are unscaled,** so large-range features dominate the distance calculations.
- **Placeholder zeros.** In this dataset, zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI usually represent missing values rather than real measurements. They were left unchanged and distort the distances.
- **Speed.** The handbuilt version uses Python loops and compares each test point against every training point, so it is far slower than scikit-learn's optimized implementation. It is intended for learning.

## Future Work

- Scale the features with `StandardScaler` (fit on the training set only) and re-run both metrics.
- Tune k over odd values (1, 3, 5, 7, 9, 11) for each metric.
- Replace placeholder zeros with column medians.
- Use k-fold cross-validation to compare metrics more reliably.
- Improve recall for the diabetic class, for example with class weighting or threshold adjustment.
- Add more distance metrics, such as Minkowski with other values of p.
- Vectorize the distance calculations with NumPy to speed up prediction.

## How to Run

1. Install the requirements:
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```
2. Place `diabetes.csv` somewhere accessible and update the path in the loading cell.
3. Open `Diabetes_Prediction_with_KNN_Classifier.ipynb` in Jupyter or VS Code.
4. Select **Restart**, then **Run All**, or run the cells in this order:
   1. imports and data loading
   2. feature/target separation and `.to_numpy()` conversion
   3. train/test split and `train_data` creation
   4. the `KNN_Classifier` class
   5. handbuilt Euclidean and Manhattan runs
   6. scikit-learn Euclidean and Manhattan runs
   7. confusion matrices and classification reports

Expected shapes before running a model: `X_train` is `(614, 8)`, `train_data` is `(614, 9)`, and `X_test` is `(154, 8)`.
