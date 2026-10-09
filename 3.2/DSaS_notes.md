can make multiple columns like this:


```python
X = np.asarray(df[['Pclass', 'Sex', 'Age', 'Siblings_SpousesAboard', 'Parents_ChildrenAboard', 'Fare']])
```
# 3.2 Trees




## GINI 
Gini = the p that 2 samples picked from the node (w replac.) belong to different classes
lower ginis are better


$$Gini=1-\Sigma_i p_i^2$$

**Classification Accuracy**


## Confusion Matrix

|                      | Predicted Positive     | Predicted Negative     |
|----------------------|------------------------|------------------------|
| **Actually Positive** | **TP** (True Positive) | **FN** (False Negative) |
| **Actually Negative** | **FP** (False Positive) | **TN** (True Negative) |

**FP**: Type I error (False alarm)  
**FN**: Type II error (Missed it)


### Accuracy
$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN} = \frac{TP+TN}{\text{Total}}$$

*Correctly classified samples out of all samples*

can do this directly with 
```python
 accuracy_score(y_true, y_pred, normalize=False)
``` 
normalise = true to return a proportion, false for raw no.

### Recall / Sensitivity

$$\text{Recall} = \text{Sensitivity} = \frac{TP}{TP+FN}$$

*Correctly classified as positives among all actual positives*

``` python
recall = recall_score(y_true, y_pred)
```

### Precision

$$\text{Precision} = \frac{TP}{TP+FP}$$

*Correctly classified as positives among classified as positives*

``` python
precision = precision_score(y_true, y_pred)
```

### Specificity

$$\text{Specificity} = \frac{TN}{TP+FP}$$

*Correctly classified as negatives among all actual negatives*


``` python
specificity = recall_score(y_true, y_pred, pos_label=0)
# by treating the neg class (0) as the pos class
```

### sklearn apps.
```python
cm = confusion_matrix(y_test, y_pred)               # rows = actual, columns = predicted
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)  # plots it
print(classification_report(y_test, y_pred))        # precision, recall, F1 per class
```


## Instantiating and fitting models

**Usually**
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="gini", max_depth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


**Tree Example**

``` python
clf3 = tree.DecisionTreeClassifier(max_leaf_nodes=3)
clf3.fit(X, y)
tree.plot_tree(clf3) # to plot
y_pred = clf3.predict(X)
```


# Validation Curves

plot model perf. against a hyperparameter (ie max leaf)

plots both training score and val. score (lines)

# Updating mean formula

$$ \text{updated mean} = \text{old mean} + \frac{\text{new val} - \text{old mean}}{\text{new num count}}$$