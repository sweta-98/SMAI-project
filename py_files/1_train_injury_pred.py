#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv('data/day_approach_maskedID_timeseries.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


import pandas as pd


# Compute stress ratio
df['stress ratio'] = df['perceived exertion'] / df['perceived recovery']

# Keep only the three relevant columns
df = df[['perceived exertion', 'perceived recovery', 'nr. sessions', 'total km', 'stress ratio', 'injury']]


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


import numpy as np
import pandas as pd
import pickle
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import datasets, metrics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import sklearn.ensemble
from xgboost import XGBClassifier #
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay # Newer way


# In[ ]:


feature_cols = df.columns[:-1]
target_col = 'injury'

X = df[feature_cols]
y = df[target_col]


# In[ ]:


df_for_balancing = df.copy()

# Separate features (X) and target (y)
y_imb = df_for_balancing['injury']
X_imb = df_for_balancing.drop(['injury'], axis=1) #


# In[ ]:


print(y_imb.value_counts())


# In[ ]:


from imblearn.under_sampling import RandomUnderSampler # For undersampling
rus = RandomUnderSampler(random_state=0)
X_balanced, y_balanced = rus.fit_resample(X_imb, y_imb)


# In[ ]:


normalized_df = pd.DataFrame(X_balanced, columns=X_imb.columns)
normalized_df['injury'] = y_balanced


# In[ ]:


# Use the balanced data created earlier
X_train_bal_u, X_test_bal_u, y_train_bal_u, y_test_bal_u = train_test_split(
             X_balanced, y_balanced, test_size = 0.3, random_state = 0, stratify=y_balanced)

import numpy as np

# Replace inf values with nan
X_train_bal_u = np.where(np.isinf(X_train_bal_u), np.nan, X_train_bal_u)
X_test_bal_u = np.where(np.isinf(X_test_bal_u), np.nan, X_test_bal_u)

# Option 1: Remove rows with NaNs
# mask = ~np.isnan(X_train_bal_u).any(axis=1)
# X_train_bal_u = X_train_bal_u[mask]
# y_train_bal_u = y_train_bal_u[mask]

# mask = ~np.isnan(X_test_bal_u).any(axis=1)
# X_test_bal_u = X_test_bal_u[mask]
# y_test_bal_u = y_test_bal_u[mask]

# OR Option 2: Replace NaNs with column mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_bal_u = imputer.fit_transform(X_train_bal_u)
X_test_bal_u = imputer.transform(X_test_bal_u)

# Scale the balanced data
sc_bal_u = StandardScaler()
X_train_bal_u_sc = sc_bal_u.fit_transform(X_train_bal_u)
X_test_bal_u_sc = sc_bal_u.transform(X_test_bal_u)


# In[ ]:


# Create oversampled data
X = df.drop(['injury'], axis=1)
y = df['injury']

# # Replace inf/-inf with NaN, then drop them
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()
y = y.loc[X.index]

sm = SMOTE(random_state = 0)
X_bal_o, y_bal_o = sm.fit_resample(X, y) # Oversample only training data

print(X_bal_o.shape)
print(y_bal_o.shape)

X_train_bal_o, X_test_bal_o, y_train_bal_o, y_test_bal_o = train_test_split(
    X_bal_o, y_bal_o, test_size=0.25, random_state=0, stratify=y_bal_o
)


print(f"Shape of original training features: {X.shape}")
print(f"Shape of oversampled training features: {X_train_bal_o.shape}")
print(f"Value counts in oversampled training target:\n{y_train_bal_o.value_counts()}")


# Scale the data
sc_bal_o = StandardScaler()
X_train_bal_o_sc = sc_bal_o.fit_transform(X_train_bal_o)
X_test_bal_o_sc = sc_bal_o.transform(X_test_bal_o) # Use the same scaler fitted on training data


# In[ ]:


joblib.dump(sc_bal_o, 'scaler_oversampled.joblib')
joblib.dump(sc_bal_u, 'scaler_undersampled.joblib')


# KNN on Balanced (Undersampled) data

# In[ ]:


print("\n--- KNN on Undersampled Balanced Data ---")

K_bal_u = []
training_bal_u = []
test_bal_u = []
scores_bal_u = {}

print("KNN Accuracy for different K on Undersampled Data:")
for k in range(2, 21):
    clf_bal_u = KNeighborsClassifier(n_neighbors = k)
    clf_bal_u.fit(X_train_bal_u_sc, y_train_bal_u) # Use scaled balanced data

    training_score_bal_u = clf_bal_u.score(X_train_bal_u_sc, y_train_bal_u)
    test_score_bal_u = clf_bal_u.score(X_test_bal_u_sc, y_test_bal_u)
    K_bal_u.append(k)

    training_bal_u.append(training_score_bal_u)
    test_bal_u.append(test_score_bal_u)
    scores_bal_u[k] = [training_score_bal_u, test_score_bal_u]

for keys, values in scores_bal_u.items():
    print(keys, ':', values)

# Here we can see that as we increase the parameter k, the training accuracy goes down and the testing accuracy increases slightly.
# confusion matrix is used to see how well the predictive model is handling the non-injured data compared to the injured data.

# --- Evaluate KNN on Undersampled Data  ---
best_k_u = 18
print(f"\n--- Evaluating KNN with k={best_k_u} on Undersampled Data ---")
clf_bal_u_best = KNeighborsClassifier(n_neighbors = best_k_u)
clf_bal_u_best.fit(X_train_bal_u_sc, y_train_bal_u)

print(f"Training Accuracy (k={best_k_u}): {clf_bal_u_best.score(X_train_bal_u_sc, y_train_bal_u):.4f}")
print(f"Test Accuracy (k={best_k_u}): {clf_bal_u_best.score(X_test_bal_u_sc, y_test_bal_u):.4f}")

y_pred_bal_u = clf_bal_u_best.predict(X_test_bal_u_sc)
print("\nConfusion Matrix (Undersampled):")
print(confusion_matrix(y_test_bal_u, y_pred_bal_u))
print("\nClassification Report (Undersampled):")
print(classification_report(y_test_bal_u, y_pred_bal_u))

with open('knn_model_balanced_undersampled.pkl', 'wb') as f:
    pickle.dump(clf_bal_u_best, f)

# Plot confusion matrix and ROC curve
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(clf_bal_u_best, X_test_bal_u_sc, y_test_bal_u, ax=ax[0])
    ax[0].set_title(f'Confusion Matrix (Undersampled, k={best_k_u})')
    RocCurveDisplay.from_estimator(clf_bal_u_best, X_test_bal_u_sc, y_test_bal_u, ax=ax[1])
    ax[1].set_title(f'ROC Curve (Undersampled, k={best_k_u})')
    plt.tight_layout()
    plt.show()
except ImportError:
    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
    plot_confusion_matrix(clf_bal_u_best, X_test_bal_u_sc, y_test_bal_u)
    plt.title(f'Confusion Matrix (Undersampled, k={best_k_u})')
    plt.show()
    plot_roc_curve(clf_bal_u_best, X_test_bal_u_sc, y_test_bal_u)
    plt.title(f'ROC Curve (Undersampled, k={best_k_u})')
    plt.show()


# KNN on Balanced (oversampled) Data

# In[ ]:


print("\n--- KNN on Oversampled Balanced Data ---")

K_bal_u = []
training_bal_u = []
test_bal_u = []
scores_bal_u = {}

print("KNN Accuracy for different K on Oversampled Data:")
for k in range(2, 21):
    clf_bal_u = KNeighborsClassifier(n_neighbors = k)
    clf_bal_u.fit(X_train_bal_o_sc, y_train_bal_o) # Use scaled balanced data

    training_score_bal_u = clf_bal_u.score(X_train_bal_o_sc, y_train_bal_o)
    test_score_bal_u = clf_bal_u.score(X_test_bal_o_sc, y_test_bal_o)
    K_bal_u.append(k)

    training_bal_u.append(training_score_bal_u)
    test_bal_u.append(test_score_bal_u)
    scores_bal_u[k] = [training_score_bal_u, test_score_bal_u]

for keys, values in scores_bal_u.items():
    print(keys, ':', values)

# Here we can see that as we increase the parameter k, the training accuracy goes down and the testing accuracy increases slightly.
# confusion matrix is used to see how well the predictive model is handling the non-injured data compared to the injured data.

# --- Evaluate KNN on Oversampled Data  ---
best_k_o = 18
print(f"\n--- Evaluating KNN with k={best_k_o} on Oversampled Data ---")
clf_bal_o_best = KNeighborsClassifier(n_neighbors = best_k_o)
clf_bal_o_best.fit(X_train_bal_o_sc, y_train_bal_o)

print(f"Training Accuracy (k={best_k_o}): {clf_bal_o_best.score(X_train_bal_o_sc, y_train_bal_o):.4f}")
print(f"Test Accuracy (k={best_k_o}): {clf_bal_o_best.score(X_test_bal_o_sc, y_test_bal_o):.4f}")

y_pred_bal_u = clf_bal_o_best.predict(X_test_bal_o_sc)
print("\nConfusion Matrix (Oversampled):")
print(confusion_matrix(y_test_bal_o, y_pred_bal_u))
print("\nClassification Report (Oversampled):")
print(classification_report(y_test_bal_o, y_pred_bal_u))

with open('knn_model_balanced_oversampled.pkl', 'wb') as f:
    pickle.dump(clf_bal_o_best, f)


# Plot confusion matrix and ROC curve
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(clf_bal_o_best, X_test_bal_o_sc, y_test_bal_o, ax=ax[0])
    ax[0].set_title(f'Confusion Matrix (Oversampled, k={best_k_o})')
    RocCurveDisplay.from_estimator(clf_bal_o_best, X_test_bal_o_sc, y_test_bal_o, ax=ax[1])
    ax[1].set_title(f'ROC Curve (Oversampled, k={best_k_o})')
    plt.tight_layout()
    plt.show()
except ImportError:
    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
    plot_confusion_matrix(clf_bal_o_best, X_test_bal_o_sc, y_test_bal_o)
    plt.title(f'Confusion Matrix (Oversampled, k={best_k_o})')
    plt.show()
    plot_roc_curve(clf_bal_o_best, X_test_bal_o_sc, y_test_bal_o)
    plt.title(f'ROC Curve (Oversampled, k={best_k_o})')
    plt.show()


# SVM on Undersampled Balanced Data

# In[ ]:


import joblib

# --- SVM classifier using undersampling ---
print("\n--- SVM on Undersampled Balanced Data ---")

svm_u = SVC(kernel = 'rbf', random_state = 0, probability=True) # probability=True for ROC curve
svm_u.fit(X_train_bal_u_sc, y_train_bal_u)

y_pred_svm_u = svm_u.predict(X_test_bal_u_sc)

joblib.dump(svm_u, 'svm_model_balanced_undersampled.joblib')


print(f"Training Accuracy: {svm_u.score(X_train_bal_u_sc, y_train_bal_u):.4f}")
print(f"Test Accuracy: {svm_u.score(X_test_bal_u_sc, y_test_bal_u):.4f}")
print("\nConfusion Matrix (SVM Undersampled):")
print(confusion_matrix(y_test_bal_u, y_pred_svm_u))
print("\nClassification Report (SVM Undersampled):")
print(classification_report(y_test_bal_u, y_pred_svm_u))

# Plot confusion matrix and ROC curve
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(svm_u, X_test_bal_u_sc, y_test_bal_u, ax=ax[0])
    ax[0].set_title('Confusion Matrix (SVM Undersampled)')
    RocCurveDisplay.from_estimator(svm_u, X_test_bal_u_sc, y_test_bal_u, ax=ax[1])
    ax[1].set_title('ROC Curve (SVM Undersampled)')
    plt.tight_layout()
    plt.show()
except ImportError:
    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
    plot_confusion_matrix(svm_u, X_test_bal_u_sc, y_test_bal_u)
    plt.title('Confusion Matrix (SVM Undersampled)')
    plt.show()
    plot_roc_curve(svm_u, X_test_bal_u_sc, y_test_bal_u)
    plt.title('ROC Curve (SVM Undersampled)')
    plt.show()




# SVM on Oversampled Balanced Data

# In[ ]:


# --- SVM Classifier using oversampling ---
print("\n--- SVM on Oversampled Balanced Data ---")

svm_o = SVC(kernel = 'rbf', random_state = 0, probability=True)
svm_o.fit(X_train_bal_o_sc, y_train_bal_o)

y_pred_svm_o = svm_o.predict(X_test_bal_o_sc)

joblib.dump(svm_o, 'svm_model_balanced_oversampled.joblib')


print(f"Training Accuracy: {svm_o.score(X_train_bal_o_sc, y_train_bal_o):.4f}")
print(f"Test Accuracy: {svm_o.score(X_test_bal_o_sc, y_test_bal_o):.4f}")
print("\nConfusion Matrix (SVM Oversampled):")
print(confusion_matrix(y_test_bal_o, y_pred_svm_o))
print("\nClassification Report (SVM Oversampled):")
print(classification_report(y_test_bal_o, y_pred_svm_o))

# Plot confusion matrix and ROC curve
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(svm_o, X_test_bal_o_sc, y_test_bal_o, ax=ax[0])
    ax[0].set_title('Confusion Matrix (SVM Oversampled)')
    RocCurveDisplay.from_estimator(svm_o, X_test_bal_o_sc, y_test_bal_o, ax=ax[1])
    ax[1].set_title('ROC Curve (SVM Oversampled)')
    plt.tight_layout()
    plt.show()
except ImportError:
    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
    plot_confusion_matrix(svm_o, X_test_bal_o_sc, y_test_bal_o)
    plt.title('Confusion Matrix (SVM Oversampled)')
    plt.show()
    plot_roc_curve(svm_o, X_test_bal_o_sc, y_test_bal_o)
    plt.title('ROC Curve (SVM Oversampled)')
    plt.show()


# Bagging on Undersampled Balanced Data

# In[ ]:


print("\n--- Bagging Classifier on Undersampled Balanced Data ---")
# Use the scaled, split undersampled data:
# X_train_bal_u_sc, X_test_bal_u_sc, y_train_bal_u, y_test_bal_u

bag_u = sklearn.ensemble.BaggingClassifier(n_estimators = 35, random_state=0) # Use random_state for reproducibility
bag_u.fit(X_train_bal_u_sc, y_train_bal_u)

y_pred_bag_u = bag_u.predict(X_test_bal_u_sc)

joblib.dump(bag_u, 'bagging_model_undersampled.joblib')


print(f"Training Accuracy: {bag_u.score(X_train_bal_u_sc, y_train_bal_u):.4f}")
print(f"Test Accuracy: {bag_u.score(X_test_bal_u_sc, y_test_bal_u):.4f}")
print("\nConfusion Matrix (Bagging Undersampled):")
print(confusion_matrix(y_test_bal_u, y_pred_bag_u))
print("\nClassification Report (Bagging Undersampled):")
print(classification_report(y_test_bal_u, y_pred_bag_u))

# Plot confusion matrix and ROC curve
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(bag_u, X_test_bal_u_sc, y_test_bal_u, ax=ax[0])
    ax[0].set_title('Confusion Matrix (Bagging Undersampled)')
    RocCurveDisplay.from_estimator(bag_u, X_test_bal_u_sc, y_test_bal_u, ax=ax[1])
    ax[1].set_title('ROC Curve (Bagging Undersampled)')
    plt.tight_layout()
    plt.show()
except ImportError:
    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
    plot_confusion_matrix(bag_u, X_test_bal_u_sc, y_test_bal_u)
    plt.title('Confusion Matrix (Bagging Undersampled)')
    plt.show()
    plot_roc_curve(bag_u, X_test_bal_u_sc, y_test_bal_u)
    plt.title('ROC Curve (Bagging Undersampled)')
    plt.show()



# Bagging on Oversampled Balanced Data

# In[ ]:


import joblib
# --- Bagging Classifier With Oversampling ---
print("\n--- Bagging Classifier on Oversampled Balanced Data ---")
# Use the scaled, split oversampled data:
# X_train_bal_o_sc, X_test_o_sc, y_train_o, y_test_o

bag_o = sklearn.ensemble.BaggingClassifier(n_estimators = 30, random_state=0)
bag_o.fit(X_train_bal_o_sc, y_train_bal_o)

y_pred_bag_o = bag_o.predict(X_test_bal_o_sc)

joblib.dump(bag_o, 'bagging_model_OVERsampled.joblib')

print(f"Training Accuracy: {bag_o.score(X_train_bal_o_sc, y_train_bal_o):.4f}")
print(f"Test Accuracy: {bag_o.score(X_test_bal_o_sc, y_test_bal_o):.4f}")
print("\nConfusion Matrix (Bagging Oversampled):")
print(confusion_matrix(y_test_bal_o, y_pred_bag_o))
print("\nClassification Report (Bagging Oversampled):")
print(classification_report(y_test_bal_o, y_pred_bag_o))

# Plot confusion matrix and ROC curve
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(bag_o, X_test_bal_o_sc, y_test_bal_o, ax=ax[0])
    ax[0].set_title('Confusion Matrix (Bagging Oversampled)')
    RocCurveDisplay.from_estimator(bag_o, X_test_bal_o_sc, y_test_bal_o, ax=ax[1])
    ax[1].set_title('ROC Curve (Bagging Oversampled)')
    plt.tight_layout()
    plt.show()
except ImportError:
    plot_confusion_matrix(bag_o, X_test_bal_o_sc, y_test_bal_o)
    plt.title('Confusion Matrix (Bagging Oversampled)')
    plt.show()
    plot_roc_curve(bag_o, X_test_bal_o_sc, y_test_bal_o)
    plt.title('ROC Curve (Bagging Oversampled)')
    plt.show()


# XGBoost on Undersampled Balanced data

# In[ ]:


# --- XGBooster model with Undersampling ---
print("\n--- XGBoost Classifier on Undersampled Balanced Data ---")
# Use the *unscaled*, split undersampled data, as XGBoost doesn't strictly require scaling
# X_train_bal_u, X_test_bal_u, y_train_bal_u, y_test_bal_u

boost_u = XGBClassifier(max_depth = 3, n_estimators = 30, random_state=0, use_label_encoder=False, eval_metric='logloss') # Added params for newer XGBoost
boost_u.fit(X_train_bal_u, y_train_bal_u) # Train on unscaled data

y_pred_boost_u = boost_u.predict(X_test_bal_u)

joblib.dump(boost_u, 'xgboost_model_undersampled.joblib')

print(f"Training Accuracy: {boost_u.score(X_train_bal_u, y_train_bal_u):.4f}")
print(f"Test Accuracy: {boost_u.score(X_test_bal_u, y_test_bal_u):.4f}")
print("\nConfusion Matrix (XGBoost Undersampled):")
print(confusion_matrix(y_test_bal_u, y_pred_boost_u))
print("\nClassification Report (XGBoost Undersampled):")
print(classification_report(y_test_bal_u, y_pred_boost_u))

# Plot confusion matrix and ROC curve
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(boost_u, X_test_bal_u, y_test_bal_u, ax=ax[0])
    ax[0].set_title('Confusion Matrix (XGBoost Undersampled)')
    RocCurveDisplay.from_estimator(boost_u, X_test_bal_u, y_test_bal_u, ax=ax[1])
    ax[1].set_title('ROC Curve (XGBoost Undersampled)')
    plt.tight_layout()
    plt.show()
except ImportError:
    plot_confusion_matrix(boost_u, X_test_bal_u, y_test_bal_u)
    plt.title('Confusion Matrix (XGBoost Undersampled)')
    plt.show()
    plot_roc_curve(boost_u, X_test_bal_u, y_test_bal_u)
    plt.title('ROC Curve (XGBoost Undersampled)')
    plt.show()


# XGBoost on Oversampled Balanced Data

# In[ ]:


print("\n--- XGBoost Classifier on Oversampled Balanced Data ---")
# Use the *unscaled*, split oversampled data
# X_train_bal_o, X_test_o, y_train_bal_o, y_test_o

boost_o = XGBClassifier(max_depth = 2, n_estimators = 30, random_state=0, use_label_encoder=False, eval_metric='logloss')
boost_o.fit(X_train_bal_o, y_train_bal_o) # Train on unscaled data

y_pred_boost_o = boost_o.predict(X_test_bal_o)

joblib.dump(boost_o, 'xgboost_model_OVERsampled.joblib')


print(f"Training Accuracy: {boost_o.score(X_train_bal_o, y_train_bal_o):.4f}")
print(f"Test Accuracy: {boost_o.score(X_test_bal_o, y_test_bal_o):.4f}")
print("\nConfusion Matrix (XGBoost Oversampled):")
print(confusion_matrix(y_test_bal_o, y_pred_boost_o))
print("\nClassification Report (XGBoost Oversampled):")
print(classification_report(y_test_bal_o, y_pred_boost_o))

# Plot confusion matrix and ROC curve
try:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(boost_o, X_test_bal_o, y_test_bal_o, ax=ax[0])
    ax[0].set_title('Confusion Matrix (XGBoost Oversampled)')
    RocCurveDisplay.from_estimator(boost_o, X_test_bal_o, y_test_bal_o, ax=ax[1])
    ax[1].set_title('ROC Curve (XGBoost Oversampled)')
    plt.tight_layout()
    plt.show()
except ImportError:
    plot_confusion_matrix(boost_o, X_test_bal_o, y_test_bal_o)
    plt.title('Confusion Matrix (XGBoost Oversampled)')
    plt.show()
    plot_roc_curve(boost_o, X_test_bal_o, y_test_bal_o)
    plt.title('ROC Curve (XGBoost Oversampled)')
    plt.show()


# In[ ]:




