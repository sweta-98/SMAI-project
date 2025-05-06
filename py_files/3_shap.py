#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import shap
import numpy as np
import joblib
import pickle


# In[2]:


df = pd.read_csv('data/data_FE.csv')


# In[3]:


model = joblib.load('models/bagging_model_balanced_oversampled.joblib')
scaler = joblib.load('models/scaler_oversampled.joblib')


# In[4]:


feat = ['perceived_exertion.0', 'perceived_recovery.0', 'nr._sessions.0', 'total_km.0', 'stress_ratio.0']
X = df[feat]

rename_map = {
    "nr._sessions.0": "nr. sessions",
    "perceived_exertion.0": "perceived exertion",
    "perceived_recovery.0": "perceived recovery",
    "stress_ratio.0": "stress ratio",
    "total_km.0": "total km"
}
X = X.rename(columns=rename_map)
X_scaled = scaler.transform(X)


# # SHAP

# In[5]:


X_scaled[0:1] 
sample = X_scaled[0:1]


# In[6]:


explainer = shap.Explainer(model.predict_proba, X_scaled, feature_names=feat)


with open('models/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)


# In[8]:


# Compute SHAP values for the single sample
shap_values = explainer(sample)

# Extract SHAP values for class 1 (probability for class 1) - All 5 features
shap_vals_class1 = shap_values.values[0, :, 1]  # This will give a 1D array of SHAP values for class 1
input_values = sample[0]  # The feature values for the sample as a 1D array

# Print lengths and shapes to debug
# print(f"Length of input_values: {len(input_values)}")
# print(f"Length of shap_vals_class1: {len(shap_vals_class1)}")
# print(f"Length of rename_map (feature names): {len(rename_map)}")

# # Print shapes for inspection
# print(f"Shape of shap_values: {shap_values.values.shape}")
# print(f"Shape of sample: {sample.shape}")


# Print SHAP values per feature
for name, value, shap_val in zip(feat, input_values, shap_vals_class1):
    print(f"{name:<20} | input = {value:>8.3f} | shap = {shap_val:>8.3f}")


# In[10]:


shap_values

