#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import numpy as np

# Load the saved model
model = joblib.load('models/bagging_model_balanced_oversampled.joblib')
scaler = joblib.load('models/scaler_oversampled.joblib')

# Get user input
print("Perceived exertion (float) (between -1 and 1)")
print("Perceived recovery (float) (between -1 and 1)")
print("Number of sessions (float) (among 0.0/1.0/2.0)")
print("Total kilometers (float) above 0.0)")
print("Stress ratio (float) (between -1 and 1)")
print()
user_input = input("Enter all the above features separated by tab space: ")

# Convert input to numpy array
try:
  try:
    features = np.array([float(x) for x in user_input.strip().split(' ')]).reshape(1, -1)
  except:
    features = np.array([float(x) for x in user_input.strip().split('\t')]).reshape(1, -1)

  # Scale the data using the original scaler
  features = scaler.transform(features)

  # Predict class and probability
  pred_class = model.predict(features)[0]
  pred_proba = model.predict_proba(features)[0]

  print("Predicted Class:", pred_class)
  print("Class Probabilities:", pred_proba)
except Exception as e:
    print("Invalid input. Please enter numbers separated by space or tab space.")
    print("Error:", e)

