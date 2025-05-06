#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('unzip /content/dataverse_files.zip -d /content/')


# In[ ]:


import pandas as pd

# Load CSV into DataFrame
data = pd.read_csv('/content/day_approach_maskedID_timeseries.csv')

print(data.keys())  # Display the first few rows


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot distribution of daily total distance
sns.histplot(data['total km'], bins=20, kde=True)
plt.title('Distribution of Daily Total Distance')
plt.xlabel('Total Distance (km)')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


# Select daily metrics for correlation analysis
daily_metrics = data[['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'perceived exertion', 'perceived recovery']]

# Compute correlation matrix
corr_matrix = daily_metrics.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Daily Metrics')
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# Initialize a dictionary to store correlation coefficients and p-values
correlation_results = {'Feature': [], 'Pearson Correlation': [], 'p-value': []}

# Calculate Pearson correlation for each feature against injury
for feature in data.columns:
    if feature != 'injury' and feature != 'Date' and feature != 'Athlete ID':  # Exclude non-feature columns
        corr, p_value = pearsonr(data[feature], data['injury'])
        correlation_results['Feature'].append(feature)
        correlation_results['Pearson Correlation'].append(corr)
        correlation_results['p-value'].append(p_value)

# Convert results to a DataFrame for better visualization
correlation_df = pd.DataFrame(correlation_results)
correlation_df = correlation_df.sort_values(by='Pearson Correlation', ascending=False)

# Display the results
print(correlation_df)

# Plot Pearson correlation coefficients
plt.figure(figsize=(20, 16))
sns.barplot(x='Pearson Correlation', y='Feature', data=correlation_df, palette='coolwarm')
plt.title('Pearson Correlation of Features with Injury')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Features')
plt.axvline(0, color='black', linestyle='--')  # Add a vertical line at 0
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# Initialize a dictionary to store correlation coefficients and p-values
correlation_results = {'Feature': [], 'Pearson Correlation': [], 'p-value': []}

# Calculate Pearson correlation for each feature against injury
for feature in data.columns:
    if feature != 'injury' and feature != 'Date' and feature != 'Athlete ID':  # Exclude non-feature columns
        corr, p_value = pearsonr(data[feature], data['injury'])
        correlation_results['Feature'].append(feature)
        correlation_results['Pearson Correlation'].append(corr)
        correlation_results['p-value'].append(p_value)

# Convert results to a DataFrame for better visualization
correlation_df = pd.DataFrame(correlation_results)
correlation_df = correlation_df.sort_values(by='Pearson Correlation', ascending=False)

# Select top 10 features
top_10_df = correlation_df.head(10)

# Display the results
print(top_10_df)

# Plot Pearson correlation coefficients for top 10 features
plt.figure(figsize=(20, 16))
sns.barplot(x='Pearson Correlation', y='Feature', data=top_10_df, palette='coolwarm')
plt.title('Top 10 Features: Pearson Correlation with Injury')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Features')
plt.axvline(0, color='black', linestyle='--')  # Add a vertical line at 0
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Select specific columns for correlation
daily_metrics = data[['injury','nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'perceived exertion', 'perceived recovery']]

# Calculate correlation with injury
correlation_with_injury = daily_metrics.corr()['injury'].drop('injury')

# Sort correlations in descending order
correlation_with_injury = correlation_with_injury.sort_values(ascending=False)

# Plot correlation with injury
plt.figure(figsize=(10, 8))
sns.barplot(x=correlation_with_injury.values, y=correlation_with_injury.index, palette='coolwarm')
plt.title('Correlation of Daily Metrics with Injury')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.axvline(0, color='black', linestyle='--')  # Add a vertical line at 0
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Prepare data
X = data.drop(columns=['injury', 'Date', 'Athlete ID'])
y = data['injury']

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)


# In[ ]:


# Extract feature importance
feature_importance = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

# Plot feature importance with increased figure size
plt.figure(figsize=(20, 16))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance for Injury Prediction')
plt.show()


# In[ ]:


sns.boxplot(x='injury', y='total km', data=data)
plt.title('Daily Total Distance vs. Injury Risk')
plt.xlabel('Injury (1 = Yes, 0 = No)')
plt.ylabel('Total Distance (km)')
plt.show()


# In[ ]:


sns.boxplot(x='injury', y='nr. sessions', data=data)
plt.title('no. of sessions vs. Injury Risk')
plt.xlabel('Injury (1 = Yes, 0 = No)')
plt.ylabel('no. of sessions')
plt.show()


# In[ ]:


sns.boxplot(x='injury', y='km Z3-4', data=data)
plt.title('km Z3-4 vs. Injury Risk')
plt.xlabel('Injury (1 = Yes, 0 = No)')
plt.ylabel('km Z3-4')
plt.show()


# In[ ]:


sns.boxplot(x='injury', y='km Z5-T1-T2', data=data)
plt.title('km Z5-T1-T2 vs. Injury Risk')
plt.xlabel('Injury (1 = Yes, 0 = No)')
plt.ylabel('km Z5-T1-T2')
plt.show()


# In[ ]:


sns.boxplot(x='injury', y='perceived exertion', data=data)
plt.title('perceived exertion vs. Injury Risk')
plt.xlabel('Injury (1 = Yes, 0 = No)')
plt.ylabel('perceived exertion')
plt.show()


# In[ ]:


sns.boxplot(x='injury', y='perceived recovery', data=data)
# plt.figure(figsize=(20, 16))
plt.title('perceived recovery vs. Injury Risk')
plt.xlabel('Injury (1 = Yes, 0 = No)')
plt.ylabel('perceived recovery')
plt.show()


# In[ ]:


from scipy.stats import ttest_ind

# Separate data into injured and non-injured groups
injured = data[data['injury'] == 1]
non_injured = data[data['injury'] == 0]

# Perform t-test for each feature
for feature in ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'perceived exertion', 'perceived recovery']:
    t_stat, p_value = ttest_ind(injured[feature], non_injured[feature])
    print(f"{feature}: t-statistic = {t_stat}, p-value = {p_value}")

