# SMAI-project - team 38  


The folder 'data' contains the raw data (day_approach_maskedID_timeseries.csv) and the feature-engineered data (data_FE.csv).  


The folder 'models' contains the best injury prediction model (bagging_model_balanced_oversampled.joblib), the feature scaler used by the injury prediction model (scaler_oversampled.joblib), the RL agent model (rl_runner_plan_advisor.zip) and the SHAP explainer (shap_explainer.pkl).


Code Files and order of execution:  
1. 0_a_EDA.ipynb - EDA
2. 0_b_EDA_feature_engg.ipynb - EDA + feature engineering. This notebook saves the feature-engineered dataset (data_FE.csv)
3. 1_train_injury_pred.ipynb - Training of injury prediction models
4. 2_train_RL_training_plan.ipynb - Training of the RL agent 
5. 3_shap.ipynb - Creates and saves the SHAP explainer of the injury prediction model
6. 4_a_inference_injury_pred.ipynb - Inference for the injury prediction model solely
7. 4_b_inference_plan_driver.ipynb - Full inference notebook for training plan generation [Final output]


The folder 'py_files' contains the python files (.py) of the notebooks mentioned above.


