#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


proj_path = '/content/drive/MyDrive/smai_project/'


# In[3]:


from google.colab import userdata
import os
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')


# In[4]:


get_ipython().system('pip install gym')
get_ipython().system('pip install stable-baselines3')
get_ipython().system('pip install shimmy>=2.0')
get_ipython().system('pip install groq')


# In[5]:


import warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*") #ignore warning when we pass numpy array without feature names instead of pandas df


# In[6]:


import shap
import pandas as pd
import numpy as np
from groq import Groq
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.model_selection import train_test_split
import pickle


# In[7]:


def convert_np_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_np_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    else:
        return obj


# In[8]:


def get_plan(rl_model, scaler, env, row):
    # Compute injury probability
    daily_features = row[[
        'perceived_exertion.0', 'perceived_recovery.0',
        'nr._sessions.0', 'total_km.0', 'stress_ratio.0'
    ]].values.reshape(1, -1)
    daily_features = scaler.transform(daily_features)
    injury_prob = env.ml_model.predict_proba(daily_features)[0][1]

    # Construct the input state
    state = np.array([row[feat] for feat in env.rl_features] + [injury_prob], dtype=np.float32)

    # Predict action
    action, _states  = rl_model.predict(state)

    state_dict = dict(zip(env.rl_features + ["injury_prob"], state))

    reward_details = reward_function(action, state_dict)
    reward = reward_details['Total reward']

    return {
        "weekly_plan": {
            "total_km": action[0],
            "km_z3-4": action[1],
            "km_z5-t1-t2": action[2],
            "km_sprinting": action[3],
            "strength_training": int(round(action[4])),  # binary
            "hours_alternative": action[5]
        },
        "injury_prob": injury_prob,
        "total_reward": reward,
        "reward_breakdown": reward_details
    }


# In[9]:


def reward_function(action, state):
    total_km, km_z34, km_z5, km_sprint, strength, alt_hrs = action

    ################################### Rewards
    reward = 0

    # Productive training reward
    prod_reward = 0.3 * km_z34 + 0.5 * km_z5 + 0.1 * total_km

    # Recovery practices reward
    rec_reward = 0.2 * alt_hrs + 0.4 * state['perceived_recovery_avg']

    reward += prod_reward + rec_reward

    ################################### Penalties

    # Deviation from baseline (Z-score) penalty with clipping
    z_penalty = 0
    for metric in ['km_sprinting', 'hours_alternative', 'perceived_exertion',
                   'perceived_recovery', 'stress_ratio', 'high_zone_pct']:
        z = state.get(f'{metric}_zcore_sum', 0)
        if abs(z) > 3.0:
            penalty = -min(1.5 * abs(z), 5.0)  # Clip to max 5
            z_penalty += penalty

    reward += z_penalty

    # Stress penalty (scaled and capped)
    stress_penalty = -min((state['stress_ratio_avg'] - 1.3) * 1.0, 2.0) if state['stress_ratio_avg'] > 1.3 else 0
    reward += stress_penalty

    # Excessive high-intensity penalty (clip overuse)
    excess_hz_penalty = 0
    if state['high_zone_pct'] > 20:
        excess_hz_penalty = -min(0.05 * (state['high_zone_pct'] - 20) ** 2, 2.0)
        reward += excess_hz_penalty

    # Low recovery with high exertion penalty (quadratic growth)
    low_rec_high_ex_penalty = 0
    if (state['perceived_recovery_avg'] < 0.25) and (state['perceived_exertion_avg'] > 0.3):
        low_rec_high_ex_penalty = -((0.25 - state['perceived_recovery_avg']) *
                                    (state['perceived_exertion_avg'] - 0.3)) * 4.0
        low_rec_high_ex_penalty = max(low_rec_high_ex_penalty, -2.0)
        reward += low_rec_high_ex_penalty

    # Injury risk proxy penalty (clip and scale down)
    injury_penalty = -min(state['perceived_recovery_avg'] * 2.5, 2.5)
    reward += injury_penalty

    return {
        'productive training reward': prod_reward,
        'good recovery practices reward': rec_reward,
        'deviation from normal baseline penalty': z_penalty,
        'high stress penalty': stress_penalty,
        'excessive high intensity penalty': excess_hz_penalty,
        'low recovery with high exertion penalty': low_rec_high_ex_penalty,
        'injury risk penalty': injury_penalty,
        'Total reward': reward
    }


# In[10]:


class PlayerTrainingEnv(gym.Env):
    def __init__(self, player_data, ml_model, scaler, rl_features, weekly_mean_stats):
        super(PlayerTrainingEnv, self).__init__()

        self.player_data = player_data
        self.ml_model = ml_model
        self.rl_features = rl_features
        self.weekly_mean_stats = weekly_mean_stats
        self.scaler = scaler

        self.current_step = 0

        # state: weekly stats + injury prob
        #['km_sprinting_avg', 'hours_alternative_avg', 'perceived_exertion_avg', 'perceived_recovery_avg',
        #'stress_ratio_avg', 'high_zone_pct', 'km_sprinting_zcore_sum', 'hours_alternative_zcore_sum',
        #'perceived_exertion_zcore_sum', 'perceived_recovery_zcore_sum', 'stress_ratio_zcore_sum', 'high_zone_pct_zcore_sum']
        self.observation_space = spaces.Box(
            low=np.zeros(len(rl_features) + 1),
            high=np.array([15, 10, 1, 1, 5, 100,
                           5, 5, 5, 5, 5, 5,
                           1
                          ]),
            dtype=np.float32
        )

        # action: trainig targets ['total_km', 'km_z3-4', 'km_z5-t1-t2', 'km_sprinting', 'strength_training', 'hours_alternative']
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([50, 30, 15, 5, 1, 5]),
            dtype=np.float32
        )



    def step(self, action):

        #injury risk pred based on latest day
        daily_features = self.player_data.iloc[self.current_step][[
            'perceived_exertion.0', 'perceived_recovery.0',
            'nr._sessions.0', 'total_km.0', 'stress_ratio.0'
        ]].values.reshape(1, -1)

        daily_features = self.scaler.transform(daily_features)
        injury_prob = self.ml_model.predict_proba(daily_features)[0][1]



        #update state (use weekly averages from the current row)
        #['km_sprinting_avg', 'hours_alternative_avg', 'perceived_exertion_avg', 'perceived_recovery_avg',
        #'stress_ratio_avg', 'high_zone_pct', 'km_sprinting_zcore_sum', 'hours_alternative_zcore_sum',
        #'perceived_exertion_zcore_sum', 'perceived_recovery_zcore_sum', 'stress_ratio_zcore_sum', 'high_zone_pct_zcore_sum']
        features = self.rl_features + ["injury_prob"]
        row = self.player_data.iloc[self.current_step]
        values = [row[feat] for feat in self.rl_features] + [injury_prob]
        state_dict = dict(zip(features, values))

        reward_breakdown = reward_function(action, state_dict)
        reward = reward_breakdown['Total reward']

        next_state = np.array(values, dtype=np.float32)

        self.current_step += 1
        done = self.current_step >= len(self.player_data)

        return next_state, reward, done, {
            'weekly_targets': action,
            'injury_prob': injury_prob,
            'reward_breakdown': reward_breakdown
        }

    def reset(self):
        self.current_step = 0
        return np.array(self.weekly_mean_stats)


# In[11]:


def get_shap_values(explainer, df_row):

  feat = ['perceived_exertion.0', 'perceived_recovery.0', 'nr._sessions.0', 'total_km.0', 'stress_ratio.0']
  X = df_row[feat]

  rename_map = {
      "nr._sessions.0": "nr. sessions",
      "perceived_exertion.0": "perceived exertion",
      "perceived_recovery.0": "perceived recovery",
      "stress_ratio.0": "stress ratio",
      "total_km.0": "total km"
  }
  X = X.rename(columns=rename_map)
  X_scaled = scaler.transform(X)

  sample = X_scaled[0:1]
  shap_values = explainer(sample)


  shap_vals_class1 = shap_values.values[0, :, 1]
  input_values = sample[0]

  shap_res={}
  for name, value, shap_val in zip(feat, input_values, shap_vals_class1):
    shap_res[name] = shap_val

  return shap_res



  # for name, value, shap_val in zip(feat, input_values, shap_vals_class1):
  #     print(f"{name:<20} | input = {value:>8.3f} | shap = {shap_val:>8.3f}")


# In[12]:


client = Groq(api_key=os.environ["GROQ_API_KEY"])

def get_groq_response(prompt, model="gemma2-9b-it"):

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# In[13]:


def get_pretty_stats(shap_values_dict, training_plan, injury_probability, reward_breakdown):
    lines = []

    lines.append("Recommended Training Plan for today:")
    for key, value in training_plan.items():
        lines.append(f"- {key}: {round(float(value), 2)}")

    lines.append("\nInjury Probability based on the runner's latest daily stats:")
    lines.append(f"- Estimated Risk: {round(injury_probability * 100, 2)}%")

    lines.append("\nReward Breakdown:")
    for key, value in reward_breakdown.items():
        lines.append(f"- {key}: {round(float(value), 2)}")

    lines.append("\nTop Contributing Factors to injury prediction (SHAP Values):")
    sorted_shap = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, shap_val in sorted_shap[:5]:  # Limit to top 5
        lines.append(f"- {feature}: {round(float(shap_val), 3)}")



    return "\n".join(lines)


# In[14]:


def inference_driver(df_test, row_idx):
  unseen_row = df_test.iloc[row_idx]

  plan_info  = get_plan(model, scaler, env, unseen_row)

  plan_info["weekly_plan"]=convert_np_floats(plan_info["weekly_plan"])
  plan_info["injury_prob"]=convert_np_floats(plan_info["injury_prob"])
  plan_info["reward_breakdown"]=convert_np_floats(plan_info["reward_breakdown"])

  # print("Recommended training plan for today:")
  # print(plan_info["weekly_plan"])
  # print("Injury Prob:", plan_info["injury_prob"])
  # print("Reward Breakdown:", plan_info["reward_breakdown"])

  shap_dict=get_shap_values(explainer, df_test.iloc[[row_idx]])
  shap_dict=convert_np_floats(shap_dict)


  print("-------------------------------------------------------")
  stats = get_pretty_stats(shap_dict, plan_info["weekly_plan"], plan_info["injury_prob"], plan_info["reward_breakdown"])
  print(stats)
  print("-------------------------------------------------------")
  print("Plan rationale:")

  prompt="""
  You are an expert sports scientist and training analyst.\n
  \nExplain the rationale behind the recommended training plan below. Focus on how and why the SHAP values and reward components contributed to the decision,
  especially in relation to injury risk and performance optimization. Be consise, as if a (non-technical) coach would be reading this analysis.

  """+ stats
  summary = get_groq_response(prompt)
  print(summary)
  print("-------------------------------------------------------")


# In[15]:


##################################


# In[15]:





# In[16]:


df = pd.read_csv(proj_path+'data/data_FE.csv')
df.info(verbose=True)


# In[17]:


rl_features1 = ['km_sprinting_avg','hours_alternative_avg','perceived_exertion_avg','perceived_recovery_avg','stress_ratio_avg','high_zone_pct']

rl_features2 = ['km_sprinting_zcore_sum','hours_alternative_zcore_sum','perceived_exertion_zcore_sum','perceived_recovery_zcore_sum','stress_ratio_zcore_sum','high_zone_pct_zcore_sum']

rl_features = rl_features1 + rl_features2

print(rl_features)

injury_pred_model_features = ['perceived_exertion.0', 'perceived_recovery.0', 'nr._sessions.0', 'total_km.0', 'stress_ratio.0']


# In[18]:


ml_model = joblib.load(proj_path+"models/bagging_model_balanced_oversampled.joblib")
scaler = joblib.load(proj_path+'models/scaler_oversampled.joblib')


# In[19]:


with open(proj_path+'models/shap_explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)


# In[20]:


model = PPO.load(proj_path+"models/rl_runner_plan_advisor")


# In[21]:


weekly_mean_stats=df[rl_features+['injury']].mean().values
weekly_mean_stats


# In[22]:


unique_players = df['athlete_id'].unique()
train_players, test_players = train_test_split(
    unique_players, test_size=0.1, random_state=42
)


df_train = df[df['athlete_id'].isin(train_players)].reset_index(drop=True)
df_test = df[df['athlete_id'].isin(test_players)].reset_index(drop=True)

df = df_train

len(df), len(df_test)


# In[23]:


env = PlayerTrainingEnv(
    player_data=df,
    ml_model=ml_model,
    scaler = scaler,
    rl_features=rl_features,
    weekly_mean_stats=weekly_mean_stats
)


# In[24]:


#########################################


# In[24]:





# In[24]:





# In[25]:


inference_driver(df_test, row_idx=45)


# In[26]:


inference_driver(df_test, row_idx=6)


# In[27]:


inference_driver(df_test, row_idx=74)


# In[26]:




