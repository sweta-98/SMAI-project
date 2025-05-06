#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install gym
#pip install stable-baselines3
#pip install shimmy>=2.0


# In[2]:


import warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*") #ignore warning when we pass numpy array without feature names instead of pandas df


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.model_selection import train_test_split


# In[4]:


df = pd.read_csv('data/data_FE.csv')
df.info(verbose=True)


# In[5]:


rl_features1 = ['km_sprinting_avg','hours_alternative_avg','perceived_exertion_avg','perceived_recovery_avg','stress_ratio_avg','high_zone_pct']


# In[6]:


rl_features2 = ['km_sprinting_zcore_sum','hours_alternative_zcore_sum','perceived_exertion_zcore_sum','perceived_recovery_zcore_sum','stress_ratio_zcore_sum','high_zone_pct_zcore_sum']


# In[7]:


rl_features = rl_features1 + rl_features2


# In[8]:


print(rl_features)


# In[9]:


injury_pred_model_features = ['perceived_exertion.0', 'perceived_recovery.0', 'nr._sessions.0', 'total_km.0', 'stress_ratio.0']


# In[10]:


df[rl_features].head()


# In[11]:


df[rl_features].describe()


# In[12]:


df[df['injury']==0][rl_features].describe()


# In[13]:


df[df['injury']==1][rl_features].describe()


# In[14]:


rl_features


# In[15]:


weekly_mean_stats=df[rl_features+['injury']].mean().values
weekly_mean_stats


# In[16]:


ml_model = joblib.load("models/bagging_model_balanced_oversampled.joblib")
scaler = joblib.load('models/scaler_oversampled.joblib')


# In[17]:


unique_players = df['athlete_id'].unique()
train_players, test_players = train_test_split(
    unique_players, test_size=0.1, random_state=42
)


# In[18]:


test_players


# In[19]:


df_train = df[df['athlete_id'].isin(train_players)].reset_index(drop=True)
df_test = df[df['athlete_id'].isin(test_players)].reset_index(drop=True)


# In[20]:


df = df_train


# In[21]:


len(df), len(df_test)


# # RL Agent

# In[22]:


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


# In[23]:


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


# In[24]:


#(self, player_data, ml_model, rl_features, weekly_mean_stats)
env = PlayerTrainingEnv(
    player_data=df, 
    ml_model=ml_model,
    scaler = scaler,
    rl_features=rl_features,
    weekly_mean_stats=weekly_mean_stats
)


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000) 
model.save("models/rl_runner_plan_advisor")


# In[ ]:





# In[25]:


#model = PPO.load("models/rl_runner_plan_advisor")


# In[26]:


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


# In[27]:


unseen_row = df_test.iloc[22]

# Get training plan
plan_info  = get_plan(model, scaler, env, unseen_row)

# Print results
print("Recommended training plan for today:")
print(plan_info["weekly_plan"])
print("Injury Prob:", plan_info["injury_prob"])
print("Reward Breakdown:", plan_info["reward_breakdown"])


# In[28]:


unseen_row = df_test.iloc[5]

# Get training plan
plan_info  = get_plan(model, scaler, env, unseen_row)

# Print results
print("Recommended training plan for today:")
print(plan_info["weekly_plan"])
print("Injury Prob:", plan_info["injury_prob"])
print("Reward Breakdown:", plan_info["reward_breakdown"])

