#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('data/day_approach_maskedID_timeseries.csv')
df.columns=['_'.join(x.lower().split()) for x in df.columns]
df.columns=[df.columns[i]+'.0' if i<=9 else df.columns[i] for i in range(len(df.columns)) ]
df.info(verbose=True)


# In[3]:


df.columns


# In[4]:


features=['nr._sessions', 'total_km', 'km_z3-4', 'km_z5-t1-t2',
       'km_sprinting', 'strength_training', 'hours_alternative',
       'perceived_exertion', 'perceived_trainingsuccess',
       'perceived_recovery']


# nr. sessions (number of trainings completed)
# 
# total km (number of kilometers covered by running)
# 
# km Z3-4 (number of kilometers covered in intensity zones three and four,
# 
# 	running on or slightly above the anaerobic threshold)
# 
# km Z5-T1-T2 (number of kilometers covered in intensity zone five, close to 
# 	     
# 	     maximum heart rate, or intensive track intervals 
# 
# 	     (T1 for longer repetitions and T2 for shorter)
# 
# km sprinting (number of kilometers covered with sprints)
# 
# strength training (whether the day included a strength training)
# 
# hours alternative (number of hours spent on cross training)
# 
# perceived exertion (athlete's own estimation of how tired they were after 
# 
# 		   completing the main session of the day. In case of of a 
# 
# 		   rest day, this value will be -0.01)
# 
# perceived trainingSuccess (athlete's own estimation of how well the session went.
# 
# 			   In case of of a rest day, this value will be -0.01)
# 
# perceived recovery (athlete's own estimation of how well rested they felt before
# 
# 	 	    the start of the session. In case of of a 
# 
# 		   rest day, this value will be -0.01)
# 

# Note: Z1–Z5 represent different heart-rate zones where Z1 is easy aerobic effort and Z5 is close to maximum heart rate. 
# 
# T1 and T2 are long and short track intervals, which are typically done at high intensity.

# In[ ]:





# In[5]:


df['athlete_id'].nunique()


# In[6]:


#df['date'].value_counts()
plt.hist(df['date'], bins=500)
plt.show()


# In[7]:


df['injury'].value_counts() / len(df) * 100


# 
# for i in range(7):
#     df['rest_day.'+str(i)] = df['perceived_recovery.'+str(i)].map(lambda x : 1 if x == -0.01 else 0)
#     
# features.append('rest_day')

# In[8]:


features


# In[9]:


for i in range(7):
    df['medium_zone_pct'+"."+str(i)] = df['km_z3-4'+"."+str(i)]/ (df['total_km'+"."+str(i)] + 1e-6) * 100
    df['high_zone_pct'+"."+str(i)] = df['km_z5-t1-t2'+"."+str(i)]/ (df['total_km'+"."+str(i)] + 1e-6) * 100
    df['sprint_pct'+"."+str(i)] = df['km_sprinting'+"."+str(i)]/ (df['total_km'+"."+str(i)] + 1e-6) * 100


# In[10]:


features.append('medium_zone_pct')
features.append('high_zone_pct')
features.append('sprint_pct')


# In[11]:


df['num_sessions'] = 0
for i in range(7):
    df['num_sessions'] += df["nr._sessions."+str(i)]
df['num_sessions_avg'] = df['num_sessions']/7


df['num_train_days'] = 0
for i in range(7):
    df['num_train_days'] += df["nr._sessions."+str(i)].map(lambda x : 1 if x>0 else 0)
    
    
df['num_rest_days'] = 7- df['num_train_days']


df['train_days_pct']= df['num_train_days']/7 * 100
df['rest_days_pct']= df['num_rest_days']/7 * 100


# In[12]:


for i in range(7):
    df['stress_ratio'+"."+str(i)] = df['perceived_exertion'+"."+str(i)].map(lambda x : 0 if x==-0.01 else x)/( df['perceived_recovery'+"."+str(i)].map(lambda x : 0 if x==-0.01 else x))
    df['stress_ratio'+"."+str(i)] = df['stress_ratio'+"."+str(i)].fillna(0.0).replace(np.inf, 0.0)
    #print(df['stress_ratio'+"."+str(i)].min(), df['stress_ratio'+"."+str(i)].max())


# In[13]:


features.append('stress_ratio')


# In[14]:


for f in features:
    if ('km' in f) or ('strength' in f) or ('alt' in f):
        df[f+'_sum'] = 0
        for i in range(7):
            df[f+'_sum'] += df[f+"."+str(i)]
        df[f+'_avg']= df[f+'_sum']/ (df['num_train_days']+1e-6)
        
    elif ('perceived_exertion' in f) or ('perceived_trainingsuccess' in f) or ('perceived_recovery' in f) or ('stress_ratio' in f):
        df[f+'_sum']=0
        for i in range(7):
            df[f+'_sum'] += df[f+"."+str(i)].map(lambda x : 0 if x==-0.01 else x)
            
        df[f+'_avg'] = df[f+'_sum']/ (df['num_train_days']+1e-6)
        
    


# In[15]:


df['medium_zone_pct'] = df['km_z3-4_sum']/ (df['total_km_sum'] + 1e-6) * 100
df['high_zone_pct'] = df['km_z5-t1-t2_sum']/ (df['total_km_sum'] + 1e-6) * 100
df['sprint_pct'] = df['km_sprinting_sum']/ (df['total_km_sum'] + 1e-6) * 100


# In[16]:


df.columns


# In[17]:


cols =['athlete_id', 'injury', 'date']
features2 = [f for f in df.columns if  not ('.' in f) and not (f in cols)]
features2


# In[18]:


print(", ".join(features))


# cols = ['athlete_id','injury', 'date']
# df_exp =pd.DataFrame()
# for i in range(7):
#     f = cols+ [x+"."+str(i) for x in features]
#     dfx = df[f].copy()
#     dfx.columns= cols+ features
#     df_exp = pd.concat([df_exp, dfx], axis=0, ignore_index=True)
# del dfx
# df_exp = df_exp.reset_index(drop=True)
# df_exp.info(verbose=True)

# In[19]:


#f = [x for x in df.columns if 'perceived_exertion' in x]
#df[f]


# In[20]:


# df0 = df[df['injury'] == 0]
# df1 = df[df['injury'] == 1]
# for f in features2:
#     sns.displot(df, x=f, hue="injury", bins=50)
#     plt.show()
    


# In[21]:


for f in features2:
#     fig, axes = plt.subplots(1, 2, figsize=(18, 5))
#     sns.histplot(df,x=f, hue='injury',stat='percent',ax=axes[0])
#     axes[0].set_title(f"Feature: {f}")
    ax= sns.histplot(data=df, x=f, hue='injury', multiple="fill", element="bars", bins=50)
    ax.set_title(f"Feature: {f}")
    
    plt.show()


# In[22]:


df0 = df[df['injury'] == 0]
df1 = df[df['injury'] == 1]
for f in features2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df0, x=f, bins=50, ax=axes[0])
    axes[0].set_title(f"Feature: {f} (No Injury)")

    sns.histplot(df1, x=f, bins=50, ax=axes[1], color='red')
    axes[1].set_title(f"Feature: {f} (Injury)")

    plt.tight_layout()
    plt.show()


# In[23]:


injured_runners = df[df['injury']==1]['athlete_id'].unique()
injured_runners


# In[24]:


len(injured_runners)


# In[25]:


df['date'].nunique()


# In[26]:


features2


# In[27]:


df_runners=df.groupby(['athlete_id'], as_index=False).agg({'injury':'sum', 'date':'nunique'})
df_runners = df_runners.rename(columns={'injury':'injury_ct'})
df_runners['injury_pct']= df_runners['injury_ct']/ df_runners['date'] * 100
df_runners.sort_values(by='injury_pct', ascending=False)


# In[28]:


plt.hist(df_runners['injury_pct'], bins=50)
plt.show()


# In[29]:


df_corr=df[features2+['injury']].corr()


# In[30]:


df_gen = pd.DataFrame()
for f in features2:
    #print(f,"-->",df_corr['injury'][f])
    #if np.abs(df_corr['injury'][f])>=0.1:
    #    print("*****************",f)
    df_gen = pd.concat([df_gen, pd.DataFrame({'Feature':[f],'Corr':[df_corr['injury'][f]],'Corr_abs':[np.abs(df_corr['injury'][f])]})],
                       axis=0,
                       ignore_index=True
                      )
df_gen.sort_values(by='Corr_abs', ascending=False)


# for f in features2:
#     feat = [f+]
#     df['km_mean'] = df[km_cols].mean(axis=1)
# df['km_std'] = df[km_cols].std(axis=1)

# In[31]:


features


# In[32]:


np.nan * 5


# In[33]:


features3=[]
for f in features:
    f_cols= [f+"."+str(i) for i in range(7)]
    df[f+'_mean'] = df[f_cols].applymap(lambda x: np.nan if x < 0 else x).mean(axis=1, skipna=True)
    df[f+'_std'] = df[f_cols].applymap(lambda x: np.nan if x < 0 else x).std(axis=1, skipna=True)
    for i in range(7):
        

        df[f+'_zcore.'+str(i)] = (df[f+"."+str(i)] - df[f+'_mean']) / df[f+'_std']
        if 'perceived' in f:
            flag = df[f+"."+str(i)].map(lambda x : np.nan if x < 0 else 1)
            df[f+'_zcore.'+str(i)] = df[f+'_zcore.'+str(i)] * flag
            
    df = df.drop([f+'_mean',f+'_std'], axis=1)
    
    f3_cols =[f+'_zcore.'+str(i) for i in range(7)]
    df[f+'_zcore_max']=df[f3_cols].abs().max(axis=1, skipna=True)
    features3.append(f+'_zcore_max')
        
    df[f+'_zcore_sum']=0
    features3.append(f+'_zcore_sum')
    

    
    for i in range(7):
        df[f+'_zcore_sum']+= df[f+'_zcore.'+str(i)].map(lambda x : x if (x>0) and not (np.isnan(x)) else 0)
    
    
        


# In[34]:


features3


# In[35]:


df[features3].describe()


# In[36]:


df_corr2=df[features3+['injury']].corr()


# In[37]:


df_runner_dev = pd.DataFrame()
for f in features3:
    #print(f,"-->",df_corr['injury'][f])
    #if np.abs(df_corr['injury'][f])>=0.1:
    #    print("*****************",f)
    df_runner_dev = pd.concat([df_runner_dev, pd.DataFrame({'Feature':[f],'Corr':[df_corr2['injury'][f]],'Corr_abs':[np.abs(df_corr2['injury'][f])]})],
                       axis=0,
                       ignore_index=True
                      )
df_runner_dev.sort_values(by='Corr_abs', ascending=False)


# In[38]:


for f in features3:
#     fig, axes = plt.subplots(1, 2, figsize=(18, 5))
#     sns.histplot(df,x=f, hue='injury',stat='percent',ax=axes[0])
#     axes[0].set_title(f"Feature: {f}")
    ax= sns.histplot(data=df, x=f, hue='injury', multiple="fill", element="bars", bins=50)
    ax.set_title(f"Feature: {f}")
    
    plt.show()


# In[39]:


df.to_csv('data/data_FE.csv', index=False)


# In[40]:


df.info(verbose=True)


# In[ ]:




