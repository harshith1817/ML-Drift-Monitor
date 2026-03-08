#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

def save_baseline_stats(train_inputs):
    stats={}
    for col in train_inputs.columns:
        stats[col]={
            "mean": float(train_inputs[col].mean()),
            "std": float(train_inputs[col].mean()),
            "min": float(train_inputs[col].min()),
            "max": float(train_inputs[col].max())
        }
    with open("reports/baseline_stats.json","w") as f:
        json.dump(stats, f, indent=4)


# In[ ]:




