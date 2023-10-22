import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import torch


df_drug = pd.read_csv('./data/entity_datasets/drug.csv')
df_drug_tr = df_drug[df_drug['is_test'] == False]
df_drug_te = df_drug[df_drug['is_test'] == True]
mapping_dict = dict(enumerate(df_drug_tr['condition'].astype('category').cat.categories))
target = df_drug['cat'].values
is_test = df_drug.is_test.values

filename = list(glob.glob('./results/tuned/Llama-2-70b-hf/drug/condition/probes/*.sav'))[0]
layer_num = int(filename.split('/')[-1].split('_')[-1].split('.')[0])
act_path = f"./activation_datasets/Llama-2-70b-hf/drug/drug.last.condition.{layer_num}.pt"
acts = torch.load(act_path)
test_activations = acts[is_test]
test_target = target[is_test]