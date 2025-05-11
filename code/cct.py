import numpy as np
import pandas as pd
import os

# loading plant_knowledge.csv
def load_plant_knowledge():
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    filepath = os.path.join(base_dir, '../data/plant_knowledge.csv')
    df = pd.read_csv(filepath)
    data = df.drop(columns=['Informant']).values
    return data

X = load_plant_knowledge()
N, M = X.shape

#print(X)


