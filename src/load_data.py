import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path='data/heart.csv'):
    df = pd.read_csv(path)
    return df

def split_data(df, target_column='target', test_size=0.3, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
