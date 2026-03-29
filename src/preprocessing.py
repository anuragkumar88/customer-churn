import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    return df

def feature_engineering(df):
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,50,100], labels=[0,1,2])
    return df

def encode_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df