from pprint import pprint
import pandas as pd

def preprocess_data(df):
    """Takes a dataframe and returns useful info and a preprocessed dataframe"""

    brief={}
    print("Preprocessing data...")
    print("-"*70)
    print("Overview...")
    df.info()
    print("-"*70)

    print("Shape and columns...")
    brief["shape"] = df.shape
    brief["columns"] = df.columns.tolist()
    print(brief["shape"])
    print(brief["columns"])
    print("-"*70)

    print("Basic stats...")
    brief["stats"] = df.describe().round(2)
    print(brief["stats"])
    print("-"*70)

    print("Null counts...")
    brief["nulls"] = df.isnull().sum()
    print(brief["nulls"])
    print("-"*70)
    
    print("Explicit duplicates...")
    brief["duplicates"] = df.duplicated().sum()
    print(brief["duplicates"])
    # show both the first occurrence and its duplicates
    dupes = df[df.duplicated(keep=False)]
    print("\nSample of duplicates:\n", dupes.sort_values(by=df.columns.tolist()).head(20))
    print("-"*70)

    print("Enforce snake_case on headers...")
    df = df.rename(columns=lambda x: x.lower().replace(" ", "_"))
    print("Remove explicit duplicates...")
    df = df.drop_duplicates()
    

    return brief, df