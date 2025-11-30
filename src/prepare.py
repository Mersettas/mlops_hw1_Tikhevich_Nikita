import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent
    params_path = base_dir / 'params.yaml'
    
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
        prepare_params = params['prepare']
    
    df = pd.read_csv(base_dir / 'data/raw/data.csv')

    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=columns_to_drop)

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=prepare_params['split_ratio'], 
        random_state=prepare_params['random_state'], 
        stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    processed_dir = base_dir / 'data/processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)

    return None

if __name__ == '__main__':
    main()

