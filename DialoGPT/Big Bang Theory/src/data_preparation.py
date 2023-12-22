import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(csv_path):
    all_rick = pd.read_csv(csv_path)
    
    contexted = []
    n = 7

    for i in range(n, len(all_rick['line'])):
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(all_rick['line'][j])
        contexted.append(row)

    columns = ['response', 'context']
    columns = columns + ['context/'+str(i) for i in range(n-1)]

    df = pd.DataFrame.from_records(contexted, columns=columns)

    trn_df, val_df = train_test_split(df, test_size=0.1)
    
    return trn_df, val_df
