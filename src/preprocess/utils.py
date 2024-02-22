from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# processing
class PreProcess:
    def process_cat_feats(self, X):
        # process categorical features
        s = (X.dtypes == 'object') 
        object_cols = list(s[s].index)

        X = pd.get_dummies(X, columns=object_cols, dtype=float)

        return X

    def split_train_test_set(self, df, month_pred):
        # Split data into features and target
        X = df.drop(['fraud_bool'], axis=1)
        y = df['fraud_bool']

        # Train test split by 'month', month 0-5 are train, 6-7 are test data as proposed in the paper
        X_train = X[X['month'] < month_pred]
        X_test = X[X['month'] == month_pred]
        y_train = y[X['month'] < month_pred]
        y_test = y[X['month'] == month_pred]

        X_train.drop('month', axis=1, inplace=True)
        X_test.drop('month', axis=1, inplace=True)

        return X_train, y_train, X_test, y_test

    def fit(self, df, month_pred):
        df = df.drop(['device_fraud_count'], axis=1, errors='ignore')

        df = self.process_cat_feats(df)

        X_train, y_train, X_test, y_test = self.split_train_test_set(df, month_pred)
        
        # list of column-names and whether they contain categorical features
        X_train = X_train.reset_index() \
                .drop(columns='index')
        X_test = X_test.reset_index() \
                .drop(columns='index')
        
        y_train = y_train.reset_index() \
                .drop(columns='index')
        y_test = y_test.reset_index() \
                .drop(columns='index')

        return X_train, y_train, X_test, y_test


# Ensemble
def hard_voting(matrix):
    # Transpose the matrix to group values by position
    transposed_matrix = list(map(list, zip(*matrix)))
    
    # Initialize a list to store the results
    result = []
    
    # Iterate through the transposed matrix
    for elements in transposed_matrix:
        # Count the occurrences of each element in the current position
        vote_counts = Counter(elements)
        
        # Find the most common element (mode)
        most_common_value, _ = vote_counts.most_common(1)[0]
        
        # Append the mode to the result list
        result.append(most_common_value)
    
    return result

def soft_voting(matrix):
    
    return [sum(col) / len(col) for col in zip(*matrix)]

# Insert into data warehouse
def insert_overwrite_partition_by_date_model(spark, df, path, col_partition_date='month', col_partition_model='model', schema=None):
    jvm = spark._jvm
    jsc = spark._jsc
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(jsc.hadoopConfiguration())
    cols = [c for c in df.columns if c != col_partition_date and c != col_partition_model]
    partition_list = df[col_partition_date].unique()

    for p in partition_list:
        df_filter = df[df[col_partition_date] == p]

        for m in df_filter[col_partition_model].unique():
            df_filter_2 = df_filter[df_filter[col_partition_model] == m]
            df_spark = spark.createDataFrame(df_filter_2[cols], schema=schema)
            df_spark.write.mode('overwrite').format('parquet').save(path + '/' + col_partition_date + '=' + str(p) + '/' + col_partition_model + '=' + m)