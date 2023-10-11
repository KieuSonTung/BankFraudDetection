from collections import Counter
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# processing
class PreProcess:
    def process_cat_feats(self, X, object_cols):
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        # Get one-hot-encoded columns
        ohe_cols = pd.DataFrame(ohe.fit_transform(X[object_cols]))
        ohe_cols.index = X.index

        X.drop(object_cols, axis=1, inplace=True)
        X[ohe_cols.columns] = ohe_cols
        X.columns = X.columns.astype(str)

        return X

    def split_train_test_set(self, df, month_pred):
        # Split data into features and target
        X = df.drop(['fraud_bool'], axis=1)
        y = df['fraud_bool']

        # Train test split by 'month', month 0-5 are train, 6-7 are test data as proposed in the paper
        X_train = X[X['month'] < month_pred]
        X_test = X[X['month'] >= month_pred]
        y_train = y[X['month'] < month_pred]
        y_test = y[X['month'] >= month_pred]

        X_train.drop('month', axis=1, inplace=True)
        X_test.drop('month', axis=1, inplace=True)

        return X_train, y_train, X_test, y_test

    def fit(self, df, month_pred):
        df = df.drop(['device_fraud_count'], axis=1, errors='ignore')

        X_train, y_train, X_test, y_test = self.split_train_test_set(df, month_pred)
        
        # list of column-names and whether they contain categorical features
        s = (X_train.dtypes == 'object') 
        object_cols = list(s[s].index)

        X_train = self.process_cat_feats(X_train, object_cols)
        X_test = self.process_cat_feats(X_test, object_cols)

        X_train = X_train.reset_index() \
                .drop(columns='index')
        X_test = X_test.reset_index() \
                .drop(columns='index')
        
        y_train = y_train.reset_index() \
                .drop(columns='index')
        y_test = y_test.reset_index() \
                .drop(columns='index')

        return X_train, y_train, X_test, y_test


# metrics
def custom_tpr(y_true, y_pred):
    fprs, tprs, thresholds = roc_curve(y_true, y_pred)
    tpr = tprs[fprs < 0.05][-1]
    
    return tpr

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

def insert_overwrite_partition_by_month(spark, df, path, col_partition='month', schema=None):
    jvm = spark._jvm
    jsc = spark._jsc
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(jsc.hadoopConfiguration())
    cols = [c for c in df.columns if c != col_partition]
    partition_list = df[col_partition].unique()

    for p in partition_list:
        df_filter = df[df[col_partition] == p]

        for m in df_filter[col_partition].unique():
            df_spark = spark.createDataFrame(df_filter[cols], schema=schema)
            df_spark.write.mode('overwrite').format('avro').save(path + '/' + col_partition + '=' + str(p))