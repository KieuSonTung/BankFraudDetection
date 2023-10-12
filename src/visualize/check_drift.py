import matplotlib.pyplot as plt
import seaborn as sns

# Compare the features's distribution in train and infer test
def plot_features_dist(train_df, infer_df):
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(15, 15))

    # Add a title to the figure
    fig.suptitle('Distribution of Numeric Features by Train and Infer set')

    for i, feature in enumerate(numeric_features):
        ax = axes[i // 3][i % 3]
        sns.kdeplot(data=train_df[feature], fill=True, ax=ax, label='Train')
        sns.kdeplot(data=infer_df[feature], fill=True, ax=ax, label='Infer')
        
        ax.set_xlabel(feature)
        ax.legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


# Compare the distribution of class 0 in train and infer test
def plot_feature_dist_each_class(train_df, infer_df, cls):
    fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(15, 15))

    # Add a title to the figure
    fig.suptitle(f'Distribution of Numeric Features of class {cls} by Train and Infer set')

    for i, feature in enumerate(numeric_features):
        ax = axes[i // 3][i % 3]
        sns.kdeplot(data=train_df[train_df['fraud_bool'] == cls][feature], fill=True, ax=ax, label='Train')
        sns.kdeplot(data=infer_df[infer_df['fraud_bool'] == cls][feature], fill=True, ax=ax, label='Infer')
        
        ax.set_xlabel(feature)
        ax.legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()