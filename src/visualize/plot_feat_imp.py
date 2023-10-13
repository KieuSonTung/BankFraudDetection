import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_feature_importance(X, feature_importance):
    feature_imp = pd.DataFrame(sorted(zip(feature_importance, X.columns)), columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()