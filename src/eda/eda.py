import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


class EDAPlot():
    def __init__(self):
        pass


    def plot_dist_by_class(self, df, feat_cols, class_col, ncols=2, kind='kde'):
        '''
        Plots the distribution of each feature by class.
        kind: 'kde' or 'boxplot'
        '''
        nrows = int(np.ceil(len(feat_cols)/ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(30,50))
        for i in range(nrows):
            for j in range(ncols):
                if i*ncols + j < len(feat_cols):
                    feat_col = feat_cols[i*ncols + j]
                    if kind=='kde':
                        y = df[(df[feat_col] >= df[feat_col].quantile(0.025)) & (df[feat_col] <= df[feat_col].quantile(0.975))][[feat_col, class_col]]
                        sns.kdeplot(data=y, x=feat_col, hue=class_col, common_norm=False, ax=axs[i,j])
                        axs[i,j].grid(True)
                        # axs[i,j].set_title(feat_col)
                    elif kind=='boxplot':
                        y = df[[feat_col, class_col]]
                        sns.boxplot(data=y, x=class_col, y=feat_col, ax=axs[i,j])
                        axs[i,j].grid(True)
                    else:
                        raise Exception('kind must be "kde" or "boxplot"')
                    
        plt.show()


    def round_func(self, x, type='down'):
        '''
        Round to the nearest 10^n.
        '''
        if x < 0:
            return -1
        else:
            z = np.power(10.0, np.floor(np.log(x)/np.log(10)))
            if type == 'down':
                x_rounded = x//z*z
            elif type == 'up':
                x_rounded = np.ceil(x//z)*z
            else:
                raise ValueError('type in ["down", "up"]')
            return x_rounded

    def create_bin(self, df_raw, feat, outlier_lower=0.025, outlier_upper=0.975, bin_lower=0.25, bin_upper=0.75, bin_size=None, log_transform=False):
        '''
        Create new column with the bins of the feature
        and another column with the order of the bins.
        '''
        
        df = df_raw.copy()

        lower_bound = self.round_func(df[feat].quantile(outlier_lower), type='down')
        upper_bound = self.round_func(df[feat].quantile(outlier_upper), type='up')

        if log_transform==False:
            # get bin_size to create bins
            if bin_size is None:
                iqr = df[feat].quantile(bin_upper) - df[feat].quantile(bin_lower)
                bin_size = self.round_func(iqr, type='down')

            df[feat+'_bin'] = np.where(
                df[feat] < lower_bound,
                'lower_outlier (<' + str(lower_bound) + ')',
                np.where(
                    df[feat] > upper_bound,
                    'upper_outlier (>' + str(upper_bound) + ')',
                    np.where(
                        df[feat]//bin_size*bin_size < lower_bound,
                        str(lower_bound),
                        (df[feat]//bin_size*bin_size).apply(str)
                    )
                    + ' - ' +
                    np.where(
                        df[feat]//bin_size*bin_size+bin_size > upper_bound,
                        '<=' + str(upper_bound),
                        '<' + (df[feat]//bin_size*bin_size+bin_size).apply(str)
                    )
                )
            )

            df[feat+'_order'] = np.where(
                df[feat] < lower_bound, lower_bound - 1,
                np.where(
                    df[feat] > upper_bound, upper_bound + 1,
                    np.where(
                        df[feat]//bin_size*bin_size < lower_bound,
                        lower_bound,
                        df[feat]//bin_size*bin_size
                    )
                )
            )
        else:
            df[feat+'_bin'] = np.where(
                df[feat] < lower_bound,
                'lower_outlier (<' + str(lower_bound) + ')',
                np.where(
                    df[feat] > upper_bound,
                    'upper_outlier (>' + str(upper_bound) + ')',
                    np.where(
                        np.power(10, np.floor(np.log(df[feat])/np.log(10))) < lower_bound,
                        str(lower_bound),
                        (np.power(10, np.floor(np.log(df[feat])/np.log(10)))).apply(str)
                    )
                    + ' - ' +
                    np.where(
                        np.power(10, np.ceil(np.log(df[feat])/np.log(10))) > upper_bound,
                        '<=' + str(upper_bound),
                        '<' + (np.power(10, np.ceil(np.log(df[feat])/np.log(10)))).apply(str)
                    )
                )
            )

            df[feat+'_order'] = np.where(
                df[feat] < lower_bound, lower_bound - 1,
                np.where(
                    df[feat] > upper_bound, upper_bound + 1,
                    np.where(
                        np.power(10, np.floor(np.log(df[feat])/np.log(10))) < lower_bound,
                        lower_bound,
                        np.power(10, np.floor(np.log(df[feat])/np.log(10)))
                    )
                )
            )
        return df


    def create_bin_dominated(self, df_raw, feat):
        # (df[feat].value_counts()/len(df)).max() >= 0.5:
        df = df_raw.copy()
        value_max = (df[feat].value_counts()/len(df)).idxmax()
        df[feat+'_bin'] = np.where(
            df[feat] == value_max, value_max,
            np.where(
                df[feat] < value_max,
                '<' + str(value_max),
                '>' + str(value_max)
            )
        )
        df[feat+'_order'] = np.where(
            df[feat] == value_max, value_max,
            np.where(
                df[feat] < value_max,
                value_max - 1,
                value_max + 1
            )
        )
        return df
        

    def create_grouped_df(self, df, feat, class_col):
        '''
        Group df by bins column.
        Return grouped df, feat to plot and bin size to position the bars.
        '''
        if (df[feat].dtype == 'O') | (df[feat].nunique() <= 10):
            df_gr = df.groupby(feat, as_index=False).agg(
                count=(class_col,'count'),
                sum=(class_col,'sum')
            ).sort_values(by=feat)

            feat_plot = feat
            if (df[feat].dtype == 'O') | (df[feat].nunique() == 2):
                bin_size = 1
            else:
                bin_size = df_gr[feat][2] - df_gr[feat][1]

        elif (df[feat].value_counts()/len(df)).max() >= 0.5:
            df_bin = self.create_bin_dominated(df, feat)
            df_gr = df_bin.groupby([feat+'_order', feat+'_bin'], as_index=False).agg(
                count=(class_col,'count'),
                sum=(class_col,'sum')
            ).sort_values(by=feat+'_order')

            feat_plot = feat+'_bin'
            bin_size = 1

        else:
            if feat == 'days_since_request':
                bin_size = 1
            elif feat == 'intended_balcon_amount':
                bin_size = 10
            elif feat == 'bank_branch_count_8w':
                bin_size = 100
            else:
                bin_size = None
            
            if feat in ['days_since_request', 'bank_branch_count_8w']:
                log_transform = True
            else:
                log_transform = False

            df_bin = self.create_bin(df, feat, bin_size=bin_size, log_transform=log_transform)

            # if (df_bin[feat+'_bin'].value_counts()/len(df_bin)).max() >= 0.5:
            #     df_bin[feat+'_transformed'] = np.where(
            #         df_bin[feat+'_bin'] == (df_bin[feat+'_bin'].value_counts()/len(df_bin)).idxmax(),
            #         df_bin[feat+'_bin'],
            #         df_bin[feat+'_order']
            #     )
            #     df_bin = self.create_bin_dominated(df_bin, feat+'_transformed')

            df_gr = df_bin.groupby([feat+'_order', feat+'_bin'], as_index=False).agg(
                count=(class_col,'count'),
                sum=(class_col,'sum')
            ).sort_values(by=feat+'_order')

            feat_plot = feat+'_bin'
            bin_size = 1

        return df_gr, feat_plot, bin_size


    def plot_dist_and_ratio(self, df, feat, class_col):
        df_gr, feat_plot, bin_size = self.create_grouped_df(df, feat, class_col)

        df_gr['perc_class'] = df_gr['sum']/df_gr['count']
        df_gr['perc_total'] = df_gr['count']/df_gr['count'].sum()
        
        # w, h = plt.figaspect(0.25)
        w = 20
        h = 5
        fig, ax = plt.subplots(1,2,figsize=(w,h))
        
        if 'customer_age' in feat_plot:
            height = 3
            shift = 11
        elif df_gr[feat_plot].nunique() >= 9:
            height = 0.05
            shift = 0.11
        else:
            height = 0.8
            shift = 0.11

        ax[0].barh(df_gr[feat_plot], df_gr['count'], color='cornflowerblue', height=height) #mediumaquamarine
        for i, v in enumerate(df_gr[['count', 'perc_total']].itertuples(index=False)):
            if 'device_distinct_emails_8w' in feat_plot:
                i -= 1
            ax[0].text(v[0], i*bin_size + shift, f'{v[0]:,}' + '(' + str(round(v[1]*100,1))+'%' + ')', fontweight='normal')
        ax[0].grid()
        ax[0].set_yticks(df_gr[feat_plot])
        ax[0].set_title(feat + ' distribution')
        ax[0].invert_yaxis()

        ax[1].barh(df_gr[feat_plot], round(df_gr['perc_class']*100,1), color='lightcoral', height=height)
        for i, v in enumerate(df_gr['perc_class']):
            if 'device_distinct_emails_8w' in feat_plot:
                i -= 1
            ax[1].text(round(v*100,1), i*bin_size + shift, str(round(v*100,1)) + '%', fontweight='normal')
        ax[1].grid()
        ax[1].set_yticks(df_gr[feat_plot])
        ax[1].set_title(feat + ' fraud ratio')
        ax[1].invert_yaxis()
        
        fig.tight_layout()

        return df_gr
        
        
    # def plot_proportion(df, feat, key_col, class_col):
    #     df_gr_perc = (df.groupby([feat+'_order', feat+'_bin'])[class_col]
    #                   .value_counts(normalize=True)
    #                   .unstack(class_col)
    #                   .sort_index(axis=1, ascending=False)
    #                   .reset_index(level=0, drop=True)*100
    #                  ).round(1)
    #     ax = df_gr_perc.plot(kind='barh', stacked=True, color=['lightcoral', 'cornflowerblue'], figsize=(8, 6), rot=0, xlabel='Group', ylabel='Proportion')
    #     for c in ax.containers:
    #         ax.bar_label(c, label_type='center', fontweight='normal')
    #     plt.title('Stacked bar chart ' + feat)
    #     plt.gca().invert_yaxis()
    #     plt.grid()
    #     plt.show()
