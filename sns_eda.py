#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
disp=pd.options.display
disp.max_rows=100
disp.max_columns=100


# In[ ]:


#df=pd.read_csv('C:\\Users\\hp\\Desktop\\Python\\Kaggle\\Housing Prices Prediction\\train.csv')
#column_summary(df, 'SalePrice')


# ## Categorical Variable

# In[ ]:


def categorical_plots(df, ind_var, d_var):
    #x='LotShape'
    #y='SalePrice'
    df_bar=df.groupby(by=[ind_var], dropna=False, sort=False).size()
    #using a subplots method
    labels=list(df_bar.index)
    records=list(list(df_bar))

    x = np.arange(len(labels))/2
    width = 0.4 #width of bars

    fig, ax = plt.subplots(2,2, figsize=(15,10))

    ##Plot 1, Bar Graph
    sns.set_theme(style="whitegrid")
    sns.set(rc={'figure.figsize':(7,5)})
    bar1 = sns.barplot(x=labels, y=records, ax=ax[0][0])

    #add some text in the chart
    bar1.set_ylabel('Number of records')
    bar1.set_xlabel(ind_var)
    #bar1.set_xticklabels(labels)

    def autolabel(bar_plot):
        #Attach a text label above each bar, data labels
        for bar in bar_plot.patches:
            height = bar.get_height()
            bar1.annotate('{}'.format(height),
                        xy=(bar.get_x()+bar.get_width()/2, height),
                        xytext=(0,2),
                        textcoords='offset points',
                        ha='center', va='bottom')
    autolabel(bar1)

    ##Plot 2, Box Plot
    box = sns.boxplot(x=ind_var, y=d_var, data=df, ax=ax[0][1])

    ##Plot 3, Pair Plot
    pair = sns.stripplot(x=ind_var, y=d_var, data=df, ax=ax[1][0])

    ##Plot 4, Violin Plot
    pair = sns.violinplot(x=ind_var, y=d_var, data=df, ax=ax[1][1])

    fig.tight_layout()
    plt.show()   


# ## Numerical Variables

# In[ ]:


def numerical_plots(df, ind_var, d_var):
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    hist = sns.histplot(data=df, x=ind_var, kde=True, ax=ax[0])
    sca = sns.scatterplot(x=ind_var, y=d_var, data=df, ax=ax[1])
    plt.show()


# ## Combine and run for a dataframe

# In[ ]:


def column_summary(df, dvar):
    import os
    os.chdir('C:\\Users\\hp\\Desktop\\Python\\Kaggle')
    import EDA_module as EDA
    df_summary=EDA.EDA_summary(df)
    #dvar='SalePrice'
    for col in list(df.columns):
        col1='\033[1m' + '\033[4m' + col + '\033[0m'
        print(col1.center(100) , end ='\n    ') 
        l=len(df[col].unique())
        print('Number of Unique Values', l)
        ind = df_summary[df_summary['Column Name']==col].index.values[0]
        dtype = df_summary.at[ind, 'Data Type']
        print('Datatype of column:' ,dtype)
        print('Null Percentage: ', df_summary.at[ind, 'Null Percentage'])
        print('Number of Non-null values', df_summary.at[ind, 'Non-null Count'])
        print('Values like ', df_summary.at[ind, 'Examples'])

        if dtype in ['object'] and l<=30:
            categorical_plots(df, col, dvar)
        if dtype in ['int64', 'float64'] and l<=30:
            categorical_plots(df, col, dvar)
        if dtype in ['int64', 'float64'] and l>30:
            numerical_plots(df, col, dvar)

        print('\n \n \n')


# In[ ]:




