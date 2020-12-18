#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns


# # General Summary

# In[ ]:


def EDA_summary(df):
    columns=['Column Name', 'Data Type', 'Non-null Count', 'Null Percentage', 'Unique Count', 'Examples']
    df_summary=pd.DataFrame(columns=columns)
    dlen=len(df)
    for col in list(df.columns):
        dtype=str(df.dtypes[col])
        count=df[col].count()
        null_perc=str(round(((dlen-count)/dlen)*100,1)) + ' %'
        try:
            uniq_val=np.sort(df[col].unique())
        except:
            uniq_val=df[col].unique()
        uniq_cnt=len(uniq_val)
        if uniq_cnt>10:
            ex=list(uniq_val[:5]) + ['.............'] + list(uniq_val[-5:])
        else:
            ex=list(uniq_val)
        arr=[col, dtype, count, null_perc, uniq_cnt, ex]
        df_temp=pd.DataFrame([arr], columns=columns)
        df_summary=df_summary.append(df_temp, ignore_index=True)  
    return df_summary
    


# # Column wise summary

# In[ ]:


def EDA_column_summary(df):
    df_summary=EDA_summary(df)
    main_col=list(df_summary['Column Name'][df_summary['Non-null Count']==len(df)])[0]
    print(main_col)
    for ind in range(len(df_summary)):
        col='\033[1m' + '\033[4m' + df_summary['Column Name'][ind] + '\033[0m'
        print(col.center(100) , end ='\n    ') 
        print('Dtype: ', df_summary['Data Type'][ind], end ='\n    ')
        print('Values: ', df_summary['Examples'][ind], end ='\n    ')

        if df_summary['Unique Count'][ind]<10:
            print('Bar Plot', end ='\n    ')
            barplot(df, df_summary['Column Name'][ind], main_col)
            print('Pie Chart')
            piechart(df, df_summary['Column Name'][ind], main_col)
        else:
            if df_summary['Data Type'][ind] in ['int64', 'float64']:
                histogram(df, df_summary['Column Name'][ind])


# # Charts

# ## Histogram 

# In[ ]:


def histogram(df, col, bins=20):
    x=list(df[col])
    #histogram of the data
    plt.figure(figsize=(bins/2,5))
    n, bins, patches = plt.hist(x, bins=20)
    r_bins=[int(round(b)) for b in bins]
    plt.xlabel(col)
    plt.ylabel('Number of records')
    plt.xticks(r_bins[::1])
    plt.show()


# ## Bar plots

# In[ ]:


def barplot(df, groupby_col, count_col):
    df_bar=df.groupby(by=[groupby_col]).count()
    df_bar[count_col]
    #using a subplots method
    labels=list(df_bar.index)
    records=list(df_bar[count_col])

    x = np.arange(len(labels))/2
    width = 0.4 #width of bars

    fig, ax = plt.subplots()
    bar1 = ax.bar(x, records, width)

    #add some text in the chart
    ax.set_ylabel('Number of records')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    def autolabel(bar_plot):
        #Attach a text label above each bar, data labels
        for bar in bar_plot:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x()+bar.get_width()/2, height),
                        xytext=(0,2),
                        textcoords='offset points',
                        ha='center', va='bottom')

    autolabel(bar1)
    fig.tight_layout()
    plt.show()    


# ## Violin Plots 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Box Plots

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Pie Chart

# In[ ]:


def piechart(df, groupby_col, count_col):
    #groupby_col='Embarked'
    #count_col='PassengerId'
    df_bar=df.groupby(by=[groupby_col]).count()
    df_bar[count_col]
    labels=list(df_bar.index)
    sizes=list(df_bar[count_col])
    #sizes=[r/sum(records) for r in records]
    figp, axp = plt.subplots()
    axp.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




