#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pandas.core.frame import DataFrame


# In[70]:


df = pd.read_excel('~/Desktop/Online_Retail.xlsx')
df['TransactionDate']= pd.to_datetime(df['TransactionDate'])
df = df.sort_values('TransactionDate')
#删除空值 清理重复值
data = df.dropna()
data = data.drop_duplicates()


# In[58]:


data


# In[151]:


# 所有用户的前30个最频繁购买项
counts = df['Item'].value_counts()
count = pd.DataFrame(counts)
count = count.head(30)
print(count)


# In[91]:


# 前30项的新数据集
keys = count.index
#de_new 删去不符合条件的数据
df_new = df[df['Item'].isin(keys)]
df_new


# In[93]:


df_new['Item'].value_counts()


# In[180]:


# 频繁项矩阵树形图
import squarify
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sizes=df_new['Item'].value_counts()
squarify.plot(sizes,value=df_new['Item'],text_kwargs={'fontsize':7},alpha=0.9,pad=True)
plt.axis('off')
plt.show()


# In[162]:


# 所有商品的购物篮分析
mybasket=(data.groupby(['TransactionNo', 'Item'])['Quan']
          .sum().unstack().reset_index().fillna(0)
          .set_index('TransactionNo'))
mybasket


# In[181]:


# converting all positive values to 1 and everything else to 0

def my_encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
    
my_basket_sets = mybasket.applymap(my_encode_units)
#my_basket_sets.drop('POSTAGE',inplace=True,axis=1) # Remove "postage" as an item

my_basket_sets


# In[182]:


# Generating frequent itemsets
my_frequent_itemsets = apriori(my_basket_sets,min_support=0.02,use_colnames=True)
my_frequent_itemsets['length'] = my_frequent_itemsets['itemsets'].apply(lambda x : len(x))
my_frequent_itemsets


# In[183]:


my_frequent_itemsets[my_frequent_itemsets['length'] >= 2]


# In[184]:


my_frequent_itemsets[(my_frequent_itemsets['length'] >= 2) &                      (my_frequent_itemsets['support'] >= 0.01)]


# In[185]:


# Generating rules
# 商品关联程度/规则
my_rules = association_rules(my_frequent_itemsets,metric='lift',min_threshold=3)
my_rules


# In[186]:


# Recommendations
my_rules[(my_rules['lift'] >= 7) &
      (my_rules['confidence'] >= 0.8)]


# In[ ]:


# 商品 PINK REGENCY TEACUP AND SAUCER 和 GREEN REGENCY TEACUP AND SAUCER 可以一起购买，商品组合的支持度为 0.020731

