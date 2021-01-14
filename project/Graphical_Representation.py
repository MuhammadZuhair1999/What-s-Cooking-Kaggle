import pandas as pd
from collections import Counter
import numpy as np
from IPython.core.display import HTML

df_train = pd.read_json('train.json')
df_train.head()

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df_train['cuisine'].value_counts().plot(kind='barh', figsize=(8,6));

counters = {}
for cuisine in df_train['cuisine'].unique():
    counters[cuisine] = Counter()
    indices = (df_train['cuisine'] == cuisine)
    for ingredients in df_train[indices]['ingredients']:
        counters[cuisine].update(ingredients)
        
        counters['italian'].most_common(10)
        
        top10 = pd.DataFrame([[items[0] for items in counters[cuisine].most_common(10)] for cuisine in counters],
            index=[cuisine for cuisine in counters],
            columns=['top{}'.format(i) for i in range(1, 11)])
print((top10.head(8)))

df_train['all_ingredients'] = df_train['ingredients'].map(";".join)
df_train.head()

indices = df_train['all_ingredients'].str.contains('garlic cloves')
df_train[indices]['cuisine'].value_counts().plot(kind='barh',
                                                 title='garlic cloves per cuisine',
                                                 figsize=(8,6));
        
relative_freq = (df_train[indices]['cuisine'].value_counts() / df_train['cuisine'].value_counts())
relative_freq.sort_values(inplace=True)
relative_freq.plot(kind='barh',figsize=(8,6));

unique = np.unique(top10.values.ravel())
unique

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
for ingredient, ax_index in zip(unique, range(4)):
    indices = df_train['all_ingredients'].str.contains(ingredient)
    relative_freq = (df_train[indices]['cuisine'].value_counts() / df_train['cuisine'].value_counts())
    relative_freq.plot(
        kind='barh',
        ax=axes.ravel()[ax_index], fontsize=24,title=ingredient);
            
def css_styling():
    styles = open("../styles/custom.css", "r").read()
    return HTML(styles)
css_styling()            