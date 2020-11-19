#%%
#Fahad _ chicked Weights Analysis
import pandas as pd # For DataFrame and handling
import numpy as np # Array and numerical processing
from scipy import stats
import seaborn as sns # High level Plotting
import statsmodels.api as sm # Modeling, e.g. ANOVA
from statsmodels.formula.api import ols 
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
#%%
# read the data
chicken_weights = pd.read_table("data/Chick Weights.txt")
#%%
#Expore the data 
chicken_weights.info()
chicken_weights.describe()
chicken_weights.head()
chicken_weights.columns
#%%
#Calculate the mean and standard deviation for each group.
chicken_weights.groupby(['feed']).agg({'weight':['mean','std']})
#%%
#Calculate the number of chicks in each group
chicken_weights['feed'].value_counts()

#%%
#Calculate a within-group z-score
numeric_columns = chicken_weights.select_dtypes(np.number)
c = numeric_columns.columns

chicken_weights[c] = chicken_weights.groupby(['feed'])[c].transform(stats.zscore)

print(chicken_weights)
#%%

#Produce a strip chart showing each chick as an individual data point

sns.stripplot(x= "feed", y = "weight", data= chicken_weights)
#%%
#Calculate a 1-way ANOVA.

model = ols("weight ~ feed", chicken_weights)
results = model.fit()
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table
#%%

#Calculate Tukey’s post-hoc test (i.e. p-values for all pair-wise t-tests)

tuky_chick = pairwise_tukeyhsd(chicken_weights['weight'], chicken_weights['feed'])
print("full tukey")
print(tuky_chick)







# %%
