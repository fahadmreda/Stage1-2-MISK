import pandas as pd # For DataFrame and handling
import numpy as np # Array and numerical processing
from scipy import stats
import seaborn as sns # High level Plotting
import statsmodels.api as sm # Modeling, e.g. ANOVA
from statsmodels.formula.api import ols 

#%%
jems = pd.read_csv("data/diamonds.csv")
#%%
#Explore the data set 
type(jems)
jems.columns # names
jems.describe()
jems.info()
jems.head()
jems.tail()
jems.sample(n = 10)
#%%
#How many diamonds with a clarity of category “IF” are present in the data-set?
len(jems[jems.clarity == "IF"])

#%%
#print all counts 
jems.clarity.value_counts()

#%%
#What fraction of the total do they represent?
len(jems[jems.clarity == "IF"])/len(jems)

#%%
# What is the cheapest diamond price overall? 

#series
min(jems.price)
#DataFrame
jems['price'].min()


#%%

#What is the range of diamond prices? 
#dangers more specified 
def getRange (x):
    low = min(x)
    high = max(x)
    return low, high

#%%

price_range = getRange(jems.price)

low, high = getRange(jems.price)

price_range

#%%
#What is the average diamond price in each category of cut and color?

jems.groupby(["cut", "color"])["price"].mean()

#%%
# Make a scatter plot that shows the diamond price described by carat.

sns.scatterplot(x = "carat", y = "price", data = jems)
#%%

#Apply a log10 transformation to both the price and carat and store these as new columns in the DataFrame: price_log10 and carat_log10.

jems["price_log10"] = np.log10(jems.price)
jems["carat_log10"] = np.log10(jems.carat)
jems

#%%

#Redraw the scatterplot using the transformed values.
sns.scatterplot(x = "carat_log10", y = "price_log10", data = jems)

#%%

# Define a linear model that describes the relatioship shown in the plot.
sns.regplot(x = "carat_log10", y = "price_log10", data = jems)
