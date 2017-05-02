import operator
from matplotlib import pylab as plt
import pandas as pd
import seaborn as sns
file1 = "lr_coef.txt"
file2 = "gbdt_coef_analysis.txt"
plt_title = "(A): logistic regression"
save_title = "%s.png" % plt_title
# Initialize the matplotlib figure
sns.set_context(rc = {'patch.linewidth': 0.1})
f, ax = plt.subplots(figsize=(12, 6))

# Initialize data
df = pd.read_csv(file1, header=None, names=['feature', 'coef'])

# normal or altered ,divided in two
#df['coef'] = -1*df['coef']   # for negative
#df = pd.DataFrame(df[df["coef"]>0], copy=True)
#normalize
#df['coef'] = df['coef'] / df['coef'].sum()
df.sort('coef', ascending=True, inplace=True)

# normal or altered ,divided in two

# Do plot
df.plot(kind='barh', x='feature', y='coef', legend=False, color="#87CEEB", figsize=(10, 5))
plt.title(plt_title)
plt.xlabel('coefficient')
plt.ylabel("")
#plt.gcf().savefig(save_title)

plt_title = "(B): gbdt-space lasso"
save_title = "%s.png" % plt_title
# Initialize the matplotlib figure
sns.set_context(rc = {'patch.linewidth': 0.1})
f, ax = plt.subplots(figsize=(12, 6))

# Initialize data
df = pd.read_csv(file2, header=None, names=['feature', 'coef'])

# normal or altered ,divided in two
#df['coef'] = -1*df['coef']   # for negative
#df = pd.DataFrame(df[df["coef"]>0], copy=True)
#normalize
#df['coef'] = df['coef'] / df['coef'].sum()
df.sort('coef', ascending=True, inplace=True)

# normal or altered ,divided in two

# Do plot
df.plot(kind='barh', x='feature', y='coef', legend=False, color="#87CEEB", figsize=(10, 5))
plt.title(plt_title)
plt.xlabel('coefficient')
plt.ylabel("")
#plt.gcf().savefig(save_title)
