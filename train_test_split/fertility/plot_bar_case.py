import operator
from matplotlib import pylab as plt
import pandas as pd
import seaborn as sns
file = "gbdt_coef_analysis.txt"
plt_title = "(A): l1 coefficient analysis"
save_title = "%s.png" % plt_title
# Initialize the matplotlib figure
sns.set_context(rc = {'patch.linewidth': 0.1})
f, ax = plt.subplots(figsize=(12, 6))

# Initialize data
df = pd.read_csv(file, header=None, names=['feature', 'coef'])

df.sort('coef', ascending=True, inplace=True)

# normal or altered ,divided in two

# Do plot
df.plot(kind='barh', x='feature', y='coef', legend=False, color="#87CEEB", figsize=(10, 5))
plt.title(plt_title)
plt.xlabel('coefficient')
plt.ylabel("")
plt.gcf().savefig(save_title)
