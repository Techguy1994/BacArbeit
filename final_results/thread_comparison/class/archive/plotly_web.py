import plotly.figure_factory as ff
import numpy as np
import pandas as pd

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4

# Group data together
hist_data = [x1, x2, x3, x4]

group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.show()

#with pandas


df = pd.DataFrame({'2012': np.random.randn(200),
                   '2013': np.random.randn(200)+1})
print(df.head)
fig = ff.create_distplot([df[c] for c in df.columns], df.columns, bin_size=.25)
fig.show()