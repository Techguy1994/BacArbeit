import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv('2024_6_18_1_22.csv')

df = df.head(5)

with open('cprofiler_example.tex', 'w') as tf:
     tf.write(df.to_latex())