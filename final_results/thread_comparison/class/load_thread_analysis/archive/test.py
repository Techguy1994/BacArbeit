import plotly.express as px
import pandas as pd

# Sample data
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5, 6],
    'y': [10, 15, 13, 17, 14, 18],
    'color': ['A', 'A', 'B', 'B', 'A', 'B'],
    'shape': ['circle', 'square', 'circle', 'square', 'circle', 'square']
})

# Create scatter plot
fig = px.scatter(df, x='x', y='y', color='color', symbol='shape')

# Update traces to connect points of the same color
for color in df['color'].unique():
    color_df = df[df['color'] == color]
    fig.add_scatter(x=color_df['x'], y=color_df['y'], mode='lines', name=f'Line for {color}')

fig.show()
