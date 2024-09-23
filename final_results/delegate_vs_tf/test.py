import pandas as pd
import plotly.express as px

# Sample data
data = {
    'Category': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'Subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Value': [10, -15, 20, -25, 15, -10, 25, -20]
}
df = pd.DataFrame(data)

# Add a column to separate positive and negative values
df['Value Type'] = df['Value'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

# Create the violin plot
fig = px.violin(df, x='Category', y='Value', color='Subcategory', box=True, points='all', facet_row='Value Type')

# Show the plot
fig.show()
