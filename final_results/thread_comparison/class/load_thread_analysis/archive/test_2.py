import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
database = pd.read_csv("load_database.csv")

# Function to create an empty DataFrame
def create_empty_dataframe():
    return pd.DataFrame(columns=["mean latency", "thread", "api", "load"])

df = create_empty_dataframe()

# Filter out PyTorch
db = database[database['api'] != "pytorch"]

# Get unique API and Load values
unique_apis = db['api'].unique()
unique_loads = db["load"].unique()

# Sort Load order
sort_loads = ["default", "one", "two", "three"]

for api in unique_apis:
    for load in unique_loads:
        filtered_db = db[(db["api"] == api) & (db["load"] == load)]
        max_value = filtered_db["latency avg"].min()
        
        filtered_max_db = filtered_db[filtered_db["latency avg"] == max_value]

        for i, r in filtered_max_db.iterrows():
            threads = r["thread count"]

        entry = {"mean latency": [max_value], "thread": [threads], "api": [api], "load": [load]}
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)

# Ensure Load is categorical and sorted
df['load'] = pd.Categorical(df['load'], categories=sort_loads, ordered=True)
df = df.sort_values('load')

# Save to CSV (optional)
df.to_csv("bar.csv")

# Create a bar plot where:
# - Load sections contain API groups
# - Each API has different bars for different thread counts
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="load", y="mean latency", hue="thread", dodge=True, ci=None)

# Adjust labels and title
plt.xlabel("Load")
plt.ylabel("Mean Latency")
plt.title("Mean Latency by Load, API, and Thread Count")
plt.legend(title="Thread Count", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show grid
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
