import pandas as pd
import sys

# Load the CSV file
file_path = "database_updated.csv"  # Update if needed
df = pd.read_csv(file_path)

# Filter only FP32 models (excluding INT8) and relevant inference types
fp32_df = df[
    (~df['Model'].str.contains("INT8", case=False)) &
    (df['inference type'].isin(['class', 'det', 'seg']))
]

# Extract relevant metrics
records = []
for _, row in fp32_df.iterrows():
    task = row['inference type']
    if row['Frameworks'] not in ['OpenVINO', 'TFLite']:
        continue

    score = None
    if task == 'class':
        score = row['Top1']
    elif task == 'det':
        score = row['mAP']
    elif task == 'seg':
        score = row['mIoU']

    if "Deeplab" in row['Model']:
        model = "DeeplabV3"
    else:
        model = row['Model']

    
    records.append({
        'Model': model,
        'Framework': row['Frameworks'],
        'Median Latency [s]': row['Median Latency [s]'],
        'StdDev [ms]': row['Standard Deviation [s]']*1000,
        'Score': score
    })



# Create a DataFrame from extracted data
metrics_df = pd.DataFrame(records)

metrics_df.to_csv("df.csv")

# Pivot the data so each row is a model and columns are framework metrics
pivot_df = metrics_df.pivot(index='Model', columns='Framework', values=['Median Latency [s]', 'StdDev [ms]', 'Score'])
pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1)

#pivot_df.to_csv("df.csv")

df_pivoted = pivot_df.copy()
df_pivoted.columns = ['_'.join(col).strip() for col in df_pivoted.columns.values]
df_pivoted = df_pivoted.reset_index()

print(df_pivoted.columns.tolist())



# Calculate differences (OpenVINO - TFLite)
df_pivoted["Δ Latency [s]"] = df_pivoted["OpenVINO_Median Latency [s]"] - df_pivoted["TFLite_Median Latency [s]"]
df_pivoted["Δ Score"] = df_pivoted["OpenVINO_Score"] - df_pivoted["TFLite_Score"]
df_pivoted["Δ StdDev [ms]"] = df_pivoted["OpenVINO_StdDev [ms]"] - df_pivoted["TFLite_StdDev [ms]"]


# Reorder and rename columns
df_final = df_pivoted[[
    "Model",
    "TFLite_Median Latency [s]", "OpenVINO_Median Latency [s]", "Δ Latency [s]",
    "TFLite_Score", "OpenVINO_Score", "Δ Score",
    "TFLite_StdDev [ms]", "OpenVINO_StdDev [ms]", "Δ StdDev [ms]"
]]

print(df_final.head())

df_final.columns = [
    "Model",
    "Latency [s] (TFLite)", "Latency [s] (OpenVINO)", "Δ Latency [s]",
    "Score (TFLite)", "Score (OpenVINO)", "Δ Score",
    "StdDev [ms] (TFLite)", "StdDev [ms] (OpenVINO)", "Δ StdDev [ms]"
]

df_final = df_final.drop(columns=[
    "Score (TFLite)", "Score (OpenVINO)", "Δ Score"
])

# Export to LaTeX
latex_table = df_final.to_latex(index=False, float_format="%.4f")
print(latex_table)

# Optional: Save to file
with open("model_comparison_with_deltas.tex", "w") as f:
    f.write(latex_table)
