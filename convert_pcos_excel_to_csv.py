import pandas as pd

# Path to the Excel file and output CSV
excel_path = r'C:\Users\Lenovo\pcos\PCOS_data_without_infertility.xlsx'
sheet = 'Full_new'
df = pd.read_excel(excel_path, sheet_name=sheet)
df.columns = [col.strip() for col in df.columns]

column_map = {
    'Age (yrs)': 'Age',
    'BMI': 'BMI',
    'Cycle(R/I)': 'Cycle_regularity',
    'Weight gain(Y/N)': 'Weight_gain',
    'hair growth(Y/N)': 'Hair_growth',
    'Skin darkening (Y/N)': 'Skin_darkening',
    'Hair loss(Y/N)': 'Hair_loss',
    'Pimples(Y/N)': 'Pimples',
    'Fast food (Y/N)': 'Fast_food',
    'Reg.Exercise(Y/N)': 'Exercise',
    'PCOS (Y/N)': 'PCOS',
}

# Extract and rename columns
df = df[list(column_map.keys())].rename(columns=column_map)

# Convert Yes/No columns to 1/0 if not already numeric, fillna(0) before astype(int)
for col in ['Weight_gain', 'Hair_growth', 'Skin_darkening', 'Hair_loss', 'Pimples', 'Fast_food', 'Exercise', 'PCOS']:
    df[col] = df[col].replace({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}).fillna(0).astype(int)

# For Cycle_regularity, convert 'R' to 5 (Regular), 'I' to 1 (Irregular), or keep as is if already numeric
if df['Cycle_regularity'].dtype == object:
    df['Cycle_regularity'] = df['Cycle_regularity'].replace({'R': 5, 'I': 1})
    df['Cycle_regularity'] = pd.to_numeric(df['Cycle_regularity'], errors='coerce').fillna(3).astype(int)

# Save to CSV
df.to_csv('PCOS_data.csv', index=False)
print('Saved cleaned dataset to PCOS_data.csv') 