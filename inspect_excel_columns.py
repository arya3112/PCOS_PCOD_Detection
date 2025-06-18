import pandas as pd

excel_path = r'C:\Users\Lenovo\pcos\PCOS_data_without_infertility.xlsx'
df = pd.read_excel(excel_path, sheet_name='Full_new')
with open('excel_columns.txt', 'w', encoding='utf-8') as f:
    f.write('COLUMNS:\n')
    for col in df.columns:
        f.write(col + '\n')
    f.write('\nSAMPLE ROWS:\n')
    f.write(df.head(5).to_string())
print('Columns and sample rows saved to excel_columns.txt') 