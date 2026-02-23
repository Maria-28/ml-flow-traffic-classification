# check_portscan.py
import pandas as pd

df = pd.read_csv("data/raw/Friday-PortScan.pcap_ISCX.csv", encoding='latin-1', low_memory=False)
df.columns = df.columns.str.strip()

print("Уникальные значения в колонке 'Label':")
print(df['Label'].value_counts())

print("\nПервые 10 строк Label:")
print(df['Label'].head(10))