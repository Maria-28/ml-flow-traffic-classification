import pandas as pd
import os

# ====== МЕНЯЙ ТОЛЬКО ЭТУ СТРОКУ ======
path = r"C:\ml_thesis\MachineLearningCVE\FridayMorning.pcap_ISCX.csv"
# ====================================

print("Проверяем, существует ли файл...")
if not os.path.exists(path):
    print("❌ Файл не найден по пути:")
    print(path)
    exit(1)

print("Загружаем файл...")
df = pd.read_csv(path, encoding="latin-1", low_memory=False)

print("Готово!")

# Чистим имена колонок
df.columns = df.columns.str.strip()

print("\nРазмер датасета (строки, колонки):")
print(df.shape)

print("\nСписок колонок:")
print(df.columns.tolist())

print("\nПервые 5 строк:")
print(df.head())

if "Label" in df.columns:
    print("\nРаспределение классов (Label):")
    print(df["Label"].value_counts())
else:
    print("\n❌ Колонка 'Label' не найдена!")

print("\nКоличество пропусков (топ-10 колонок):")
print(df.isna().sum().sort_values(ascending=False).head(10))
