import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path("data/cicids2017")

arquivos = sorted(DATA_DIR.glob("*.parquet"))

print("=" * 60)
print("EXPLORAÇÃO — CIC-IDS2017")
print("=" * 60)

dfs = []
for f in arquivos:
    df = pd.read_parquet(f)
    label = "BENIGN" if "Benign" in f.name else "ATTACK"
    df["_arquivo"] = f.stem
    df["_label_bin"] = 0 if label == "BENIGN" else 1
    dfs.append(df)
    print(f"\n{f.name}")
    print(f"  Linhas : {len(df):,}")
    print(f"  Colunas: {len(df.columns)}")
    if "Label" in df.columns:
        print(f"  Classes: {df['Label'].value_counts().to_dict()}")

print("\n" + "=" * 60)
print("DATASET COMPLETO")
print("=" * 60)

full = pd.concat(dfs, ignore_index=True)
print(f"Total de amostras : {len(full):,}")
print(f"Total de colunas  : {len(full.columns)}")
print(f"\nBalanceamento binário:")
vc = full["_label_bin"].value_counts()
print(f"  Benign (0): {vc.get(0,0):,} ({vc.get(0,0)/len(full)*100:.1f}%)")
print(f"  Attack (1): {vc.get(1,0):,} ({vc.get(1,0)/len(full)*100:.1f}%)")

print(f"\nColunas disponíveis:")
for col in full.columns:
    if not col.startswith("_"):
        dtype = str(full[col].dtype)
        nulls = full[col].isnull().sum()
        print(f"  {col:<45} {dtype:<10} nulls={nulls}")

print("\n" + "=" * 60)
print("VALORES INFINITOS por coluna")
print("=" * 60)
num_cols = full.select_dtypes(include="number").columns
inf_counts = ((full[num_cols] == float("inf")) | (full[num_cols] == float("-inf"))).sum()
inf_counts = inf_counts[inf_counts > 0].sort_values(ascending=False)
if len(inf_counts):
    print(inf_counts.head(10))
else:
    print("Nenhum valor infinito encontrado.")
