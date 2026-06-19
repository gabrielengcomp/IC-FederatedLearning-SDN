import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

DATA_DIR = Path("data/cicids2017")

FEATURES = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Bwd IAT Total", "Bwd IAT Mean", "Fwd PSH Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count",
    "Down/Up Ratio", "Avg Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init Fwd Win Bytes", "Init Bwd Win Bytes",
    "Active Mean", "Active Std", "Idle Mean", "Idle Std",
]

print("Carregando arquivos...")
dfs = []
for f in sorted(DATA_DIR.glob("*.parquet")):
    df = pd.read_parquet(f, columns=FEATURES + ["Label"])
    dfs.append(df)
    print(f"  {f.name}: {len(df):,} linhas")

full = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(full):,} amostras")

full["label"] = (full["Label"] != "Benign").astype(int)
vc = full["label"].value_counts()
print(f"Benign (0): {vc[0]:,} ({vc[0]/len(full)*100:.1f}%)")
print(f"Attack (1): {vc[1]:,} ({vc[1]/len(full)*100:.1f}%)")

X = full[FEATURES].astype(np.float32)
y = full["label"].values

print("\nNormalizando features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Dividindo em cliente 1 e cliente 2 (Non-IID)...")
benign_idx = np.where(y == 0)[0]
attack_idx = np.where(y == 1)[0]

np.random.seed(42)
np.random.shuffle(benign_idx)
np.random.shuffle(attack_idx)

c1_benign = benign_idx[:int(len(benign_idx) * 0.7)]
c2_benign = benign_idx[int(len(benign_idx) * 0.7):]
c1_attack = attack_idx[:int(len(attack_idx) * 0.3)]
c2_attack = attack_idx[int(len(attack_idx) * 0.7):]

c1_idx = np.concatenate([c1_benign, c1_attack])
c2_idx = np.concatenate([c2_benign, c2_attack])

np.random.shuffle(c1_idx)
np.random.shuffle(c2_idx)

for i, idx in enumerate([c1_idx, c2_idx], 1):
    X_c = X_scaled[idx]
    y_c = y[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)
    np.save(f"data/client{i}_X_train.npy", X_tr)
    np.save(f"data/client{i}_X_test.npy",  X_te)
    np.save(f"data/client{i}_y_train.npy", y_tr)
    np.save(f"data/client{i}_y_test.npy",  y_te)
    vc_c = pd.Series(y_c).value_counts()
    print(f"  Cliente {i}: {len(idx):,} amostras — Benign={vc_c.get(0,0):,} Attack={vc_c.get(1,0):,}")

print("\nPré-processamento concluído. Arquivos salvos em data/")
print(f"Features usadas: {len(FEATURES)}")
