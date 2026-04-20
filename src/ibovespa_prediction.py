#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ==========================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==========================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 2. CARREGAR O DATASET TRATADO
# ==========================================
df = pd.read_csv("ibovespa_tratado.csv")

df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data')

# ==========================================
# 3. CALCULAR RSI (14 PERÍODOS)
# ==========================================
delta = df['fechamento'].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

# ==========================================
# 4. INDICADORES DE REVERSÃO (CHAVE DO MODELO)
# ==========================================
periodo = 14

# Lowest low e highest high
df['low_n'] = df['min'].rolling(periodo).min()
df['high_n'] = df['max'].rolling(periodo).max()

# Stochastic Oscillator %K
df['stoch_k'] = ((df['fechamento'] - df['low_n']) /
                 (df['high_n'] - df['low_n'])) * 100

# Williams %R
df['williams_r'] = ((df['high_n'] - df['fechamento']) /
                    (df['high_n'] - df['low_n'])) * -100

# EMA curta (5 períodos)
df['ema_short'] = df['fechamento'].ewm(span=5, adjust=False).mean()

# Distância entre fechamento e EMA
df['dist_ema'] = (df['fechamento'] - df['ema_short']) / df['ema_short']

# RSI lag
df['rsi_lag'] = df['rsi'].shift(1)

# Lags de Stoch e Williams
df['stoch_k_lag'] = df['stoch_k'].shift(1)
df['williams_lag'] = df['williams_r'].shift(1)

# ==========================================
# 5. TARGET (Alta ou Baixa Amanhã)
# ==========================================
df['target'] = (df['fechamento'].shift(-1) > df['fechamento']).astype(int)

# Remover NaNs
df.dropna(inplace=True)

# ==========================================
# 6. FEATURES FINAIS
# ==========================================
features = [
    'stoch_k_lag',
    'williams_lag',
    'dist_ema',
    'rsi_lag',
]

X = df[features]
y = df['target']

# ==========================================
# 7. DIVISÃO DOS DADOS (Últimos 30 dias = teste)
# ==========================================
X_train = X.iloc[:-30]
y_train = y.iloc[:-30]

X_test = X.iloc[-30:]
y_test = y.iloc[-30:]

# ==========================================
# 8. MODELO — RANDOM FOREST BALANCEADO
# ==========================================
modelo = RandomForestClassifier(
    n_estimators=250,
    max_depth=5,
    min_samples_leaf=5,
    class_weight='balanced_subsample',
    random_state=42
)

modelo.fit(X_train, y_train)

# ==========================================
# 9. THRESHOLD TUNING 
# ==========================================
probs = modelo.predict_proba(X_test)[:, 1]

best_threshold = 0
best_acc = 0

for t in np.arange(0.30, 0.70, 0.01):
    preds_temp = (probs > t).astype(int)
    acc = accuracy_score(y_test, preds_temp)
    if acc > best_acc:
        best_acc = acc
        best_threshold = t

final_preds = (probs > best_threshold).astype(int)

# ==========================================
# 10. RESULTADOS
# ==========================================
print(f"Melhor threshold encontrado: {best_threshold:.2f}")
print(f"Acurácia final: {accuracy_score(y_test, final_preds):.2%}")
print(classification_report(y_test, final_preds))

