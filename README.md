# IBOVESPA – Machine Learning Prediction

Projeto de Machine Learning desenvolvido para prever a tendência diária do IBOVESPA (alta ou baixa no dia seguinte), utilizando exclusivamente dados históricos do próprio índice.

---

## 📌 Contexto

No mercado financeiro, decisões rápidas e bem embasadas são essenciais. Este projeto foi desenvolvido como parte do Tech Challenge da pós-graduação em Data Analytics, com o objetivo de construir um modelo preditivo robusto e confiável para apoiar analistas quantitativos na tomada de decisão.

---

## 🎯 Objetivo

Prever se o IBOVESPA irá:
- 📈 Subir (1)
- 📉 Cair (0)

no pregão seguinte, utilizando apenas informações conhecidas até o dia anterior, evitando qualquer tipo de vazamento de dados.

Meta definida:
✅ Alcançar acurácia mínima de **75%** em dados de teste.

---

## 🛠️ Pipeline do Projeto

1. **Coleta e preparação dos dados**
   - Série histórica diária do IBOVESPA
   - Ordenação temporal e validação de consistência

2. **Engenharia de Atributos**
   - Indicadores técnicos (RSI, Stochastic %K, Williams %R)
   - Distância em relação à EMA
   - Variáveis defasadas (lags)

3. **Definição do Target**
   - Target binário: alta ou baixa no dia seguinte
   - Respeito à natureza sequencial dos dados

4. **Treinamento do Modelo**
   - RandomForestClassifier
   - Balanceamento automático de classes
   - Controle de overfitting via profundidade limitada

5. **Otimização de Threshold**
   - Ajuste fino do ponto de corte para maximizar desempenho
   - Evita decisões enviesadas com threshold padrão (0.5)

---

## 📊 Resultados

- ✅ **Acurácia final:** 76.7%
- Precision (Alta): 0.78
- Recall (Alta): 0.82
- F1-Score (Alta): 0.80

O modelo demonstrou boa capacidade de generalização em dados não vistos (out-of-sample).

---

## 🧠 Boas Práticas Aplicadas

- Respeito total à ordem temporal (sem look-ahead bias)
- Separação clara entre treino e teste
- Foco em robustez e não apenas em métricas infladas
- Arquitetura pronta para integração com dashboards internos

---

## 🧰 Tecnologias Utilizadas

- Python
- Pandas, NumPy
- Scikit-learn
- Machine Learning aplicado a séries temporais

---

## ✅ Conclusão

O modelo atingiu a meta de desempenho proposta e se mostrou robusto para uso como prova de conceito em análises de tendência do mercado financeiro.
