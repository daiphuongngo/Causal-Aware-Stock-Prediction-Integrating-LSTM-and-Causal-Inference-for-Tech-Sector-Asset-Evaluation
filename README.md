# Causal-Aware-Stock-Prediction-Integrating-LSTM-and-Causal-Inference-for-Tech-Sector-Asset-Evaluation

![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master, Data Science**

## CSCI S-278	**Applied Quantitative Finance and Machine Learning** (Python)

## Professors: MarcAntonio, PhD

**Head of Research and Data Science, Digital Data Design Institute, Harvard University**

## Author: **Dai-Phuong Ngo (Liam)**

## Timeline: June 23 - August 12, 2025

---

## Executive Summary

In this project, I aimed to enhance the accuracy and interpretability of stock market forecasting by combining traditional time series deep learning techniques with modern causal inference methodologies. Specifically, I explored how causal reasoning can uncover true treatment effects (e.g., changes in macroeconomic indicators like interest rates) on stock returns, while LSTM models capture nonlinear time-dependent patterns in historical prices and momentum indicators.

The core innovation lies in detecting when both the **causal effect of a treatment** and **forecasted return direction** agree, indicating high-confidence trading signals. I trained causal forest models and uplift random forest classifiers to uncover heterogeneous treatment effects (HTEs) on stock returns. Meanwhile, I deployed LSTM models to generate return forecasts based on past patterns.

By integrating these two perspectives, I developed a framework that flags **aligned vs. misaligned periods**, visualizes their impact on cumulative profit/loss, and simulates trading strategies accordingly. This hybrid approach offers both **interpretability (via causal inference)** and **predictive strength (via RNN-LSTM)** for portfolio management decisions.

---

## Data Overview

To maintain consistency and relevance, I focused on **stock data from four major U.S. technology companies**:

* **Apple (AAPL)**
* **Microsoft (MSFT)**
* **Amazon (AMZN)**
* **Google/Alphabet (GOOGL)**

These firms were selected due to their **high market capitalization**, **liquidity**, and **global economic influence**, making them ideal candidates for testing economic causal effects on financial returns.


![download (33)](https://github.com/user-attachments/assets/af7171be-a45c-4769-ba4b-f73e33f77aff)

![download (34)](https://github.com/user-attachments/assets/d167a28f-2842-4ac9-b73d-7ebca74dbbd6)

**Data Sources and Structure:**

* Daily historical stock prices (Open, High, Low, Close, Volume)
* Derived features such as daily **returns**, **momentum**, and **volatility**
* Macro factors: **Interest rates**, **market sentiment scores**, and **news sentiment polarity**

Each stock's time series data was aligned with **external economic events**, enabling both **supervised learning** and **causal analysis of treatment effects** (e.g., how a spike in interest rates affects AAPL returns).

---

## Feature Selection Rationale

To support **causal modeling and predictive learning**, I included the following features:

### Outcome Variable:

* **Stock Returns**: Logarithmic return between consecutive closing prices.

### Treatment Variable:

* **Interest Rate Shock Indicator**: A binary variable indicating whether the interest rate is above its rolling median. This simulates the "treatment" in causal inference.

### Covariates:

* **Momentum**: Captures recent price trends; useful as a proxy for investor behavior.
* **Market Sentiment**: Derived from news APIs or public index (e.g., VIX, sentiment score); controls for public perception not captured in price data.
* **Trading Volume**: As a liquidity proxy, it helps isolate genuine price movements from noise.

These variables were selected to satisfy **backdoor criteria** in causal graphs, allowing estimation of the treatment effect while adjusting for potential confounders.

![download (37)](https://github.com/user-attachments/assets/d4db4883-4fe6-4b1a-a990-4ddd4690013f)

### Why Combine with LSTM?

While causal models offer insight into the **direction and heterogeneity** of effects, they lack temporal modeling capacity. To address this, I used **LSTM networks** to:

* Forecast daily returns based on lagged time series inputs
* Complement the causal model by detecting patterns that do not necessarily stem from external interventions

Together, this architecture balances **why** stock returns change (causal reasoning) with **how** they are likely to evolve (deep learning forecast), offering a powerful decision framework for financial applications.

![download (38)](https://github.com/user-attachments/assets/316444a6-651b-4743-b62a-2f7bd4be1beb)

![download (36)](https://github.com/user-attachments/assets/dae4bf76-0232-4e67-a6c5-d7d1ea25e652)

![download (35)](https://github.com/user-attachments/assets/f848265f-6259-487a-b371-c39f596a96db)

---

