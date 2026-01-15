# Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network

## Paper Analysis

**Title:** Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network

**Authors:** Jack Fanshawe, Rumi Masih, Alexander Cameron (University of Queensland)

**Publication:** arXiv:2601.04602v1, January 8, 2026

---

## Core Problem & Motivation

The paper addresses a fundamental limitation in statistical arbitrage (StatArb): **backward-looking dependence measures**. Traditional approaches using rolling-window correlations are inherently lagged—they continue reflecting previous regimes even after market conditions shift. During crises like COVID-19, correlations spike within days while windowed estimators take weeks to adjust, causing:
- Inaccurate cluster formation
- Collapsed trading baskets
- Significant trading losses

---

## Proposed Solution: THGNN Architecture

The authors propose a **Temporal-Heterogeneous Graph Neural Network (THGNN)** combining:

### 1. Temporal Encoder (Transformer)
| Component | Specification |
|-----------|---------------|
| Input | 30-day sequences x 37 features per stock |
| Projection | Linear to d_model = 128 |
| Layers | 4 pre-norm Transformer encoder layers |
| Attention heads | 8 (d_k = 16 per head) |
| Output | 512-dimensional node embeddings |
| Dropout | 0.2 |

### 2. Graph Construction (Daily)
- **Edge sampling:** Top 50 + bottom 50 correlations + 75 random mid-strength partners per node
- **Edge attributes:** Baseline correlation, |correlation|, sign indicator, same-sector flags
- **Relation class:** Low/neutral/high (0/1/2) based on correlation terciles

### 3. Relational Encoder (GAT)
| Component | Specification |
|-----------|---------------|
| Layers | 3 graph attention layers |
| Attention heads | 4 per layer |
| Node dimension | 512 |
| Expert heads | 3 MLPs routed by relation type (neg/mid/pos) |

### 4. Loss Function
- **Huber loss** for edge-wise predictions (robust to outliers)
- **Histogram matching loss** with Gaussian soft binning to prevent mode collapse
- Predictions made in **Fisher-z space** (variance stabilization)

---

## Data & Features (37 total)

| Category | Features |
|----------|----------|
| Price/Volume | Closing price, trading volume |
| Technical | 5/20/60-day momentum, RSI-14, ATR-14, 5-day reversal |
| Firm Characteristics | Market cap, book-to-market ratio |
| Factor Exposures | Rolling betas to FF3 (mkt, smb, hml) |
| Macro/Risk | VIX, crude oil, 10Y Treasury, dollar index, GARCH volatility |
| Returns | Daily excess return, raw return, SPY return |
| Sector | gsector, gsubind codes |
| Correlation Context | Rolling correlations with market (10/21/63-day), sector/subindustry correlations |

**Data Sources:** CRSP/Compustat via WRDS, FRED for macroeconomic variables

**Split:** Training 2006-2018, Out-of-sample 2019-2024

---

## Trading Framework

The model integrates with **SPONGEsym clustering** (Cartea et al., 2023):

1. THGNN produces NxN predicted correlation matrix
2. SPONGEsym clustering forms long/short baskets
3. Cluster count k determined by 90% explained variance criterion
4. **ML ensemble filtering** (5 classifiers: HGB, AdaBoost, MLP, LogReg, SGD)
5. Only **top 10% signals** by predicted profitability traded (~45 stocks)
6. **10-day rebalancing** with 4% take-profit threshold

---

## Results

### Correlation Prediction Performance (269M+ edges evaluated)

| Metric | Persistence Baseline | THGNN Model | Improvement |
|--------|---------------------|-------------|-------------|
| MAE | 0.3071 | 0.2302 | -25.0% |
| RMSE | 0.3852 | 0.2940 | -23.7% |
| Bias | -0.0049 | -0.0012 | -75.5% |
| Pearson r | 0.310 | 0.778 | +151% |
| Spearman rho | 0.314 | 0.795 | +153% |

**Key finding:** Variance explained increases from 9.6% to 60.5%

### Trading Performance (2019-2024)

| Metric | Strategy | S&P 500 | Improvement |
|--------|----------|---------|-------------|
| Annualized Return | 19.20% | 14.43% | +4.77% |
| Sharpe Ratio | 1.837 | 0.647 | +184% |
| Sortino Ratio | 2.473 | 0.786 | +215% |
| Maximum Drawdown | -9.43% | -33.93% | +24.5% |
| Max Loss Duration | 82 days | 512 days | -84% |
| Calmar Ratio | 2.035 | 0.425 | +379% |

**Statistical Significance:** Ledoit-Wolf bootstrap test p-value = 0.0200

---

## Interpretability Findings

### Feature Importance (Gradient x Input)
**Top 5 drivers:**
1. VIX (volatility)
2. Crude oil prices (DCOILWTICO)
3. 10-year Treasury yield (DGS10)
4. Trade-weighted dollar index (DTWEXBGS)
5. Sub-industry binding (gsubind)

**Regime-dependent behavior:** Macro factors dominate during stability; sector identifiers become crucial during crises.

### Sector Attention Dynamics
- **Energy:** Intra-sector attention spikes during COVID-19 (0.05 to 0.35), reflecting oil demand collapse
- **Financials:** Diversified attention (~75% across top 5 sectors), stable during systemic shocks
- **Real Estate:** Highly volatile attention, complex interdependencies, unexplained Healthcare dominance

---

## Limitations Acknowledged

1. No slippage/transaction costs in backtest
2. No comparison with alternative architectures (LSTM, CNN, other GNNs)
3. Single hyperparameter configuration (not optimized)
4. Limited interpretability tools (only Gradient x Input)
5. Unable to fully replicate Korniejczuk & Slepaczuk (2024) ML-filtered results

---

## Key Contributions

1. **First application** of GNNs to direct short-medium horizon equity correlation forecasting
2. **Demonstrated** that forward-looking correlations materially improve StatArb clustering
3. **Provided** interpretable attention-based insights into regime-dependent sector relationships
4. **Validated** that correlation should be treated as a predictable variable, not a fixed historical summary

---

## Practical Implications

For quantitative traders and portfolio managers:
- Replace rolling-window correlations with forward predictions
- Feed macro variables to ensure regime adaptability
- Inform hedging decisions by predicted (not historical) dependencies
- Use attention weights as regime-shift indicators

---

## References

- Original Paper: [arXiv:2601.04602v1](https://arxiv.org/abs/2601.04602)
- SPONGEsym Clustering: Cartea et al. (2023)
- ML-Filtered StatArb: Korniejczuk & Slepaczuk (2024)

---

*This README provides an analysis of the research paper included in this repository.*
