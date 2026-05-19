------ Disclaimer ------- 

This project is for **educational and research purposes only**. Nothing in this repository constitutes financial advice. Trading futures involves significant risk of loss. Use at your own risk.

"ICT" refers to the trading concepts developed by Michael Huddleston (Inner Circle Trader). This project is not affiliated with, endorsed by, or promoting ICT or any related products/services.

**This code was developed with the assistance of AI.**

> **⚠️ Data not included**
> The CSV files are **not part of this repository**. Historical NQ futures data is licensed and must be purchased separately. The files exceed GitHub's file size limit and cannot be shared here. 
 **PURCHASE THE  CSV DATA FROM : https://www.backtestmarket.com/en/nasdaq-pack 
  
AND SAVE IT UNDER A FOLDER NAMED "DATA/"



# NQ ICT Strategy v3 – Backtest

Python backtest for an NQ futures strategy. For a full explanation of the strategy see the included Word documents.

---

## Installation

```bash
pip install numpy pandas matplotlib
```

Python 3.8 or higher required.

---

## Data

Place three CSV files in a `data/` folder next to `Backtest3.py`:

```
data/
  nq-5m.csv
  nq-15m.csv
  nq-1h.csv
```

**Format** – semicolon-separated, no header row, columns: `date;time;open;high;low;close;volume`

Date format: `DD/MM/YYYY` · Time format: `HH:MM:SS`

---

## Usage

```bash
python3 Backtest3.py
```

---

## Output

Results are saved automatically to the `results/` folder:

| File | Content |
|---|---|
| `ict_v3_trades.csv` | All trades with entry/exit details |
| `ict_v3_walkforward.csv` | Walk-forward results per period |
| `ict_v3_dashboard.png` | Full results dashboard |

---

## Sample Results

| Metric | Value |
|---|---|
| Trades | 138 |
| Win Rate | 32.6% |
| Profit Factor | 1.70 |
| Sharpe Ratio | 0.882 |
| Max Drawdown | −6.28% |
| CAGR | 4.70% |
| Final Equity | $125,798 |

---

