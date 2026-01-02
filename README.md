# PI Hunter

**NIH Grant Recruitment Value Estimator**

A Streamlit web application that searches NIH Reporter for a researcher's active grants and estimates the recruitment value (portable funding) they could bring if recruited to your institution.

![PI Hunter Screenshot](https://github.com/zhizhid/PI-hunter/raw/main/screenshot.png)

## Features

- **Search by PI name** with automatic disambiguation for common names
- **Profile-based tracking** using NIH profile_id to identify researchers across institutions
- **Remaining funds calculation** based on time remaining on each grant
- **Multi-year funding detection** for RF1, DP1, DP2, and R01s with large awards
- **Equal split for Multi-PI grants** - divides funds equally among all PIs
- **Portability scoring** based on grant mechanism (R01: 95%, U01: 50%, P30: 10%, etc.)
- **Deduplication** of multi-year grant records (keeps only latest fiscal year)
- **Export to CSV** for further analysis

## Installation

```bash
git clone https://github.com/zhizhid/PI-hunter.git
cd PI-hunter
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## How It Works

### Remaining Funds Calculation

**Standard grants (R01, R21, U01, etc.):**
```
Remaining = Annual Award × Years Remaining
```

**Multi-year funded grants (RF1, DP1, DP2, or detected large R01s):**
```
Remaining = Total Award × (Years Remaining ÷ Total Project Years)
```

### Multi-PI Split

For grants with multiple PIs, funds are divided equally:
```
PI Share = Remaining Funds ÷ Number of PIs
```

### Portability Scores

| Grant Type | Score | Notes |
|------------|-------|-------|
| R01, R21, R03, R37 | 0.95 | Highly portable |
| RF1, R35 (MIRA) | 0.90 | Portable |
| K-series | 0.85 | Follow the researcher |
| U01 | 0.50 | Depends on structure |
| P-series | 0.10-0.30 | Mostly institutional |
| T-series | 0.05 | Stay with institution |

### Final Calculation

```
Portable Value = PI's Remaining Funds × Portability Score
```

## Data Source

All data is sourced from [NIH Reporter](https://reporter.nih.gov/) via their public API.

## Limitations

- Does not account for pending renewals or new submissions
- Cannot predict carryover balances or unobligated funds
- Multi-PI splits assume equal distribution (actual splits may vary)
- Some grants have institutional dependencies not captured in grant type alone

## License

MIT License
