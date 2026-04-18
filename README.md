# Chicago Tourist Safety Analyzer

A data-driven tool that helps tourists assess neighborhood safety in Chicago by analyzing historical crime data, generating heatmaps, and predicting crime risk by location and time of day.

## Project Overview

Chicago attracts millions of visitors each year, but crime remains a concern for tourists unfamiliar with the city. This project uses the **Chicago Police Department's "Crimes — 2001 to Present"** dataset to provide actionable safety insights:

- Identification of high-risk blocks and neighborhoods
- Crime density heatmaps segmented by severity tier
- A machine learning model that predicts crime frequency for any block at any hour
- Data-driven police deployment recommendations

## Crime Tier System

Crimes are categorized into two tiers based on their relevance and severity for tourists:

**Tier 1 — High-frequency, direct-contact crimes:** Robbery, Theft, Battery, Assault

**Tier 2 — Lower-frequency but severe crimes:** Burglary, Motor Vehicle Theft, Criminal Sexual Assault, Kidnapping, Stalking, Human Trafficking

## Project Structure

```
crime-prediction/
├── notebooks/
│   ├── eda.ipynb                 # Exploratory data analysis & visualization
│   ├── train.ipynb               # Random Forest model training & evaluation
│   ├── safe.ipynb                # Safety heatmaps with KDE density estimation
│   ├── deploy.ipynb              # Crime prediction model & hourly risk profiles
│   └── police_deployment.ipynb  # Police deployment recommendation system
├── scripts/
│   ├── eda.py                    # EDA script (coordinate conversion, top blocks)
│   ├── deploy.py                 # Optimized prediction for top 30 blocks
│   └── police_deployment.py     # Extended model with deployment recommendations
├── images/
│   ├── chicago_tier_heatmap.png
│   ├── tier_heatmap.png
│   ├── rf_crime_prediction.png
│   ├── test_prediction.png
│   ├── prediction_deployment_optimized.png
│   └── police_deployment_recommendation.png
├── requirements.txt
└── README.md
```

## Notebooks

### `eda.ipynb` — Exploratory Data Analysis

- Loads and cleans the crime dataset (coordinate conversion from EPSG:3435 → WGS84)
- Plots geographic distribution of the top 5 crime types
- Identifies the top 10 most dangerous blocks
- Analyzes year-over-year crime trends
- Generates side-by-side KDE heatmaps for Tier 1 vs Tier 2 crimes

### `train.ipynb` — Model Training & Evaluation

- Feature engineering: block encoding, hour of day, arrest rate, block-level crime totals
- Trains a **Random Forest Regressor** (100 estimators, max_depth=15)
- Achieves **R² = 0.626** on the held-out test set
- Produces 4-panel evaluation plot: feature importance, actual vs. predicted, residuals, metrics

### `safe.ipynb` — Safety Heatmap Generation

- Builds KDE-based (Gaussian Kernel Density Estimation) heatmaps for each crime tier
- Overlays crime density on geographic scatter plots using red (Tier 1) and blue (Tier 2) color scales
- Uses KMeans clustering and custom scoring for zone-level risk assessment
- Outputs `images/tier_heatmap.png`

### `deploy.ipynb` — Crime Prediction & Risk Profiles

- Predicts crime count per block per hour for the top 30 highest-risk blocks
- Generates 24-hour crime risk profiles and hourly trend visualizations

### `police_deployment.ipynb` — Police Deployment System

- Extends the prediction model with 7 features (block, hour, arrest rate, community area, IUCR, location description)
- Allocates officers at **1 officer per 2 predicted crimes**
- Outputs top 15 deployment recommendations with officer counts

## Model Performance

| Metric | Training | Test |
|--------|----------|------|
| R²     | 0.854    | 0.626 |
| RMSE   | 0.418    | 0.641 |
| MAE    | 0.160    | 0.236 |

**Top features by importance:** Block location · Hour of day · Historical block crime total · Arrest rate

## Sample Output

| Visualization | Description |
|---|---|
| `chicago_tier_heatmap.png` | Side-by-side Tier 1 / Tier 2 geographic heatmaps |
| `tier_heatmap.png` | KDE crime density overlays |
| `rf_crime_prediction.png` | Model evaluation: feature importance & residuals |
| `prediction_deployment_optimized.png` | 4-panel deployment overview |
| `police_deployment_recommendation.png` | Officer allocation by block |

## Tech Stack

- **Data Processing:** Pandas, NumPy, GeoPandas
- **Visualization:** Matplotlib, Seaborn, SciPy (Gaussian KDE)
- **Machine Learning:** scikit-learn (RandomForestRegressor, KMeans, LabelEncoder)
- **Geospatial:** GeoPandas (CRS transformation EPSG:3435 → EPSG:4326)

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Data

Download the dataset from the [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) or via KaggleHub, and place the CSV in the project root.

### Usage

Run notebooks in order:

```
1. notebooks/eda.ipynb          → explore data & distributions
2. notebooks/train.ipynb        → train & evaluate the model
3. notebooks/safe.ipynb         → generate safety heatmaps
4. notebooks/deploy.ipynb       → hourly crime risk profiles
5. notebooks/police_deployment.ipynb  → deployment recommendations
```

Or run the standalone scripts directly:

```bash
python scripts/eda.py
python scripts/deploy.py
python scripts/police_deployment.py
```

## Authors

CSE 163 Project Team

## License

This project is for educational purposes (CSE 163 coursework).
