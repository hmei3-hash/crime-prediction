# Chicago Tourist Safety Analyzer

A data-driven tool that helps tourists assess neighborhood safety in Chicago by analyzing historical crime data, generating heatmaps, and predicting crime risk by location and time of day.

## Project Overview

Chicago attracts millions of visitors each year, but crime remains a concern for tourists unfamiliar with the city. This project uses the **Chicago Police Department's "Crimes — 2001 to Present"** dataset to provide actionable safety insights, including:

- Identification of high-risk blocks and neighborhoods
- Crime density heatmaps segmented by severity tier
- A machine learning model that predicts crime frequency for any block at any hour

## Crime Tier System

Crimes are categorized into two tiers based on their relevance and severity for tourists:

**Tier 1 — High-frequency, direct-contact crimes:** Robbery, Theft, Battery, Assault

**Tier 2 — Lower-frequency but severe crimes:** Burglary, Motor Vehicle Theft, Criminal Sexual Assault, Kidnapping, Stalking, Human Trafficking

## Project Structure

```
├── eda.ipynb      # Exploratory data analysis & visualization
├── safe.ipynb     # Safety heatmaps with KDE density estimation
├── deploy.ipynb   # Crime prediction model (Random Forest)
└── README.md
```

### `eda.ipynb` — Exploratory Data Analysis

- Loads and cleans the crime dataset (coordinate conversion from EPSG:3435 → WGS84)
- Plots geographic distribution of the top 5 crime types
- Identifies the top 10 most dangerous blocks
- Analyzes year-over-year crime trends
- Generates side-by-side KDE heatmaps for Tier 1 vs Tier 2 crimes

### `safe.ipynb` — Safety Heatmap Generation

- Builds KDE-based (Gaussian Kernel Density Estimation) heatmaps for each crime tier
- Overlays crime density on geographic scatter plots using red (Tier 1) and blue (Tier 2) color scales
- Uses KMeans clustering and custom scoring for zone-level risk assessment
- Outputs publication-quality heatmap images (`tier_heatmap.png`)

### `deploy.ipynb` — Predictive Model

- Engineers features from raw crime records: block encoding, hour of day, arrest rate, block-level crime totals, and coordinates
- Trains a **Random Forest Regressor** to predict crime count per block per hour
- Evaluates model performance with R² score on a held-out test set
- Generates 24-hour crime risk profiles for the top 30 most dangerous blocks
- Visualizes predicted hourly crime trends

## Tech Stack

- **Data Processing:** Pandas, NumPy, GeoPandas
- **Visualization:** Matplotlib, Seaborn, SciPy (Gaussian KDE)
- **Machine Learning:** scikit-learn (RandomForestRegressor, KMeans, LabelEncoder)
- **Geospatial:** GeoPandas (CRS transformation EPSG:3435 → EPSG:4326)

## Getting Started

### Prerequisites

```
pip install pandas numpy geopandas matplotlib seaborn scipy scikit-learn
```

### Data

Download the dataset from the [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) or via KaggleHub, and place the CSV file in your working directory.

### Usage

1. Run `eda.ipynb` to explore the data and understand crime distributions
2. Run `safe.ipynb` to generate safety heatmaps
3. Run `deploy.ipynb` to train the prediction model and generate hourly risk profiles

## Sample Output

The project produces:

- Geographic scatter plots colored by crime type
- Bar charts of the top 10 highest-crime blocks
- Side-by-side Tier 1 (red) / Tier 2 (blue) KDE heatmaps
- Hourly crime prediction curves for high-risk blocks

## Authors

CSE 163 Project Team

## License

This project is for educational purposes (CSE 163 coursework).
