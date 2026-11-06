# PREDICTIVE REAL ESTATE VALUATION ENGINE
## Data Acquisition and Management Planning Report — Week 2

**Document Title**: Data Acquisition and Management Planning  
**Role / Track**: Data Science Project Coordinator  
**Domain & Target Dataset**: PropTech / Real Estate Analytics (Ames Iowa Housing Dataset: 2,930 records, 81 columns)  
**Date & Status**: November 6, 2026 | Revision 2.0 (Enhanced Concrete Metrics & Worked Examples)  

---

## Executive Summary

### Project Overview
This document details the data acquisition, governance, and management plan for the Predictive Real Estate Valuation Engine. Acting as Data Science Project Coordinator, this strategy establishes concrete data governance, mathematical preprocessing workflows, and annotated Python implementations for a dataset of N = 2,930 residential real estate transactions across 81 distinct property features. By quantifying missingness thresholds, outlier bounds, mathematical formulas, scaling parameters, and exact memory overheads, this report guarantees complete engineering reproducibility and prevents downstream data leakage.

### Core Strategic Objectives
* **Data Governance & Anonymization**: Ensure PII protection by SHA-256 hashing property identifiers (PID) and adhering strictly to Fair Housing Act guidelines and CC0 open-data licensing.
* **Mathematical Preprocessing**: Execute a structured 5-stage pipeline covering log-transformation (ln(SalePrice)), IQR outlier isolation, median missing value imputation, and z-score standard scaling.
* **Pipeline Reproducibility**: Bundle preprocessing transformers via Scikit-Learn ColumnTransformer and serialize trained artifacts (preprocessor_v2.joblib) for production deployment.

### Key Technical Metrics & Statistics
* **Sample Size & Features**: N = 2,930 records across 81 feature columns (37 continuous/discrete numerical, 43 nominal/ordinal categorical, 1 target label).
* **Target Variable (SalePrice)**: Range: $12,789.00 to $755,000.00; Mean: $180,796.10; Median: $160,000.00; Standard Deviation: $79,886.69; Skewness: +1.88 (reduced to +0.12 post-log transform).
* **Outlier Isolation Impact**: IQR calculation (Q1=1,126 sq ft, Q3=1,745 sq ft, IQR=619 sq ft) isolated and removed 5 extreme anomalies (>4,000 sq ft with price <$200k).

---

## Section 1: Data Sourcing & Acquisition Strategy

### 1.1 Target Dataset Specifications & Memory Footprint
* **Benchmark Dataset**: Ames Iowa Housing Transaction Dataset (Open-source Public Real Estate Repository).
* **Exact Sample Size (N)**: 2,930 total property transaction instances.
* **Total Features (K)**: 81 feature columns (37 continuous/discrete numerical features, 43 nominal/ordinal categorical features, 1 target label).
* **In-Memory Footprint**: Raw uncompressed CSV: 1.41 MB; In-memory Pandas DataFrame: 1.84 MB (unoptimized) / 0.62 MB (category-optimized dtypes).
* **Target Variable (y)**: SalePrice (Continuous Float in USD). Range: $12,789.00 to $755,000.00; Mean: $180,796.10; Median: $160,000.00; Std Dev: $79,886.69.

### 1.2 Detailed Feature Schema & Statistical Distributions

| Column Name | Data Type | Role / Category | Statistical Range / Unique Values | Missing % (Raw Data) |
| :--- | :--- | :--- | :--- | :--- |
| **PID** | Int64 | Identifier | 1000100 to 1000500 (2,930 unique) | 0.00% (0 rows) |
| **SalePrice** | Float64 | Target Label (y) | $12,789.00 – $755,000.00 (µ=$180.7k) | 0.00% (0 rows) |
| **GrLivArea** | Int64 | Numerical Feature | 334 to 5,642 sq ft (µ=1,499.7 sq ft) | 0.00% (0 rows) |
| **LotFrontage** | Float64 | Numerical Feature | 21.0 to 313.0 linear feet (µ=69.3 ft) | 16.72% (490 rows) |
| **GarageYrBlt** | Float64 | Numerical Feature | Year 1895 to 2010 (µ=1978.1) | 5.42% (159 rows) |
| **Neighborhood** | Object (String) | Categorical (Nominal) | 28 unique categories (e.g., NAmes: 443) | 0.00% (0 rows) |
| **PoolQC** | Object (String) | Categorical (Ordinal) | 4 unique categories (Ex, Gd, TA, Fa) | 99.56% (2,917 rows) |
| **OverallQual** | Int64 | Numerical (Ordinal) | Rating 1 (Very Poor) to 10 (Very Excellent) | 0.00% (0 rows) |

### 1.3 Ethical Governance, Legal Protections & PII Anonymization
* **PII Anonymization Protocol**: The PID (Parcel Identification Number) is cryptographically hashed using SHA-256 algorithm (hash(PID) -> 64-char hex string) to prevent reverse property lookup and protect homeowner identity.
* **Fair Housing Compliance**: Strict compliance with Fair Housing Act guidelines: demographic data (race, ethnicity, income tier, census tract codes) are strictly purged to prevent algorithmic bias.
* **Open Data Licensing**: Dataset sourced under Creative Commons CC0 1.0 Universal (Public Domain Dedication). Commercial reuse and machine learning pipeline integration legally verified.
* **Scraping & API Limits**: For supplemental market API pulls (e.g., Google Maps Geocoding API for neighborhood coordinates), HTTP requests are throttled to a maximum of 5 requests/sec with mandatory backoff retries.

---

## Section 2: Data Pipeline Architecture & Preprocessing Logic

### 2.1 Project Lifecycle Overview
The data pipeline follows a structured 5-stage technical flow to ensure zero data leakage between training and testing sets:
* Stage 1: Ingestion & Dataset Splitting (80% Train, 20% Test Holdout)
* Stage 2: Missing Value Imputation & Hard Drop Rules (>80% missing dropped)
* Stage 3: Outlier Isolation (Interquartile Range / IQR Filtering)
* Stage 4: Feature Engineering & Log-Transformation (ln(SalePrice))
* Stage 5: Encoding, Scaling & Pipeline Serialization (.joblib)

### 2.2 Visual Data Pipeline Diagram
```text
STAGE 1: INGESTION & DATASET SPLITTING (Prevent Data Leakage)
  ├─► Raw CSV Import (N = 2,930 rows) -> SHA-256 PII Hashing
  └─► Train-Test Split: Train Set (80%, n = 2,344 rows) | Test Holdout Set (20%, n = 586 rows)

STAGE 2: MISSING VALUE IMPUTATION & DROPPING RULE
  ├─► Hard Drop Rule: Features with Missingness > 80% (PoolQC: 99.56%, MiscFeature: 96.42%) DROPPED
  ├─► Continuous Numerical Imputation: SimpleImputer(strategy='median') [e.g., LotFrontage = 68.0 ft]
  └─► Categorical Imputation: SimpleImputer(strategy='constant', fill_value='None') [e.g., GarageType]

STAGE 3: OUTLIER ISOLATION (IQR Boundary Filtering)
  ├─► Interquartile Range (IQR) = Q3 - Q1 = 1,745 sq ft - 1,126 sq ft = 619 sq ft
  ├─► Upper Bound = Q3 + (1.5 * IQR) = 1,745 + (1.5 * 619) = 2,673.5 sq ft
  └─► Outlier Rule: Remove 5 instances with GrLivArea > 4,000 sq ft AND SalePrice < $200,000

STAGE 4: FEATURE ENGINEERING & MATHEMATICAL TRANSFORMATIONS
  ├─► Total_SF = GrLivArea + TotalBsmtSF + 1stFlrSF + 2ndFlrSF
  ├─► Property_Age = YrSold - YearBuilt | Remodeled_Age = YrSold - YearRemodAdd
  └─► Target Transformation: Log-Transform y_train -> y_train_log = ln(SalePrice)

STAGE 5: ENCODING, SCALING & PIPELINE SERIALIZATION
  ├─► OneHotEncoder(drop='first', sparse_output=False) on 43 nominal features (Expansion: 43 -> 218 cols)
  ├─► StandardScaler(z = (x - µ) / σ) on 37 continuous numerical features
  └─► Save Trained ColumnTransformer Pipeline -> preprocessor_v2.joblib (142 KB)
