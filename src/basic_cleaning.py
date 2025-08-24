import pandas as pd 
import numpy as np

## data cleaning frame work 

def clean_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    # log helper
    def log(msg):
        if verbose:
            print(f"[INFO] {msg}")

    # 1. standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    log("Standardized column names.")

    # 2. remove exact duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        log(f"Removed {dup_count} duplicate rows.")

    # 3. trim and lowercase all string (object) values
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip().str.lower()
    log("Standardized string columns (lowercase + trimmed).")

    # 4. detect missing values (including blanks and placeholders)
    placeholder_values = ['n/a', 'na', '--', '-', 'none', 'null', '', 'nan']
    df.replace(placeholder_values, np.nan, inplace=True)
    null_report = df.isnull().sum()
    null_report = null_report[null_report > 0]
    if not null_report.empty:
        log(f"Missing values found in columns:\n{null_report}")

    # 5. flag constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        log(f"Constant columns (consider removing): {constant_cols}")

    # 6. flag high cardinality categorical columns
    high_card_cols = [col for col in df.select_dtypes(include='object') if df[col].nunique() > 100]
    if high_card_cols:
        log(f"High-cardinality columns (consider encoding strategies): {high_card_cols}")

    # 7. detect numeric outliers using IQR
    num_cols = df.select_dtypes(include=np.number).columns
    outlier_report = {}
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
        if outliers > 0:
            outlier_report[col] = outliers
    if outlier_report:
        log(f"Potential numeric outliers detected:\n{outlier_report}")
    
    # 8. convert applicable columns to category
    for col in df.select_dtypes(include='object'):
        n_unique = df[col].nunique()
        if n_unique < len(df) * 0.05:
            df[col] = df[col].astype('category')
    log("Converted suitable object columns to category dtype.")

    log("Data cleaning complete.")
    return df
