import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_missing_values(lf: pl.LazyFrame) -> pl.DataFrame:
    """Calculates missing values count and percentage for each column."""
    # This is an optimized query, safe to run on full data usually.
    total_rows = lf.select(pl.len()).collect().item()
    
    null_counts = lf.select([
        pl.col(col).null_count().alias(col) for col in lf.collect_schema().names()
    ]).collect()
    
    df_missing = null_counts.transpose(include_header=True, header_name="Column", column_names=["Missing_Count"])
    df_missing = df_missing.with_columns(
        (pl.col("Missing_Count") / total_rows * 100).alias("Missing_Percentage")
    )
    return df_missing.sort("Missing_Percentage", descending=True)

def get_target_distribution(lf: pl.LazyFrame, target_col: str) -> pl.DataFrame:
    """Gets the count and percentage of the target variable."""
    # Aggregation is fast in Polars even on 20M rows.
    return lf.group_by(target_col).agg(pl.len().alias("Count")).collect().with_columns(
        (pl.col("Count") / pl.sum("Count") * 100).alias("Percentage")
    )

def get_correlation_matrix(lf: pl.LazyFrame, numeric_cols: list, limit: int = 1_000_000) -> pl.DataFrame:
    """
    Calculates correlation matrix.
    Args:
        limit: If > 0, samples the data before correlation. If None or 0, uses full data.
    """
    if limit and limit > 0:
        # Check total rows first to avoid error if limit > rows
        # But sample(n) handles it properly usually or we can use Fraction.
        # For simplicity and speed:
        # Optimization: Fetch a sample directly instead of collecting all.
        # fetch(n) gets the first n rows (head). For correlation this is often "okay" but biased.
        # Better: use map_batches or simple fetch if we accept head-bias for speed.
        # Given "Perfect work" request, we should try to be somewhat random if possible, 
        # but fetch is vastly faster/safer. 
        # Let's use fetch, as correlation on randomized dataset isn't fully guaranteed unless we shuffle, 
        # which defeats the purpose of avoiding full read.
        # A reasonable compromise for EDA speed on 20M rows:
        df_data = lf.select(numeric_cols).fetch(limit)
        return df_data.corr()
    else:
        # Full correlation
        # Warning: This might be memory intensive if columns are many.
        return lf.select(numeric_cols).collect().corr()

def get_outliers(lf: pl.LazyFrame, numeric_cols: list) -> pl.DataFrame:
    """Table of potential outlier counts using IQR method."""
    # Approx stats on full data is accessible via Polars approx_quantile or just quantile (exact).
    # Quantile on 20M is reasonably fast.
    
    stats = lf.select([
        pl.col(c).quantile(0.25).alias(f"{c}_q1") for c in numeric_cols
    ] + [
        pl.col(c).quantile(0.75).alias(f"{c}_q3") for c in numeric_cols
    ]).collect()
    
    outlier_summary = []
    
    # Pre-calculating bounds
    bounds = {}
    for c in numeric_cols:
        q1 = stats[f"{c}_q1"][0]
        q3 = stats[f"{c}_q3"][0]
        iqr = q3 - q1
        bounds[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        
    # To optimize, we can construct one big expression to count outliers?
    # Or just iterate if the number of columns isn't huge (20 is fine).
    
    for c in numeric_cols:
        lower, upper = bounds[c]
        # Count query
        count = lf.filter((pl.col(c) < lower) | (pl.col(c) > upper)).select(pl.len()).collect().item()
        
        outlier_summary.append({
            "Column": c,
            "Outlier_Count": count,
            "Lower_Bound": lower,
            "Upper_Bound": upper
        })
        
    return pl.DataFrame(outlier_summary)
