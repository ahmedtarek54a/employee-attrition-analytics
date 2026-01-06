import polars as pl

def compute_attrition_kpis(lf: pl.LazyFrame):
    return (
        lf.select([
            pl.count().alias("total_employees"),
            pl.col("Attrition").mean().alias("attrition_rate")
        ])
    )

def attrition_by_department(lf: pl.LazyFrame):
    return (
        lf.groupby("Department")
        .agg([
            pl.count().alias("employees"),
            pl.col("Attrition").mean().alias("attrition_rate")
        ])
        .sort("attrition_rate", descending=True)
    )

def add_tenure_bucket(lf: pl.LazyFrame):
    return lf.with_columns(
        pl.when(pl.col("YearsAtCompany") < 1).then("0-1 year")
        .when(pl.col("YearsAtCompany") < 3).then("1-3 years")
        .when(pl.col("YearsAtCompany") < 5).then("3-5 years")
        .otherwise("5+ years")
        .alias("TenureBucket")
    )

def attrition_by_tenure(lf: pl.LazyFrame):
    return (
        lf.groupby("TenureBucket")
        .agg([
            pl.count().alias("employees"),
            pl.col("Attrition").mean().alias("attrition_rate")
        ])
        .sort("TenureBucket")
    )
