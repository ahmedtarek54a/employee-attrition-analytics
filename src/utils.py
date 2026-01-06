import polars as pl
import os

import os
import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "employee_attrition_20M_FINAL.csv")

def load_data(filepath: str = None) -> pl.LazyFrame:
    """
    Loads data lazily using Polars.
    """
    if filepath is None:
        filepath = DEFAULT_DATA_PATH

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")

    return pl.scan_csv(
        filepath,
        ignore_errors=True,
        infer_schema_length=10000
    )



def get_column_types(lf: pl.LazyFrame):
    """
    Returns lists of numerical and categorical columns.
    """
    schema = lf.collect_schema()
    num_cols = []
    cat_cols = []
    
    for name, dtype in schema.items():
        if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            num_cols.append(name)
        else:
            cat_cols.append(name)
            
    return num_cols, cat_cols

# --- Feature Configuration for Streamlit App ---
FEATURE_CONFIG = [
    {"name": "Age", "type": "number", "min": 18, "max": 100, "default": 30},
    {"name": "BusinessTravel", "type": "selectbox", "options": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]},
    {"name": "DailyRate", "type": "number", "min": 0, "max": 2000, "default": 800},
    {"name": "Department", "type": "selectbox", "options": ["Sales", "Research & Development", "Human Resources"]},
    {"name": "DistanceFromHome", "type": "number", "min": 1, "max": 100, "default": 5},
    {"name": "Education", "type": "selectbox", "options": [1, 2, 3, 4, 5], "help": "1:Below College, 2:College, 3:Bachelor, 4:Master, 5:Doctor"},
    {"name": "EducationField", "type": "selectbox", "options": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]},
    {"name": "EnvironmentSatisfaction", "type": "slider", "min": 1, "max": 4, "default": 3},
    {"name": "Gender", "type": "selectbox", "options": ["Male", "Female"]},
    {"name": "JobInvolvement", "type": "slider", "min": 1, "max": 4, "default": 3},
    {"name": "JobLevel", "type": "slider", "min": 1, "max": 5, "default": 2},
    {"name": "JobRole", "type": "selectbox", "options": ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"]},
    {"name": "JobSatisfaction", "type": "slider", "min": 1, "max": 4, "default": 3},
    {"name": "MaritalStatus", "type": "selectbox", "options": ["Single", "Married", "Divorced"]},
    {"name": "MonthlyIncome", "type": "number", "min": 1000, "max": 50000, "default": 5000},
    {"name": "NumCompaniesWorked", "type": "number", "min": 0, "max": 20, "default": 1},
    {"name": "OverTime", "type": "selectbox", "options": ["Yes", "No"]},
    {"name": "PercentSalaryHike", "type": "number", "min": 0, "max": 100, "default": 15},
    {"name": "PerformanceRating", "type": "slider", "min": 1, "max": 4, "default": 3},
    {"name": "TotalWorkingYears", "type": "number", "min": 0, "max": 50, "default": 10},
    {"name": "WorkLifeBalance", "type": "slider", "min": 1, "max": 4, "default": 3},
    {"name": "YearsAtCompany", "type": "number", "min": 0, "max": 40, "default": 5},
    {"name": "YearsInCurrentRole", "type": "number", "min": 0, "max": 40, "default": 3},
    {"name": "YearsSinceLastPromotion", "type": "number", "min": 0, "max": 40, "default": 1},
    {"name": "YearsWithCurrManager", "type": "number", "min": 0, "max": 40, "default": 2},
]
