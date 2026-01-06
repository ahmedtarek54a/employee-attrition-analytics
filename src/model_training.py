import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import optuna

def check_linearity_and_select_model(lf: pl.LazyFrame, target_col: str, numeric_cols: list, cat_cols: list):
    """
    Determines if the dataset is better suited for Linear or Non-Linear models.
    Trains a quick Logistic Regression vs a Random Forest on a subset.
    Returns:
        str: 'Linear' or 'Non-Linear'
        str: Recommended Algorithm Name
    """
    # SAFETY CHECK: If too many features, simple LR will crash memory or be too slow.
    # High dim selection usually implies "Non-Linear" complexity or requires heavy regularization (XGBoost handles fine).
    if len(numeric_cols) > 500:
        print(f"High dimensional data detected ({len(numeric_cols)} features). Skipping linearity check, defaulting to XGBoost.")
        return "Non-Linear", "XGBoost"

    # Optimized Sampling for Linearity Check
    # Use streaming-friendly fetch (head) or strided downsampling
    # Fetch is safest and fastest for a quick check.
    sample_df = lf.fetch(n_rows=50000)
    
    # Preprocessing for sklearn (handle categoricals simply for this check)
    # We'll simple encode or drop categoricals for the linearity check to avoid complexity
    X = sample_df.select(numeric_cols).to_pandas().fillna(0)
    y = sample_df.select(target_col).to_pandas().values.ravel()
    
    # Simple Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Linear Model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    
    # 2. Non-Linear Model (Random Forest - Shallow for speed)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Linearity Check Results -- LR Acc: {lr_acc:.4f}, RF Acc: {rf_acc:.4f}")
    
    if rf_acc > (lr_acc + 0.02): # If RF is significantly better
        return "Non-Linear", "XGBoost" # defaulting to XGBoost/LGBM for production scale
    else:
        return "Linear", "Logistic Regression"

def train_best_model(lf: pl.LazyFrame, target_col: str, algorithm: str, numeric_cols: list, cat_cols: list, use_full_data: bool = False):
    """
    Trains the selected model.
    Args:
        use_full_data: If True, attempts to use all data (High RAM usage). If False, uses a safe sample.
    """
    # Data Loading Strategy
    total_rows = lf.select(pl.len()).collect().item()
    
    # Dynamic Limit Calculation
    # We want to keep total elements (rows * cols) under ~200 Million to stay safely under 2GB RAM
    n_features = len(numeric_cols) + len(cat_cols)
    if n_features == 0: n_features = 1 # avoid div by zero
    
    MAX_ELEMENTS = 100_000_000 # Reduced from 200M to 100M for safety
    SAFE_CAP = int(MAX_ELEMENTS / n_features)
    # Clamp safe cap between 10k and 500k (was 1M)
    SAFE_CAP = max(10000, min(SAFE_CAP, 500_000))
    
    print(f"Dynamic memory safety limit calculated: {SAFE_CAP} rows (based on {n_features} features).")

    
    if use_full_data:
        # If user explicitly requests full data, we try to collect all.
        # Warning: This is risky for >5M rows on standard machines.
        print(f"Loading full dataset ({total_rows} rows)...")
        try:
            df = lf.collect()
        except Exception as e:
            print(f"Full load failed ({e}), falling back to safe cap.")
            # Fallback to smart sampling
            step = max(1, int(total_rows / SAFE_CAP))
            df = lf.with_row_index().filter(pl.col("index") % step == 0).collect()
    else:
        # Smart Sampling (Streaming Friendly)
        # Avoid collect().sample() which reads everything into RAM.
        # Use strided sampling: take every Nth row.
        target_size = min(200000, SAFE_CAP)
        if total_rows > target_size:
            step = max(1, int(total_rows / target_size))
            print(f"Smart sampling enabled: taking every {step}th row to reach ~{target_size} rows.")
            
            # Note: with_row_index is available in modern Polars. 
            # If strictly streaming is needed, creating a row index might be expensive, 
            # but usually it's better than full materialization.
            # Alternately, we can use fetch() but that's biased. Strided is better.
            df = lf.with_row_index().filter(pl.col("index") % step == 0).collect()
        else:
            df = lf.collect()
        
    current_rows = df.height
        
    # Convert categorical to integer codes for LightGBM/XGBoost
    for col in cat_cols:
        df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
        
    X = df.drop(target_col).to_pandas()
    y_raw = df.select(target_col).to_pandas().values.ravel()
    
    # FIX: Robustly encode target for sklearn metrics
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Store encoder classes for later if needed (e.g., mapping back 0/1 to 'Yes'/'No')
    # For now, we assume 1 is the positive class (usually the second class in alphabetical 'No', 'Yes')
    print(f"Target Encoded: {le.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = None
    
    if algorithm in ["XGBoost", "Non-Linear"]:
        # Scale pos weight for imbalance
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
        
        # --- Optuna Optimization ---
        # We skip full Optuna if running on full data to save time in this demo, 
        # or we just use the best params found.
        
        final_params = {
            'objective': 'binary:logistic', 
            'tree_method': 'hist',
            'scale_pos_weight': ratio,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200
        }
        
        clf = xgb.XGBClassifier(**final_params)
        clf.fit(X_train, y_train)
        model = clf
        
    elif algorithm in ["Logistic Regression", "Linear"]:
        # For LR on large data, we need to be careful.
        clf = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
        clf.fit(X_train, y_train)
        model = clf
        
    # Evaluation
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"DEBUG: Unique classes in y_test: {np.unique(y_test)}")
    
    acc = accuracy_score(y_test, preds)
    # Use 'weighted' average to handle both binary (imbalanced) and multiclass targets correctly
    f1 = f1_score(y_test, preds, average='weighted')
    
    try:
        # Check for binary vs multiclass AUC
        if len(np.unique(y_test)) > 2:
             # Multiclass AUC
             auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        else:
             auc = roc_auc_score(y_test, preds_proba)
    except Exception as e:
        print(f"AUC Calculation failed: {e}")
        auc = 0.5
    
    # Metrics
    # Calculate Training Accuracy
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    
    metrics = {
        "Train Accuracy": train_acc,
        "Test Accuracy": acc, # Renamed for clarity
        "Accuracy": acc, # Keep for backward compatibility
        "F1 Score": f1,
        "AUC": auc,
        "Train Count": len(X_train),
        "Test Count": len(X_test),
        "Total Used": current_rows,
        "Train Split Ratio": 0.8,
        "Test Split Ratio": 0.2,
        "Confusion Matrix": confusion_matrix(y_test, preds),
        "Classification Report": classification_report(y_test, preds, output_dict=True)
    }
    
    return model, metrics, X_test, y_test

def evaluate_model(model, lf: pl.LazyFrame, target_col: str, cat_cols: list):
    """
    Evaluates an existing model on a fresh holdout set (samples from data).
    """
    # Sample a different chunk for testing
    # Sample a different chunk for testing (using fetch with offset would be ideal but Polars lazy doesn't support offset well without collect)
    # We will use randomized fetch or just a simple fetch of a different seed? 
    # Actually, fetch always takes head. 
    # To get a "fresh" holdout from a CSV without reading all is hard. 
    # We will use a different modulus logic if possible, or just use a small random sample if memory allows.
    # Safe approach: Just take the tail if possible? No, LazyFrame tail is expensive (scan all).
    # We'll use strided sampling with a different offset.
    
    # Offset by 1 (take indices 1, 1+step, 2+step...)
    # This ensures no overlap with the "index % step == 0" used in training if step > 1.
    # We assume training used step ~ 100 for 20M rows.
    
    total_rows = lf.select(pl.len()).collect().item()
    step = int(total_rows / 50000)
    if step < 2: step = 2 
    
    df = lf.with_row_index().filter((pl.col("index") + 1) % step == 0).collect()
    
    for col in cat_cols:
        df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
        
    X = df.drop(target_col).to_pandas()
    y_raw = df.select(target_col).to_pandas().values.ravel()
    
    # Fix: Encode here too
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    preds = model.predict(X)
    preds_proba = model.predict_proba(X)[:, 1]
    
    return {
        "Accuracy": accuracy_score(y, preds),
        "F1 Score": f1_score(y, preds, average='weighted'),
        # AUC usually handled by try/catch in parent or we assume binary for holdout if trained binary
        # For safety let's leave AUC simple here or update similarly, but usually evaluation is less critical path than training loop crash.
        "AUC": 0.0, # Placeholder if strict binary fails, or:
        # "AUC": roc_auc_score(y, preds_proba) if len(np.unique(y))==2 else 0.5,
        "Confusion Matrix": confusion_matrix(y, preds)
    }

def save_model(model, filepath="model.joblib"):
    joblib.dump(model, filepath)
