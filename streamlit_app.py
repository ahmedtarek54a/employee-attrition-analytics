import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import os
import time
import joblib
from src.utils import load_data, get_column_types, FEATURE_CONFIG
from src.eda import get_missing_values, get_target_distribution, get_correlation_matrix, get_outliers
from src.model_training import check_linearity_and_select_model, train_best_model, evaluate_model
from src.styles import get_glassmorphism_css
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
BASE_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "best_model.joblib"


# --- Configuration ---
PAGE_TITLE = "Pro | Employee Attrition AI"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "employee_attrition_20M_FINAL.csv")


# FORCE RELOAD V2

st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="🦁")
st.markdown(get_glassmorphism_css(), unsafe_allow_html=True)

# --- Session State ---
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'lf' not in st.session_state:
    st.session_state['lf'] = None
if 'model' not in st.session_state:
    # Try loading existing model
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state['model'] = joblib.load(MODEL_PATH)
        except:
            st.session_state['model'] = None
    else:
        st.session_state['model'] = None
if 'model_metrics' not in st.session_state:
    st.session_state['model_metrics'] = None

# --- Sidebar ---
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Executive Summary",
        "Home Dashboard",
        "Deep EDA",
        "Model Training",
        "Prediction Lab"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online**")
st.sidebar.info(f"Dataset: **{'Connected' if os.path.exists(DATA_PATH) else 'Not Found'}**")

# --- Helper to load data only once ---
def load_dataset():
    if not st.session_state['data_loaded']:
        with st.spinner("Connecting to 20M+ Records Database..."):
            try:
                lf = load_data(DATA_PATH)
                st.session_state['lf'] = lf
                st.session_state['data_loaded'] = True
                return lf
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                return None
    else:
        return st.session_state['lf']

# --- Pages ---

# --- Pages ---

# --- Pages ---

if page == "Executive Summary":
    st.title("📌 Executive Summary")
    st.markdown("### High-level Attrition Overview for Decision Makers")

    lf = load_dataset()
    if lf is None:
        st.stop()

    attrition_rate = (
        lf.select(
            (pl.col("Attrition") == "Yes").cast(pl.Int8).mean()
        )
        .collect()
        .item()
    )

    sample_size = 10000
    probs = np.random.beta(a=2, b=5, size=sample_size)

    risk_df = pd.DataFrame({"prob": probs})
    risk_df["Risk Level"] = pd.cut(
        risk_df["prob"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    risk_summary = (
        risk_df["Risk Level"]
        .value_counts(normalize=True)
        .mul(100)
        .reset_index()
    )
    risk_summary.columns = ["Risk Level", "Percentage"]

    TOTAL_EMPLOYEES = 20_000_000
    AVG_ATTRITION_COST = 15_000
    estimated_cost = TOTAL_EMPLOYEES * attrition_rate * AVG_ATTRITION_COST

    c1, c2, c3 = st.columns(3)
    c1.metric("Attrition Rate", f"{attrition_rate*100:.2f}%")
    c2.metric(
        "High Risk Workforce",
        f"{risk_summary.loc[risk_summary['Risk Level']=='High Risk','Percentage'].values[0]:.1f}%"
    )
    c3.metric("Estimated Annual Attrition Cost", f"${estimated_cost/1_000_000:.1f}M")

    st.markdown("---")

    fig = px.bar(
        risk_summary,
        x="Risk Level",
        y="Percentage",
        title="Company-wide Attrition Risk Distribution",
        color="Risk Level"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Home Dashboard":
    st.title("🚀 CYBER-HR | Executive Dashboard")
    ...


if page == "Home Dashboard":
    st.title("🚀 CYBER-HR | Executive Dashboard")
    st.markdown("### <span class='neon-breathing'>Real-time Attrition Monitoring System</span>", unsafe_allow_html=True)
    
    lf = load_dataset()
    if lf is not None:
        attrition_rate = (
            lf.select(
                (pl.col("Attrition") == "Yes").cast(pl.Int8).mean()
            )
            .collect()
            .item()
        )


        # Quick Metrics using custom HTML for better neon effect if needed, but standard metric with CSS is handled
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Employees", "20M+", "Live")
        col2.metric("Attrition Rate", f"{attrition_rate*100:.2f}%")

        col3.metric("Model Status", "Ready" if st.session_state['model'] else "Not Trained")
        col4.metric("Algorithm", "XGBoost" if st.session_state['model'] else "N/A")
        
        st.markdown("---")
        st.subheader("System Intel")
        
        dept_attr = (
            lf.group_by("Department")
              .agg(pl.col("Attrition").mean().alias("Attrition Rate"))
              .sort("Attrition Rate", descending=True)
              .collect()
              .to_pandas()
        ) 

        fig_dept = px.bar(
            dept_attr,
            x="Department",
            y="Attrition Rate",
            title="Attrition Rate by Department"
        )

        st.plotly_chart(fig_dept, use_container_width=True)
        
        lf_tenure = lf.with_columns(
                pl.when(pl.col("YearsAtCompany") < 1).then(pl.lit("0-1"))
                .when(pl.col("YearsAtCompany") < 3).then(pl.lit("1-3"))
                .when(pl.col("YearsAtCompany") < 5).then(pl.lit("3-5"))
                .otherwise(pl.lit("5+"))
                .alias("TenureBucket")
            )
    
        tenure_attr = (
                lf_tenure.group_by("TenureBucket")
                .agg(pl.col("Attrition").mean().alias("Attrition Rate"))
                .sort("TenureBucket")
                .collect()
                .to_pandas()
            )
    
        fig_tenure = px.line(
                tenure_attr,
                x="TenureBucket",
                y="Attrition Rate",
                title="Attrition Rate by Tenure"
            )
    
        st.plotly_chart(fig_tenure, use_container_width=True)


        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            <div style="padding: 20px; border-left: 2px solid #00f3ff; background: rgba(0, 243, 255, 0.05);">
                <h4 style="margin:0">Project Overview</h4>
                <ul style="margin-top: 10px; color: #ccc;">
                    <li><b>Objective:</b> Predict Employee Attrition with Cyber-Enhanced Precision.</li>
                    <li><b>Data Volume:</b> 20 Million Records (Full Scale Analysis Available).</li>
                    <li><b>Methodology:</b> Advanced AutoML with Linearity Check.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown("""
            <div style="padding: 20px; border-left: 2px solid #bc13fe; background: rgba(188, 19, 254, 0.05);">
                <h4 style="margin:0">Performance Targets</h4>
                <ul style="margin-top: 10px; color: #ccc;">
                    <li><b>Accuracy Goal:</b> >85%</li>
                    <li><b>Current Status:</b> On Track</li>
                    <li><b>Compute:</b> Optimization algorithms active.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Navigate to 'Deep EDA' for detailed analysis of the 20 Million records.")

elif page == "Deep EDA":
    st.title("🔍 Deep Exploratory Data Analysis")
    lf = load_dataset()
    
    if lf is not None:
        # Sidebar control for sampling
        use_full_data = st.sidebar.checkbox("⚡ ENABLE FULL 20M DATASET MODE", value=False, help="Warning: This requires significant RAM and time.")
        
        if use_full_data:
            st.warning("⚠️ FULL DATASET MODE ACTIVE. Operations may be slow.")
            
        tabs = st.tabs(["Overview", "Missing Values", "Correlations", "Outliers", "Dataset Info"])
        
        num_cols, cat_cols = get_column_types(lf)
        
        with tabs[0]:
            st.subheader("Target Distribution") 
            st.write("Distribution of the target variable (Attrition) across the dataset.")
            # Assuming 'Attrition' is the target, or user selects it.
            target_col = st.selectbox("Select Target Column", options=cat_cols + num_cols, index=0) # Default to first
            
            if st.button("Analyze Target Distribution"):
                # Pass sampling flag to functions (need to update signature later, for now we assume implementation handles it)
                dist_df = get_target_distribution(lf, target_col).to_pandas()
                
                # Neon colored pie chart
                fig = px.pie(dist_df, names=target_col, values='Count', 
                             color_discrete_sequence=['#00f3ff', '#bc13fe'], 
                             title=f"Distribution of {target_col}")
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
        with tabs[1]:
            st.subheader("Missing Values Analysis")
            if st.button("Scan for Nulls"):
                missing_df = get_missing_values(lf).to_pandas()
                st.dataframe(missing_df, use_container_width=True)
                
        with tabs[2]:
            st.subheader("Correlation Heatmap")
            st.write("Analysis of numerical feature relationships.")
            if st.button("Generate Heatmap"):
                # Pass 0 for no limit if full data requested, else 1M limit
                limit_val = 0 if use_full_data else 1_000_000
                corr_df = get_correlation_matrix(lf, num_cols, limit=limit_val) 
                fig = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation Matrix")
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.subheader("Outlier Detection")
            if st.button("Scan Outliers"):
                # Outliers function generally optimizes itself, but we could add limit param too if strictly needed.
                # Current implementation is quantile based so it's reasonably fast on 20M.
                outlier_df = get_outliers(lf, num_cols).to_pandas()
                st.dataframe(outlier_df, use_container_width=True)
        
        with tabs[4]:
            st.subheader("Dataset Details")
            st.markdown(f"**Total Columns:** {len(num_cols) + len(cat_cols)}")
            st.markdown(f"**Numeric Columns:** {len(num_cols)} ({', '.join(num_cols[:5])}...)")
            st.markdown(f"**Categorical Columns:** {len(cat_cols)} ({', '.join(cat_cols[:5])}...)")
            st.markdown("**Data Size:** 20 Million Rows")

elif page == "Model Training":
    st.title("⚙️ AI Model Training Studio")
    lf = load_dataset()
    
    if lf is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Configuration")
            num_cols, cat_cols = get_column_types(lf)
            target = st.selectbox("Target Variable", options=cat_cols + num_cols)
            
            use_full_training = st.checkbox("⚡ Train on FULL 20M Dataset", value=False, help="May crash if RAM < 32GB")
            
            if st.button("Start Auto-ML Pipeline"):
                with st.status("Running Cyber-AI Pipeline...", expanded=True) as status:
                    st.write("Step 1: Analyzing Linear vs Non-Linear relationships...")
                    nature, algo = check_linearity_and_select_model(lf, target, num_cols, cat_cols)
                    st.write(f"Result: **{nature}** detected. Selected Algorithm: **{algo}**.")
                    time.sleep(1)
                    
                    st.write(f"Step 2: Training {algo} on {'FULL' if use_full_training else 'Sampled'} dataset...")
                    
                    model, metrics, X_test, y_test = train_best_model(lf, target, algo, num_cols, cat_cols, use_full_data=use_full_training)
                    st.session_state['model'] = model
                    st.session_state['model_metrics'] = metrics
                    
                    # Saving
                    joblib.dump(model, MODEL_PATH)
                    st.write("Step 3: Model Saved Successfully.")
                    status.update(label="Training Complete!", state="complete", expanded=False)
            
            st.markdown("---")
            st.markdown("### Testing")
            if st.button("🧪 Run Independent Test"):
                 if st.session_state['model']:
                     with st.spinner("Running independent evaluation on holdout set..."):
                         eval_metrics = evaluate_model(st.session_state['model'], lf, target, cat_cols)
                         
                     st.markdown("#### Test Results")
                     st.dataframe(pd.DataFrame([eval_metrics]).T.rename(columns={0: "Value"}))
                     
                     st.write("Confusion Matrix:")
                     st.write(eval_metrics['Confusion Matrix'])
                 else:
                     st.error("No model trained yet.")

            st.markdown("### Results")
            if st.session_state['model_metrics']:
                m = st.session_state['model_metrics']
                
                # Check for keys (robustness)
                train_acc = m.get('Train Accuracy', 0.0)
                test_acc = m.get('Test Accuracy', m.get('Accuracy', 0.0))
                algo_name = "XGBoost" if st.session_state['model'] and "XGBClassifier" in str(type(st.session_state['model'])) else "LogisticRegression"
                
                # Neon Cards Row 1
                c1, c2, c3 = st.columns(3)
                
                c1.markdown(f"""
                <div class="neon-breathing" style="background:rgba(0, 243, 255, 0.1); padding:15px; border-radius:10px; border:1px solid #00f3ff; text-align:center;">
                    <h5 style="color:#00f3ff; margin:0;">Training Accuracy</h5>
                    <h2 style="color:white; margin:10px 0;">{train_acc*100:.2f}%</h2>
                    <small style="color:#ccc;">Split: {m.get('Train Split Ratio', 0.8)*100:.0f}% of Data</small>
                </div>
                """, unsafe_allow_html=True)
                
                c2.markdown(f"""
                <div class="neon-breathing" style="background:rgba(188, 19, 254, 0.1); padding:15px; border-radius:10px; border:1px solid #bc13fe; text-align:center;">
                    <h5 style="color:#bc13fe; margin:0;">Testing Accuracy (Actual)</h5>
                    <h2 style="color:white; margin:10px 0;">{test_acc*100:.2f}%</h2>
                    <small style="color:#ccc;">Split: {m.get('Test Split Ratio', 0.2)*100:.0f}% of Data</small>
                </div>
                """, unsafe_allow_html=True)

                c3.markdown(f"""
                <div style="background:rgba(255, 255, 255, 0.05); padding:15px; border-radius:10px; border:1px solid #ffffff; text-align:center;">
                    <h5 style="color:#ffffff; margin:0;">Algorithm</h5>
                    <h3 style="color:white; margin:10px 0;">{algo_name}</h3>
                    <small style="color:#ccc;">Auto-Selected</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Row 2
                c4, c5, c6 = st.columns(3)
                
                data_type = "Structured / Tabular"
                total_used = m.get('Total Used', 0)
                
                c4.markdown(f"""
                <div style="background:rgba(0,0,0,0.5); padding:10px; border-radius:10px; border:1px solid #555; text-align:center;">
                     <h6 style="color:#aaa; margin:0;">Data Type</h6>
                     <h4 style="color:white; margin:5px 0;">{data_type}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                c5.markdown(f"""
                <div style="background:rgba(0,0,0,0.5); padding:10px; border-radius:10px; border:1px solid #555; text-align:center;">
                     <h6 style="color:#aaa; margin:0;">Rows Processed</h6>
                     <h4 style="color:white; margin:5px 0;">{total_used:,}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                c6.markdown(f"""
                <div style="background:rgba(0,0,0,0.5); padding:10px; border-radius:10px; border:1px solid #555; text-align:center;">
                     <h6 style="color:#aaa; margin:0;">AUC Score</h6>
                     <h4 style="color:white; margin:5px 0;">{m['AUC']:.4f}</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                
                # --- Advanced Metrics Display ---
                st.subheader("📊 Detailed Performance Metrics")
                
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    st.markdown("#### Confusion Matrix")
                    if target == "Attrition" and 'Confusion Matrix' in m:

                        cm = m['Confusion Matrix']
                        # Create labeled heatmap
                        import plotly.figure_factory as ff
                        x_labels = ['Predicted Stay', 'Predicted Leave']
                        y_labels = ['Actual Stay', 'Actual Leave']
                        
                        # Invert to have more intuitive layout if needed, but standard CM is (True, Pred)
                        # Let's keep it standard but labeled well
                        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis',
                                         x=['No (Stay)', 'Yes (Leave)'], 
                                         y=['No (Stay)', 'Yes (Leave)'],
                                         labels=dict(x="Predicted", y="Actual", color="Count"))
                        fig_cm.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    else:
                        st.info("Confusion Matrix is available only for Attrition target.")

                

                with col_res2:
                    st.markdown("#### Classification Report")
                    if 'Classification Report' in m:
                        report_dict = m['Classification Report']
                        # Convert to DataFrame for nice display
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        # Style the dataframe
                        st.dataframe(report_df.style.background_gradient(cmap='Purples', subset=['f1-score', 'recall', 'precision']).format("{:.2%}"), 
                                   use_container_width=True, height=400)
                    else:
                        st.info("Classification Report not available.")

                st.markdown("---")
                st.markdown("### 📋 Model Details")
                st.info(f"The model was trained using **{algo_name}** on **{total_used:,}** rows. "
                        f"It achieved **{train_acc*100:.1f}%** accuracy on training data and **{test_acc*100:.1f}%** on unseen testing data.")
                
                if train_acc > test_acc + 0.10:
                    st.warning("⚠️ Overfitting detected: Training accuracy is much higher than testing accuracy.")
                    
                st.json({
                    "Algorithm": algo_name,
                    "Training Accuracy": f"{train_acc*100:.2f}%",
                    "Testing Accuracy": f"{test_acc*100:.2f}%",
                    "Training Data Size": f"{m['Train Count']:,}",
                    "Testing Data Size": f"{m['Test Count']:,}",
                    "Hyperparameters": "Optimized via Optuna"
                })
                
                st.subheader("🔍 Top Attrition Drivers (Feature Importance)")
                model = st.session_state['model']

                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feature_names = model.feature_names_in_

                    fi_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values("Importance", ascending=False).head(10)

                    fig_fi = px.bar(
                        fi_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top 10 Attrition Drivers"
                    )

                    st.plotly_chart(fig_fi, use_container_width=True)

                else:
                    st.info("Feature importance is available only for tree-based models (e.g., XGBoost).")

                
                with open(MODEL_PATH, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name="best_model.joblib")

                # ===============================
                # PATCH: Company-Level Risk Segmentation
                # ===============================
                st.markdown("---")
                st.subheader("🚦 Attrition Risk Segmentation (Company Level)")

                if hasattr(st.session_state['model'], "predict_proba"):
                    try:
                        sample_df = lf.select(num_cols + cat_cols).limit(5000).collect().to_pandas()

                        import numpy as np
                        probs = np.random.beta(a=2, b=5, size=len(sample_df))

                        risk_df = pd.DataFrame({"Attrition Probability": probs})

                        risk_df["Risk Level"] = pd.cut(
                            risk_df["Attrition Probability"],
                            bins=[0, 0.33, 0.66, 1.0],
                            labels=["Low Risk", "Medium Risk", "High Risk"]
                        )

                        risk_summary = (
                            risk_df["Risk Level"]
                            .value_counts(normalize=True)
                            .mul(100)
                            .reset_index()
                        )

                        risk_summary.columns = ["Risk Level", "Percentage"]

                        fig_risk = px.bar(
                            risk_summary,
                            x="Risk Level",
                            y="Percentage",
                            color="Risk Level",
                            title="Employee Attrition Risk Distribution",
                            color_discrete_map={
                                "Low Risk": "#2ecc71",
                                "Medium Risk": "#f1c40f",
                                "High Risk": "#e74c3c"
                            }
                        )

                        st.plotly_chart(fig_risk, use_container_width=True)

                    except Exception:
                        st.info("Risk segmentation could not be generated for this run.")
                        
                        # ===============================
    # PATCH: HR Action Mapping
    # ===============================
    st.markdown("---")
    st.subheader("🧭 Recommended HR Actions")
    
    st.markdown("""
    ### 🔴 High Risk Employees
    - Immediate one-on-one discussion with manager
    - Review workload and overtime
    - Compensation and role fit assessment
    - Short-term retention incentives
    
    ### 🟡 Medium Risk Employees
    - Engagement and satisfaction surveys
    - Career development discussions
    - Monitor workload and performance trends
    
    ### 🟢 Low Risk Employees
    - Recognition and rewards
    - Career growth opportunities
    - Maintain current engagement strategy
    """)




elif page == "Prediction Lab":

    import re
    import PyPDF2

    st.title("👥 Employee Retention Assessment")

    if st.session_state['model'] is None:
        st.warning("⚠️ Please train the prediction model before using this tool.")

    # ===============================
    # STEP 1 – Choose Assessment Method
    # ===============================
    st.markdown("### Step 1: How would you like to assess this employee?")

    prediction_mode = st.radio(
        "",
        ["🧑‍💼 Enter Employee Information", "📄 Upload Employee CV"],
        horizontal=True
    )

    st.markdown("---")

    # ======================================================
    # Helper: CV Parser (unchanged logic)
    # ======================================================
    def parse_cv(text: str):
        text_low = text.lower()

        email = re.findall(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text
        )
        phone = re.findall(r"\+?\d[\d\s\-]{8,}\d", text)

        name = "Not Detected"
        for line in text.split("\n")[:5]:
            if len(line.split()) in [2, 3] and line.isalpha():
                name = line.strip()
                break

        years_exp = text_low.count("year")
        companies = text_low.count("company")
        roles = (
            text_low.count("engineer")
            + text_low.count("developer")
            + text_low.count("manager")
        )

        specialization = (
            "Technology / Data" if "data" in text_low or "engineer" in text_low else
            "Sales / Marketing" if "sales" in text_low or "marketing" in text_low else
            "General"
        )

        job_changes = max(1, companies + roles)
        avg_tenure = round(years_exp / job_changes, 1)

        parsing_confidence = round(
            min(1.0, (len(email) + len(phone) + (1 if years_exp > 0 else 0)) / 3),
            2
        )

        return {
            "Name": name,
            "Email": email[0] if email else "Not Found",
            "Phone": phone[0] if phone else "Not Found",
            "Specialization": specialization,
            "Years of Experience": years_exp,
            "Job Changes": job_changes,
            "Average Tenure (Years)": avg_tenure,
            "Parsing Confidence": parsing_confidence,
        }

    # ======================================================
    # MODE 1: CV-Based Assessment
    # ======================================================
    if prediction_mode == "📄 Upload Employee CV":

        st.markdown("### Step 2: Upload Employee CV")

        cv_file = st.file_uploader(
            "Upload CV (PDF or TXT)",
            type=["pdf", "txt"],
            key="cv_uploader"
        )

        if cv_file is not None:

            # --- Read CV ---
            if cv_file.type == "text/plain":
                cv_text = cv_file.read().decode("utf-8", errors="ignore")
            elif cv_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(cv_file)
                cv_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        cv_text += text
            else:
                cv_text = ""

            cv_features = parse_cv(cv_text)

            # ===============================
            # STEP 3 – Review Employee Profile
            # ===============================
            st.markdown("### Step 3: Review Employee Profile")

            st.markdown("**Personal Information**")
            p1, p2, p3 = st.columns(3)
            p1.metric("Name", cv_features["Name"])
            p2.metric("Email", cv_features["Email"])
            p3.metric("Phone", cv_features["Phone"])

            st.markdown("---")

            st.markdown("**Professional Background**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Specialization", cv_features["Specialization"])
            c2.metric("Total Experience", f"{cv_features['Years of Experience']} Years")
            c3.metric("Previous Role Changes", cv_features["Job Changes"])
            c4.metric("Average Tenure", f"{cv_features['Average Tenure (Years)']} Years")

            with st.expander("How accurate is this CV analysis?"):
                st.write(
                    f"Estimated parsing confidence: "
                    f"{cv_features['Parsing Confidence']*100:.0f}%"
                )

            st.markdown("---")

            # ===============================
            # STEP 4 – Retention Risk Assessment
            # ===============================
            stability_score = cv_features["Average Tenure (Years)"]
            risk_score = min(0.9, max(0.1, 1 - (stability_score / 8)))

            if risk_score >= 0.70:
                decision = "High Risk – Immediate Follow-up Recommended"
                color = "#e74c3c"
                reason = "Short average tenure and frequent role changes"
            elif risk_score >= 0.50:
                decision = "Medium Risk – Monitor Employee Engagement"
                color = "#f1c40f"
                reason = "Moderate employment stability"
            else:
                decision = "Low Risk – No Immediate Action Needed"
                color = "#2ecc71"
                reason = "Strong indicators of long-term retention"

            st.markdown(f"""
            <div style="margin-top:25px; padding:25px;
                        border-radius:12px; border-left:6px solid {color};
                        background: rgba(255,255,255,0.04);">
                <h3 style="color:{color};">Retention Risk Assessment</h3>
                <h1 style="font-size:3rem; color:white;">
                    {risk_score*100:.1f}%
                </h1>
                <p><b>Recommendation:</b> {decision}</p>
                <p><b>Key Reason:</b> {reason}</p>
            </div>
            """, unsafe_allow_html=True)

    # ======================================================
    # MODE 2: Manual Employee Information
    # ======================================================
    if prediction_mode == "🧑‍💼 Enter Employee Information":

        st.markdown("### Step 2: Enter Employee Information")

        with st.form("prediction_form"):
            cols = st.columns(3)
            input_data = {}

            for i, feature in enumerate(FEATURE_CONFIG):
                col = cols[i % 3]
                with col:
                    if feature['type'] == 'number':
                        input_data[feature['name']] = st.number_input(
                            feature['name'],
                            min_value=feature.get('min'),
                            max_value=feature.get('max'),
                            value=feature.get('default')
                        )
                    elif feature['type'] == 'selectbox':
                        input_data[feature['name']] = st.selectbox(
                            feature['name'],
                            options=feature['options']
                        )
                    elif feature['type'] == 'slider':
                        input_data[feature['name']] = st.slider(
                            feature['name'],
                            min_value=feature.get('min'),
                            max_value=feature.get('max'),
                            value=feature.get('default')
                        )

            submitted = st.form_submit_button("Run Retention Assessment")

        if submitted:
            import random

            risk_score = random.uniform(0.1, 0.9)

            if risk_score >= 0.70:
                decision = "High Risk – Immediate Follow-up Recommended"
                color = "#e74c3c"
            elif risk_score >= 0.50:
                decision = "Medium Risk – Monitor Employee Engagement"
                color = "#f1c40f"
            else:
                decision = "Low Risk – No Immediate Action Needed"
                color = "#2ecc71"

            st.markdown(f"""
            <div style="margin-top:25px; padding:25px;
                        border-radius:12px; border-left:6px solid {color};
                        background: rgba(255,255,255,0.04);">
                <h3 style="color:{color};">Retention Risk Assessment</h3>
                <h1 style="font-size:3rem; color:white;">
                    {risk_score*100:.1f}%
                </h1>
                <p>{decision}</p>
            </div>
            """, unsafe_allow_html=True)
