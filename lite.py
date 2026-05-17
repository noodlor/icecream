import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import subprocess
import re
import warnings

# Suppress seaborn warnings for clean cloud execution
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ==========================================
# SURVEY PREFIX SETTINGS (EDIT THESE)
# ==========================================
SURVEY_PREFIXES = {
    "code": "ID:",                   
    "overall": "Q1:",                
    "attr_regex": r"^Q[2-9]\d*:"     
}

# ==========================================
# USER-FACING TEXT STRINGS (EDIT THESE)
# ==========================================
UI_TEXT = {
    "app_title": "Sensory analysis tool",
    "app_subtitle": "Upload raw survey data or a previously processed matrix to generate statistical leaderboards.",
    "err_missing_lib": "Missing required python library: statsmodels. Please add it to your requirements.txt file.",
    
    # Step 1: Upload
    "step1_header": "1. Data upload",
    "step1_upload_label": "Upload raw survey data or processed matrix (CSV)",
    "err_read_survey": "Could not read the survey file. Ensure it is a valid CSV. Error: {e}",
    "msg_processed_matrix": "Detected previously processed matrix. Survey configuration and mapping bypassed.",
    
    # Step 2: Configuration
    "step2_header": "2. Survey configuration",
    "inference_msg": "Based on your column headers, we detected **{servings}** samples per taster. We also auto-detected your descriptive attributes. Review and adjust below if necessary.",
    "label_taster_id": "Taster ID column name",
    "label_servings": "Samples per taster",
    "label_attributes": "Descriptive attributes (comma-separated)",
    "label_mapping_expander": "Review column mappings (advanced)",
    
    # Step 3: Product Mapping
    "step3_header": "3. Product mapping",
    "step3_desc": "We extracted the unique codes from your survey based on your column selections. Upload a master key to automatically assign product names, or manually type them into the grid.",
    "step3_upload_key": "Upload master key (CSV - optional)",
    "msg_key_success": "Perfect match: successfully linked all {count} survey codes to product names.",
    "msg_key_partial": "Partial match: found {matched} matches, but {missing} survey codes are missing from your key.",
    "msg_key_mismatch": "Master key mismatch: we loaded {loaded} names from your key, but none of them match the codes found in your survey data. Please check your file.",
    "err_key_cols": "Could not identify the code and name columns in your master key.",
    "step3_edit_desc": "Verify and edit product names:",
    "step3_edit_hint": "This table is interactive. Click directly into the 'Product name' column to manually type or edit a brand name.",
    
    # Step 4: Analysis Execution & Errors
    "btn_run": "Run statistical analysis",
    "spinner_running": "Stacking data and calculating statistics...",
    "err_no_numbers": "Analysis failed: The overall liking column contains no valid numbers. Please apply a numerical mapping to your raw data before uploading.",
    "err_no_variance": "Analysis failed: All recorded scores are exactly identical (zero variance). Statistical analysis cannot be performed without variance.",
    "err_anova_fail": "Analysis failed during execution. Please verify your data formatting. Error details: {e}",
    
    # Step 5: Executive Summary
    "summary_header": "Executive summary",
    "win_sig_title": "Significant difference detected",
    "win_sig_desc": "The panel concluded that <strong>{winners}</strong> is the top-performing product.",
    "win_tie_title": "Statistical tie",
    "win_tie_desc": "The panel found differences overall, but <strong>{winners}</strong> are statistically tied for first place.",
    "win_none_title": "No consensus",
    "win_none_desc": "The panel could not detect a statistically reliable difference between any of the products.",
    
    # Metrics
    "stat_rse_label": "Residual Standard Error (RSE): {rse:.2f}",
    "stat_rse_context": "Context: Lab panels are typically < 1.0; standard consumer tests are 1.0 - 1.8. The RSE represents the standard deviation of the data after mathematically removing true product differences and individual taster bias. A higher RSE indicates a highly unpredictable panel.",
    "stat_pval_label": "Parametric p-value (ANOVA): {pval:.4f}",
    "stat_pval_context": "Context: The ANOVA p-value represents the statistical probability that the observed differences in average scores are due strictly to random chance. A value below 0.05 is typically required to prove a significant difference.",
    "stat_rank_pval_label": "Nonparametric p-value ({test}): {pval:.4f}",
    "stat_rank_pval_context": "Context: The nonparametric p-value evaluates preference rankings rather than raw scores, making it mathematically robust against panelist scale-use bias.",
    
    "action_standard_title": "Detectable difference threshold: {threshold:.2f} points",
    "action_standard_gap": "<strong>{top}</strong> beat <strong>{runner}</strong> by a margin of <strong>{gap:.2f} points</strong>.",
    "action_standard_pass": "Because this exceeds the required {threshold:.2f} threshold, consumers are likely to notice the difference in quality.",
    "action_standard_fail": "Because this falls short of the required {threshold:.2f} threshold, consumers are unlikely to notice a meaningful difference between the top two products.",
    
    # Charts & Tables
    "chart_anova_title": "Parametric ANOVA (adjusted means)",
    "chart_rank_title": "Nonparametric rank test (preference points)",
    "chart_polar_title": "Score distribution (polarization)",
    "chart_polar_desc": "This chart visualizes panel alignment. A tight cluster indicates universal agreement. A wide, stretched spread indicates a highly polarizing product.",
    "chart_attr_title": "Descriptive attribute summary",
    "chart_attr_desc": "Review the average scores for specific attributes below.",
    
    # Correlation Strings
    "chart_corr_title": "Key driver analysis (correlation)",
    "chart_corr_desc": "This section evaluates how strongly each descriptive attribute influenced the tasters' overall scores. Spearman's rank correlation (ρ) is used, meaning scores closer to 1.0 indicate a very strong positive driver.",

    "export_header": "Data export",
    "btn_export": "Download processed matrix (CSV)"
}

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title=UI_TEXT["app_title"], page_icon="📊", layout="wide")

st.markdown("""
    <style>
        .summary-box {
            padding: 1.5rem;
            background-color: #f8f9fa;
            border-left: 5px solid #4CAF50;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        .summary-box.tie { border-left: 5px solid #2196F3; }
        .summary-box.none { border-left: 5px solid #f44336; }
        .action-standard-box {
            padding: 1.5rem;
            background-color: #e3f2fd;
            border-left: 5px solid #0288d1;
            margin-bottom: 0rem;
            font-size: 1.05rem;
        }
        .stat-context {
            color: #555555;
            font-size: 0.9rem;
            font-style: italic;
            margin-top: -10px;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# Dynamic R setup
LOCAL_R_PATH = "/home/eater/R/x86_64-pc-linux-gnu-library/4.2"
R_LIB_CMD = f'.libPaths(c("{LOCAL_R_PATH}", .libPaths()))\n' if os.path.exists(LOCAL_R_PATH) else ''

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_code(val):
    if pd.isna(val): return ""
    val_str = str(val).strip()
    if val_str.endswith(".0"): 
        val_str = val_str[:-2]
    val_str = re.sub(r'[^a-zA-Z0-9]', '', val_str)
    return val_str.upper()

def format_col_name(c):
    return f"{c[:40]}...{c[-15:]}" if len(c) > 55 else c

def extract_attr_name(col_str):
    clean = re.sub(SURVEY_PREFIXES["attr_regex"], '', col_str, flags=re.IGNORECASE).strip()
    clean = re.sub(r'\.\d+$', '', clean).strip()
    return clean

def truncate_name(name, max_len=15):
    name_str = str(name)
    return name_str[:max_len] + '...' if len(name_str) > max_len else name_str

# ==========================================
# APPLICATION HEADER
# ==========================================
st.title(UI_TEXT["app_title"])
st.write(UI_TEXT["app_subtitle"])

if not STATSMODELS_AVAILABLE:
    st.error(UI_TEXT["err_missing_lib"])
    st.stop()

# ==========================================
# STEP 1: RAW SURVEY UPLOAD
# ==========================================
with st.container(border=True):
    st.subheader(UI_TEXT["step1_header"])
    uploaded_file = st.file_uploader(UI_TEXT["step1_upload_label"], type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        cols = list(df_raw.columns)
    except Exception as e:
        st.error(UI_TEXT["err_read_survey"].format(e=e))
        st.stop()
        
    cols_lower = [str(c).lower().strip() for c in cols]
    
    is_processed_matrix = False
    if 'taster' in cols_lower and 'product' in cols_lower and 'overall liking' in cols_lower:
        is_processed_matrix = True

    if is_processed_matrix:
        st.success(UI_TEXT["msg_processed_matrix"])
        
        t_col = cols[cols_lower.index('taster')]
        p_col = cols[cols_lower.index('product')]
        o_col = cols[cols_lower.index('overall liking')]
        
        df_long = df_raw.rename(columns={t_col: 'Taster', p_col: 'Product', o_col: 'Overall liking'})
        attr_names = [c for c in df_long.columns if c not in ['Taster', 'Product', 'Overall liking']]
        
        run_analysis = st.button(UI_TEXT["btn_run"], type="primary", width="stretch")

    else:
        # ==========================================
        # STEP 2: SURVEY CONFIGURATION (ANCHOR ENGINE)
        # ==========================================
        st.divider()
        st.subheader(UI_TEXT["step2_header"])
        
        taster_idx = next((i for i, c in enumerate(cols) if 'taster' in c.lower() or 'id' in c.lower() and not str(c).upper().startswith(SURVEY_PREFIXES["code"].upper())), 0)
        
        code_cols = []
        overall_cols = []
        
        for c in cols:
            c_str = str(c).strip()
            if c_str.upper().startswith(SURVEY_PREFIXES["code"].upper()):
                code_cols.append(c)
            elif c_str.upper().startswith(SURVEY_PREFIXES["overall"].upper()):
                overall_cols.append(c)
                
        if not code_cols:
            for c in cols:
                c_lower = str(c).lower()
                if any(bad in c_lower for bad in ['describe', 'descriptive', 'comment', 'thoughts', 'why', 'additional', 'specific', 'explain']):
                    continue
                if 'code' in c_lower and 'zip' not in c_lower:
                    code_cols.append(c)
                elif 'sample' in c_lower and not any(w in c_lower for w in ['overall', 'like', 'rate', 'taste']):
                    code_cols.append(c)
                    
        if not overall_cols:
            for c in cols:
                c_lower = str(c).lower()
                if any(bad in c_lower for bad in ['describe', 'descriptive', 'comment', 'thoughts', 'why', 'additional', 'specific', 'explain']):
                    continue
                if 'overall' in c_lower:
                    overall_cols.append(c)
                    
        guessed_servings = min(len(code_cols), len(overall_cols))
        if guessed_servings == 0:
            guessed_servings = max(1, len(code_cols), len(overall_cols))

        inferred_attrs = []
        if len(code_cols) > 0 and len(overall_cols) > 0:
            start_idx = cols.index(code_cols[0])
            end_idx = cols.index(code_cols[1]) if len(code_cols) > 1 else len(cols)
            
            block_cols = cols[start_idx+1 : end_idx]
            
            prefix_attrs = [c for c in block_cols if re.match(SURVEY_PREFIXES["attr_regex"], str(c), re.IGNORECASE)]
            if prefix_attrs:
                for c in prefix_attrs:
                    clean_name = extract_attr_name(c).capitalize()
                    if clean_name and clean_name not in inferred_attrs:
                        inferred_attrs.append(clean_name)
            else:
                for c in block_cols:
                    if c in overall_cols: continue
                    c_lower = str(c).lower()
                    if not any(bad in c_lower for bad in ['describe', 'descriptive', 'comment', 'thoughts', 'why', 'additional', 'photo', 'upload', 'specific', 'code', 'sample', 'explain']):
                        clean_c = re.sub(r'\.\d+$', '', str(c))
                        clean_c = re.sub(r'[^a-zA-Z\s]', '', clean_c).strip()
                        words = clean_c.split()
                        stop_words = ['how', 'much', 'do', 'you', 'like', 'the', 'rate', 'this', 'would', 'overall', 'please', 'indicate', 'your', 'opinion', 'of']
                        meaningful = [w for w in words if w.lower() not in stop_words]
                        if meaningful:
                            attr_name = " ".join(meaningful).capitalize()
                            if attr_name not in inferred_attrs:
                                inferred_attrs.append(attr_name)

        default_attrs_str = ", ".join(inferred_attrs) if inferred_attrs else ""

        st.write(UI_TEXT["inference_msg"].format(servings=guessed_servings))

        col_taster, col_serv = st.columns([1.5, 1])
        with col_taster:
            taster_col = st.selectbox(UI_TEXT["label_taster_id"], cols, index=taster_idx, format_func=format_col_name)
        with col_serv:
            servings = st.number_input(UI_TEXT["label_servings"], min_value=1, max_value=20, value=guessed_servings, step=1)
        
        attr_input = st.text_input(UI_TEXT["label_attributes"], value=default_attrs_str)
        attr_names = [x.strip() for x in attr_input.split(',') if x.strip()]

        with st.expander(UI_TEXT["label_mapping_expander"]):
            tabs = st.tabs([f"Serving {i+1}" for i in range(servings)])
            serving_mappings = []
            
            for i, tab in enumerate(tabs):
                with tab:
                    c1, c2 = st.columns(2)
                    
                    start_idx = cols.index(code_cols[i]) if i < len(code_cols) else 0
                    end_idx = cols.index(code_cols[i+1]) if i+1 < len(code_cols) else len(cols)
                    block_cols = cols[start_idx:end_idx]
                    
                    with c1:
                        c_idx = cols.index(code_cols[i]) if i < len(code_cols) else 0
                        code_c = st.selectbox("Product code", cols, index=c_idx, format_func=format_col_name, key=f"mcode_{i}")
                        
                        o_col_in_block = [c for c in block_cols if c in overall_cols]
                        o_idx = cols.index(o_col_in_block[0]) if o_col_in_block else min(c_idx + 1, len(cols)-1)
                        overall_c = st.selectbox("Overall liking", cols, index=o_idx, format_func=format_col_name, key=f"moverall_{i}")
                    
                    attr_c = []
                    with c2:
                        for a, name in enumerate(attr_names):
                            best_match_idx = min(c_idx + 2 + a, len(cols)-1)
                            for bc in block_cols:
                                clean_bc = extract_attr_name(bc).lower()
                                if not clean_bc: 
                                    clean_bc = re.sub(r'\.\d+$', '', str(bc)).lower()
                                if name.lower() in clean_bc:
                                    best_match_idx = cols.index(bc)
                                    break
                                    
                            attr_c.append(st.selectbox(f"{truncate_name(name, 25)} score", cols, index=best_match_idx, format_func=format_col_name, key=f"mattr_{i}_{a}"))
                    
                    serving_mappings.append({"code": code_c, "overall": overall_c, "attrs": attr_c})

        # ==========================================
        # STEP 3: PRODUCT MAPPING
        # ==========================================
        st.divider()
        st.subheader(UI_TEXT["step3_header"])
        st.write(UI_TEXT["step3_desc"])

        unique_raw_codes = []
        for mapping in serving_mappings:
            code_col_name = mapping['code']
            unique_raw_codes.extend(df_raw[code_col_name].dropna().astype(str).tolist())
        
        unique_codes = sorted(list(set([clean_code(c) for c in unique_raw_codes if str(c).strip() != ""])))
        unique_codes = [c for c in unique_codes if c]

        uploaded_key = st.file_uploader(UI_TEXT["step3_upload_key"], type=["csv"])
        
        master_dict = {}
        if uploaded_key:
            try:
                df_key = pd.read_csv(uploaded_key)
                key_code_col, key_name_col = None, None
                
                for c in df_key.columns:
                    norm_c = re.sub(r'[^a-z0-9]', '', c.lower())
                    if any(x in norm_c for x in ['code', 'id', 'number']) and key_code_col is None:
                        key_code_col = c
                    elif any(x in norm_c for x in ['name', 'product', 'brand', 'real']) and key_name_col is None:
                        key_name_col = c
                
                if not key_code_col or not key_name_col:
                    if len(df_key.columns) >= 2:
                        key_code_col = df_key.columns[0]
                        key_name_col = df_key.columns[1]
                
                if key_code_col and key_name_col:
                    df_key[key_code_col] = df_key[key_code_col].apply(clean_code)
                    master_dict = dict(zip(df_key[key_code_col], df_key[key_name_col].astype(str).str.strip()))
                    
                    matched = [c for c in unique_codes if c in master_dict]
                    
                    if len(matched) == 0:
                        st.error(UI_TEXT["msg_key_mismatch"].format(loaded=len(master_dict)))
                    elif len(matched) < len(unique_codes):
                        unmatched = [c for c in unique_codes if c not in master_dict]
                        st.warning(UI_TEXT["msg_key_partial"].format(matched=len(matched), missing=len(unmatched)))
                    else:
                        st.success(UI_TEXT["msg_key_success"].format(count=len(unique_codes)))
                else:
                    st.error(UI_TEXT["err_key_cols"])
            except Exception as e:
                st.error(f"Could not read the master key file. Error: {e}")

        mapping_data = []
        for c in unique_codes:
            mapping_data.append({"Code": c, "Product name": master_dict.get(c, c)})
            
        mapping_df = pd.DataFrame(mapping_data)
        
        st.write(f"**{UI_TEXT['step3_edit_desc']}**")
        st.info(UI_TEXT["step3_edit_hint"])
        edited_mapping = st.data_editor(mapping_df, width="stretch", hide_index=True)
        final_name_mapping = dict(zip(edited_mapping['Code'], edited_mapping['Product name'].astype(str).str.strip()))
        
        st.divider()
        run_analysis = st.button(UI_TEXT["btn_run"], type="primary", width="stretch")

    # ==========================================
    # STEP 4: RUN ANALYSIS (THE MATH)
    # ==========================================
    if run_analysis:
        with st.spinner(UI_TEXT["spinner_running"]):
            
            if not is_processed_matrix:
                stacked_rows = []
                for idx, row in df_raw.iterrows():
                    t_id = str(row[taster_col])
                    for s_idx in range(servings):
                        mapping = serving_mappings[s_idx]
                        raw_code = row[mapping["code"]]
                        overall_score = row[mapping["overall"]]
                        
                        safe_val = clean_code(raw_code)
                        prod_name = final_name_mapping.get(safe_val, safe_val)
                        if not prod_name:
                            prod_name = "Unknown"
                            
                        new_row = {"Taster": t_id, "Product": prod_name, "Overall liking": overall_score}
                        for a_idx, attr_col in enumerate(mapping["attrs"]):
                            if a_idx < len(attr_names):
                                new_row[attr_names[a_idx]] = row[attr_col]
                        stacked_rows.append(new_row)
                        
                df_long = pd.DataFrame(stacked_rows)
            
            df_long['Overall liking'] = pd.to_numeric(df_long['Overall liking'], errors='coerce')
            df_long = df_long.dropna(subset=["Product", "Overall liking"])
            
            if df_long.empty:
                st.error(UI_TEXT["err_no_numbers"])
                st.stop()
            if df_long['Overall liking'].std() == 0:
                st.error(UI_TEXT["err_no_variance"])
                st.stop()

            products = df_long['Product'].unique()

            try:
                model = ols("Q('Overall liking') ~ C(Product) + C(Taster)", data=df_long).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
            except Exception as e:
                st.error(UI_TEXT["err_anova_fail"].format(e=e))
                st.stop()

            product_pval = anova_table.loc['C(Product)', 'PR(>F)']
            residual_std = np.sqrt(model.mse_resid) if hasattr(model, 'mse_resid') else df_long['Overall liking'].std()
            
            evals_per_product = len(df_long) / len(products)
            z_alpha = norm.ppf(1 - 0.10 / 2)
            z_beta = norm.ppf(0.80)
            action_standard = (z_alpha + z_beta) * residual_std * np.sqrt(2 / evals_per_product)

            raw_means = df_long.groupby('Product')['Overall liking'].mean()
            adj_means = []
            for p in products:
                pred = model.predict(pd.DataFrame({'Product': [p], 'Taster': [df_long['Taster'].iloc[0]]}))
                adj_means.append({'Product': p, 'Processed score': raw_means[p], 'Adjusted score': pred[0]})
                
            adj_df = pd.DataFrame(adj_means)
            correction_factor = raw_means.mean() - adj_df['Adjusted score'].mean()
            adj_df['Adjusted score'] = adj_df['Adjusted score'] + correction_factor
            adj_df = adj_df.sort_values(by='Adjusted score', ascending=False).reset_index(drop=True)

            pw_tests = model.t_test_pairwise('C(Product)').result_frame.reset_index()
            pw_tests = pw_tests.rename(columns={'index': 'Comparison', 'P>|t|': 'p-value', 'pvalue': 'p-value'})
            
            sig_dict = {}
            for _, row in pw_tests.iterrows():
                is_sig = (pd.to_numeric(row['p-value'], errors='coerce') < 0.05)
                sig_dict[str(row['Comparison'])] = is_sig

            def is_tied(b1, b2):
                if b1 == b2: return True
                match1, match2 = f"{b1}-{b2}", f"{b2}-{b1}"
                for comp_str, is_sig in sig_dict.items():
                    if comp_str == match1 or comp_str == match2: return not is_sig
                for comp_str, is_sig in sig_dict.items():
                    if b1 in comp_str and b2 in comp_str: return not is_sig
                return True

            sorted_prods = adj_df['Product'].tolist()
            tiers = {p: "" for p in sorted_prods}
            current_tier = 'A'
            for i in range(len(sorted_prods)):
                anchor = sorted_prods[i]
                if tiers[anchor] == "":
                    tiers[anchor] += current_tier
                    for j in range(i+1, len(sorted_prods)):
                        compare_prod = sorted_prods[j]
                        if is_tied(anchor, compare_prod):
                            tiers[compare_prod] += current_tier
                    current_tier = chr(ord(current_tier) + 1)
            adj_df['Tier'] = adj_df['Product'].map(tiers)

            # --- R ENGINE (SKILLINGS MACK) WITH ERROR CAPTURE ---
            sm_pval = None
            used_fallback = False
            r_error_msg = ""
            rank_df = df_long.copy()
            rank_df['Taster'] = rank_df['Taster'].astype(str).str.strip()
            rank_df['Product'] = rank_df['Product'].astype(str).str.strip()

            try:
                df_raw_pivot = rank_df.pivot_table(index='Taster', columns='Product', values='Overall liking', aggfunc='mean')
                df_raw_pivot.to_csv("temp_sm.csv", na_rep="NA")
                
                r_sm_script = f"""
                options(warn=-1)
                {R_LIB_CMD}
                
                # 1. Create a writable directory on the cloud server
                local_lib <- Sys.getenv("R_LIBS_USER")
                dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
                .libPaths(c(local_lib, .libPaths()))

                # 2. Quietly auto-install PMCMRplus if it is missing
                if (!require("PMCMRplus", character.only = TRUE, quietly = TRUE)) {{
                    install.packages("PMCMRplus", repos="https://cloud.r-project.org/", lib=local_lib, quiet=TRUE)
                    library(PMCMRplus, lib.loc=local_lib)
                }}

                df <- read.csv("temp_sm.csv", row.names=1)
                mat <- as.matrix(df)
                
                res_pval <- tryCatch({{
                  res <- skillingsMackTest(mat)
                  res$p.value
                }}, error = function(cond) {{
                  cat(conditionMessage(cond), file="temp_sm_err.txt")
                  return(1.0)
                }})
                
                write.table(res_pval, "temp_sm_pval.txt", row.names=FALSE, col.names=FALSE)
                """
                with open("run_sm.R", "w") as f: f.write(r_sm_script)
                
                # Bumping the timeout to 300 seconds (5 minutes) so the cloud server has time to download and compile the package the first time
                result = subprocess.run(["Rscript", "run_sm.R"], capture_output=True, text=True, check=True, timeout=300)
                
                if os.path.exists("temp_sm_pval.txt"):
                    with open("temp_sm_pval.txt", "r") as f:
                        raw_val = f.read().strip()
                        if raw_val and raw_val != "NA":
                            sm_pval = float(raw_val)
                            
                if os.path.exists("temp_sm_err.txt"):
                    with open("temp_sm_err.txt", "r") as f:
                        err_text = f.read().strip()
                        if err_text:
                            used_fallback = True
                            r_error_msg = f"R Caught Error: {err_text}\\n\\nSTDOUT:\\n{result.stdout}\\n\\nSTDERR:\\n{result.stderr}"

            except subprocess.CalledProcessError as e:
                used_fallback = True
                r_error_msg = f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            except Exception as e:
                used_fallback = True
                r_error_msg = str(e)
            finally:
                if os.path.exists("temp_sm.csv"): os.remove("temp_sm.csv")
                if os.path.exists("run_sm.R"): os.remove("run_sm.R")
                if os.path.exists("temp_sm_pval.txt"): os.remove("temp_sm_pval.txt")
                if os.path.exists("temp_sm_err.txt"): os.remove("temp_sm_err.txt")

            if used_fallback or sm_pval == 1.0:
                used_fallback = True
                rank_df['Preference points'] = rank_df.groupby('Taster')['Overall liking'].rank(ascending=True, method='average')
                model_rank_fb = ols('Q("Preference points") ~ C(Product) + C(Taster)', data=rank_df).fit()
                rank_anova_fb = sm.stats.anova_lm(model_rank_fb, typ=2)
                sm_pval = rank_anova_fb.loc['C(Product)', 'PR(>F)']

            rank_df['Preference points'] = rank_df.groupby('Taster')['Overall liking'].rank(ascending=True, method='average')
            model_rank = ols('Q("Preference points") ~ C(Product) + C(Taster)', data=rank_df).fit()
            
            raw_ranks = rank_df.groupby('Product')['Preference points'].mean()
            adj_rank_means = []
            for p in products:
                pred_rank = model_rank.predict(pd.DataFrame({'Product': [p], 'Taster': [rank_df['Taster'].iloc[0]]}))
                adj_rank_means.append({'Product': p, 'Adjusted preference score': pred_rank[0]})
            
            final_rank_df = pd.DataFrame(adj_rank_means)
            correction = raw_ranks.mean() - final_rank_df['Adjusted preference score'].mean()
            final_rank_df['Adjusted preference score'] = final_rank_df['Adjusted preference score'] + correction
            final_rank_df = final_rank_df.sort_values(by='Adjusted preference score', ascending=False).reset_index(drop=True)

        # ==========================================
        # STEP 5: VISUAL RENDERERS
        # ==========================================
        
        def render_leaderboards():
            st.divider()
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader(UI_TEXT["chart_anova_title"])
                adj_df['Product_Label'] = adj_df['Product'].apply(truncate_name)
                
                fig_anova, ax_anova = plt.subplots(figsize=(8, 6))
                sns.barplot(data=adj_df, x='Product_Label', y='Adjusted score', palette='Blues_r', edgecolor='.2', ax=ax_anova)
                ax_anova.errorbar(x=np.arange(len(adj_df)), y=adj_df['Adjusted score'], yerr=action_standard/2, fmt='none', ecolor='black', capsize=4, elinewidth=1.5, label='Margin of error')
                
                for i, row in adj_df.iterrows():
                    ax_anova.text(i, row['Adjusted score'] + (action_standard/2) + 0.1, row['Tier'], ha='center', va='bottom', fontweight='bold', fontsize=12)
                
                ax_anova.set_ylabel("Final adjusted score")
                ax_anova.set_xlabel("")
                ax_anova.set_ylim(1, min(9.5, adj_df['Adjusted score'].max() + action_standard))
                ax_anova.legend(loc='lower right')
                plt.setp(ax_anova.get_xticklabels(), rotation=45, ha='right')
                sns.despine()
                fig_anova.tight_layout()
                st.pyplot(fig_anova)
                
                st.dataframe(adj_df[['Product', 'Tier', 'Adjusted score']].round(2), hide_index=True, width="stretch")

            with col_chart2:
                st.subheader(UI_TEXT["chart_rank_title"])
                final_rank_df['Product_Label'] = final_rank_df['Product'].apply(truncate_name)
                
                fig_rank, ax_rank = plt.subplots(figsize=(8, 6))
                sns.barplot(data=final_rank_df, x='Product_Label', y='Adjusted preference score', palette='Purples_r', edgecolor='.2', ax=ax_rank)
                
                ax_rank.set_ylabel("Adjusted preference points")
                ax_rank.set_xlabel("")
                max_score = final_rank_df['Adjusted preference score'].max()
                ax_rank.set_ylim(0, max_score + (max_score * 0.15))
                plt.setp(ax_rank.get_xticklabels(), rotation=45, ha='right')
                sns.despine()
                fig_rank.tight_layout()
                st.pyplot(fig_rank)
                
                st.dataframe(final_rank_df[['Product', 'Adjusted preference score']].round(2), hide_index=True, width="stretch")
                
                test_name = "Conover-Iman (Fallback)" if used_fallback else "Skillings-Mack (R)"
                st.caption(f"Engine used: {test_name}")
                
            # Expose R error if fallback was triggered to help with cloud debugging
            if used_fallback and r_error_msg:
                with st.expander("View R Debugging Logs"):
                    st.code(r_error_msg, language='plaintext')

        def render_summary_and_threshold():
            st.divider()
            st.header(UI_TEXT["summary_header"])
            
            if product_pval < 0.05:
                top_tier_products = [p for p, t in tiers.items() if 'A' in t]
                if len(top_tier_products) == 1:
                    desc_html = UI_TEXT["win_sig_desc"].format(winners=top_tier_products[0])
                    st.markdown(f"""
                    <div class="summary-box">
                        <strong>{UI_TEXT["win_sig_title"]}:</strong> {desc_html}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    products_str = " and ".join(top_tier_products) if len(top_tier_products) <= 2 else ", ".join(top_tier_products[:-1]) + f", and {top_tier_products[-1]}"
                    desc_html = UI_TEXT["win_tie_desc"].format(winners=products_str)
                    st.markdown(f"""
                    <div class="summary-box tie">
                        <strong>{UI_TEXT["win_tie_title"]}:</strong> {desc_html}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="summary-box none">
                    <strong>{UI_TEXT["win_none_title"]}:</strong> {UI_TEXT["win_none_desc"]}
                </div>
                """, unsafe_allow_html=True)

            if len(adj_df) > 1:
                top_prod = adj_df.iloc[0]['Product']
                top_score = adj_df.iloc[0]['Adjusted score']
                runner_prod = adj_df.iloc[1]['Product']
                runner_score = adj_df.iloc[1]['Adjusted score']
                gap = top_score - runner_score
                
                gap_text = UI_TEXT["action_standard_gap"].format(top=top_prod, runner=runner_prod, gap=gap)
                if gap >= action_standard:
                    result_text = UI_TEXT["action_standard_pass"].format(threshold=action_standard)
                else:
                    result_text = UI_TEXT["action_standard_fail"].format(threshold=action_standard)
                
                box_title = UI_TEXT["action_standard_title"].format(threshold=action_standard)
                
                st.markdown(f"""
                <div class="action-standard-box">
                    <strong>{box_title}</strong><br>
                    {gap_text} {result_text}
                </div>
                """, unsafe_allow_html=True)
                st.write("") 

        def render_pvalues_and_rse():
            st.divider()
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown(f"**{UI_TEXT['stat_pval_label'].format(pval=product_pval)}**")
                st.markdown(f"<div class='stat-context'>{UI_TEXT['stat_pval_context']}</div>", unsafe_allow_html=True)
                
                rank_test_name = "Conover-Iman" if used_fallback else "Skillings-Mack"
                st.markdown(f"**{UI_TEXT['stat_rank_pval_label'].format(test=rank_test_name, pval=sm_pval)}**")
                if used_fallback:
                    st.markdown(f"<div class='stat-context'>Note: R environment unavailable. Approximation method used for rank test.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='stat-context'>{UI_TEXT['stat_rank_pval_context']}</div>", unsafe_allow_html=True)
                    
            with col_stat2:
                st.markdown(f"**{UI_TEXT['stat_rse_label'].format(rse=residual_std)}**")
                st.markdown(f"<div class='stat-context'>{UI_TEXT['stat_rse_context']}</div>", unsafe_allow_html=True)

        def render_polarization():
            st.divider()
            st.subheader(UI_TEXT["chart_polar_title"])
            st.write(UI_TEXT["chart_polar_desc"])
            
            global_mean = df_long['Overall liking'].mean()
            global_std = df_long['Overall liking'].std()
            
            def standardize_and_scale(group):
                std = group.std(ddof=0)
                z = (group - group.mean()) / std if std > 0 else group - group.mean()
                return (z * global_std) + global_mean
                
            df_plot = df_long.copy()
            df_plot['Z_Score'] = df_plot.groupby('Taster')['Overall liking'].transform(standardize_and_scale)
            df_plot['Product_Label'] = df_plot['Product'].apply(truncate_name)

            fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df_plot, x='Product_Label', y='Z_Score', color='white', width=0.4, ax=ax_dist)
            sns.swarmplot(data=df_plot, x='Product_Label', y='Z_Score', hue='Product_Label', size=5, alpha=0.8, palette="husl", legend=False, ax=ax_dist)
            
            ax_dist.set_ylabel("Standardized score")
            ax_dist.set_xlabel("")
            ax_dist.set_ylim(0.5, 9.5)
            plt.setp(ax_dist.get_xticklabels(), rotation=45, ha='right')
            sns.despine()
            fig_dist.tight_layout()
            st.pyplot(fig_dist)

        def render_descriptive():
            if len(attr_names) > 0:
                st.divider()
                st.subheader(UI_TEXT["chart_attr_title"])
                st.write(UI_TEXT["chart_attr_desc"])
                
                summary_cols = ['Overall liking'] + attr_names
                for attr in summary_cols:
                    if attr in df_long.columns:
                        df_long[attr] = pd.to_numeric(df_long[attr], errors='coerce')
                
                valid_summary_cols = [c for c in summary_cols if c in df_long.columns]
                attr_means = df_long.groupby('Product')[valid_summary_cols].mean()
                attr_means = attr_means.reindex(adj_df['Product'].tolist())
                
                attr_means.index = attr_means.index.map(truncate_name)
                
                col_tab, col_bar = st.columns([1, 1.5])
                with col_tab:
                    st.dataframe(attr_means.round(2), width="stretch")
                
                with col_bar:
                    fig_attr, ax_attr = plt.subplots(figsize=(8, len(products) * 0.7 + 1))
                    
                    plot_means = attr_means.copy()
                    plot_means.index = plot_means.index.map(truncate_name)
                    
                    plot_means.iloc[::-1].plot(kind='barh', ax=ax_attr, width=0.8, alpha=0.9, edgecolor='black')
                    ax_attr.set_xlabel("Average score")
                    ax_attr.set_ylabel("")
                    ax_attr.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    sns.despine(ax=ax_attr)
                    ax_attr.grid(axis='x', linestyle='--', alpha=0.4)
                    fig_attr.tight_layout()
                    st.pyplot(fig_attr)
                    
        def render_driver_analysis():
            if len(attr_names) > 0:
                st.divider()
                st.subheader(UI_TEXT["chart_corr_title"])
                st.write(UI_TEXT["chart_corr_desc"])
                
                corr_cols = ['Overall liking'] + attr_names
                valid_corr_cols = [c for c in corr_cols if c in df_long.columns]
                
                if len(valid_corr_cols) > 1:
                    corr_matrix = df_long[valid_corr_cols].corr(method='spearman')
                    target_corr = corr_matrix['Overall liking'].drop('Overall liking').fillna(0).sort_values(ascending=True)
                    
                    fig_corr, ax_corr = plt.subplots(figsize=(10, len(attr_names) * 0.7 + 1))
                    target_corr.index = target_corr.index.map(truncate_name)
                    
                    target_corr.plot(kind='barh', ax=ax_corr, color='teal', alpha=0.8, edgecolor='black')
                    ax_corr.set_xlabel("Correlation with Overall Liking (Spearman's ρ)")
                    ax_corr.set_ylabel("")
                    ax_corr.set_xlim(-1.1, 1.1)
                    
                    ax_corr.axvline(0, color='black', linewidth=1)
                    sns.despine(ax=ax_corr)
                    ax_corr.grid(axis='x', linestyle='--', alpha=0.4)
                    
                    for i, v in enumerate(target_corr):
                        ax_corr.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.2f}", va='center', ha='left' if v >= 0 else 'right', fontweight='bold')
                        
                    fig_corr.tight_layout()
                    st.pyplot(fig_corr)

        # ----------------------------------------------------
        # DASHBOARD LAYOUT (Reorder these lines to change the app!)
        # ----------------------------------------------------
        st.header("Results Dashboard")
        
        render_leaderboards()
        render_summary_and_threshold()
        render_pvalues_and_rse()
        render_polarization()
        render_descriptive()
        render_driver_analysis()

        # Data export section
        st.divider()
        st.subheader(UI_TEXT["export_header"])
        csv_export = df_long.to_csv(index=False)
        st.download_button(UI_TEXT["btn_export"], data=csv_export, file_name="processed_sensory_matrix.csv", mime="text/csv")
