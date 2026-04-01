import streamlit as st
import os
import pandas as pd
import numpy as np
import subprocess
import random
import re

# ==========================================
# CENTRALIZED TEXT CONFIGURATION
# ==========================================
LANG = {
    "page_title": "Sensory Panel Designer",
    "page_subtitle": "Easily build fair, randomized serving designs for tasting panels. *(Powered by R's AlgDesign statistical package).* ",
    
    "step_1_title": "Step 1: Panel Logistics",
    "step_1_desc": "Input the number of tasters and how many samples they will evaluate.",
    "num_tasters": "Expected number of tasters",
    "servings_per": "Samples evaluated per taster",
    "serving_error": "A taster cannot evaluate more samples than the total number of products available.",

    "step_2_title": "Step 2: Define Your Products",
    "step_2_desc": "How would you like to enter the items being tasted today?",
    "input_mode_label": "Product Entry Method",
    "mode_manual": "Type them in manually",
    "mode_csv": "Upload a master list (CSV)",
    "csv_help": "Upload a CSV containing your product names and blind codes.",
    "csv_format_guide": "**Required CSV Format:**\n\nYour file must contain exactly these three column headers (case-sensitive):\n\n| Product | 3-Digit Code | Real Name |\n| :--- | :--- | :--- |\n| A | 492 | Brand X Vanilla |\n| B | 184 | Brand Y Vanilla |",
    "csv_error": "We couldn't read your CSV. Please make sure the headers are spelled exactly: Product, 3-Digit Code, Real Name.",
    "csv_duplicate_error": "Your CSV has duplicate 3-digit codes. Every product needs a unique code.",
    "manual_num_products": "Total number of products to test",
    "manual_enter_names_instruction": "Enter actual product names:",
    "manual_name_prefix": "Product",
    "auto_code_checkbox": "Automatically generate random 3-digit blind codes",
    
    "step_3_title": "Step 3: Generate Design",
    "btn_generate": "Generate Serving Design",
    "loading_msg": "Initializing R environment and calculating D-optimal incomplete block design via the AlgDesign package...",
    "timeout_error": "The statistical engine timed out. The mathematical combination requested may not be optimally resolvable. Please adjust your panel size or serving counts.",
    "r_missing_error": "R is not installed or not found in the system PATH. Please ensure R is configured.",
    "blank_name_error": "Please ensure all product names are filled in before continuing.",
    "duplicate_name_error": "Duplicate product names detected. Please ensure each product name is unique.",
    
    "results_title": "Experimental Block Design",
    "results_stats_header": "Design Quality (D-Optimal Matrix):",
    "results_disclaimer_codes": "Note: Only the blind codes are shown below to prevent bias.",
    "results_disclaimer_names": "Note: Product names are shown below.",
    "btn_download_sched": "Download Design (CSV)",
    
    "key_title": "Master Key",
    "key_desc": "Save this key to translate 3-digit codes back to real product names after the tasting.",
    "btn_download_key": "Download Master Key (CSV)"
}

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title=LANG["page_title"], page_icon="🍦", layout="centered")

st.markdown("""
    <style>
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div,
        div[data-baseweb="number_input"] > div,
        .stButton > button,
        .stAlert {
            border-radius: 0px !important;
        }
        div[role="radiogroup"] {
            gap: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DYNAMIC R ENVIRONMENT SETUP
# ==========================================
LOCAL_R_PATH = "/home/eater/R/x86_64-pc-linux-gnu-library/4.2"
R_LIB_CMD = f'.libPaths(c("{LOCAL_R_PATH}", .libPaths()))\n' if os.path.exists(LOCAL_R_PATH) else ''

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_3_digit_code(val):
    if pd.isna(val): return ""
    val_str = str(val).strip()
    if val_str.endswith(".0"): 
        val_str = val_str[:-2]
    val_str = re.sub(r'[^a-zA-Z0-9]', '', val_str)
    if val_str.isdigit():
        return val_str.zfill(3)
    return val_str.upper()

def generate_d_optimal_matrix(v_count, b_count, k_count, r_lib_cmd):
    # Added robust installation logic for AlgDesign to handle Streamlit Cloud
    r_script = f"""
    options(warn=-1, repos=c(CRAN="http://cran.us.r-project.org"))
    {r_lib_cmd}
    
    if (!requireNamespace("AlgDesign", quietly = TRUE)) {{
        install.packages("AlgDesign")
    }}
    library(AlgDesign)

    V <- {int(v_count)}
    B <- {int(b_count)}
    K <- {int(k_count)}
    N <- B * K

    pool <- rep(1:V, length.out = N)
    within_data <- data.frame(Trt = as.factor(pool))

    blocksizes <- rep(K, B)
    b_opt <- optBlock(~., withinData=within_data, blocksizes=blocksizes, nRepeats=100)

    out_matrix <- matrix(nrow=B, ncol=K)
    for(i in 1:B) {{
      out_matrix[i, ] <- as.numeric(as.character(b_opt$Blocks[[i]]$Trt))
    }}
    write.csv(out_matrix, "temp_design.csv", row.names=FALSE)
    """
    
    with open("generate_design.R", "w") as f:
        f.write(r_script)

    subprocess.run(["Rscript", "generate_design.R"], capture_output=True, text=True, check=True, timeout=60)
    df_result = pd.read_csv("temp_design.csv")
    
    if os.path.exists("generate_design.R"):
        os.remove("generate_design.R")
    if os.path.exists("temp_design.csv"):
        os.remove("temp_design.csv")
        
    return df_result

# ==========================================
# MAIN APPLICATION UI
# ==========================================
st.title(LANG["page_title"])
st.markdown(LANG["page_subtitle"])
st.divider()

# --- STEP 1: DEFINE PANEL LOGISTICS ---
st.subheader(LANG["step_1_title"])
st.markdown(LANG["step_1_desc"])

with st.container(border=True):
    num_products_input = st.number_input(LANG["manual_num_products"], min_value=2, max_value=26, value=4, step=1)
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        num_tasters = st.number_input(LANG["num_tasters"], min_value=1, value=30, step=1)
    with col2:
        servings_per_taster = st.number_input(LANG["servings_per"], min_value=1, value=min(4, int(num_products_input)), step=1)

# --- STEP 2: DEFINE PRODUCTS ---
st.subheader(LANG["step_2_title"])
st.markdown(LANG["step_2_desc"])

with st.container(border=True):
    product_def_mode = st.radio(LANG["input_mode_label"], [LANG["mode_manual"], LANG["mode_csv"]], horizontal=True)
    st.divider()

    df_master = None
    product_names = []
    assign_codes = True
    blind_codes_from_csv = []
    num_products_val = int(num_products_input)

    if product_def_mode == LANG["mode_csv"]:
        st.info(LANG["csv_format_guide"])
        uploaded_master = st.file_uploader(LANG["mode_csv"], type=["csv"], help=LANG["csv_help"])
        
        if uploaded_master is not None:
            try:
                df_raw = pd.read_csv(uploaded_master)
                
                # Fuzzy matching for column headers
                mapped_cols = {}
                for c in df_raw.columns:
                    norm = re.sub(r'[^a-z0-9]', '', c.lower())
                    if 'code' in norm:
                        mapped_cols['Code'] = c
                    elif 'name' in norm or 'real' in norm:
                        mapped_cols['Name'] = c
                    elif 'product' in norm or 'id' in norm or 'letter' in norm:
                        mapped_cols['Product'] = c

                if 'Code' not in mapped_cols or 'Name' not in mapped_cols or 'Product' not in mapped_cols:
                    st.error(LANG["csv_error"])
                    df_master = None
                else:
                    # Rename to internal standard names to keep the logic clean
                    df_master = df_raw.rename(columns={
                        mapped_cols['Product']: 'Product',
                        mapped_cols['Code']: '3-Digit Code',
                        mapped_cols['Name']: 'Real Name'
                    })
                    
                    st.dataframe(df_master[['Product', '3-Digit Code', 'Real Name']], hide_index=True)
                    product_names = df_master['Real Name'].astype(str).tolist()
                    blind_codes_from_csv = df_master['3-Digit Code'].apply(clean_3_digit_code).tolist()
                    
                    num_products_val = len(df_master)
                    if num_products_val != num_products_input:
                        st.caption(f"*(Note: Overriding your 'Total products' selection from Step 1. We found {num_products_val} products in your CSV.)*")
                    assign_codes = False
                    
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df_master = None

    else:
        st.markdown(f"**{LANG['manual_enter_names_instruction']}**")
        n_p = int(num_products_val)
        
        for i in range(0, n_p, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < n_p:
                    idx = i + j
                    with cols[j]:
                        p_n = st.text_input(f"{LANG['manual_name_prefix']} {chr(65+idx)}", value=f"Product {chr(65+idx)}")
                        product_names.append(p_n)
                        
        st.write("")
        assign_codes = st.checkbox(LANG["auto_code_checkbox"], value=True)


# --- STEP 3: EXECUTION ---
st.subheader(LANG["step_3_title"])

if st.button(LANG["btn_generate"], type="primary", use_container_width=True):
    clean_names = [p.strip() for p in product_names]
    
    if product_def_mode == LANG["mode_csv"] and df_master is None:
        st.error(LANG["csv_error"])
    elif "" in clean_names:
        st.error(LANG["blank_name_error"])
    elif len(set(clean_names)) < len(clean_names):
        st.error(LANG["duplicate_name_error"])
    elif servings_per_taster > num_products_val:
        st.error(LANG["serving_error"])
    elif product_def_mode == LANG["mode_csv"] and len(set(blind_codes_from_csv)) < len(blind_codes_from_csv):
        st.error(LANG["csv_duplicate_error"])
    else:
        with st.spinner(LANG["loading_msg"]):
            try:
                # 1. Generate Matrix
                df_r = generate_d_optimal_matrix(num_products_val, num_tasters, servings_per_taster, R_LIB_CMD)
                
                # 2. Pre-shuffle rows to protect against dropouts
                df_r = df_r.sample(frac=1).reset_index(drop=True)
                
                # 3. Calculate statistics
                expected_count = (num_tasters * servings_per_taster) / num_products_val
                expected_pairs = (expected_count * (servings_per_taster - 1)) / (num_products_val - 1) if num_products_val > 1 else 0
                
                counts = [0] * num_products_val
                pairs = [[0] * num_products_val for _ in range(num_products_val)]

                for idx, row in df_r.iterrows():
                    block = [int(x) - 1 for x in row.values]
                    for i in range(len(block)):
                        counts[block[i]] += 1
                        for j in range(i + 1, len(block)):
                            pairs[block[i]][block[j]] += 1
                            pairs[block[j]][block[i]] += 1

                actual_min_count = min(counts)
                actual_max_count = max(counts)
                
                actual_pair_counts = []
                for i in range(num_products_val):
                    for j in range(i + 1, num_products_val):
                        actual_pair_counts.append(pairs[i][j])
                
                actual_min_pairs = min(actual_pair_counts) if actual_pair_counts else 0
                actual_max_pairs = max(actual_pair_counts) if actual_pair_counts else 0

                count_text = f"Exactly {actual_min_count}" if actual_min_count == actual_max_count else f"Between {actual_min_count} and {actual_max_count}"
                pair_text = f"Exactly {actual_min_pairs}" if actual_min_pairs == actual_max_pairs else f"Between {actual_min_pairs} and {actual_max_pairs}"

                # 4. Apply Codes
                if product_def_mode == LANG["mode_manual"] and assign_codes:
                    blind_codes = [str(x).zfill(3) for x in random.sample(range(100, 1000), num_products_val)]
                elif product_def_mode == LANG["mode_csv"]:
                    blind_codes = blind_codes_from_csv
                else:
                    blind_codes = None
                
                table_data = []
                for i, row in df_r.iterrows():
                    block_row = {"Taster ID": f"Taster {str(i + 1).zfill(2)}"}
                    for j, val in enumerate(row.values):
                        product_idx = int(val) - 1
                        if blind_codes:
                            block_row[f"Sample {j+1}"] = blind_codes[product_idx]
                        else:
                            block_row[f"Sample {j+1}"] = clean_names[product_idx]
                    table_data.append(block_row)

                final_df = pd.DataFrame(table_data)

                if blind_codes:
                    key_df = pd.DataFrame({
                        "Product Name": clean_names,
                        "3-Digit Code": blind_codes
                    })
                else:
                    key_df = None

                # --- DISPLAY RESULTS ---
                st.divider()
                st.subheader(LANG["results_title"])
                st.markdown(f"""
                **{LANG["results_stats_header"]}**
                * Target appearances per product: {count_text} (Theoretical optimal: {expected_count:.2f})
                * Pairwise balance (products served together): {pair_text} (Theoretical optimal: {expected_pairs:.2f})
                """)
                
                st.dataframe(final_df, hide_index=True)
                
                if key_df is not None:
                    st.caption(LANG["results_disclaimer_codes"])
                else:
                    st.caption(LANG["results_disclaimer_names"])
                
                csv_export = final_df.to_csv(index=False)
                st.download_button(LANG["btn_download_sched"], data=csv_export, file_name="tasting_design.csv", mime="text/csv")

                if key_df is not None:
                    st.divider()
                    st.subheader(LANG["key_title"])
                    st.markdown(LANG["key_desc"])
                    st.dataframe(key_df, hide_index=True)
                    
                    master_key_csv = key_df.to_csv(index=False)
                    st.download_button(LANG["btn_download_key"], data=master_key_csv, file_name="master_key.csv", mime="text/csv")

            except subprocess.TimeoutExpired:
                st.error(LANG["timeout_error"])
            except FileNotFoundError:
                st.error(LANG["r_missing_error"])
            except subprocess.CalledProcessError as e:
                st.error("An error occurred while executing the statistical engine.")
                st.code(e.stderr)
