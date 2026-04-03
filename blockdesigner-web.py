import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import numpy as np
import subprocess
import random
import re

# ==========================================
# FEATURE FLAGS & THRESHOLDS
# Toggle these to True/False to turn UI features on or off
# ==========================================
FEATURES = {
    "show_detectable_difference": True,
    "show_fatigue_warning": False,
    "fatigue_moderate_threshold": 5,
    "fatigue_high_threshold": 7,
    "color_safe": "transparent",
    "color_warning": "#fef08a", # Sleek pastel yellow
    "color_risky": "#fbcfe8"    # Sleek pastel pink
}

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
    "detectable_diff_caption": "Based on these parameters, the panel will be able to reliably detect a quality difference of **{delta:.1f} points** (on a 9-point scale) between any two products.",

    "step_2_title": "Step 2: Input Products",
    "step_2_desc": "How would you like to enter the items being tasted?",
    "input_mode_label": "Product Entry Method",
    "mode_manual": "Type them in manually",
    "mode_csv": "Upload a master list (CSV)",
    "csv_help": "Upload a CSV containing product names.",
    "csv_format_guide": "**Required CSV Format:**\n\nFile must contain a column for the Name. Columns for the Product (e.g., A, B, C) and Code are optional. They will be auto-generated if not found.\n\n| Product | Code | Name |\n| :--- | :--- | :--- |\n| A | 492 | Brand X Vanilla |\n| B | 184 | Brand Y Vanilla |",
    "csv_error": "Couldn't read CSV. Please make sure it has a column containing the word 'Name'.",
    "csv_duplicate_error": "CSV has duplicate codes. Every product needs a unique code.",
    "csv_success_all": "Found {count} products with custom letters and blind codes.",
    "csv_info_no_code": "Found {count} products. No custom codes detected, so we will automatically generate 3-digit blind codes for you.",
    "csv_info_only_names": "Found {count} products. We will automatically assign them letters (A, B, C...) and random 3-digit blind codes.",
    "manual_num_products": "Total number of products to test",
    "manual_enter_names_instruction": "Enter actual product names:",
    "manual_name_prefix": "Product",
    "auto_code_checkbox": "Automatically generate unique 3-digit blind codes",
    
    "step_3_title": "Step 3: Generate Design",
    "btn_generate": "Generate Block Design",
    "loading_msg": "Initializing R environment and calculating D-optimal incomplete block design via the AlgDesign package...",
    "timeout_error": "The statistical engine timed out. The mathematical combination requested may not be optimally resolvable. Please adjust panel size or serving counts.",
    "r_missing_error": "R is not installed or not found in the system PATH. Please ensure R is configured.",
    "blank_name_error": "Please ensure all product names are filled in before continuing.",
    "duplicate_name_error": "Duplicate product names detected. Please ensure each product name is unique.",
    "missing_selection_error": "Please select a Product Entry Method in Step 2 before generating the design.",
    
    "results_title": "Experimental Block Design",
    "results_stats_header": "Design Quality (D-Optimal Matrix):",
    "results_disclaimer_codes": "Note: Only the blind codes are shown to prevent bias.",
    "results_disclaimer_names": "Note: Product names are shown below.",
    "btn_download_sched": "Download Design (CSV)",
    
    "key_title": "Master Key & Prep Sheet",
    "key_desc": "Save this key to translate codes back to real product names after the tasting, and use the 'Total Servings' column to prepare the exact number of samples needed.",
    "btn_download_key": "Download Master Key (CSV)",
    
    "copy_instruction": "*(Tip: To copy and paste to a spreadsheet, click any cell, press **Ctrl+A** / **Cmd+A** to select all, then **Ctrl+C** / **Cmd+C** to copy.)*"
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
# GOATCOUNTER ANALYTICS
# ==========================================
components.html("""
    <script>
        window.goatcounter = {
            path: function(p) { return location.host + p }
        }
    </script>
    <script data-goatcounter="https://eater.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
""", height=0)

# ==========================================
# DYNAMIC R ENVIRONMENT SETUP
# ==========================================
# Create a local, writable directory for R packages to prevent permission errors on Streamlit Cloud
LOCAL_R_LIB = os.path.join(os.getcwd(), "r_packages")
os.makedirs(LOCAL_R_LIB, exist_ok=True)
R_LIB_CMD = f'.libPaths(c("{LOCAL_R_LIB}", .libPaths()))\n'

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

def generate_d_optimal_matrix(v_count, b_count, k_count, r_lib_cmd, local_r_lib):
    # R script dynamically installs into the writable local directory
    r_script = f"""
    options(warn=-1, repos=c(CRAN="https://cloud.r-project.org"))
    dir.create("{local_r_lib}", showWarnings = FALSE, recursive = TRUE)
    {r_lib_cmd}
    
    if (!requireNamespace("AlgDesign", quietly = TRUE)) {{
        install.packages("AlgDesign", lib="{local_r_lib}")
    }}
    library(AlgDesign, lib.loc="{local_r_lib}")

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

    subprocess.run(["Rscript", "generate_design.R"], capture_output=True, text=True, check=True, timeout=520)
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
    col1, col2, col3 = st.columns(3)
    with col1:
        num_products_input = st.number_input(LANG["manual_num_products"], min_value=2, max_value=26, value=9, step=1)
    with col2:
        num_tasters = st.number_input(LANG["num_tasters"], min_value=1, value=21, step=1)
    with col3:
        servings_per_taster = st.number_input(LANG["servings_per"], min_value=1, value=min(4, int(num_products_input)), step=1)

    # Live Detectable Difference Calculation
    if FEATURES["show_detectable_difference"]:
        evals_per_product = (num_tasters * servings_per_taster) / num_products_input
        if evals_per_product > 0:
            z_alpha = 1.960
            z_beta = 0.842
            sigma = 1.3
            delta = (z_alpha + z_beta) * sigma * np.sqrt(2 / evals_per_product)
            st.caption(LANG["detectable_diff_caption"].format(delta=delta))

    # Dynamic Fatigue Ribbon Footer
    if FEATURES["show_fatigue_warning"]:
        if servings_per_taster >= FEATURES["fatigue_high_threshold"]:
            ribbon_color = FEATURES["color_risky"]
            ribbon_text = "<strong>High Fatigue Risk:</strong> Tasting this many samples will dull palates."
        elif servings_per_taster >= FEATURES["fatigue_moderate_threshold"]:
            ribbon_color = FEATURES["color_warning"]
            ribbon_text = "<strong>Moderate Fatigue Risk:</strong> Ensure tasters use palate cleansers between samples."
        else:
            ribbon_color = FEATURES["color_safe"]
            ribbon_text = ""
            
        if ribbon_color != "transparent":
            st.markdown(f"""
                <div style="margin-top: 8px; margin-bottom: 8px; padding: 6px 12px; border-radius: 4px; background-color: {ribbon_color}; color: #453000; font-size: 0.85em; font-weight: 500;">
                    {ribbon_text}
                </div>
            """, unsafe_allow_html=True)
        else:
            # When safe, just render a tiny 2px spacer with the same margins to keep layout consistent
            st.markdown(f'<div style="height: 2px; margin-top: 8px; margin-bottom: 8px; background-color: {ribbon_color};"></div>', unsafe_allow_html=True)

# --- STEP 2: DEFINE PRODUCTS ---
st.subheader(LANG["step_2_title"])
st.markdown(LANG["step_2_desc"])

with st.container(border=True):
    # Radio button defaults to None, keeping the UI clean until a selection is made
    product_def_mode = st.radio(
        LANG["input_mode_label"], 
        [LANG["mode_manual"], LANG["mode_csv"]], 
        horizontal=True,
        index=None
    )

    df_master = None
    product_names = []
    assign_codes = True
    blind_codes_from_csv = []
    num_products_val = int(num_products_input)

    if product_def_mode is not None:
        st.divider()

    if product_def_mode == LANG["mode_csv"]:
        st.info(LANG["csv_format_guide"])
        uploaded_master = st.file_uploader(LANG["mode_csv"], type=["csv"], help=LANG["csv_help"], label_visibility="collapsed")
        
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

                if 'Name' not in mapped_cols:
                    st.error(LANG["csv_error"])
                    df_master = None
                else:
                    num_prods = len(df_raw)
                    
                    # 1. Extract or Generate Real Names
                    real_names = df_raw[mapped_cols['Name']].astype(str).tolist()
                    
                    # 2. Extract or Generate Product Letters
                    if 'Product' in mapped_cols:
                        product_letters = df_raw[mapped_cols['Product']].astype(str).tolist()
                        has_products = True
                    else:
                        product_letters = [chr(65+i) for i in range(num_prods)]
                        has_products = False
                        
                    # 3. Extract or Generate 3-Digit Codes
                    if 'Code' in mapped_cols:
                        blind_codes = df_raw[mapped_cols['Code']].apply(clean_3_digit_code).tolist()
                        has_codes = True
                    else:
                        blind_codes = [str(x).zfill(3) for x in random.sample(range(100, 1000), num_prods)]
                        has_codes = False

                    # Display Smart Feedback Messages
                    if has_products and has_codes:
                        st.success(LANG["csv_success_all"].format(count=num_prods))
                    elif has_products and not has_codes:
                        st.info(LANG["csv_info_no_code"].format(count=num_prods))
                    else:
                        st.info(LANG["csv_info_only_names"].format(count=num_prods))

                    # Build the standardized master dataframe
                    df_master = pd.DataFrame({
                        'Product': product_letters,
                        'Code': blind_codes,
                        'Name': real_names
                    })
                    
                    st.dataframe(df_master, hide_index=True)
                    
                    # Pass values downstream
                    product_names = real_names
                    blind_codes_from_csv = blind_codes
                    num_products_val = num_prods
                    assign_codes = False # We already generated codes in the block above if needed
                    
                    if num_products_val != num_products_input:
                        st.caption(f"*(Note: Overriding 'Total products' selection from Step 1. {num_products_val} products found in the CSV.)*")
                    
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df_master = None

    elif product_def_mode == LANG["mode_manual"]:
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
    if product_def_mode is None:
        st.error(LANG["missing_selection_error"])
    else:
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
                    df_r = generate_d_optimal_matrix(num_products_val, num_tasters, servings_per_taster, R_LIB_CMD, LOCAL_R_LIB)
                    
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

                    count_text = f"exactly {actual_min_count}" if actual_min_count == actual_max_count else f"between {actual_min_count} and {actual_max_count}"
                    pair_text = f"exactly {actual_min_pairs}" if actual_min_pairs == actual_max_pairs else f"between {actual_min_pairs} and {actual_max_pairs}"

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

                    # Generate the Master Key & Prep Sheet Tally
                    if blind_codes:
                        key_df = pd.DataFrame({
                            "Product Name": clean_names,
                            "3-Digit Code": blind_codes,
                            "Total Servings": counts
                        })
                    else:
                        key_df = pd.DataFrame({
                            "Product Name": clean_names,
                            "Total Servings": counts
                        })

                    # --- DISPLAY RESULTS ---
                    st.divider()
                    st.subheader(LANG["results_title"])
                    st.markdown(f"""
                    **{LANG["results_stats_header"]}**
                    * **Target appearances:** Each product is served {count_text} times across the entire panel (Theoretical target: {expected_count:.2f}).
                    * **Pairwise balance:** Every product is evaluated alongside every other product {pair_text} times (Theoretical target: {expected_pairs:.2f}).

                    """)
                    
                    st.dataframe(final_df, hide_index=True)
                    
                    if blind_codes:
                        st.caption(LANG["results_disclaimer_codes"])
                    else:
                        st.caption(LANG["results_disclaimer_names"])
                    
                    st.caption(LANG["copy_instruction"])
                    
                    csv_export = final_df.to_csv(index=False)
                    st.download_button(LANG["btn_download_sched"], data=csv_export, file_name="tasting_design.csv", mime="text/csv")

                    st.divider()
                    st.subheader(LANG["key_title"])
                    st.markdown(LANG["key_desc"])
                    st.dataframe(key_df, hide_index=True)
                    
                    st.caption(LANG["copy_instruction"])
                    
                    master_key_csv = key_df.to_csv(index=False)
                    st.download_button(LANG["btn_download_key"], data=master_key_csv, file_name="master_key_prep_sheet.csv", mime="text/csv")

                except subprocess.TimeoutExpired:
                    st.error(LANG["timeout_error"])
                except FileNotFoundError:
                    st.error(LANG["r_missing_error"])
                except subprocess.CalledProcessError as e:
                    st.error("An error occurred while executing the statistical engine.")
                    st.code(e.stderr)
