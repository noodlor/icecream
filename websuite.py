import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import binom, norm
from scipy.cluster.vq import kmeans2
from numpy.linalg import svd
import subprocess
import warnings
import random
import re

# Silence the Swarmplot point-placement warnings
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')

try:
    from pingouin import multivariate_normality
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Sensory Science Suite", page_icon="📊", layout="wide")

# The Invisible Top Anchor (For the scroll hack)
st.markdown("<div id='top-of-page'></div>", unsafe_allow_html=True)

st.markdown("""
    <style>
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div,
        div[data-baseweb="number_input"] > div,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] > div,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        .stButton > button,
        .stAlert,
        div[data-testid="stMetric"] {
            border-radius: 0px !important;
        }
        div[role="radiogroup"] {
            gap: 1.5rem;
        }
        /* Hides the "Press Enter to apply" hint on text inputs */
        div[data-testid="InputInstructions"] { 
            display: none !important; 
        }
        .bottom-line-box {
            padding: 1.5rem;
            background-color: #f8f9fa;
            border-left: 5px solid #4CAF50;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }
        .bottom-line-box.tie {
            border-left: 5px solid #2196F3;
        }
        .bottom-line-box.none {
            border-left: 5px solid #f44336;
        }
        .advanced-test-box {
            padding: 1rem;
            background-color: #f1f3f4;
            border-left: 4px solid #6c757d;
            margin-bottom: 1rem;
            font-size: 0.95rem;
        }
        .action-standard-box {
            padding: 1.5rem;
            background-color: #e3f2fd;
            border-left: 5px solid #0288d1;
            margin-bottom: 2rem;
            font-size: 1.05rem;
        }
    </style>
""", unsafe_allow_html=True)

# JS Injection for Global Auto-Highlight on Focus
components.html("""
    <script>
        var parentDoc = window.parent.document;
        parentDoc.addEventListener('focusin', function(e) {
            if (e.target && e.target.tagName === 'INPUT' && e.target.type === 'text') {
                e.target.select();
            }
        });
    </script>
""", height=0)

# ==========================================
# DYNAMIC R ENVIRONMENT SETUP
# ==========================================
LOCAL_R_PATH = "/home/eater/R/x86_64-pc-linux-gnu-library/4.2"
R_LIB_CMD = f'.libPaths(c("{LOCAL_R_PATH}", .libPaths()))\n' if os.path.exists(LOCAL_R_PATH) else ''

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_data(uploaded_file, gsheet_url):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif gsheet_url.strip():
        try:
            if "/edit" in gsheet_url:
                export_url = gsheet_url.split('/edit')[0] + '/export?format=csv'
            else:
                export_url = gsheet_url
            return pd.read_csv(export_url)
        except Exception as e:
            st.error(f"Could not load data from Google Sheets. Ensure the link is set to 'Anyone with the link can view'. Error details: {e}")
            return None
    return None

def clean_3_digit_code(val):
    """Aggressively cleans floats, ints, spaces, and strings to match perfectly."""
    if pd.isna(val): return ""
    val_str = str(val).strip()
    if val_str.endswith(".0"): 
        val_str = val_str[:-2]
    val_str = re.sub(r'[^a-zA-Z0-9]', '', val_str)
    if val_str.isdigit():
        return val_str.zfill(3)
    return val_str.upper()

def clear_state_keys(keys_to_clear):
    """Safely purges specific keys from the session state to reset a module."""
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def generate_d_optimal_matrix(v_count, b_count, k_count, r_lib_cmd):
    """Abstracts the R script generation for D-Optimal matrix designs."""
    r_script = f"""
    options(warn=-1)
    {r_lib_cmd}
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

    subprocess.run(["Rscript", "generate_design.R"], capture_output=True, text=True, check=True, timeout=30)
    df_result = pd.read_csv("temp_design.csv")
    
    if os.path.exists("generate_design.R"):
        os.remove("generate_design.R")
    if os.path.exists("temp_design.csv"):
        os.remove("temp_design.csv")
        
    return df_result

# ==========================================
# SESSION STATE MEMORY & CALLBACKS
# ==========================================
if "active_tool" not in st.session_state:
    st.session_state.active_tool = "Panel Size Optimizer"
if "transfer_brands" not in st.session_state:
    st.session_state.transfer_brands = 5
if "transfer_tasters" not in st.session_state:
    st.session_state.transfer_tasters = 20
if "transfer_servings" not in st.session_state:
    st.session_state.transfer_servings = 4
if "decoded_df" not in st.session_state:
    st.session_state.decoded_df = None
if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False
if "decoder_mode" not in st.session_state:
    st.session_state.decoder_mode = "start"

# Lightweight keys for UI tools that don't need full state retention
if "corr_upload_key" not in st.session_state:
    st.session_state.corr_upload_key = 0
if "discrim_key" not in st.session_state:
    st.session_state.discrim_key = 0

def go_to_designer(b, t, s):
    st.session_state.transfer_brands = int(b)
    st.session_state.transfer_tasters = int(t)
    st.session_state.transfer_servings = int(s)
    st.session_state.active_tool = "Experimental Block Designer"
    st.session_state.scroll_to_top = True

def go_to_analyzer():
    st.session_state.active_tool = "Hedonic Analyzer"
    st.session_state.scroll_to_top = True

def send_sim_to_analyzer():
    st.session_state.decoded_df = st.session_state.sim_decoded_df.copy()
    go_to_analyzer()

def send_sim_to_profiler():
    st.session_state.active_tool = "Flavor Profiler"
    st.session_state.scroll_to_top = True

def reset_decoder_state():
    st.session_state.decoded_df = None

def reset_raw_survey_state():
    st.session_state.decoded_df = None
    st.session_state.decoder_mode = "start"

if st.session_state.scroll_to_top:
    scroll_script = """
    <script>
        setTimeout(function() {
            var parentDoc = window.parent.document;
            var anchor = parentDoc.getElementById('top-of-page');
            if (anchor) {
                anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
            } else {
                var main = parentDoc.querySelector('section.main') || parentDoc.querySelector('[data-testid="stMainBlockContainer"]');
                if (main) main.scrollTo({top: 0, behavior: 'smooth'});
            }
        }, 200);
    </script>
    """
    components.html(scroll_script, height=0)
    st.session_state.scroll_to_top = False

# ==========================================
# ROBUST NAVIGATION MENU
# ==========================================
st.sidebar.title("Sensory Lab Suite")

def nav_btn(tool_name):
    btn_type = "primary" if st.session_state.active_tool == tool_name else "secondary"
    if st.sidebar.button(tool_name, type=btn_type, width='stretch'):
        st.session_state.active_tool = tool_name
        st.session_state.scroll_to_top = True
        st.rerun()

st.sidebar.markdown("**Study Design**")
nav_btn("Panel Size Optimizer")
nav_btn("Experimental Block Designer")

st.sidebar.markdown("**Affective (Hedonic) Liking**")
nav_btn("Hedonic Simulator")
nav_btn("Survey Decoder")
nav_btn("Hedonic Analyzer")

st.sidebar.markdown("**Descriptive Flavor Profiling**")
nav_btn("Descriptive Simulator")
nav_btn("Flavor Profiler")

st.sidebar.markdown("**Specialty Tests**")
nav_btn("Discrimination Test")
nav_btn("Correlation Matrix")
nav_btn("Documentation")

tool = st.session_state.active_tool


# ==========================================
# TOOL 1: PANEL SIZE OPTIMIZER
# ==========================================
if tool == "Panel Size Optimizer":
    st.title("Panel Size Optimizer")
    st.markdown("Calculate the required panel size, evaluate statistical power, or determine the expected detectable difference.")
    
    with st.container(border=True):
        col_radio, _ = st.columns([1, 1])
        with col_radio:
            calc_mode = st.radio(
                "Calculation Mode", 
                [
                    "Calculate required panel size", 
                    "Calculate statistical power (fixed panel size)",
                    "Calculate detectable difference (fixed panel size)"
                ]
            )
        st.divider()
        
        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            products = st.number_input("Total Products to Test", min_value=2, value=4, step=1)
        with col2:
            servings_per_taster = st.number_input("Servings Evaluated Per Taster", min_value=1, max_value=int(products), value=min(4, int(products)), step=1)
            
        st.divider()
        col3, col4, _ = st.columns([1, 1, 2])
        
        if calc_mode == "Calculate required panel size":
            with col3:
                delta = st.number_input("Target Detectable Difference", min_value=0.1, value=1.0, step=0.1)
            with col4:
                target_power = st.slider("Target statistical power", min_value=0.50, max_value=0.99, value=0.80, step=0.01)
                unbalanced = st.checkbox("Allow unbalanced design (skip serving multiplier)", value=False)
            
        elif calc_mode == "Calculate statistical power (fixed panel size)":
            with col3:
                delta = st.number_input("Target Detectable Difference", min_value=0.1, value=1.0, step=0.1)
            with col4:
                fixed_tasters = st.number_input("Available Tasters (fixed panel size)", min_value=2, value=20, step=1)
            
        elif calc_mode == "Calculate detectable difference (fixed panel size)":
            with col3:
                fixed_tasters = st.number_input("Available Tasters (fixed panel size)", min_value=2, value=20, step=1)
            with col4:
                target_power = st.slider("Target statistical power", min_value=0.50, max_value=0.99, value=0.80, step=0.01)

    with st.expander("Advanced Statistical Settings"):
        col_adv1, col_adv2, _ = st.columns([1, 1, 2])
        with col_adv1:
            stdev = st.number_input("Estimated Standard Deviation", min_value=0.1, value=1.3, step=0.1)
        with col_adv2:
            alpha = st.slider("Significance Level (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

    z_alpha = norm.ppf(1 - alpha / 2)

    st.subheader("Results")
    if calc_mode == "Calculate required panel size":
        z_beta = norm.ppf(target_power)
        
        raw_n_per_product = 2 * ((z_alpha + z_beta) ** 2) * ((stdev / delta) ** 2)
        min_evals_per_product = int(np.ceil(raw_n_per_product))
        
        total_evals_needed = min_evals_per_product * products
        min_tasters = int(np.ceil(total_evals_needed / servings_per_taster))
        
        optimal_tasters = min_tasters
        if not unbalanced:
            while (optimal_tasters * servings_per_taster) % products != 0:
                optimal_tasters += 1

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Target Evaluations Per Product", min_evals_per_product)
        res_col2.metric("Minimum Tasters", min_tasters)
        res_col3.metric("Recommended Balanced Panel Size", optimal_tasters)
        
        if optimal_tasters == min_tasters and not unbalanced:
            st.markdown(f"**Balanced design:** {min_tasters} tasters each evaluating {servings_per_taster} samples perfectly balances across {products} products.")
        elif unbalanced:
            st.markdown("**Unbalanced design selected.** Serving orders should be carefully randomized to minimize bias.")
        else:
            st.markdown(f"**Balanced recommendation:** We rounded the panel size up to **{optimal_tasters}** so every product is evaluated an equal number of times.")

        col_btn, _1, _2 = st.columns([1, 2, 2])
        with col_btn:
            st.button("Send to Block Designer", type="primary", on_click=go_to_designer, args=(products, optimal_tasters, servings_per_taster), width='stretch')

    elif calc_mode == "Calculate statistical power (fixed panel size)":
        evals_per_product = (fixed_tasters * servings_per_taster) / products
        variance_factor = 2 * ((stdev / delta) ** 2)
        
        if evals_per_product > 0 and variance_factor > 0:
            z_beta_calc = np.sqrt(evals_per_product / variance_factor) - z_alpha
            calculated_power = norm.cdf(z_beta_calc)
        else:
            calculated_power = 0.0

        st.metric("Estimated Statistical Power", f"{calculated_power * 100:.1f}%")
        st.markdown(f"*(Based on {evals_per_product:.1f} expected evaluations per product)*")
        
        col_btn, _1, _2 = st.columns([1, 2, 2])
        with col_btn:
            st.button("Send to Block Designer", type="primary", on_click=go_to_designer, args=(products, fixed_tasters, servings_per_taster), width='stretch')

    elif calc_mode == "Calculate detectable difference (fixed panel size)":
        z_beta = norm.ppf(target_power)
        evals_per_product = (fixed_tasters * servings_per_taster) / products
        
        if evals_per_product > 0:
            calculated_delta = (z_alpha + z_beta) * stdev * np.sqrt(2 / evals_per_product)
        else:
            calculated_delta = 0.0

        st.metric("Expected Detectable Difference (Delta)", f"{calculated_delta:.2f} points")
        st.markdown(f"*(Based on {evals_per_product:.1f} expected evaluations per product)*")
        
        col_btn, _1, _2 = st.columns([1, 2, 2])
        with col_btn:
            st.button("Send to Block Designer", type="primary", on_click=go_to_designer, args=(products, fixed_tasters, servings_per_taster), width='stretch')

# ==========================================
# TOOL 2: EXPERIMENTAL BLOCK DESIGNER
# ==========================================
elif tool == "Experimental Block Designer":
    st.title("Experimental Block Designer")
    st.markdown("Generate a D-optimal serving schedule utilizing R's `AlgDesign` statistical package.")
    
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Design Data"):
            clear_state_keys(['bd_generated', 'bd_final_df', 'bd_key_df', 'bd_stats'])

    default_servings = min(4, st.session_state.transfer_brands - 1)
    if default_servings < 1: 
        default_servings = 1

    with st.container(border=True):
        product_def_mode = st.radio("Product Definition Method", ["Manual Entry & Auto-Codes", "Upload CSV Master List"], horizontal=True)
        st.divider()

        df_master = None
        if product_def_mode == "Upload CSV Master List":
            uploaded_master = st.file_uploader("Upload Master List (CSV)", type=["csv"], help="Must contain columns: Product, 3-Digit Code, Real Name")
            if uploaded_master is not None:
                try:
                    df_master = pd.read_csv(uploaded_master)
                    req_cols = ['Product', '3-Digit Code', 'Real Name']
                    if not all(col in df_master.columns for col in req_cols):
                        st.error("Validation Error: CSV must contain exactly these headers: Product, 3-Digit Code, Real Name")
                        df_master = None
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    df_master = None
                    
            if df_master is not None:
                num_products_val = len(df_master)
            else:
                num_products_val = st.session_state.transfer_brands
        else:
            num_products_val = st.session_state.transfer_brands

        col1, col2, col3, _ = st.columns([1, 1, 1, 1])
        with col1:
            if product_def_mode == "Upload CSV Master List" and df_master is not None:
                num_products = st.number_input("Total Products to Test", value=num_products_val, disabled=True)
            else:
                num_products = st.number_input("Total Products to Test", min_value=2, max_value=26, value=num_products_val, step=1)
        with col2:
            num_tasters = st.number_input("Total Tasters", min_value=1, value=st.session_state.transfer_tasters, step=1)
        with col3:
            transfer_s = st.session_state.get('transfer_servings', 4)
            safe_servings = min(transfer_s, int(num_products))
            servings_per_taster = st.number_input("Servings Evaluated Per Taster", min_value=1, max_value=int(num_products), value=safe_servings, step=1)
        
        st.divider()
        
        product_names = []
        assign_codes = True
        blind_codes_from_csv = []
        
        if product_def_mode == "Manual Entry & Auto-Codes":
            st.markdown("**1. Define Product Names (Optional)**")
            n_p = int(num_products)
            with st.container(border=True):
                for i in range(0, n_p, 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < n_p:
                            idx = i + j
                            key = f"design_pname_{idx}"
                            if key not in st.session_state: 
                                st.session_state[key] = f"Product {chr(65+idx)}"
                            with cols[j]:
                                st.markdown(f"<div style='text-align: center; font-weight: bold; margin-bottom: 5px; color: #333;'>Product {chr(65+idx)}</div>", unsafe_allow_html=True)
                                p_n = st.text_input(f"Name {idx}", key=key, label_visibility="collapsed")
                                product_names.append(p_n)
                                st.write("")
            
            st.markdown("**2. Assign Blind Codes**")
            assign_codes = st.checkbox("Automatically assign random 3-digit blind codes to products", value=True)
            
        else:
            if df_master is not None:
                st.markdown("**Loaded Products Master List:**")
                st.dataframe(df_master, hide_index=True)
                product_names = df_master['Real Name'].astype(str).tolist()
                blind_codes_from_csv = df_master['3-Digit Code'].apply(clean_3_digit_code).tolist()
                assign_codes = False
            else:
                st.info("Please upload a CSV file containing your product list to proceed.")

    col_btn, _1, _2 = st.columns([1, 2, 2])
    with col_btn:
        generate_clicked = st.button("Generate D-Optimal Design", type="primary", width='stretch')

    if generate_clicked:
        clean_names = [p.strip() for p in product_names]
        
        if product_def_mode == "Upload CSV Master List" and df_master is None:
            st.error("Validation Error: Please upload a valid CSV master list before generating.")
        elif "" in clean_names:
            st.error("Validation Error: One or more Product Names are blank. Please fill in all names before generating.")
        elif len(set(clean_names)) < len(clean_names):
            st.error("Validation Error: You have duplicate Product Names. Please ensure every product has a unique name.")
        elif servings_per_taster > num_products:
            st.error("A taster cannot evaluate more servings than the total number of products available.")
        elif product_def_mode == "Upload CSV Master List" and len(set(blind_codes_from_csv)) < len(blind_codes_from_csv):
            st.error("Validation Error: You have duplicate 3-Digit Codes in your CSV.")
        else:
            with st.spinner('Engaging R backend and calculating D-optimal matrix...'):
                try:
                    df_r = generate_d_optimal_matrix(num_products, num_tasters, servings_per_taster, R_LIB_CMD)
                    
                    # Pre-shuffle the rows to protect against panelist dropouts
                    df_r = df_r.sample(frac=1).reset_index(drop=True)
                    
                    expected_count = (num_tasters * servings_per_taster) / num_products
                    expected_pairs = (expected_count * (servings_per_taster - 1)) / (num_products - 1) if num_products > 1 else 0
                    
                    counts = [0] * num_products
                    pairs = [[0] * num_products for _ in range(num_products)]

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
                    for i in range(num_products):
                        for j in range(i + 1, num_products):
                            actual_pair_counts.append(pairs[i][j])
                    
                    actual_min_pairs = min(actual_pair_counts) if actual_pair_counts else 0
                    actual_max_pairs = max(actual_pair_counts) if actual_pair_counts else 0

                    count_text = f"exactly **{actual_min_count}**" if actual_min_count == actual_max_count else f"between **{actual_min_count}** and **{actual_max_count}**"
                    pair_text = f"exactly **{actual_min_pairs}**" if actual_min_pairs == actual_max_pairs else f"between **{actual_min_pairs}** and **{actual_max_pairs}**"

                    if product_def_mode == "Manual Entry & Auto-Codes" and assign_codes:
                        blind_codes = [str(x).zfill(3) for x in random.sample(range(100, 1000), num_products)]
                    elif product_def_mode == "Upload CSV Master List":
                        blind_codes = blind_codes_from_csv
                    else:
                        blind_codes = None
                    
                    table_data = []
                    for i, row in df_r.iterrows():
                        block_row = {"Taster": f"Taster {str(i + 1).zfill(2)}"}
                        for j, val in enumerate(row.values):
                            product_idx = int(val) - 1
                            if blind_codes:
                                block_row[f"Serving {j+1}"] = blind_codes[product_idx]
                            else:
                                block_row[f"Serving {j+1}"] = clean_names[product_idx]
                        table_data.append(block_row)

                    final_df = pd.DataFrame(table_data)

                    st.session_state.bd_final_df = final_df
                    if blind_codes:
                        key_df = pd.DataFrame({
                            "Product Name": clean_names,
                            "Code": blind_codes
                        })
                        st.session_state.bd_key_df = key_df
                    else:
                        st.session_state.bd_key_df = None
                        
                    st.session_state.bd_stats = (count_text, expected_count, pair_text, expected_pairs)
                    st.session_state.bd_generated = True

                except subprocess.TimeoutExpired:
                    st.error("The R engine timed out. The mathematical combination you requested is too complex or impossible to resolve. Please adjust your Taster or Serving counts.")
                except FileNotFoundError:
                    st.error("R is not installed or not found in your system PATH. Please install R to use this tool.")
                except subprocess.CalledProcessError as e:
                    st.error("An error occurred while executing the R script. Please check your terminal output.")
                    st.code(e.stderr)

    if st.session_state.get('bd_generated', False):
        count_text, expected_count, pair_text, expected_pairs = st.session_state.bd_stats
        
        st.divider()
        st.subheader("Optimized Serving Schedule")
        st.markdown(f"""
        **Design Statistics (D-Optimal):**
        * **Target appearances:** Each product is served {count_text} times across the entire panel (Theoretical target: {expected_count:.2f}).
        * **Pairwise balance:** Every product is evaluated alongside every other product {pair_text} times (Theoretical target: {expected_pairs:.2f}).
        """)
        
        st.dataframe(st.session_state.bd_final_df, hide_index=True)
        
        if st.session_state.bd_key_df is not None:
            st.markdown("**(Only the final blind codes are shown above to prevent tester bias.)**")
        else:
            st.markdown("**(Product names are shown above as requested.)**")
        
        col_dl_sched, _1, _2 = st.columns([1, 2, 2])
        with col_dl_sched:
            csv_export = st.session_state.bd_final_df.to_csv(index=False)
            st.download_button("Download Serving Schedule (CSV)", data=csv_export, file_name="serving_schedule.csv", mime="text/csv")

        if st.session_state.bd_key_df is not None:
            st.divider()
            st.subheader("Blind Code Master Key")
            st.markdown("Export this Master Key to securely import into the Survey Decoder later.")
            st.dataframe(st.session_state.bd_key_df, hide_index=True)
            
            col_dl_key, _1, _2 = st.columns([1, 2, 2])
            with col_dl_key:
                master_key_csv = st.session_state.bd_key_df.to_csv(index=False)
                st.download_button("Download Master Key (CSV)", data=master_key_csv, file_name="master_key.csv", mime="text/csv")

# ==========================================
# TOOL 2.5: HEDONIC SIMULATION ENGINE
# ==========================================
elif tool == "Hedonic Simulator":
    st.title("Hedonic Simulator (Dummy Data Generator)")
    st.markdown("Generate highly realistic, mathematically messy 'Overall Liking' survey data to train your team or stress-test the Hedonic Analyzer.")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Simulation Data"):
            clear_state_keys(['sim_generated', 'sim_final_df', 'sim_key_df', 'sim_decoded_df'])

    with st.container(border=True):
        col1, col2, col3, _ = st.columns([1, 1, 1, 1])
        with col1:
            sim_products = st.number_input("Total Products to Test", min_value=2, max_value=26, value=6, step=1)
        with col2:
            sim_tasters = st.number_input("Total Tasters", min_value=1, max_value=500, value=30, step=1)
        with col3:
            sim_servings = st.number_input("Servings Evaluated Per Taster", min_value=1, max_value=int(sim_products), value=min(4, int(sim_products)), step=1)

        st.divider()
        st.markdown("**1. Product Names & Distribution Profile**")
        st.markdown("Choose how the underlying 'true' scores are generated. Bimodal distributions simulate a highly polarized market where tasters strongly disagree.")
        
        dist_choice = st.radio("Market Distribution Type", ["Normal (Consensus)", "Bimodal (Highly Polarized)", "Mixed (Realistic Market)"], horizontal=True)
        
        sim_names = []
        
        with st.container(border=True):
            for i in range(0, int(sim_products), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < int(sim_products):
                        idx = i + j
                        default_name = f"Product {chr(65+idx)}"
                        
                        with cols[j]:
                            p_name = st.text_input(f"Product {idx+1} Name", value=default_name, key=f"sim_name_{idx}")
                            sim_names.append(p_name)

        st.divider()
        st.markdown("**2. Inject Statistical Messiness**")
        col_knob1, col_knob2 = st.columns(2)
        with col_knob1:
            taster_bias = st.slider("Taster Bias (Severity/Generosity Variance)", min_value=0.0, max_value=3.0, value=1.2, step=0.1, help="Higher values assign extreme harshness or generosity baselines to specific tasters.")
        with col_knob2:
            random_noise = st.slider("Random Palate Noise (Standard Deviation)", min_value=0.0, max_value=3.0, value=1.2, step=0.1, help="Simulates inconsistency in 1-to-9 Hedonic consumer liking scores. (Usually around 1.2 to 1.3)")

    col_btn, _1, _2 = st.columns([1, 2, 2])
    with col_btn:
        generate_sim = st.button("Generate Simulation", type="primary", width='stretch')

    if generate_sim:
        clean_names = [p.strip() for p in sim_names]
        
        if "" in clean_names:
            st.error("Validation Error: One or more Product Names are blank. Please fill in all names before generating.")
        elif len(set(clean_names)) < len(clean_names):
            st.error("Validation Error: You have duplicate Product Names. Please ensure every product has a unique name.")
        elif sim_servings > sim_products:
            st.error("A taster cannot evaluate more servings than the total number of products available.")
        else:
            with st.spinner('Running D-Optimal design and simulating scores...'):
                try:
                    df_r = generate_d_optimal_matrix(sim_products, sim_tasters, sim_servings, R_LIB_CMD)
                    
                    # Pre-shuffle the rows to protect against panelist dropouts in simulation too
                    df_r = df_r.sample(frac=1).reset_index(drop=True)
                    
                    blind_codes = [str(x).zfill(3) for x in random.sample(range(100, 1000), int(sim_products))]
                    
                    taster_profiles = np.random.choice([0, 1], size=int(sim_tasters))
                    taster_hidden_biases = np.random.normal(0, taster_bias, int(sim_tasters))

                    true_scores_A = []
                    true_scores_B = []

                    for i in range(int(sim_products)):
                        if "Normal" in dist_choice:
                            score = round(random.uniform(4.0, 8.0), 1)
                            true_scores_A.append(score)
                            true_scores_B.append(score)
                        elif "Bimodal" in dist_choice:
                            score_A = round(random.uniform(7.0, 9.0), 1)
                            score_B = round(random.uniform(2.0, 4.0), 1)
                            if random.choice([True, False]):
                                true_scores_A.append(score_A)
                                true_scores_B.append(score_B)
                            else:
                                true_scores_A.append(score_B)
                                true_scores_B.append(score_A)
                        else: # Mixed
                            is_bimodal = random.choice([True, False])
                            if is_bimodal:
                                score_A = round(random.uniform(7.0, 9.0), 1)
                                score_B = round(random.uniform(2.0, 4.0), 1)
                                if random.choice([True, False]):
                                    true_scores_A.append(score_A)
                                    true_scores_B.append(score_B)
                                else:
                                    true_scores_A.append(score_B)
                                    true_scores_B.append(score_A)
                            else:
                                score = round(random.uniform(4.0, 8.0), 1)
                                true_scores_A.append(score)
                                true_scores_B.append(score)

                    table_data = []
                    decoded_data = []
                    
                    for i, row in df_r.iterrows():
                        taster_id = f"Taster {str(i + 1).zfill(2)}"
                        block_row = {"Taster ID": taster_id}
                        my_bias = taster_hidden_biases[i]
                        my_profile = taster_profiles[i]
                        
                        for j, val in enumerate(row.values):
                            product_idx = int(val) - 1
                            
                            if my_profile == 0:
                                base_score = true_scores_A[product_idx]
                            else:
                                base_score = true_scores_B[product_idx]
                                
                            noise = np.random.normal(0, random_noise)
                            
                            raw_score = base_score + my_bias + noise
                            clamped_score = max(1, min(9, round(raw_score)))
                            
                            block_row[f"Serving {j+1} Code"] = blind_codes[product_idx]
                            block_row[f"Serving {j+1} Score"] = clamped_score
                            
                            decoded_data.append({
                                "Taster": taster_id,
                                "Product": clean_names[product_idx],
                                "Score": clamped_score
                            })
                            
                        table_data.append(block_row)

                    final_df = pd.DataFrame(table_data)
                    key_df = pd.DataFrame({
                        "Product Name": clean_names,
                        "Code": blind_codes
                    })
                    
                    df_stacked_sim = pd.DataFrame(decoded_data)
                    df_pivot_sim = df_stacked_sim.pivot_table(index="Taster", columns="Product", values="Score", aggfunc='mean').reset_index()
                    df_pivot_sim.columns.name = None

                    st.session_state.sim_final_df = final_df
                    st.session_state.sim_key_df = key_df
                    st.session_state.sim_decoded_df = df_pivot_sim
                    st.session_state.sim_generated = True

                except Exception as e:
                    st.error(f"An error occurred during simulation generation: {e}")

    if st.session_state.get('sim_generated', False):
        st.success("Simulation Complete! Download your files below to test the workflow.")
        
        st.subheader("1. The Raw Survey Data")
        st.markdown("This wide-format file perfectly mimics an export from Qualtrics or Google Forms. Upload this to the **Survey Decoder**.")
        st.dataframe(st.session_state.sim_final_df.head(10), hide_index=True)
        
        csv_export = st.session_state.sim_final_df.to_csv(index=False)
        st.download_button("Download simulated_raw_survey.csv", data=csv_export, file_name="simulated_raw_survey.csv", mime="text/csv")

        st.divider()
        st.subheader("2. The Master Key")
        st.markdown("Upload this directly into the **Survey Decoder** to map the blind codes back to your product names automatically.")
        st.dataframe(st.session_state.sim_key_df, hide_index=True)
        
        master_key_csv = st.session_state.sim_key_df.to_csv(index=False)
        st.download_button("Download simulated_master_key.csv", data=master_key_csv, file_name="simulated_master_key.csv", mime="text/csv")
        
        st.divider()
        st.subheader("3. The Decoded Data Matrix (Fast-Track)")
        st.markdown("Want to skip the Survey Decoder entirely? This file is already stacked, translated, and perfectly formatted. Upload this directly into the **Hedonic Analyzer**.")
        st.dataframe(st.session_state.sim_decoded_df.head(10), hide_index=True)
        
        col_sim_dl, col_sim_send, _ = st.columns([1, 1, 2])
        with col_sim_dl:
            decoded_csv_export = st.session_state.sim_decoded_df.to_csv(index=False)
            st.download_button("Download Matrix (CSV)", data=decoded_csv_export, file_name="simulated_decoded_matrix.csv", mime="text/csv")
        with col_sim_send:
            st.button("Send to Hedonic Analyzer", type="primary", on_click=send_sim_to_analyzer, width='stretch')

# ==========================================
# TOOL 2.6: DESCRIPTIVE SIMULATOR
# ==========================================
elif tool == "Descriptive Simulator":
    st.title("Descriptive Simulator (Flavor Profiling)")
    st.markdown("Generate multivariate data to test the Flavor Profiler. This generates a Complete Block Design mapping specific flavor attributes using a consumer-friendly 1-to-7 intensity scale.")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Simulation Data"):
            clear_state_keys(['desc_generated', 'desc_sim_df'])

    with st.container(border=True):
        col1, col2, col3, _ = st.columns([1, 1, 1, 1])
        with col1:
            desc_products = st.number_input("Total Products", min_value=2, max_value=10, value=3, step=1)
        with col2:
            desc_tasters = st.number_input("Total Tasters", min_value=5, max_value=200, value=30, step=1)
        with col3:
            desc_attrs = st.number_input("Number of Attributes", min_value=3, max_value=8, value=5, step=1)

        st.divider()
        st.markdown("**1. Define Attributes & True Intensities (1-7 Scale)**")
        
        distinct_choice = st.radio("Flavor Distinctiveness", ["Subtle Differences (Products taste very similar)", "Extreme Profiles (Highly distinct, spiky flavor profiles)"], horizontal=True)
        st.markdown("We will automatically assign 'True' intensity scores based on your choice above, but you can fine-tune them here. (1 = Not at all, 7 = Extremely intense).")

        dist_key = "subtle" if "Subtle" in distinct_choice else "extreme"

        attr_names = []
        prod_names = []
        true_intensities = {}

        c_attr = st.columns(int(desc_attrs))
        default_attrs = ["Sweetness", "Saltiness", "Crunchiness", "Bitterness", "Chocolate", "Chewiness", "Acidity", "Moisture"]
        for j in range(int(desc_attrs)):
            with c_attr[j]:
                attr_name = st.text_input(f"Attribute {j+1}", value=default_attrs[j], key=f"desc_attr_{j}")
                attr_names.append(attr_name)

        for i in range(int(desc_products)):
            with st.container(border=True):
                p_name = st.text_input(f"Product {i+1} Name", value=f"Product {chr(65+i)}", key=f"desc_pname_{i}")
                prod_names.append(p_name)
                
                c_vals = st.columns(int(desc_attrs))
                true_intensities[p_name] = []
                for j in range(int(desc_attrs)):
                    with c_vals[j]:
                        if "Subtle" in distinct_choice:
                            default_val = round(random.uniform(3.5, 5.0), 1)
                        else:
                            default_val = round(random.choice([random.uniform(1.0, 3.0), random.uniform(5.0, 7.0)]), 1)
                        
                        val = st.number_input(f"{attr_names[j]}", min_value=1.0, max_value=7.0, value=default_val, step=0.1, key=f"desc_val_{i}_{j}_{dist_key}")
                        true_intensities[p_name].append(val)

        st.divider()
        st.markdown("**2. Inject Panel Noise**")
        desc_noise = st.slider("Human Inconsistency (Standard Deviation)", min_value=0.0, max_value=4.0, value=1.5, step=0.1, help="Untrained tasters have higher standard deviations (noise) than calibrated lab panels. (Usually around 1.5 or higher for intensity scales)")

    col_btn, _1, _2 = st.columns([1, 2, 2])
    with col_btn:
        generate_desc = st.button("Generate Descriptive Data", type="primary", width='stretch')

    if generate_desc:
        clean_pnames = [p.strip() for p in prod_names]
        clean_anames = [a.strip() for a in attr_names]
        
        if "" in clean_pnames or "" in clean_anames:
            st.error("Validation Error: Please fill in all Product and Attribute names.")
        elif len(set(clean_pnames)) < len(clean_pnames) or len(set(clean_anames)) < len(clean_anames):
            st.error("Validation Error: You have duplicate Product or Attribute Names.")
        else:
            with st.spinner("Simulating taster scoring..."):
                raw_data = []
                
                for t in range(int(desc_tasters)):
                    taster_id = f"Taster {str(t+1).zfill(2)}"
                    scale_bias = np.random.normal(0, 0.5) 
                    
                    for p in clean_pnames:
                        row = {"Taster": taster_id, "Product": p}
                        for a_idx, attr in enumerate(clean_anames):
                            true_score = true_intensities[p][a_idx]
                            noise = np.random.normal(0, desc_noise)
                            raw_score = true_score + scale_bias + noise
                            clamped_score = max(1, min(7, round(raw_score)))
                            row[attr] = clamped_score
                        raw_data.append(row)
                
                df_raw_desc = pd.DataFrame(raw_data)
                
                st.session_state.desc_sim_df = df_raw_desc
                st.session_state.desc_generated = True

    if st.session_state.get('desc_generated', False):
        st.success("Simulation Complete! Download the raw survey below.")
        st.dataframe(st.session_state.desc_sim_df.head(10), hide_index=True)
        
        col_desc_dl, col_desc_send, _ = st.columns([1, 1, 2])
        with col_desc_dl:
            desc_csv = st.session_state.desc_sim_df.to_csv(index=False)
            st.download_button("Download Matrix (CSV)", data=desc_csv, file_name="simulated_descriptive_survey.csv", mime="text/csv")
        with col_desc_send:
            st.button("Send to Flavor Profiler", type="primary", on_click=send_sim_to_profiler, width='stretch')


# ==========================================
# TOOL 5: SURVEY DECODER
# ==========================================
elif tool == "Survey Decoder":
    st.title("Survey Decoder (Stack & Pivot)")
    st.markdown("Convert raw, unorganized survey exports into a clean, analysis-ready matrix.")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Decoder Memory"):
            clear_state_keys(['decoded_df', 'master_names', 'key_version'])
            st.session_state.decoder_mode = "start"

    if "key_version" not in st.session_state:
        st.session_state.key_version = 0
    if "master_names" not in st.session_state:
        st.session_state.master_names = {}
    if "last_key_id" not in st.session_state:
        st.session_state.last_key_id = None

    with st.container(border=True):
        col_upload, col_url = st.columns(2)
        with col_upload:
            uploaded_file = st.file_uploader("Upload Raw Survey Data (CSV)", type=["csv"], key="raw_survey", on_change=reset_raw_survey_state)
        with col_url:
            gsheet_url = st.text_input("OR Paste Public Google Sheet URL", placeholder="https://docs.google.com/...", key="raw_url", on_change=reset_raw_survey_state)

    df_raw = load_data(uploaded_file, gsheet_url)

    if df_raw is not None:
        st.subheader("1. Identify Columns")
        cols = list(df_raw.columns)
        
        guessed_servings = max(1, (len(cols) - 1) // 2)
        
        dataset_signature = "".join(cols)

        with st.container(border=True):
            col_taster, col_serv, _ = st.columns([1, 1, 2])
            with col_taster:
                taster_col = st.selectbox(
                    "Which column contains the Taster IDs?", 
                    cols, 
                    on_change=reset_decoder_state, 
                    key=f"dec_taster_{dataset_signature}"
                )
            with col_serv:
                servings = st.number_input(
                    "How many servings did each taster evaluate?", 
                    min_value=1, max_value=20, value=guessed_servings, step=1, 
                    on_change=reset_decoder_state, 
                    key=f"dec_serv_{dataset_signature}"
                )

            st.markdown("Map your survey columns for each serving:")
            serving_cols = []
            for i in range(servings):
                c1, c2, _ = st.columns([1, 1, 2])
                with c1:
                    code_c = st.selectbox(
                        f"Serving {i+1} - 3-Digit Code Column", 
                        cols, index=min(i*2 + 1, len(cols)-1), 
                        on_change=reset_decoder_state, 
                        key=f"code_{i}_{dataset_signature}"
                    )
                with c2:
                    score_c = st.selectbox(
                        f"Serving {i+1} - Score Column", 
                        cols, index=min(i*2 + 2, len(cols)-1), 
                        on_change=reset_decoder_state, 
                        key=f"score_{i}_{dataset_signature}"
                    )
                serving_cols.append((code_c, score_c))

        st.subheader("2. Master Key (Code Translation)")

        unique_codes = set()
        for code_c, _ in serving_cols:
            codes = df_raw[code_c].apply(clean_3_digit_code).replace("", np.nan).dropna().unique()
            unique_codes.update(codes)
        unique_codes = sorted(list(unique_codes))

        if unique_codes:
            if st.session_state.decoder_mode == "start":
                st.markdown("We extracted the unique blind codes from your data. How would you like to assign the Product Names?")
                c_btn1, c_btn2, _ = st.columns([1, 1, 2])
                with c_btn1:
                    if st.button("Upload Master Key File", width='stretch'):
                        st.session_state.decoder_mode = "upload"
                        st.rerun()
                with c_btn2:
                    if st.button("Manually Enter Names", width='stretch'):
                        st.session_state.decoder_mode = "manual"
                        st.rerun()

            elif st.session_state.decoder_mode == "upload":
                key_up = st.file_uploader("Upload the master_key.csv exported from the Block Designer:", type=["csv"])
                if key_up:
                    try:
                        kdf = pd.read_csv(key_up)
                        if "Code" not in kdf.columns or "Product Name" not in kdf.columns:
                            st.error("Invalid file format. The file must contain exactly 'Code' and 'Product Name' columns.")
                        else:
                            imported_dict = {}
                            for _, r in kdf.iterrows():
                                c = clean_3_digit_code(r["Code"])
                                imported_dict[c] = str(r["Product Name"])
                            
                            missing_in_key = set(unique_codes) - set(imported_dict.keys())
                            
                            if missing_in_key:
                                st.error(f"Mismatch Error! The uploaded Master Key is missing definitions for these codes found in your survey: {sorted(list(missing_in_key))}. Please ensure you are uploading the correct file.")
                            else:
                                for code in unique_codes:
                                    if code in imported_dict:
                                        st.session_state.master_names[code] = imported_dict[code]
                                
                                st.session_state.key_version += 1
                                st.session_state.decoded_df = None
                                st.session_state.decoder_mode = "manual"
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                
                if st.button("← Cancel / Back"):
                    st.session_state.decoder_mode = "start"
                    st.rerun()

            if st.session_state.decoder_mode == "manual":
                if st.button("← Back / Start Over"):
                    st.session_state.decoder_mode = "start"
                    st.rerun()
                    
                st.markdown("Confirm or assign the corresponding product names below:")
                default_names = [f"Product {str(x).zfill(2)}" if len(unique_codes) > 9 else f"Product {x}" for x in range(1, len(unique_codes)+1)]
                final_mapped_names = {}
                
                with st.container(border=True):
                    for i in range(0, len(unique_codes), 3):
                        grid = st.columns(3)
                        for j in range(3):
                            if i + j < len(unique_codes):
                                code = unique_codes[i+j]
                                current_val = st.session_state.master_names.get(code, default_names[i+j])
                                
                                with grid[j]:
                                    st.markdown(f"<h3 style='text-align: center; color: #333; margin-bottom: 0px;'>{code}</h3>", unsafe_allow_html=True)
                                    val = st.text_input(f"Name for {code}", value=current_val, key=f"ti_{code}_{st.session_state.key_version}", label_visibility="collapsed", on_change=reset_decoder_state)
                                    st.session_state.master_names[code] = val
                                    final_mapped_names[code] = val
                                    st.write("")

                col_btn, _1, _2 = st.columns([1, 2, 2])
                with col_btn:
                    decode_clicked = st.button("Decode & Format Data", type="primary", width='stretch')

                if decode_clicked:
                    with st.spinner("Stacking and pivoting..."):
                        code_map = {str(k).strip(): str(v).strip() for k, v in final_mapped_names.items()}

                        stacked_data = []
                        for idx, row in df_raw.iterrows():
                            taster_val = row[taster_col]
                            taster = str(taster_val).strip() if isinstance(taster_val, str) else taster_val

                            for code_col, score_col in serving_cols:
                                code_val = clean_3_digit_code(row[code_col])
                                score_val = row[score_col]
                                
                                if code_val and pd.notna(score_val) and code_val.lower() != 'nan':
                                    product_name = code_map.get(code_val, code_val)
                                    stacked_data.append({"Taster": taster, "Product": product_name, "Score": score_val})

                        df_stacked = pd.DataFrame(stacked_data)
                        df_stacked["Score"] = pd.to_numeric(df_stacked["Score"], errors='coerce')
                        df_stacked = df_stacked.dropna(subset=["Score"])

                        df_pivot = df_stacked.pivot_table(index="Taster", columns="Product", values="Score", aggfunc='mean')
                        df_pivot = df_pivot.reset_index()
                        df_pivot.columns.name = None
                        
                        st.session_state.decoded_df = df_pivot

                if st.session_state.decoded_df is not None:
                    st.success("**Successfully decoded!**")
                    
                    st.subheader("Next Steps")
                    col_next1, col_next2, _ = st.columns([1, 1, 2])
                    with col_next1:
                        st.button("Send to Hedonic Analyzer", type="primary", on_click=go_to_analyzer, width='stretch')
                    with col_next2:
                        csv_data = st.session_state.decoded_df.to_csv(sep=',', index=False)
                        st.download_button("Download as CSV", data=csv_data, file_name="decoded_survey_matrix.csv", mime="text/csv")
                    
                    with st.expander("View Data Matrix"):
                        st.dataframe(st.session_state.decoded_df, hide_index=True)
                    
        else:
            st.warning("No codes found in the selected columns. Double check your column mappings above.")

# ==========================================
# TOOL 6: HEDONIC ANALYZER (TWO-WAY ANOVA)
# ==========================================
elif tool == "Hedonic Analyzer":
    st.title("Hedonic Analyzer (Two-Way ANOVA)")
    st.markdown("Analyze incomplete block data by isolating product differences from taster biases.")
    
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Analyzer Data"):
            clear_state_keys(['decoded_df'])

    if not STATSMODELS_AVAILABLE:
        st.error("Missing library. Please run `pip install statsmodels` to use this tool.")
        st.stop()

    if st.session_state.decoded_df is None:
        st.info("Formatting requirement: Ensure your dataset has a column named exactly 'Taster', followed by the products as columns.")

    with st.container(border=True):
        col_cb, _ = st.columns([2, 1])
        with col_cb:
            apply_zscore = st.checkbox("Standardize data using Z-scores (Neutralize taster harshness/generosity to level the playing field)", value=True)
    
        df = None
        transformed_df_display = None
        
        if st.session_state.decoded_df is not None:
            st.success("**Successfully loaded decoded survey data from memory.**")
            df = st.session_state.decoded_df.copy()
        else:
            col_upload, col_url = st.columns(2)
            with col_upload:
                uploaded_file = st.file_uploader("Upload Tasting Scores (CSV)", type=["csv"])
            with col_url:
                gsheet_url = st.text_input("OR Paste Public Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")
            
            df = load_data(uploaded_file, gsheet_url)
    
    if df is not None:
        if 'taster' in df.columns:
            df.rename(columns={'taster': 'Taster'}, inplace=True)
            
        if 'Taster' not in df.columns:
            st.error("Error: Could not find a 'Taster' column. Please check your formatting.")
        else:
            if df['Taster'].dtype == 'object':
                df['Taster'] = df['Taster'].astype(str).str.strip()
            df.columns = [str(c).strip() for c in df.columns]

            df_numeric_raw = df.drop(columns=['Taster'])
            df_numeric = df_numeric_raw.apply(pd.to_numeric, errors='coerce')

            valid_values_check = df_numeric.values[~np.isnan(df_numeric.values)]
            if len(valid_values_check) == 0 or np.nanstd(valid_values_check) == 0:
                st.error("Error: Insufficient variance in data. All scores are identical or invalid. Statistical analysis cannot be performed.")
                st.stop()

            # Store a long-form version of the purely raw data for the Calibration Table
            df_raw_long_source = df_numeric.copy()
            df_raw_long_source.insert(0, 'Taster', df['Taster'])
            df_raw_long = df_raw_long_source.melt(id_vars=['Taster'], var_name='Product', value_name='Score').dropna()

            if apply_zscore:
                raw_values = df_numeric.values
                valid_values = raw_values[~np.isnan(raw_values)]
                global_mean = valid_values.mean()
                global_std = valid_values.std()

                def standardize_and_scale(row):
                    row_std = row.std(ddof=0)
                    if row_std > 0:
                        z = (row - row.mean()) / row_std
                    else:
                        z = row - row.mean()
                    return (z * global_std) + global_mean
                
                df_numeric = df_numeric.apply(standardize_and_scale, axis=1)
                
                transformed_df_display = df_numeric.copy()
                transformed_df_display.insert(0, 'Taster', df['Taster'])

            products = list(df_numeric.columns)
            
            df_long_source = df_numeric.copy()
            df_long_source.insert(0, 'Taster', df['Taster'])

            df_long = df_long_source.melt(id_vars=['Taster'], var_name='Product', value_name='Score').dropna()
            df_long['Taster'] = df_long['Taster'].astype(str).str.strip()
            df_long['Product'] = df_long['Product'].astype(str).str.strip()
            
            try:
                model = ols('Score ~ C(Product) + C(Taster)', data=df_long).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
            except Exception as e:
                st.error("An error occurred during ANOVA execution. Please check your data formatting.")
                st.stop()
            
            # --- ACTION STANDARD (DETECTABLE DIFFERENCE) CALCULATION ---
            # Using 80% power, 95% confidence
            z_alpha = norm.ppf(1 - 0.05 / 2) # approx 1.96
            z_beta = norm.ppf(0.80)          # approx 0.84
            evals_per_product = len(df_long) / len(products)
            
            # Extract residual standard error from the ANOVA model
            residual_std = np.sqrt(model.mse_resid) if hasattr(model, 'mse_resid') else df_long['Score'].std()
            action_standard = (z_alpha + z_beta) * residual_std * np.sqrt(2 / evals_per_product)
            
            product_pval = anova_table.loc['C(Product)', 'PR(>F)']
            taster_pval = anova_table.loc['C(Taster)', 'PR(>F)']
            
            raw_means = df_long.groupby('Product')['Score'].mean()
            adj_means = []
            for p in products:
                pred = model.predict(pd.DataFrame({'Product': [p], 'Taster': [df_long['Taster'].iloc[0]]}))
                adj_means.append({'Product': p, 'Processed Score': raw_means[p], 'Adjusted Score': pred[0]})
                
            adj_df = pd.DataFrame(adj_means)
            correction_factor = raw_means.mean() - adj_df['Adjusted Score'].mean()
            adj_df['Adjusted Score'] = adj_df['Adjusted Score'] + correction_factor
            adj_df = adj_df.sort_values(by='Adjusted Score', ascending=False).reset_index(drop=True)

            pw_tests_raw = model.t_test_pairwise('C(Product)').result_frame.reset_index()
            rename_dict = {
                'index': 'Comparison',
                'coef': 'Difference',
                'P>|t|': 'p-value',
                'pvalue': 'p-value'
            }
            pw_tests_raw = pw_tests_raw.rename(columns=rename_dict)
            
            sig_dict = {}
            if 'p-value' in pw_tests_raw.columns:
                for _, row in pw_tests_raw.iterrows():
                    comp_str = str(row['Comparison'])
                    is_sig = (pd.to_numeric(row['p-value'], errors='coerce') < 0.05)
                    sig_dict[comp_str] = is_sig

            def is_tied(b1, b2):
                if b1 == b2: return True
                match1 = f"{b1}-{b2}"
                match2 = f"{b2}-{b1}"
                for comp_str, is_sig in sig_dict.items():
                    if comp_str == match1 or comp_str == match2:
                        return not is_sig
                for comp_str, is_sig in sig_dict.items():
                    if b1 in comp_str and b2 in comp_str:
                        return not is_sig
                return True

            sorted_products = adj_df['Product'].tolist()
            tiers = {p: "" for p in sorted_products}
            current_tier = 'A'
            
            for i in range(len(sorted_products)):
                anchor = sorted_products[i]
                if tiers[anchor] == "": 
                    tiers[anchor] += current_tier
                    for j in range(i+1, len(sorted_products)):
                        compare_product = sorted_products[j]
                        if is_tied(anchor, compare_product):
                            tiers[compare_product] += current_tier
                    current_tier = chr(ord(current_tier) + 1)
            
            adj_df['Tier'] = adj_df['Product'].map(tiers)

            st.divider()
            
            st.subheader("The Bottom Line")
            
            if product_pval < 0.05:
                top_tier_products = [p for p, t in tiers.items() if 'A' in t]
                if len(top_tier_products) == 1:
                    st.markdown(f"""
                    <div class="bottom-line-box">
                        <strong>Significant Difference Detected:</strong> The panel concluded that <strong>{top_tier_products[0]}</strong> is the undisputed top-performing product.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    products_str = ", ".join(top_tier_products[:-1]) + f" and {top_tier_products[-1]}" if len(top_tier_products) > 2 else f"{top_tier_products[0]} and {top_tier_products[1]}"
                    st.markdown(f"""
                    <div class="bottom-line-box tie">
                        <strong>Statistical Tie:</strong> The panel found a difference between the products overall, but <strong>{products_str}</strong> are statistically tied for first place.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="bottom-line-box none">
                    <strong>No Consensus:</strong> The panel could not detect a statistically reliable difference between any of the products.
                </div>
                """, unsafe_allow_html=True)

            # ==========================================
            # SIDE-BY-SIDE PERFORMANCE RANKINGS
            # ==========================================
            st.subheader("Performance Rankings")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                show_error_bars = st.toggle("Show Error Bars", value=True, help="Displays the error margins (± half the detectable difference) on the ANOVA chart.")
            with col_t2:
                compare_rank = st.toggle("Compare with Rank-Based Preference Test", help="Switch on to compare the Averages (ANOVA) against strict Head-to-Head Rankings (Non-Parametric) side-by-side.")
            
            if compare_rank:
                with st.spinner("Running true Skillings-Mack omnibus test via R..."):
                    sm_pval = 1.0
                    r_error_msg = ""
                    used_fallback = False
                    
                    df_rank = df_long_source.melt(id_vars=['Taster'], var_name='Product', value_name='Score').dropna()
                    df_rank['Taster'] = df_rank['Taster'].astype(str).str.strip()
                    df_rank['Product'] = df_rank['Product'].astype(str).str.strip()
                    
                    try:
                        df_raw_pivot = df_rank.pivot_table(index='Taster', columns='Product', values='Score', aggfunc='mean')
                        df_raw_pivot.to_csv("temp_sm.csv", na_rep="NA")
                        
                        r_sm_script = f"""
                        options(warn=-1)
                        {R_LIB_CMD}
                        library(PMCMRplus)

                        # row.names=1 skips the 'Taster' column so matrix is purely numeric
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
                        with open("run_sm.R", "w") as f:
                            f.write(r_sm_script)
                            
                        result = subprocess.run(["Rscript", "run_sm.R"], capture_output=True, text=True, check=True, timeout=120)
                        
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
                                    r_error_msg = f"R Caught Error: {err_text}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                            
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
                        df_rank['Preference Points'] = df_rank.groupby('Taster')['Score'].rank(ascending=True, method='average')
                        model_rank_fb = ols('Q("Preference Points") ~ C(Product) + C(Taster)', data=df_rank).fit()
                        rank_anova_fb = sm.stats.anova_lm(model_rank_fb, typ=2)
                        sm_pval = rank_anova_fb.loc['C(Product)', 'PR(>F)']

                    df_rank['Preference Points'] = df_rank.groupby('Taster')['Score'].rank(ascending=True, method='average')
                    
                    try:
                        model_rank = ols('Q("Preference Points") ~ C(Product) + C(Taster)', data=df_rank).fit()
                        
                        raw_ranks = df_rank.groupby('Product')['Preference Points'].mean()
                        adj_rank_means = []
                        for p in products:
                            pred_rank = model_rank.predict(pd.DataFrame({'Product': [p], 'Taster': [df_rank['Taster'].iloc[0]]}))
                            adj_rank_means.append({'Product': p, 'Adjusted Preference Score': pred_rank[0]})
                        
                        rank_df = pd.DataFrame(adj_rank_means)
                        correction = raw_ranks.mean() - rank_df['Adjusted Preference Score'].mean()
                        rank_df['Adjusted Preference Score'] = rank_df['Adjusted Preference Score'] + correction
                        rank_df = rank_df.sort_values(by='Adjusted Preference Score', ascending=False).reset_index(drop=True)
                        
                        pw_tests_rank = model_rank.t_test_pairwise('C(Product)').result_frame.reset_index()
                        rename_dict_rank = {'index': 'Comparison', 'coef': 'Difference', 'P>|t|': 'p-value', 'pvalue': 'p-value'}
                        pw_tests_rank = pw_tests_rank.rename(columns=rename_dict_rank)
                        
                        sig_dict_rank = {}
                        if 'p-value' in pw_tests_rank.columns:
                            for _, row in pw_tests_rank.iterrows():
                                comp_str = str(row['Comparison'])
                                is_sig = (pd.to_numeric(row['p-value'], errors='coerce') < 0.05)
                                sig_dict_rank[comp_str] = is_sig
                        
                        def is_tied_rank(b1, b2):
                            if b1 == b2: return True
                            match1 = f"{b1}-{b2}"
                            match2 = f"{b2}-{b1}"
                            for comp_str, is_sig in sig_dict_rank.items():
                                if comp_str == match1 or comp_str == match2:
                                    return not is_sig
                            for comp_str, is_sig in sig_dict_rank.items():
                                if b1 in comp_str and b2 in comp_str:
                                    return not is_sig
                            return True
                        
                        sorted_rank_products = rank_df['Product'].tolist()
                        rank_tiers = {p: "" for p in sorted_rank_products}
                        current_rank_tier = 'A'
                        for i in range(len(sorted_rank_products)):
                            anchor = sorted_rank_products[i]
                            if rank_tiers[anchor] == "": 
                                rank_tiers[anchor] += current_rank_tier
                                for j in range(i+1, len(sorted_rank_products)):
                                    compare_product = sorted_rank_products[j]
                                    if is_tied_rank(anchor, compare_product):
                                        rank_tiers[compare_product] += current_rank_tier
                                current_rank_tier = chr(ord(current_rank_tier) + 1)
                        rank_df['Tier'] = rank_df['Product'].map(rank_tiers)
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig, ax = plt.subplots(figsize=(8, 6))

                            sns.barplot(data=adj_df, x='Product', y='Adjusted Score', hue='Product', palette='Blues_r', edgecolor='.2', dodge=False, ax=ax, zorder=3)
                            
                            if show_error_bars:
                                ax.errorbar(x=np.arange(len(adj_df)), y=adj_df['Adjusted Score'], yerr=action_standard/2, fmt='none', ecolor='black', capsize=4, elinewidth=1.5, zorder=5)
                            
                            if ax.get_legend() is not None: ax.get_legend().remove()
                            for i, row in adj_df.iterrows():
                                y_pos = row['Adjusted Score'] + (action_standard/2 if show_error_bars else 0) + 0.1
                                ax.text(i, y_pos, row['Tier'], ha='center', va='bottom', fontweight='bold', fontsize=12, clip_on=False)
                                
                            ax.set_ylabel("Final Adjusted Score", fontsize=11)
                            ax.set_xlabel("")
                            ax.set_title("ANOVA (Average Scores)", fontsize=13, pad=15)
                            
                            y_max = 9.0
                            if show_error_bars:
                                max_bar_height = adj_df['Adjusted Score'].max() + (action_standard / 2)
                                if max_bar_height > 8.5:
                                    y_max = max_bar_height + 0.5
                            ax.set_ylim(1, y_max)
                            
                            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                            sns.despine()
                            fig.tight_layout()
                            st.pyplot(fig)
                            
                            st.markdown(f"<div style='text-align: center; color: #777; font-size: 0.95rem; margin-top: 10px; margin-bottom: 20px;'>ANOVA p-value: {product_pval:.5f} {('(Significant)' if product_pval < 0.05 else '(Not Significant)')}</div>", unsafe_allow_html=True)
                            st.dataframe(adj_df[['Product', 'Tier', 'Adjusted Score']].round(2), hide_index=True, width='stretch')

                        with col_chart2:
                            fig_rank, ax_rank = plt.subplots(figsize=(8, 6))
                            sns.barplot(data=rank_df, x='Product', y='Adjusted Preference Score', hue='Product', palette='Purples_r', edgecolor='.2', dodge=False, ax=ax_rank)
                            if ax_rank.get_legend() is not None: ax_rank.get_legend().remove()
                            for i, row in rank_df.iterrows():
                                ax_rank.text(i, row['Adjusted Preference Score'] + 0.1, row['Tier'], ha='center', va='bottom', fontweight='bold', fontsize=12, clip_on=False)
                            ax_rank.set_ylabel("Adjusted Preference Points", fontsize=11)
                            ax_rank.set_xlabel("")
                            ax_rank.set_title("Rank Test (Preference Points)", fontsize=13, pad=15)
                            max_score = rank_df['Adjusted Preference Score'].max()
                            ax_rank.set_ylim(0, max_score + (max_score * 0.15))
                            plt.setp(ax_rank.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                            sns.despine()
                            fig_rank.tight_layout()
                            st.pyplot(fig_rank)
                            
                            if sm_pval is not None:
                                if used_fallback:
                                    st.markdown(f"<div style='text-align: center; color: #777; font-size: 0.95rem; margin-top: 10px; margin-bottom: 2px;'>Conover-Iman p-value*: {sm_pval:.5f} {('(Significant)' if sm_pval < 0.05 else '(Not Significant)')}</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='text-align: center; color: #999; font-size: 0.8rem; margin-bottom: 20px;'>*R framework unavailable. Falling back to Conover approximation.</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div style='text-align: center; color: #777; font-size: 0.95rem; margin-top: 10px; margin-bottom: 20px;'>Skillings-Mack p-value: {sm_pval:.5f} {('(Significant)' if sm_pval < 0.05 else '(Not Significant)')}</div>", unsafe_allow_html=True)
                            
                            st.dataframe(rank_df[['Product', 'Tier', 'Adjusted Preference Score']].round(2), hide_index=True, width='stretch')
                            
                        if used_fallback and r_error_msg:
                            with st.expander("View R Debugging Logs"):
                                st.code(r_error_msg, language='plaintext')
                            
                    except Exception as e:
                        st.error(f"Could not calculate rank ANOVA: {e}")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))

                sns.barplot(data=adj_df, x='Product', y='Adjusted Score', hue='Product', palette='Blues_r', edgecolor='.2', dodge=False, ax=ax, zorder=3)
                
                if show_error_bars:
                    ax.errorbar(x=np.arange(len(adj_df)), y=adj_df['Adjusted Score'], yerr=action_standard/2, fmt='none', ecolor='black', capsize=4, elinewidth=1.5, zorder=5)
                
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                for i, row in adj_df.iterrows():
                    y_pos = row['Adjusted Score'] + (action_standard/2 if show_error_bars else 0) + 0.1
                    ax.text(i, y_pos, row['Tier'], ha='center', va='bottom', fontweight='bold', fontsize=12, clip_on=False)
                    
                ax.set_ylabel("Final Adjusted Score", fontsize=11)
                ax.set_xlabel("Product", fontsize=11)
                ax.set_title("Adjusted Mean Scores & Quality Tiers", fontsize=14, pad=15)
                
                y_max = 9.0
                if show_error_bars:
                    max_bar_height = adj_df['Adjusted Score'].max() + (action_standard / 2)
                    if max_bar_height > 8.5:
                        y_max = max_bar_height + 0.5
                ax.set_ylim(1, y_max)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                sns.despine()
                fig.tight_layout()
                st.pyplot(fig)
                
                st.markdown(f"<div style='text-align: center; color: #777; font-size: 0.95rem; margin-top: 10px; margin-bottom: 20px;'>ANOVA p-value: {product_pval:.5f} {('(Significant)' if product_pval < 0.05 else '(Not Significant)')}</div>", unsafe_allow_html=True)
                
                st.markdown("**Adjusted Means Table**")
                display_df = adj_df[['Product', 'Tier', 'Adjusted Score']].round(2)
                st.dataframe(display_df, hide_index=True)

            # ==========================================
            # ACTION STANDARD SUMMARY (DETECTABLE DIFFERENCE)
            # ==========================================
            st.divider()
            st.subheader("Detectable Difference")
            
            top_product = adj_df.iloc[0]['Product']
            top_score = adj_df.iloc[0]['Adjusted Score']
            
            if len(adj_df) > 1:
                runner_up = adj_df.iloc[1]['Product']
                runner_up_score = adj_df.iloc[1]['Adjusted Score']
                gap = top_score - runner_up_score
                
                conclusion_text = f"<strong>{top_product}</strong> beat <strong>{runner_up}</strong> by a margin of <strong>{gap:.2f} points</strong>."
                if gap >= action_standard:
                    conclusion_text += f" Because this exceeds the {action_standard:.2f} threshold, <strong>readers are likely to notice the difference</strong>."
                else:
                    conclusion_text += f" Because this falls short of the {action_standard:.2f} threshold, <strong>readers are unlikely to notice a meaningful difference</strong> between the top two brands."
            else:
                gap = 0
                conclusion_text = "Not enough data to calculate a gap between products."
                
            st.markdown(f"""
            <div class="action-standard-box">
                Based on the size of your panel and the calculated variance, you need a gap of <strong>{action_standard:.2f} points</strong> to confidently declare a noticeable difference.<br><br>
                {conclusion_text}
            </div>
            """, unsafe_allow_html=True)


            st.divider()
            st.subheader("Score Distribution (Polarization)")
            st.markdown("This chart visualizes the spread of opinions. A tight cluster means universal agreement. A wide spread means a polarizing product.")
            
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_long, x='Product', y='Score', color='white', width=0.4, ax=ax_dist)
            sns.swarmplot(data=df_long, x='Product', y='Score', hue='Product', size=5.5, alpha=0.8, palette="husl", ax=ax_dist)
            if ax_dist.get_legend() is not None:
                ax_dist.get_legend().remove()
            
            ax_dist.set_ylabel("Standardized Score" if apply_zscore else "Raw Score", fontsize=11)
            ax_dist.set_xlabel("Product", fontsize=11)
            ax_dist.set_ylim(1, 9)
            
            plt.setp(ax_dist.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            sns.despine()
            fig_dist.tight_layout()
            st.pyplot(fig_dist)

            # ==========================================
            # ADVANCED EDITORIAL ANALYTICS
            # ==========================================
            st.divider()
            st.markdown("### Advanced Editorial Analytics (Optional)")
            st.markdown("Dive deeper into your data to uncover hidden taster groups.")

            with st.expander("Taster Segmentation (Taste Profiles)", expanded=st.session_state.get('run_cluster', False)):
                st.markdown("""
                <div class="advanced-test-box">
                    <strong>Why run this?</strong> If a product has a mediocre average score (e.g., 5.0), it might actually be highly polarizing. This tool splits your tasters into two distinct groups to reveal if a "niche audience" obsessed over a specific product while others hated it. 
                </div>
                """, unsafe_allow_html=True)
                
                run_cluster = st.toggle("Enable Taster Segmentation", key="run_cluster")
                
                if run_cluster:
                    with st.spinner("Finding niche audiences..."):
                        cluster_df = df_numeric.copy()
                        cluster_df.index = df['Taster']
                        
                        cluster_df = cluster_df.fillna(cluster_df.mean())
                        if cluster_df.isnull().values.any():
                            cluster_df = cluster_df.fillna(cluster_df.values.mean())
                        
                        try:
                            data_matrix = cluster_df.values.astype(float)
                            data_matrix += np.random.rand(*data_matrix.shape) * 0.0001 
                            
                            centroids, labels = kmeans2(data_matrix, 2, minit='points')
                            cluster_df['Taste Profile'] = [f"Profile 1" if l == 0 else f"Profile 2" for l in labels]
                            
                            c_counts = cluster_df['Taste Profile'].value_counts()
                            total_tasters = len(cluster_df)
                            p1_pct = (c_counts.get('Profile 1', 0) / total_tasters) * 100
                            p2_pct = (c_counts.get('Profile 2', 0) / total_tasters) * 100
                            
                            plot_df = cluster_df.reset_index().melt(id_vars=['Taster', 'Taste Profile'], var_name='Product', value_name='Average Score')
                            
                            fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))
                            sns.barplot(data=plot_df, x='Product', y='Average Score', hue='Taste Profile', palette='Set2', errorbar=None, ax=ax_cluster)
                            
                            ax_cluster.set_ylim(1, 9)
                            ax_cluster.set_ylabel("Average Score within Profile")
                            ax_cluster.set_title("Polarization Check: How Different Taster Groups Voted", pad=15)
                            plt.setp(ax_cluster.get_xticklabels(), rotation=45, ha='right')
                            sns.despine()
                            st.pyplot(fig_cluster)
                            
                            col_p1, col_p2 = st.columns(2)
                            col_p1.metric("Profile 1", f"{p1_pct:.1f}% of tasters")
                            col_p2.metric("Profile 2", f"{p2_pct:.1f}% of tasters")
                            
                            st.markdown("**How to read this:** The algorithm mathematically divided your panel into two distinct groups based on their voting behavior. Look for products where the green and orange bars are dramatically different—these are your highly polarizing 'niche favorites'.")
                            
                        except Exception as e:
                            st.error(f"Clustering failed (likely due to a small or uniform dataset): {e}")

            # ==========================================
            # QUALITY CONTROL
            # ==========================================
            st.divider()
            st.subheader("Under the Hood: Quality Control")
            
            st.markdown("**Taster Severity Calibration**")
            
            taster_means = df_raw_long.groupby('Taster')['Score'].mean()
            panel_mean = df_raw_long['Score'].mean()
            
            taster_df = pd.DataFrame({
                'Taster ID': taster_means.index,
                'Average Score Given': taster_means.values,
                'Deviation from Panel': taster_means.values - panel_mean
            }).round(2)
            
            taster_df['Profile'] = taster_df['Deviation from Panel'].apply(
                lambda x: "Very Harsh" if x <= -1.5 else ("Harsh" if x < -0.5 else ("Generous" if x > 0.5 else ("Very Generous" if x >= 1.5 else "Average")))
            )
            
            taster_df['Taster ID'] = pd.to_numeric(taster_df['Taster ID'], errors='ignore')
            st.dataframe(taster_df.sort_values('Deviation from Panel'), hide_index=True)

            if apply_zscore and transformed_df_display is not None:
                st.divider()
                st.subheader("Raw Data Matrices")
                with st.expander("View Standardized Z-Score Matrix"):
                    st.markdown("Here is the fully back-transformed, standardized dataset. Each taster's scores have been mean-centered and variance-adjusted, then rescaled to the global 1-to-9 range.")
                    
                    z_df_rounded = transformed_df_display.round(2)
                    st.dataframe(z_df_rounded, hide_index=True)
                    
                    st.markdown("Hover over the block below and click the **Copy** icon to export these standard scores.")
                    st.code(z_df_rounded.to_csv(sep='\t', index=False, float_format='%.2f'), language='plaintext')

# ==========================================
# TOOL 7: FLAVOR PROFILER (PCA & RADAR)
# ==========================================
elif tool == "Flavor Profiler":
    st.title("Flavor Profiler (PCA & Radar Charts)")
    st.markdown("Upload a raw descriptive survey (Taster | Product | Attribute 1 | Attribute 2...) to generate overlapping flavor Radar Charts and a 2D PCA Sensory Map.")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Profiler Data"):
            clear_state_keys(['desc_sim_df'])

    df_desc = None
    if st.session_state.get('desc_sim_df') is not None:
        st.success("**Successfully loaded simulated descriptive data.**")
        df_desc = st.session_state.desc_sim_df.copy()
    else:
        with st.container(border=True):
            uploaded_desc = st.file_uploader("Upload Multivariate Raw Survey (CSV)", type=["csv"], key="desc_survey")
        df_desc = load_data(uploaded_desc, "")
    
    if df_desc is not None:
        st.subheader("1. Map Survey Columns")
        cols = list(df_desc.columns)
        
        dataset_signature = "".join(cols)
        
        c1, c2 = st.columns(2)
        with c1:
            prod_col = st.selectbox(
                "Product Column", 
                cols, 
                index=1 if len(cols)>1 else 0,
                key=f"fp_prod_{dataset_signature}"
            )
        with c2:
            attr_cols = st.multiselect(
                "Select Flavor Attribute Columns", 
                [c for c in cols if c != prod_col], 
                default=[c for c in cols if c != prod_col and c != 'Taster'],
                key=f"fp_attr_{dataset_signature}"
            )
            
        if len(attr_cols) >= 3:
            if st.button("Generate Flavor Profiles", type="primary", width='stretch'):
                with st.spinner("Calculating aggregate means and running Principal Component Analysis..."):
                    df_desc[prod_col] = df_desc[prod_col].astype(str).str.strip()
                    for c in attr_cols:
                        df_desc[c] = pd.to_numeric(df_desc[c], errors='coerce')
                    
                    df_clean = df_desc.dropna(subset=attr_cols)
                    prod_means = df_clean.groupby(prod_col)[attr_cols].mean()
                    products_list = prod_means.index.tolist()
                    
                    st.divider()
                    st.subheader("Overlapping Radar Charts")
                    
                    angles = np.linspace(0, 2 * np.pi, len(attr_cols), endpoint=False).tolist()
                    angles += angles[:1] 
                    
                    fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                    
                    for prod in products_list:
                        values = prod_means.loc[prod].tolist()
                        values += values[:1]
                        ax_radar.plot(angles, values, label=prod, linewidth=2)
                        ax_radar.fill(angles, values, alpha=0.1)
                        
                    ax_radar.set_xticks(angles[:-1])
                    ax_radar.set_xticklabels(attr_cols, fontsize=11, fontweight='bold')
                    
                    ax_radar.set_ylim(1, 7)
                    
                    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                    st.pyplot(fig_radar)
                    
                    st.divider()
                    st.subheader("PCA Sensory Map (Principal Component Analysis)")
                    st.markdown("This algorithm mathematically compresses all flavor attributes onto a 2D grid. Products grouped close together taste similar. The red arrows indicate which flavor attributes are 'pulling' the products in that direction.")
                    
                    X = prod_means.values
                    X_centered = X - np.mean(X, axis=0)
                    
                    try:
                        U, S, Vt = svd(X_centered, full_matrices=False)
                        scores = U * S
                        loadings = Vt.T
                        
                        pc1_var = (S[0]**2 / np.sum(S**2)) * 100
                        pc2_var = (S[1]**2 / np.sum(S**2)) * 100
                        
                        fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
                        
                        colors = plt.cm.tab10(np.linspace(0, 1, len(products_list)))
                        for i, prod in enumerate(products_list):
                            ax_pca.scatter(scores[i, 0], scores[i, 1], marker='o', s=150, color=colors[i], label=prod, edgecolor='black', zorder=5)
                            ax_pca.text(scores[i, 0]+0.1, scores[i, 1]+0.1, prod, fontsize=11, fontweight='bold')
                            
                        scale_factor = np.max(np.abs(scores[:, :2])) / np.max(np.abs(loadings[:, :2]))
                        for j, attr in enumerate(attr_cols):
                            ax_pca.arrow(0, 0, loadings[j, 0]*scale_factor, loadings[j, 1]*scale_factor, color='red', alpha=0.6, head_width=0.1, zorder=4)
                            ax_pca.text(loadings[j, 0]*scale_factor*1.15, loadings[j, 1]*scale_factor*1.15, attr, color='darkred', fontsize=11, ha='center')
                            
                        ax_pca.set_xlabel(f"Principal Component 1 ({pc1_var:.1f}%)", fontsize=12)
                        ax_pca.set_ylabel(f"Principal Component 2 ({pc2_var:.1f}%)", fontsize=12)
                        ax_pca.axhline(0, color='grey', linestyle='--', alpha=0.5)
                        ax_pca.axvline(0, color='grey', linestyle='--', alpha=0.5)
                        ax_pca.set_title("2D Flavor Landscape", pad=15, fontsize=14)
                        
                        sns.despine()
                        fig_pca.tight_layout()
                        st.pyplot(fig_pca)
                        
                    except Exception as e:
                        st.error(f"Not enough variance in the data to run PCA. Need distinct product differences. Error: {e}")

        else:
            st.warning("You must select at least 3 Flavor Attribute Columns to generate Radar Charts and PCA.")

# ==========================================
# TOOL 8: DISCRIMINATION TEST
# ==========================================
elif tool == "Discrimination Test":
    st.title("Discrimination Test Analyzer")
    st.markdown("Calculate statistical significance for difference testing.")
    
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Test Data"):
            st.session_state.discrim_key += 1
            st.rerun()

    with st.container(border=True):
        col_radio, _ = st.columns([1, 1])
        with col_radio:
            method = st.radio("Test Method", ["Triangle", "Tetrad", "Duo-Trio"], horizontal=True, key=f"dt_rad_{st.session_state.discrim_key}")
            
        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            tasters = st.number_input("Total Tasters", min_value=1, value=30, step=1, key=f"dt_tast_{st.session_state.discrim_key}")
        with col2:
            correct = st.number_input("Correct Guesses (leave 0 for planning mode)", min_value=0, max_value=tasters, value=0, step=1, key=f"dt_corr_{st.session_state.discrim_key}")

    with st.expander("Advanced Statistical Settings"):
        col_alpha, _ = st.columns([1, 2])
        with col_alpha:
            alpha = st.slider("Significance Level (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01, key=f"dt_alpha_{st.session_state.discrim_key}")

    p_guess = 0.5 if method == "Duo-Trio" else (1/3)
    min_correct = int(binom.ppf(1 - alpha, tasters, p_guess)) + 1

    st.subheader("Results")
    st.metric("Target Correct Guesses Required", min_correct)

    if correct > 0:
        p_value = binom.sf(correct - 1, tasters, p_guess)
        st.markdown("### Post-Test Analysis")
        st.write(f"**Calculated p-value:** {p_value:.5f}")
        
        if p_value < alpha:
            st.markdown("**Statistically significant:** The panel detected a difference.")
        else:
            st.markdown("**Not significant:** No reliable difference was detected by the panel.")

# ==========================================
# TOOL 9: CORRELATION MATRIX
# ==========================================
elif tool == "Correlation Matrix":
    st.title("Correlation Matrix & Heatmap")
    st.markdown("Analyze relationships between variables using a CSV or public Google Sheet.")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Correlation Data"):
            st.session_state.corr_upload_key += 1
            st.rerun()

    with st.container(border=True):
        col_method, col_cmap, _ = st.columns([1.5, 1, 1])
        with col_method:
            method_choice = st.radio(
                "Correlation Method", 
                ["Auto (Pingouin Test)", "Pearson (Parametric)", "Spearman (Non-Parametric)"],
                horizontal=True
            )
        with col_cmap:
            cmap_choice_raw = st.selectbox(
                "Heatmap Color Palette",
                ["PiYG (Pink-Green)", "coolwarm (Blue-Red)", "RdBu (Red-Blue)", "BrBG (Brown-Green)", "PurpleYellows (Custom)"],
                index=0
            )
            
            if "PurpleYellows" in cmap_choice_raw:
                cmap_choice = mcolors.LinearSegmentedColormap.from_list(
                    "PurpleYellows", ["#51247A", "#FFFFFF", "#E8E29D"]
                )
            else:
                cmap_choice = cmap_choice_raw.split(" ")[0]

        st.divider()
        col_upload, col_url = st.columns(2)
        with col_upload:
            uploaded_file = st.file_uploader("Upload Data (CSV)", type=["csv"], key=f"corr_upload_{st.session_state.corr_upload_key}")
        with col_url:
            gsheet_url = st.text_input("OR Paste Public Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...", key=f"corr_url_{st.session_state.corr_upload_key}")
    
    df = load_data(uploaded_file, gsheet_url)
    
    if df is not None:
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            st.error("No numeric columns found. Please ensure the dataset contains numerical values.")
        else:
            corr_method = "pearson" 
            
            if "Auto" in method_choice:
                if PINGOUIN_AVAILABLE:
                    try:
                        norm_res = multivariate_normality(df_numeric, alpha=.05)
                        corr_method = "pearson" if norm_res.normal else "spearman"
                        st.info(f"Auto-normality test result: Using **{corr_method.title()}** correlation.")
                    except Exception:
                        corr_method = "spearman"
                        st.warning("Normality test failed (likely due to sample size). Defaulting to Spearman correlation.")
                else:
                    corr_method = "spearman"
                    st.warning("Pingouin library not installed. Defaulting to Spearman. Install via `pip install pingouin` to enable the auto mode.")
            else:
                corr_method = method_choice.split()[0].lower()

            corr = df_numeric.corr(method=corr_method)
            mask = np.triu(np.ones_like(corr, dtype=bool))

            st.divider()
            
            st.subheader("Heatmap")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', center=0, cmap=cmap_choice, cbar_kws={"shrink": .5}, ax=ax)
            st.pyplot(fig)

            st.divider()

            st.subheader("Top Correlations")
            df_corr_stacked = corr.stack().reset_index()
            df_corr_stacked.columns = ['FEATURE_1', 'FEATURE_2', 'CORRELATION']
            
            mask_dups = (df_corr_stacked[['FEATURE_1', 'FEATURE_2']].apply(frozenset, axis=1).duplicated()) | (df_corr_stacked['FEATURE_1'] == df_corr_stacked['FEATURE_2'])
            df_corr_stacked = df_corr_stacked[~mask_dups]
            
            sorted_df = df_corr_stacked.sort_values(by="CORRELATION", key=abs, ascending=False).reset_index(drop=True)
            st.dataframe(sorted_df, hide_index=True)
            
            st.divider()
            st.subheader("Export to Spreadsheets")
            st.code(sorted_df.to_csv(sep='\t', index=False), language='plaintext')
