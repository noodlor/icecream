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
nav_btn("Descriptive Analyzer")

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
# TOOL 5: SURVEY DECODER (UNIVERSAL HUB)
# ==========================================
elif tool == "Survey Decoder":
    st.title("Survey Decoder (Universal Hub)")
    st.markdown("Convert raw, unorganized survey exports into a clean, analysis-ready master matrix.")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Decoder Memory"):
            clear_state_keys(['smart_matrix'])
            st.rerun()

    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload Raw Survey Data (CSV)", type=["csv"], key="raw_survey")

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        cols = list(df_raw.columns)
        
        st.subheader("1. Master Key (Optional)")
        st.markdown("Upload your blinding key to automatically convert 3-digit codes into real brand names.")
        uploaded_key = st.file_uploader("Upload Master Key (CSV)", type=["csv"], key="master_key")
        master_dict = {}
        
        def clean_3_digit_code(val):
            if pd.isna(val): return ""
            return str(val).split('.')[0].strip()

        if uploaded_key:
            df_key = pd.read_csv(uploaded_key)
            mapped_cols = {}
            for c in df_key.columns:
                norm = re.sub(r'[^a-z0-9]', '', c.lower())
                if 'code' in norm:
                    mapped_cols['Code'] = c
                elif 'name' in norm or 'real' in norm or 'product' in norm:
                    mapped_cols['Name'] = c
                    
            if 'Code' in mapped_cols and 'Name' in mapped_cols:
                df_key[mapped_cols['Code']] = df_key[mapped_cols['Code']].apply(clean_3_digit_code)
                master_dict = dict(zip(df_key[mapped_cols['Code']], df_key[mapped_cols['Name']].astype(str).str.strip()))
                st.success(f"✅ Loaded {len(master_dict)} brand names from Master Key.")
            elif len(df_key.columns) >= 2:
                # Fallback
                master_dict = dict(zip(df_key.iloc[:, 0].apply(clean_3_digit_code), df_key.iloc[:, 1].astype(str).str.strip()))
                st.success(f"✅ Loaded {len(master_dict)} brand names from Master Key.")

        st.divider()
        st.subheader("2. Map Survey Columns")
        
        # The Format Squeezer (Keeps the beginning, shows the end!)
        def format_col_name(c):
            if len(c) > 55:
                return f"{c[:35]}...{c[-15:]}"
            return c

        # The "Strict Rule" Smart Guesser
        def guess_col_index(search_term, serving_index, columns, fallback_index):
            matches = []
            search_core = search_term.lower()[:5]
            
            for i, c in enumerate(columns):
                c_lower = c.lower()
                
                # If this column contains any of these "open ended" prompt words, ban it completely
                if any(bad in c_lower for bad in ['describe', 'descriptive', 'thoughts', 'why', 'additional']):
                    if search_core not in ['descr', 'thoug', 'text ']:
                        continue
                        
                if search_core in c_lower:
                    matches.append(i)
                    
            if len(matches) > serving_index:
                return matches[serving_index]
            return fallback_index

        taster_idx = next((i for i, c in enumerate(cols) if 'taster' in c.lower() or 'id' in c.lower()), 0)
        code_match_count = sum(1 for c in cols if 'code' in c.lower())
        guessed_servings = code_match_count if code_match_count > 0 else max(1, (len(cols) - 1) // 2)

        col_taster, col_serv, col_attrs = st.columns([1.5, 1, 1])
        with col_taster:
            taster_col = st.selectbox("Taster ID Column", cols, index=taster_idx, format_func=format_col_name)
        with col_serv:
            servings = st.number_input("Number of Servings", min_value=1, max_value=20, value=guessed_servings, step=1)
        with col_attrs:
            num_attrs = st.number_input("Descriptive Attributes", min_value=0, max_value=10, value=3, help="Do not count Product Code or Overall Liking here.")

        st.markdown("**Define your Custom Descriptive Attributes:**")
        default_names = ["Flavor", "Texture", "Sweetness", "Appearance", "Aroma"]
        attr_names = []
        if num_attrs > 0:
            name_cols = st.columns(int(num_attrs))
            for a in range(int(num_attrs)):
                default_val = default_names[a] if a < len(default_names) else f"Attribute {a+1}"
                with name_cols[a]:
                    attr_names.append(st.text_input(f"Attr {a+1} Name", value=default_val, key=f"attr_name_{a}"))

        st.markdown("**Map Columns for Each Serving:**")
        st.markdown("*Verify the auto-mapped columns below. The dropdowns are shortened to show the end of the text so you can spot the '.1' and '.2' labels easily.*")
        
        serving_mappings = []
        
        for i in range(servings):
            st.markdown(f"#### Serving {i+1}")
            
            # The Waterfall Layout
            left_col, right_col = st.columns(2)
            
            with left_col:
                c_idx = guess_col_index('code', i, cols, min(i*3 + 1, len(cols)-1))
                code_c = st.selectbox("Code Col", cols, index=c_idx, format_func=format_col_name, key=f"mcode_{i}")
                
                o_idx = guess_col_index('overall', i, cols, min(i*3 + 2, len(cols)-1))
                overall_c = st.selectbox("Overall Liking Col", cols, index=o_idx, format_func=format_col_name, key=f"moverall_{i}")
            
            attr_c = []
            with right_col:
                for a, name in enumerate(attr_names):
                    a_idx = guess_col_index(name, i, cols, min(i*3 + 3 + a, len(cols)-1))
                    attr_c.append(st.selectbox(f"{name} Col", cols, index=a_idx, format_func=format_col_name, key=f"mattr_{i}_{a}"))
            
            serving_mappings.append({
                "code": code_c, 
                "overall": overall_c,
                "attrs": attr_c
            })
            st.write("")

        st.divider()
        
        if st.button("Stack Data into Master Matrix", type="primary", width="stretch"):
            with st.spinner("Stacking and decoding..."):
                stacked_rows = []
                for idx, row in df_raw.iterrows():
                    taster_id = row[taster_col]
                    for s_idx in range(servings):
                        mapping = serving_mappings[s_idx]
                        
                        raw_code = row[mapping["code"]]
                        overall_score = row[mapping["overall"]]
                        
                        # Decode product names if we have a master key
                        prod_name = str(raw_code).strip()
                        if master_dict:
                            safe_val = clean_3_digit_code(raw_code)
                            prod_name = master_dict.get(safe_val, safe_val)
                            
                        new_row = {
                            "Taster": taster_id,
                            "Product": prod_name,
                            "Overall Liking": overall_score
                        }
                        
                        # Add custom attributes
                        for a_idx, attr_col in enumerate(mapping["attrs"]):
                            attr_name = attr_names[a_idx]
                            new_row[attr_name] = row[attr_col]
                            
                        stacked_rows.append(new_row)
                        
                df_master = pd.DataFrame(stacked_rows)
                df_master = df_master.dropna(subset=["Product", "Overall Liking"], how='any')
                st.session_state.smart_matrix = df_master
                
        if 'smart_matrix' in st.session_state:
            st.success("✅ Master Matrix Successfully Built!")
            st.dataframe(st.session_state.smart_matrix.head(8), hide_index=True)
            
            dl_csv = st.session_state.smart_matrix.to_csv(index=False)
            st.download_button("Download Master Matrix (CSV)", data=dl_csv, file_name="master_decoded_matrix.csv", mime="text/csv")
            
            st.markdown("### Send to Analyzer")
            h_col, d_col = st.columns(2)
            with h_col:
                if st.button("Send to Hedonic Analyzer (Overall Winners)", type="primary", width="stretch"):
                    st.session_state.smart_matrix = st.session_state.smart_matrix # Keep it alive
                    st.session_state.active_tool = "Hedonic Analyzer"
                    st.rerun()
            with d_col:
                if st.button("Send to Descriptive Analyzer (Flavor Profiles)", type="primary", width="stretch"):
                    st.session_state.desc_sim_df = st.session_state.smart_matrix # Preload it!
                    st.session_state.active_tool = "Descriptive Analyzer"
                    st.rerun()

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
        
        # Catch the baton from the Survey Decoder
        if st.session_state.get('smart_matrix') is not None:
            st.success("✅ **Successfully loaded Master Matrix from the Survey Decoder.**")
            df_raw = st.session_state.smart_matrix.copy()
            if st.button("Clear Imported Data & Upload a New CSV"):
                st.session_state.smart_matrix = None
                st.rerun()
        elif st.session_state.get('decoded_df') is not None: # Legacy fallback
            st.success("**Successfully loaded decoded survey data from memory.**")
            df_raw = st.session_state.decoded_df.copy()
        else:
            col_upload, col_url = st.columns(2)
            with col_upload:
                uploaded_file = st.file_uploader("Upload Tasting Scores (CSV)", type=["csv"])
            with col_url:
                gsheet_url = st.text_input("OR Paste Public Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")
            
            df_raw = load_data(uploaded_file, gsheet_url)

        if df_raw is not None:
            st.subheader("1. Map Survey Columns")
            cols = list(df_raw.columns)
            
            # Check if it's long format (has Product and Overall Liking) or wide format (Taster + Products)
            is_long_format = False
            if any('product' in c.lower() for c in cols) and any('overall' in c.lower() or 'liking' in c.lower() or 'score' in c.lower() for c in cols):
                is_long_format = True
                
            if is_long_format:
                c1, c2, c3 = st.columns(3)
                with c1:
                    taster_idx = next((i for i, c in enumerate(cols) if 'taster' in c.lower() or 'id' in c.lower()), 0)
                    taster_col = st.selectbox("Taster ID Column", cols, index=taster_idx)
                with c2:
                    prod_idx = next((i for i, c in enumerate(cols) if 'product' in c.lower() or 'brand' in c.lower()), 1)
                    prod_col = st.selectbox("Product Column", cols, index=prod_idx)
                with c3:
                    default_score = next((i for i, c in enumerate(cols) if 'overall' in c.lower() or 'liking' in c.lower() or 'score' in c.lower()), 2)
                    score_col = st.selectbox("Overall Liking Score", cols, index=default_score)

                # Ensure scores are numeric so math doesn't crash
                df_long_in = df_raw[[taster_col, prod_col, score_col]].rename(columns={taster_col: 'Taster', prod_col: 'Product', score_col: 'Score'})
                df_long_in['Score'] = pd.to_numeric(df_long_in['Score'], errors='coerce')
                df_long_in = df_long_in.dropna(subset=['Score'])
                
                # Convert to wide format so the rest of the legacy Hedonic Analyzer works perfectly!
                df = df_long_in.pivot_table(index='Taster', columns='Product', values='Score', aggfunc='mean').reset_index()
                df.columns.name = None
            else:
                st.info("Wide-format detected. Proceeding with legacy ingestion.")
                df = df_raw.copy()
                
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

            # Create the RAW long dataframe (This will be our single source of truth for all ANOVA math)
            df_raw_long_source = df_numeric.copy()
            df_raw_long_source.insert(0, 'Taster', df['Taster'])
            df_long = df_raw_long_source.melt(id_vars=['Taster'], var_name='Product', value_name='Score').dropna()
            df_long['Taster'] = df_long['Taster'].astype(str).str.strip()
            df_long['Product'] = df_long['Product'].astype(str).str.strip()
            df_raw_long = df_long.copy() # Keep a specific alias for the calibration table

            # Create a separate Z-scored dataframe strictly for visual plotting (if requested)
            df_plot_long = df_long.copy()
            
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
                
                df_z_numeric = df_numeric.apply(standardize_and_scale, axis=1)
                
                transformed_df_display = df_z_numeric.copy()
                transformed_df_display.insert(0, 'Taster', df['Taster'])
                
                df_plot_long = transformed_df_display.melt(id_vars=['Taster'], var_name='Product', value_name='Score').dropna()
                df_plot_long['Taster'] = df_plot_long['Taster'].astype(str).str.strip()
                df_plot_long['Product'] = df_plot_long['Product'].astype(str).str.strip()

            products = list(df_numeric.columns)
            
            # Run the ANOVA on the RAW, un-transformed data
            try:
                model = ols('Score ~ C(Product) + C(Taster)', data=df_long).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
            except Exception as e:
                st.error("An error occurred during ANOVA execution. Please check your data formatting.")
                st.stop()
            
            # --- ACTION STANDARD (DETECTABLE DIFFERENCE) CALCULATION ---
            # Using 80% power, 95% confidence
            z_alpha = norm.ppf(1 - 0.10 / 2) # approx 1.96
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
                    
                    df_rank = df_long.copy() # Uses the mathematically pure raw data for rank conversion
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
            
            st.caption(f"**Residual Standard Error (Noise):** {residual_std:.3f} points")
            
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
            sns.boxplot(data=df_plot_long, x='Product', y='Score', color='white', width=0.4, ax=ax_dist)
            sns.swarmplot(data=df_plot_long, x='Product', y='Score', hue='Product', size=5.5, alpha=0.8, palette="husl", ax=ax_dist)
            if ax_dist.get_legend() is not None:
                ax_dist.get_legend().remove()
            
            ax_dist.set_ylabel("Standardized Score" if apply_zscore else "Raw Score", fontsize=11)
            ax_dist.set_xlabel("Product", fontsize=11)
            ax_dist.set_ylim(0.5, 9.5)
            
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

            with st.expander("Taster Segmentation (Taste Profiles)"):
                st.markdown("""
                <div class="advanced-test-box">
                    <strong>⚠️ INCOMPLETE BLOCK WARNING:</strong> Because tasters did not evaluate every product, this clustering algorithm has to artificially infer the missing scores. This can distort the "Taste Tribes." Use this heatmap for visual exploration, but do not rely on it for strict statistical conclusions.
                </div>
                <div class="advanced-test-box" style="margin-top: -10px;">
                    <strong>Why run this?</strong> If a product has a mediocre average score (e.g., 5.0), it might actually be highly polarizing. This tool mathematically splits your tasters into distinct flavor camps to reveal if a "niche audience" obsessed over a specific product while others hated it. 
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Finding niche audiences..."):
                    cluster_df = df_numeric.copy()
                    cluster_df.index = df['Taster']
                    
                    cluster_df = cluster_df.fillna(cluster_df.mean())
                    if cluster_df.isnull().values.any():
                        cluster_df = cluster_df.fillna(cluster_df.values.mean())
                    
                    try:
                        from sklearn.cluster import KMeans
                        from sklearn.metrics import silhouette_score
                        
                        data_matrix = cluster_df.values.astype(float)
                        data_matrix += np.random.rand(*data_matrix.shape) * 0.0001 
                        
                        max_k = min(6, len(data_matrix) - 1)
                        if max_k >= 3:
                            distortions = []
                            silhouettes = []
                            K_range = range(2, max_k + 1)
                            
                            best_k = 2
                            best_sil = -1
                            
                            for k in K_range:
                                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                                km.fit(data_matrix)
                                distortions.append(km.inertia_)
                                sil = silhouette_score(data_matrix, km.labels_)
                                silhouettes.append(sil)
                                if sil > best_sil:
                                    best_sil = sil
                                    best_k = k
                        else:
                            K_range = [2]
                            best_k = 2
                            distortions = []
                            silhouettes = []
                            
                        st.subheader("Polarization Check")
                        st.markdown("**How Different Taster Groups Voted**")
                        
                        col_dial, col_warn = st.columns([1, 2])
                        
                        # THE FIX: Add '1' to the options list for the Magic Dial
                        dial_options = [1] + list(K_range)
                        default_idx = dial_options.index(best_k) if best_k in dial_options else 0
                        
                        with col_dial:
                            selected_k = st.selectbox(
                                "How many flavor profiles? (Magic Dial)", 
                                options=dial_options, 
                                index=default_idx
                            )
                        with col_warn:
                            st.info(f"**Note:** Math says **{best_k}** is optimal for clear data separation, but you can adjust this if a different grouping is simpler to explain in your report.")
                            
                        # THE FIX: Bypass KMeans if the user selects 1 profile
                        if selected_k == 1:
                            labels = np.zeros(len(data_matrix), dtype=int)
                        else:
                            km_final = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
                            labels = km_final.fit_predict(data_matrix)
                        
                        unique, counts = np.unique(labels, return_counts=True)
                        total_tasters = len(labels)
                        profile_names = {}
                        for l, c in zip(unique, counts):
                            pct = (c / total_tasters) * 100
                            # Clean up the name if it's just 1 profile
                            profile_names[l] = "Entire Panel (100.0%)" if selected_k == 1 else f"Profile {l+1} ({pct:.1f}%)"
                            
                        cluster_df['Taste Profile'] = [profile_names[l] for l in labels]
                        
                        plot_df = cluster_df.reset_index().melt(id_vars=['Taster', 'Taste Profile'], var_name='Product', value_name='Average Score')
                        plot_df = plot_df.sort_values(by='Taste Profile')
                        
                        fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=plot_df, x='Product', y='Average Score', hue='Taste Profile', palette='Set2', errorbar=None, ax=ax_cluster)
                        
                        ax_cluster.set_ylim(1, 9)
                        ax_cluster.set_ylabel("Average Score within Profile")
                        plt.setp(ax_cluster.get_xticklabels(), rotation=45, ha='right')
                        sns.despine()
                        st.pyplot(fig_cluster)
                        
                        st.markdown("**How to read this:** The algorithm mathematically divided your panel into distinct groups based on their voting behavior. Look for products where the bars are dramatically different—these are your highly polarizing 'niche favorites'.")
                        
                        # ==========================================
                        # NEW TASTE TRIBE HEATMAP
                        # ==========================================
                        st.divider()
                        st.subheader("The 'Taste Tribe' Heatmap")
                        st.markdown("This chart plots every single vote from the panel. The products (columns) are sorted left-to-right by their ultimate rank. The tasters (rows) have been mathematically reorganized and grouped by their Taste Profile. Look for massive blocks of solid color to see exactly where the tribes agreed or went to war over specific ice creams.")
                        
                        # THE FIX: Add a toggle to show/hide the inferred scores
                        show_imputed = st.checkbox("Mark mathematically inferred scores with an asterisk (*)", value=True)
                        
                        # Sort the dataframe so tasters in the same profile are grouped together visually
                        heatmap_data = cluster_df.copy()
                        
                        # THE FIX: Force Python to treat the Taster IDs as integers so they sort 1, 2, 10 instead of 1, 10, 2
                        heatmap_data['Taster_Num'] = pd.to_numeric(heatmap_data.index, errors='coerce')
                        heatmap_data = heatmap_data.sort_values(by=['Taste Profile', 'Taster_Num'])
                        
                        # Clean up Y-axis labels so they don't redundantly say "Profile 1" if K=1
                        if selected_k == 1:
                            y_labels = [f"Taster {idx}" for idx, row in heatmap_data.iterrows()]
                        else:
                            y_labels = [f"Taster {idx} ({row['Taste Profile']})" for idx, row in heatmap_data.iterrows()]
                        
                        # Drop the string and sorting columns
                        heatmap_numeric = heatmap_data.drop(columns=['Taste Profile', 'Taster_Num'])
                        
                        # Grab the final rank order and force the columns to match
                        rank_ordered_products = adj_df['Product'].tolist()
                        safe_ordered_cols = [p for p in rank_ordered_products if p in heatmap_numeric.columns]
                        heatmap_numeric = heatmap_numeric[safe_ordered_cols]
                        
                        # THE FIX: Build the transparent overlay to mark the inferred scores
                        if show_imputed:
                            # Safely grab the raw data with the missing NaNs intact
                            raw_for_heatmap = df_numeric_raw.copy()
                            raw_for_heatmap.index = df['Taster']
                            # Align it perfectly with our newly sorted heatmap
                            raw_aligned = raw_for_heatmap.reindex(index=heatmap_numeric.index, columns=heatmap_numeric.columns)
                            # Create an array of asterisks wherever the raw data was blank
                            annot_labels = np.where(raw_aligned.isna(), "*", "")
                        else:
                            # Give it a blank overlay if the toggle is off
                            annot_labels = np.full(heatmap_numeric.shape, "")
                        
                        fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
                        
                        # Draw the heatmap (RdBu_r: Red = High Score/Hot, Blue = Low Score/Cold)
                        # Notice we pass `annot=annot_labels` to draw our asterisks
                        sns.heatmap(heatmap_numeric, cmap="RdBu_r", center=5, vmin=1, vmax=9, 
                                    yticklabels=y_labels, cbar_kws={'label': 'Score (1 = Dislike, 9 = Like)'}, 
                                    annot=annot_labels, fmt="", annot_kws={'size': 18, 'va': 'center'}, ax=ax_heat)
                        
                        ax_heat.set_ylabel("Tasters (Grouped by Tribe)" if selected_k > 1 else "Tasters", fontsize=11)
                        ax_heat.set_xlabel("Product (Ranked 1st to Last)", fontsize=11)
                        plt.setp(ax_heat.get_xticklabels(), rotation=45, ha='right')
                        fig_heat.tight_layout()
                        st.pyplot(fig_heat)
                        # ==========================================
                        # END NEW HEATMAP CODE
                        # ==========================================
                        # ==========================================
                        # MAGAZINE-STYLE VISUALIZATIONS
                        # ==========================================
                        st.divider()
                        st.subheader("Magazine-Style Editorial Visualizations")
                        st.markdown("These charts strip away the heavy statistics to focus purely on visual storytelling for your readers. Toggle them on below:")
                        
                        col_chk1, col_chk2 = st.columns(2)
                        with col_chk1:
                            show_ridge = st.checkbox("Ridge Plot (Joyplot)")
                        with col_chk2:
                            show_slope = st.checkbox("The Great Divide (Slopegraph)")
                        show_pca = False # Disabled for Hedonic Data due to incomplete block invalidity
                        
                        if show_ridge:
                            st.markdown("#### 1. The Ridge Plot")
                            st.markdown("Look for tall, skinny peaks (consensus) vs. wide double-peaks (highly polarizing).")
                            
                            # Reverse order so the winner is at the top of the chart
                            ranked_prods = adj_df['Product'].tolist()[::-1] 
                            fig_ridge, axes_ridge = plt.subplots(len(ranked_prods), 1, figsize=(10, 0.8 * len(ranked_prods)), sharex=True, gridspec_kw={'hspace': -0.4})
                            
                            if len(ranked_prods) == 1:
                                axes_ridge = [axes_ridge]
                                
                            for i, p in enumerate(ranked_prods):
                                subset = df_plot_long[df_plot_long['Product'] == p]['Score'].dropna()
                                if len(subset) > 1:
                                    sns.kdeplot(subset, ax=axes_ridge[i], fill=True, clip=(1,9), bw_adjust=1.2, color="#4c72b0", alpha=0.7, linewidth=1.5)
                                axes_ridge[i].set_ylabel(p, rotation=0, ha='right', va='center', fontsize=10, fontweight='bold')
                                axes_ridge[i].set_yticks([])
                                axes_ridge[i].set_xlim(1, 9)
                                axes_ridge[i].spines['top'].set_visible(False)
                                axes_ridge[i].spines['right'].set_visible(False)
                                axes_ridge[i].spines['left'].set_visible(False)
                                axes_ridge[i].patch.set_alpha(0) # Makes the overlapping transparent
                            
                            axes_ridge[-1].set_xlabel("Score (1 to 9)")
                            st.pyplot(fig_ridge)

                        if show_slope:
                            st.markdown("#### 2. The Great Divide (Slopegraph)")
                            if selected_k >= 2:
                                profiles = cluster_df['Taste Profile'].unique()[:2]
                                p1_name, p2_name = profiles[0], profiles[1]
                                
                                # Calculate average scores per profile, then rank them 1 to 10
                                p1_scores = cluster_df[cluster_df['Taste Profile'] == p1_name].drop(columns=['Taste Profile', 'Taster_Num'], errors='ignore').mean(numeric_only=True)
                                p2_scores = cluster_df[cluster_df['Taste Profile'] == p2_name].drop(columns=['Taste Profile', 'Taster_Num'], errors='ignore').mean(numeric_only=True)
                                
                                # THE FIX: Calculate the "Real" rank for the text labels, and a "Plot" rank to prevent overlapping
                                p1_ranks_real = p1_scores.rank(ascending=False, method='min')
                                p2_ranks_real = p2_scores.rank(ascending=False, method='min')
                                
                                p1_ranks_plot = p1_scores.rank(ascending=False, method='first')
                                p2_ranks_plot = p2_scores.rank(ascending=False, method='first')
                                
                                fig_slope, ax_slope = plt.subplots(figsize=(8, 8))
                                
                                for prod in p1_ranks_real.index:
                                    r1_real = p1_ranks_real[prod]
                                    r2_real = p2_ranks_real[prod]
                                    
                                    r1_plot = p1_ranks_plot[prod]
                                    r2_plot = p2_ranks_plot[prod]
                                    
                                    # Color logic: Red = Polarizing, Green = Agreement, Gray = Minor shift
                                    color = "gray"
                                    if abs(r1_real - r2_real) >= 4:
                                        color = "#d62728" 
                                    elif r1_real == r2_real:
                                        color = "#2ca02c" 
                                        
                                    # Plot lines using the unique Y-coordinates
                                    ax_slope.plot([1, 2], [r1_plot, r2_plot], marker='o', color=color, linewidth=2, markersize=8)
                                    
                                    # Print text using the real rank numbers
                                    if r1_real == r2_real:
                                        ax_slope.text(0.95, r1_plot, prod, ha='right', va='center', fontsize=10)
                                        ax_slope.text(2.05, r2_plot, prod, ha='left', va='center', fontsize=10)
                                    else:
                                        ax_slope.text(0.95, r1_plot, f"{prod}  (#{int(r1_real)})", ha='right', va='center', fontsize=10)
                                        ax_slope.text(2.05, r2_plot, f"(#{int(r2_real)})  {prod}", ha='left', va='center', fontsize=10)
                                
                                ax_slope.set_xticks([1, 2])
                                ax_slope.set_xticklabels([p1_name, p2_name], fontsize=12, fontweight='bold')
                                ax_slope.set_yticks([])
                                ax_slope.set_xlim(0.5, 2.5)
                                ax_slope.invert_yaxis() 
                                ax_slope.spines['top'].set_visible(False)
                                ax_slope.spines['right'].set_visible(False)
                                ax_slope.spines['bottom'].set_visible(False)
                                ax_slope.spines['left'].set_visible(False)
                                st.pyplot(fig_slope)
                                st.markdown("*(**Red lines** indicate a massive shift in preference of 4+ ranks. **Green lines** indicate perfect tribal agreement.)*")
                            else:
                                st.warning("The Slopegraph requires at least 2 tribes to compare. Please set the Magic Dial to 2 or more!")

                        if show_pca:
                            st.markdown("#### 3. The Vanilla Constellation (PCA Flavor Map)")
                            st.markdown("Products that are closer together share a very similar fanbase. Products far apart appeal to opposite palates.")
                            
                            from sklearn.decomposition import PCA
                            
                            # Safely build the data map directly from df_long
                            pca_pivot = df_long.pivot_table(index='Product', columns='Taster', values='Score', aggfunc='mean')
                            pca_data = pca_pivot.fillna(pca_pivot.median(axis=1)).fillna(5) # Fill blanks neutrally
                            
                            if len(pca_data) >= 3:
                                pca = PCA(n_components=2)
                                coords = pca.fit_transform(pca_data)
                                
                                fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
                                ax_pca.scatter(coords[:, 0], coords[:, 1], s=150, color='#ff7f0e', edgecolor='black', zorder=3)
                                
                                # Annotate the dots with product names
                                for i, txt in enumerate(pca_data.index):
                                    ax_pca.annotate(txt, (coords[i, 0], coords[i, 1]), xytext=(8, 8), 
                                                    textcoords='offset points', fontsize=11, fontweight='bold',
                                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                                
                                ax_pca.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
                                ax_pca.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
                                ax_pca.set_xlabel(f"Primary Preference Axis ({pca.explained_variance_ratio_[0]*100:.1f}% of variance)")
                                ax_pca.set_ylabel(f"Secondary Preference Axis ({pca.explained_variance_ratio_[1]*100:.1f}% of variance)")
                                sns.despine()
                                st.pyplot(fig_pca)
                            else:
                                st.warning("Not enough products to build a flavor map.")
                        # ==========================================
                        # END MAGAZINE VISUALIZATIONS
                        # ==========================================

                        if len(K_range) > 1:
                            with st.expander("Advanced Clustering Diagnostics"):
                                fig_diag, (ax_elb, ax_sil) = plt.subplots(1, 2, figsize=(10, 4))
                                
                                ax_elb.plot(K_range, distortions, marker='o', color='#1f77b4')
                                ax_elb.set_title("Elbow Method (Look for the bend)", fontsize=11)
                                ax_elb.set_xlabel("Number of Profiles (k)")
                                ax_elb.set_ylabel("Distortion / Inertia")
                                ax_elb.set_xticks(K_range)
                                
                                colors = ['#ff7f0e' if k != best_k else '#2ca02c' for k in K_range]
                                ax_sil.bar(K_range, silhouettes, color=colors, alpha=0.8)
                                ax_sil.set_title("Silhouette Score (Higher is better)", fontsize=11)
                                ax_sil.set_xlabel("Number of Profiles (k)")
                                ax_sil.set_ylabel("Silhouette Score")
                                ax_sil.set_xticks(K_range)
                                
                                sns.despine(fig=fig_diag)
                                fig_diag.tight_layout()
                                st.pyplot(fig_diag)
                                
                                st.markdown(f"The algorithm tested models from {min(K_range)} to {max(K_range)} profiles. The Silhouette Score peaked at **{best_k} profiles**, making it the mathematically optimal choice.")
                        
                    except ImportError:
                        st.error("Missing library. Please run `pip install scikit-learn` to use the advanced segmentation tool.")
                    except Exception as e:
                        st.error(f"Clustering failed (likely due to a small or uniform dataset): {e}")

                    if len(products) > 1:
                            st.write("")
                            with st.expander("Analyze Rank Agreement vs. Scaling Noise", expanded=False):
                                st.markdown("""
                                <div class="advanced-test-box">
                                    <strong>Why run this?</strong> We already know the overall rankings (A, B, C...). But we have a high "Noise Meter" score of 2.16. <br><br>This visual proves that the noise isn't because tasters disagreed on <em>what</em> was good. It proves they <strong>perfectly agreed on the ranking</strong>, they just used the 1-to-9 scale differently (the "Fanning Effect"). Tasters agreed Product A was 1st and Product J was last; they just disagreed on whether A was "6 points better" or only "1 point better" than J.
                                </div>
                                """, unsafe_allow_html=True)
                                
                                rank_ordered_products = adj_df['Product'].tolist()
                                
                                # Build the pivot table directly from the clean df_plot_long
                                ordered_pivot = df_plot_long.pivot_table(index='Taster', columns='Product', values='Score', aggfunc='mean')
                                
                                # Reorder columns to match the final ranking
                                ordered_pivot = ordered_pivot[rank_ordered_products]
                                
                                # Transpose to make Products the X-axis and Tasters the lines
                                df_spag = ordered_pivot.T
                                
                                fig_spag, ax_spag = plt.subplots(figsize=(10, 6))
                                
                                # THE FIX: Convert the Pandas Index (Ice Cream Names) to a raw Numpy array
                                x_axis_vals = df_spag.index.to_numpy()
                                
                                # Plot every taster line in light gray
                                for column in df_spag.columns:
                                    y_vals = df_spag[column].to_numpy() # THE FIX: Convert to raw Numpy array
                                    ax_spag.plot(x_axis_vals, y_vals, color='gray', alpha=0.3, linewidth=1)
                                
                                # Plot the overall Median in bold black
                                median_line = ordered_pivot.median()
                                ax_spag.plot(x_axis_vals, median_line.to_numpy(), color='black', linewidth=3.5, label='Panel Median (The consensus)')
                                
                                # Plot the best taster in green (most consistent)
                                correlations = ordered_pivot.apply(lambda row: row.corr(median_line), axis=1)
                                try:
                                    best_index_pos = correlations.argmax()
                                    best_taster_id = correlations.index[best_index_pos]
                                    best_taster_series = ordered_pivot.iloc[best_index_pos]
                                    ax_spag.plot(x_axis_vals, best_taster_series.to_numpy(), color='#2ca02c', linewidth=2.5, linestyle=':', label=f'{best_taster_id} (Most Consistent)')
                                except Exception:
                                    pass

                                ax_spag.set_ylabel("Standardized Score" if apply_zscore else "Raw Score", fontsize=11)
                                ax_spag.set_ylim(0.5, 9.5)
                                ax_spag.set_title("Rank Agreement Check (Spaghetti Plot)", fontsize=13, pad=15)
                                ax_spag.legend(loc='lower left')
                                plt.setp(ax_spag.get_xticklabels(), rotation=45, ha='right')
                                sns.despine()
                                fig_spag.tight_layout()
                                st.pyplot(fig_spag)
                                
                                st.markdown("**How to read this:** The products on the bottom are ordered from the #1 Winner (left) to the Last Place (right). Look at the gray lines. They 'fan out' wildly (high noise), but they almost all generally follow the black median line **downhill.** This proves they agreed on the *ranking*, they just violently disagreed on *magnitude*.")

            # ==========================================
            # QUALITY CONTROL
            # ==========================================
            st.divider()
            st.subheader("Under the Hood: Quality Control")
            
            st.markdown("**Panel Noise Meter**")
            
            if residual_std < 1.0:
                noise_status = "🌟 **Lab Quality (Very Low Noise)** - This panel was incredibly consistent!"
            elif residual_std < 1.8:
                noise_status = "✅ **Standard Consumer Test (Normal Noise)** - Typical human inconsistency."
            else:
                noise_status = "⚠️ **High Noise** - Tasters were highly unpredictable. Expect a larger required gap to prove a winner."
            
            st.info(f"**Residual Standard Error:** {residual_std:.2f} \n\n {noise_status}")

            with st.expander("View Consistency Map (Predicted vs. Actual)", expanded=False):
                st.markdown("This chart plots what the math *expected* each taster to say vs. what they *actually* said. Dots hugging the red line represent perfect consistency. A wide, scattered cloud indicates highly unpredictable tasters (noise).")
                
                fig_qc, ax_qc = plt.subplots(figsize=(8, 6))
                predicted = model.fittedvalues
                actual = df_long['Score']
                
                sns.scatterplot(x=predicted, y=actual, alpha=0.7, color='#1f77b4', s=70, edgecolor='black', ax=ax_qc)
                
                # Diagonal line
                min_val = min(predicted.min(), actual.min()) - 0.5
                max_val = max(predicted.max(), actual.max()) + 0.5
                ax_qc.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', alpha=0.6, label='Perfect Consistency')
                
                ax_qc.set_xlabel("Predicted Score (Model Expectation)", fontsize=11)
                ax_qc.set_ylabel("Actual Score (What Taster Said)", fontsize=11)
                ax_qc.set_title("Predicted vs. Actual Scores", fontsize=13, pad=15)
                ax_qc.legend()
                sns.despine()
                fig_qc.tight_layout()
                st.pyplot(fig_qc)

            st.write("")
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
# TOOL 7: FLAVOR PROFILER (AUTOMATED ANOVA, PCA & RADAR)
# ==========================================
elif tool == "Descriptive Analyzer":
    st.title("Descriptive Analyzer (Automated ANOVA, PCA & Radar)")
    st.markdown("Analyze stacked descriptive data. Automatically runs significance testing on every attribute and renders 2D PCA Sensory Maps to visualize product similarities.")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear Profiler Data"):
            clear_state_keys(['desc_sim_df'])

    # Safely load advanced mathematical libraries
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("Missing libraries. Please ensure `statsmodels`, `scikit-learn`, and `matplotlib` are installed to run this tool.")
        st.stop()

    df_desc = None
    
    # Catch the baton from the Survey Decoder
    if st.session_state.get('desc_sim_df') is not None:
        st.success("✅ **Successfully loaded descriptive data matrix from the Survey Decoder.**")
        df_desc = st.session_state.desc_sim_df.copy()
        
        if st.button("Clear Imported Data & Upload a New CSV"):
            st.session_state.desc_sim_df = None
            st.rerun()
    else:
        with st.container(border=True):
            uploaded_desc = st.file_uploader("Upload Decoded Master Matrix (CSV)", type=["csv"], key="desc_survey")
            if uploaded_desc is not None:
                df_desc = pd.read_csv(uploaded_desc)
    
    if df_desc is not None:
        st.divider()
        st.subheader("1. Map Survey Columns & Settings")
        
        # Z-Score Toggle
        apply_zscore = st.checkbox("Standardize data using Z-scores before plotting (Neutralizes taster harshness/generosity for cleaner charts)", value=True)
        st.write("")

        cols = list(df_desc.columns)
        
        # Smart guessing for Taster and Product columns
        taster_idx = next((i for i, c in enumerate(cols) if 'taster' in c.lower() or 'id' in c.lower() or 'panelist' in c.lower()), 0)
        prod_idx = next((i for i, c in enumerate(cols) if 'product' in c.lower() or 'brand' in c.lower() or 'sample' in c.lower()), 1 if len(cols)>1 else 0)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            taster_col = st.selectbox("Taster Column (Critical for filtering out human bias)", cols, index=taster_idx)
        with c2:
            prod_col = st.selectbox("Product Column", cols, index=prod_idx)
        with c3:
            liking_options = ["None (Do not map overall liking)"] + cols
            default_liking_idx = next((i + 1 for i, c in enumerate(cols) if 'overall' in c.lower() or 'liking' in c.lower()), 0)
            liking_col = st.selectbox("Overall Liking Column (Optional)", liking_options, index=default_liking_idx)
            
        excluded = [prod_col, taster_col]
        # We explicitly DO NOT exclude liking_col anymore, so it appears on all charts
            
        default_attrs = [c for c in cols if c not in excluded]
        attr_cols = st.multiselect("Select Descriptive Attributes", default_attrs, default=default_attrs)
            
        if len(attr_cols) >= 3:
            if st.button("Generate Statistical Flavor Profiles", type="primary", width='stretch'):
                st.session_state.desc_profiles_generated = True

            if st.session_state.get('desc_profiles_generated', False):
                with st.spinner("Crunching automated ANOVAs and rendering PCA geometry..."):
                    
                    # Clean data types
                    df_desc[prod_col] = df_desc[prod_col].astype(str).str.strip()
                    for c in attr_cols:
                        df_desc[c] = pd.to_numeric(df_desc[c], errors='coerce')
                    
                    if liking_col != "None (Do not map overall liking)":
                        df_desc[liking_col] = pd.to_numeric(df_desc[liking_col], errors='coerce')
                        df_clean = df_desc.dropna(subset=attr_cols + [prod_col, taster_col, liking_col])
                    else:
                        df_clean = df_desc.dropna(subset=attr_cols + [prod_col, taster_col])
                    
                    # ==========================================
                    # 1. THE AUTOMATED ANOVA LOOP (Uses raw data)
                    # ==========================================
                    st.divider()
                    st.subheader("Statistical Significance (Automated Two-Way ANOVA)")
                    st.markdown("We ran an independent Two-Way ANOVA on every attribute to determine if the panel actually detected a real difference between the products, mathematically filtering out individual taster bias.")
                    
                    anova_results = []
                    for attr in attr_cols:
                        try:
                            formula = f"Q('{attr}') ~ C(Q('{prod_col}')) + C(Q('{taster_col}'))"
                            model = smf.ols(formula, data=df_clean).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            
                            p_val = anova_table.loc[f"C(Q('{prod_col}'))", 'PR(>F)']
                            f_val = anova_table.loc[f"C(Q('{prod_col}'))", 'F']
                            
                            if p_val < 0.01:
                                sig_label = "🌟 Highly Significant (p < 0.01)"
                            elif p_val < 0.05:
                                sig_label = "✅ Significant (p < 0.05)"
                            else:
                                sig_label = "❌ Not Significant"
                                
                            anova_results.append({
                                "Attribute": attr,
                                "Result": sig_label,
                                "p-value": f"{p_val:.4f}",
                                "F-statistic": f"{f_val:.2f}"
                            })
                        except Exception as e:
                            anova_results.append({"Attribute": attr, "Result": "Error parsing data", "p-value": "N/A", "F-statistic": "N/A"})
                            
                    st.dataframe(pd.DataFrame(anova_results), hide_index=True)
                    
                    # ==========================================
                    # 2. Z-SCORE STANDARDIZATION (For visual charts)
                    # ==========================================
                    df_plot = df_clean.copy()
                    
                    if apply_zscore:
                        cols_to_z = attr_cols.copy()
                        if liking_col != "None (Do not map overall liking)":
                            cols_to_z.append(liking_col)
                            
                        for attr in cols_to_z:
                            global_mean = df_plot[attr].mean()
                            global_std = df_plot[attr].std()
                            
                            def standardize_taster(group):
                                std = group.std(ddof=0)
                                if std > 0:
                                    z = (group - group.mean()) / std
                                else:
                                    z = group - group.mean()
                                return (z * global_std) + global_mean
                                
                            df_plot[attr] = df_plot.groupby(taster_col)[attr].transform(standardize_taster)

                    # Calculate means and standard errors for the charts
                    prod_means = df_plot.groupby(prod_col)[attr_cols].mean()
                    prod_sems = df_plot.groupby(prod_col)[attr_cols].sem().fillna(0)
                    products_list = prod_means.index.tolist()
                    
                    if liking_col != "None (Do not map overall liking)":
                        overall_means = df_plot.groupby(prod_col)[liking_col].mean()
                        # Sort products by overall liking
                        products_list = overall_means.sort_values(ascending=False).index.tolist()
                        prod_means = prod_means.reindex(products_list)
                        prod_sems = prod_sems.reindex(products_list)
                    
                    # ==========================================
                    # 3. AGGREGATE MEANS TABLE
                    # ==========================================
                    st.divider()
                    st.subheader("Average Scores")
                    if apply_zscore:
                        st.caption("*(Adjusted via Z-score)*")
                        
                    display_means = prod_means.copy()
                    if liking_col != "None (Do not map overall liking)":
                        display_means['Overall Liking'] = overall_means.reindex(products_list)
                    
                    st.dataframe(display_means.round(2))
                    
                    # ==========================================
                    # NEW: MAGAZINE-STYLE ATTRIBUTE VS LIKING CHARTS
                    # ==========================================
                    if liking_col != "None (Do not map overall liking)":
                        st.divider()
                        st.subheader("Editorial Deep Dives")
                        
                        col_chk1, col_chk2 = st.columns(2)
                        with col_chk1:
                            show_bubble = st.checkbox("Bubble Matrix (Attribute Battlefield)", value=True)
                        with col_chk2:
                            show_triple = st.checkbox("Ranked Attribute Breakdown (Horizontal Bar)", value=True)
                            
                        if show_bubble and len(attr_cols) >= 2:
                            st.markdown("#### The Attribute Battlefield (Bubble Matrix)")
                            st.markdown("Maps two specific descriptive attributes against Overall Liking. **Bubble Size and Color represent the Overall Liking score.** Look for patterns: does a massive green bubble still appear even when one attribute is rated poorly? That proves which attribute matters more to your tasters!")
                            
                            # Give the user drop downs to pick their X and Y for the bubble matrix
                            b1, b2 = st.columns(2)
                            with b1:
                                x_attr = st.selectbox("X-Axis Attribute", attr_cols, index=0)
                            with b2:
                                y_attr = st.selectbox("Y-Axis Attribute", attr_cols, index=1 if len(attr_cols) > 1 else 0)
                            
                            fig_bub, ax_bub = plt.subplots(figsize=(10, 7))
                            
                            x_vals = prod_means[x_attr]
                            y_vals = prod_means[y_attr]
                            sizes = overall_means.reindex(products_list)
                            
                            # Normalize sizes for plotting
                            min_s = sizes.min()
                            max_s = sizes.max()
                            if max_s > min_s:
                                plot_sizes = ((sizes - min_s) / (max_s - min_s)) * 1000 + 200
                            else:
                                plot_sizes = [500] * len(sizes)
                                
                            scatter = ax_bub.scatter(x_vals, y_vals, s=plot_sizes, c=sizes, cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidth=1.5)
                            
                            for i, p in enumerate(products_list):
                                letter = chr(65 + i)
                                ax_bub.annotate(letter, (x_vals[p], y_vals[p]), xytext=(0, 0), textcoords='offset points', 
                                                ha='center', va='center', fontsize=12, fontweight='bold', 
                                                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", alpha=0.85))
                            
                            ax_bub.set_xlabel(f"Average {x_attr}", fontsize=11, fontweight='bold')
                            ax_bub.set_ylabel(f"Average {y_attr}", fontsize=11, fontweight='bold')
                            fig_bub.colorbar(scatter, ax=ax_bub, label="Overall Liking Score")
                            
                            import seaborn as sns
                            sns.despine(ax=ax_bub)
                            ax_bub.grid(True, linestyle='--', alpha=0.4)
                            st.pyplot(fig_bub)
                            
                            st.markdown("**Product Legend:**")
                            legend_cols = st.columns(3)
                            for i, p in enumerate(products_list):
                                with legend_cols[i % 3]:
                                    st.markdown(f"**{chr(65 + i)}:** {p}")
                            
                        if show_triple:
                            st.markdown("#### Ranked Attribute Breakdown")
                            st.markdown("Products are sorted top-to-bottom by Overall Liking. See exactly which descriptive attributes dragged down the losers or propelled the winners.")
                            
                            # Limit to top 3 attributes if there are many, plus overall liking
                            display_attrs = attr_cols[:3]
                            
                            bar_df = prod_means[display_attrs].copy()
                            bar_df['Overall Liking'] = overall_means.reindex(products_list)
                            
                            fig_trip, ax_trip = plt.subplots(figsize=(10, len(products_list) * 0.8 + 1))
                            
                            # Reverse order so winner is at top
                            bar_df = bar_df.iloc[::-1]
                            
                            bar_df.plot(kind='barh', ax=ax_trip, width=0.8, alpha=0.9, edgecolor='black')
                            ax_trip.set_xlabel("Average Score", fontsize=11)
                            ax_trip.set_ylabel("")
                            ax_trip.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            
                            sns.despine(ax=ax_trip)
                            ax_trip.grid(axis='x', linestyle='--', alpha=0.4)
                            st.pyplot(fig_trip)

                    # ==========================================
                    # 4. BAR CHART WITH ERROR BARS
                    # ==========================================
                    st.divider()
                    st.subheader("Attribute Comparison (Grouped Bar Chart)")
                    show_error = st.checkbox("Show error bars (Standard Error of the Mean)", value=True)
                    
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                    x = np.arange(len(attr_cols))
                    width = 0.8 / len(products_list)
                    offset = (len(products_list) - 1) / 2
                    
                    for i, prod in enumerate(products_list):
                        means = prod_means.loc[prod].values
                        errs = prod_sems.loc[prod].values if show_error else None
                        pos = x + (i - offset) * width
                        ax_bar.bar(pos, means, width, label=prod, yerr=errs, capsize=4, alpha=0.85, edgecolor='black')
                        
                    ax_bar.set_xticks(x)
                    ax_bar.set_xticklabels(attr_cols, fontweight='bold', fontsize=11)
                    ax_bar.set_ylabel("Score", fontweight='bold')
                    ax_bar.legend(title=prod_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
                    ax_bar.spines['top'].set_visible(False)
                    ax_bar.spines['right'].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig_bar)

                    # ==========================================
                    # 5. OVERLAPPING RADAR CHART
                    # ==========================================
                    st.divider()
                    st.subheader("Visual Profile (Radar Chart)")
                    
                    fig_radar, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
                    angles = np.linspace(0, 2 * np.pi, len(attr_cols), endpoint=False).tolist()
                    angles += angles[:1] 
                    
                    for prod in products_list:
                        values = prod_means.loc[prod].values.flatten().tolist()
                        values += values[:1]
                        ax_radar.plot(angles, values, linewidth=2.5, label=prod)
                        ax_radar.fill(angles, values, alpha=0.1)
                        
                    ax_radar.set_xticks(angles[:-1])
                    ax_radar.set_xticklabels(attr_cols, fontsize=11, fontweight='bold')
                    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                    st.pyplot(fig_radar)
                    
                    # ==========================================
                    # 6. PCA SENSORY MAP
                    # ==========================================
                    st.divider()
                    st.subheader("2D PCA Sensory Map")
                    st.markdown("This map uses Principal Component Analysis to squash your multidimensional attributes down to an X/Y grid. **Products located close together taste similar.** The red arrows act as gravity—pulling products in the direction of that specific attribute.")
                    
                    if len(products_list) >= 3:
                        scaler = StandardScaler()
                        scaled_means = scaler.fit_transform(prod_means)
                        
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(scaled_means)
                        
                        fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
                        
                        ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], s=150, alpha=0.8, color='#1f77b4', edgecolors='black')
                        for i, prod in enumerate(products_list):
                            letter = chr(65 + i)
                            ax_pca.annotate(letter, (pca_result[i, 0], pca_result[i, 1]), xytext=(8, 5), textcoords='offset points', fontsize=12, fontweight='bold', bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", alpha=0.85))
                            
                        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                        for i, attr in enumerate(attr_cols):
                            ax_pca.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='#d62728', alpha=0.6, width=0.015, head_width=0.08)
                            ax_pca.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, attr, color='#d62728', fontsize=12, fontweight='bold')
                            
                        ax_pca.axhline(0, color='black', linestyle='--', alpha=0.3)
                        ax_pca.axvline(0, color='black', linestyle='--', alpha=0.3)
                        ax_pca.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% of variance)")
                        ax_pca.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% of variance)")
                        ax_pca.grid(alpha=0.2)
                        
                        st.pyplot(fig_pca)
                        
                        st.markdown("**Product Legend:**")
                        legend_cols_pca = st.columns(3)
                        for i, p in enumerate(products_list):
                            with legend_cols_pca[i % 3]:
                                st.markdown(f"**{chr(65 + i)}:** {p}")
                    else:
                        st.warning("You need at least 3 distinct products to generate a mathematical PCA map.")
        else:
            st.info("Please select at least 3 descriptive attributes to map.")

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
