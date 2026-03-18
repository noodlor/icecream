import streamlit as st
import math
import random
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Forces web-safe chart rendering, silencing the context warning
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from scipy.stats import binom, norm
import subprocess

# Try to import advanced stats libraries, but don't crash if missing
try:
    from pingouin import multivariate_normality
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import scipy.stats as stats
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ==========================================
# HELPER FUNCTION: LOAD DATA
# ==========================================
def load_data(uploaded_file, gsheet_url):
    """Loads a DataFrame from either an uploaded CSV or a public Google Sheet URL."""
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

# ==========================================
# SESSION STATE MEMORY
# ==========================================
if "active_tool" not in st.session_state:
    st.session_state.active_tool = "Panel Size Optimizer"
if "transfer_brands" not in st.session_state:
    st.session_state.transfer_brands = 5
if "transfer_tasters" not in st.session_state:
    st.session_state.transfer_tasters = 20

def go_to_designer(b, t):
    st.session_state.transfer_brands = int(b)
    st.session_state.transfer_tasters = int(t)
    st.session_state.active_tool = "Experimental Block Designer"

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Sensory Science Suite", layout="wide")

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
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Sensory Lab Suite")
st.sidebar.markdown("Select an analysis tool:")

st.sidebar.radio("Navigation", [
    "Documentation",
    "Panel Size Optimizer", 
    "Experimental Block Designer",
    "Discrimination Test", 
    "Correlation Matrix",
    "Hedonic Analyzer"
], label_visibility="collapsed", key="active_tool")

tool = st.session_state.active_tool

# ==========================================
# TOOL 0: DOCUMENTATION READER
# ==========================================
if tool == "Documentation":
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_text = f.read()
        st.markdown(readme_text)
    else:
        st.title("Documentation")
        st.warning("Could not find the `README.md` file. Ensure it is saved in the exact same directory as this script.")

# ==========================================
# TOOL 1: PANEL SIZE OPTIMIZER
# ==========================================
elif tool == "Panel Size Optimizer":
    st.title("Panel Size Optimizer")
    st.markdown("Calculate the required panel size or evaluate the statistical power of a fixed panel.")
    
    calc_mode = st.radio(
        "Calculation Mode", 
        ["Calculate required panel size", "Calculate statistical power (fixed panel size)"], 
        horizontal=True
    )

    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        brands = st.number_input("Number of Samples", min_value=2, value=4, step=1)
    with col2:
        delta = st.number_input("Target Detectable Difference", min_value=0.1, value=1.0, step=0.1)
    with col3:
        stdev = st.number_input("Estimated Standard Deviation", min_value=0.1, value=1.3, step=0.1)
        
    alpha = 0.05
    z_alpha = norm.ppf(1 - alpha / 2)

    if calc_mode == "Calculate required panel size":
        target_power = st.slider("Target statistical power", min_value=0.50, max_value=0.99, value=0.80, step=0.01)
        unbalanced = st.checkbox("Allow unbalanced design (skip serving multiplier)", value=True)

        z_beta = norm.ppf(target_power)
        raw_n = 2 * ((z_alpha + z_beta) ** 2) * ((stdev / delta) ** 2)
        min_tasters = math.ceil(raw_n)
        
        remainder = min_tasters % brands
        if unbalanced or remainder == 0:
            optimal_tasters = min_tasters
        else:
            optimal_tasters = min_tasters + (brands - remainder)

        st.divider()
        st.subheader("Results")
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Minimum Tasters", min_tasters)
        res_col2.metric("Recommended Panel Size", optimal_tasters)
        
        with st.expander("Interpretation notes"):
            st.markdown("""
            * **Minimum tasters:** The calculated mathematical threshold required to achieve your target statistical power. A panel smaller than this risks a Type II error.
            * **Recommended size:** To account for first-sample bias, every product should be served first, second, and third an equal number of times. This value rounds the minimum up to the nearest multiple of your sample count to allow for a balanced design.
            """)
        
        if optimal_tasters == min_tasters and not unbalanced:
            st.success(f"Balanced design: {min_tasters} is a multiple of {brands}.")
        elif unbalanced:
            st.warning("Unbalanced design selected. Serving orders should be randomized manually to minimize first-sample bias.")

        st.divider()
        st.button(
            "Send to Block Designer", 
            type="primary", 
            width='stretch',
            on_click=go_to_designer, 
            args=(brands, optimal_tasters)
        )

    else:
        fixed_tasters = st.number_input("Available Tasters (fixed panel size)", min_value=2, value=20, step=1)
        
        variance_factor = 2 * ((stdev / delta) ** 2)
        if variance_factor > 0:
            z_beta_calc = math.sqrt(fixed_tasters / variance_factor) - z_alpha
            calculated_power = norm.cdf(z_beta_calc)
        else:
            calculated_power = 0.0

        st.divider()
        st.subheader("Results")
        st.metric("Estimated Statistical Power", f"{calculated_power * 100:.1f}%")
        
        with st.expander("Interpretation notes"):
            st.markdown("""
            * **Statistical power:** The probability of correctly detecting your target difference if it truly exists.
            * A standard benchmark in sensory science is 80%. If your calculated power is significantly lower, the test carries a high risk of a Type II error (failing to detect a real difference due to insufficient sample size).
            """)
            
        st.divider()
        st.button(
            "Send to Block Designer", 
            type="primary", 
            width='stretch',
            on_click=go_to_designer, 
            args=(brands, fixed_tasters)
        )

# ==========================================
# TOOL 2: EXPERIMENTAL BLOCK DESIGNER (R-POWERED)
# ==========================================
elif tool == "Experimental Block Designer":
    st.title("Experimental Block Designer")
    st.markdown("Generate a D-optimal serving schedule utilizing R's `AlgDesign` statistical package.")
    
    default_cups = min(4, st.session_state.transfer_brands - 1)
    if default_cups < 1: 
        default_cups = 1

    col1, col2, col3 = st.columns(3)
    with col1:
        num_brands = st.number_input("Total Samples Tested", min_value=2, max_value=26, value=st.session_state.transfer_brands, step=1)
    with col2:
        num_tasters = st.number_input("Total Tasters", min_value=1, value=st.session_state.transfer_tasters, step=1)
    with col3:
        cups_per_taster = st.number_input("Samples Evaluated Per Taster", min_value=1, max_value=int(num_brands), value=default_cups, step=1)

    if st.button("Generate D-Optimal Design", type="primary", width='stretch'):
        if cups_per_taster > num_brands:
            st.error("A taster cannot evaluate more samples than the total number of samples available.")
        else:
            with st.spinner('Engaging R backend and calculating D-optimal matrix...'):
                
                # 1. Write the R script dynamically
                r_script = f"""
                options(warn=-1)
                # Auto-install the package if it's missing
                if (!require("AlgDesign", quietly = TRUE)) {{
                  install.packages("AlgDesign", repos="https://cloud.r-project.org", quiet=TRUE)
                  library(AlgDesign)
                }}

                V <- {int(num_brands)}
                B <- {int(num_tasters)}
                K <- {int(cups_per_taster)}
                N <- B * K

                # Step 1: Create a perfectly balanced pool of items mathematically
                pool <- rep(1:V, length.out = N)
                within_data <- data.frame(Trt = as.factor(pool))

                # Step 2: Partition that perfect pool into optimized blocks
                blocksizes <- rep(K, B)
                b_opt <- optBlock(~., withinData=within_data, blocksizes=blocksizes, nRepeats=100)

                # Extract the matrix and export to CSV
                out_matrix <- matrix(nrow=B, ncol=K)
                for(i in 1:B) {{
                  out_matrix[i, ] <- as.numeric(as.character(b_opt$Blocks[[i]]$Trt))
                }}
                write.csv(out_matrix, "temp_design.csv", row.names=FALSE)
                """
                
                try:
                    with open("generate_design.R", "w") as f:
                        f.write(r_script)

                    # 2. Execute the R script via terminal subprocess
                    subprocess.run(["Rscript", "generate_design.R"], capture_output=True, text=True, check=True)

                    # 3. Read the perfect matrix back into Python
                    df_r = pd.read_csv("temp_design.csv")
                    
                    # 4. Analyze the exact counts to prove the balance
                    expected_count = (num_tasters * cups_per_taster) / num_brands
                    expected_pairs = (expected_count * (cups_per_taster - 1)) / (num_brands - 1) if num_brands > 1 else 0
                    
                    counts = [0] * num_brands
                    pairs = [[0] * num_brands for _ in range(num_brands)]

                    for idx, row in df_r.iterrows():
                        block = [int(x) - 1 for x in row.values] # Convert 1-based R output to 0-based Python
                        for i in range(len(block)):
                            counts[block[i]] += 1
                            for j in range(i + 1, len(block)):
                                pairs[block[i]][block[j]] += 1
                                pairs[block[j]][block[i]] += 1

                    actual_min_count = min(counts)
                    actual_max_count = max(counts)
                    
                    actual_pair_counts = []
                    for i in range(num_brands):
                        for j in range(i + 1, num_brands):
                            actual_pair_counts.append(pairs[i][j])
                    
                    actual_min_pairs = min(actual_pair_counts) if actual_pair_counts else 0
                    actual_max_pairs = max(actual_pair_counts) if actual_pair_counts else 0

                    count_text = f"exactly **{actual_min_count}**" if actual_min_count == actual_max_count else f"between **{actual_min_count}** and **{actual_max_count}**"
                    pair_text = f"exactly **{actual_min_pairs}**" if actual_min_pairs == actual_max_pairs else f"between **{actual_min_pairs}** and **{actual_max_pairs}**"

                    # 5. Format for the UI
                    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    table_data = []
                    for i, row in df_r.iterrows():
                        block_row = {"Taster": str(i + 1)}
                        for j, val in enumerate(row.values):
                            block_row[f"Sample {j+1}"] = letters[int(val) - 1]
                        table_data.append(block_row)

                    final_df = pd.DataFrame(table_data)

                    st.divider()
                    st.subheader("Optimized Serving Schedule")
                    
                    st.markdown(f"""
                    **Design Statistics (D-Optimal):**
                    * **Target appearances:** Each sample is served {count_text} times across the entire panel (Theoretical target: {expected_count:.2f}).
                    * **Pairwise balance:** Every sample is evaluated alongside every other sample {pair_text} times (Theoretical target: {expected_pairs:.2f}).
                    """)
                    
                    st.dataframe(final_df, hide_index=True, width='stretch')
                    
                    with st.expander("Interpretation notes"):
                        st.markdown("""
                        * **Mathematical limits:** If your theoretical targets are fractions (e.g., 1.33), a spread of 1 (e.g., between 1 and 2 times) is the absolute best balance physically possible. 
                        * **Taster row:** Each row corresponds to a single participant. Read left-to-right to determine the serving order.
                        """)
                    
                    st.divider()
                    st.subheader("Export to Spreadsheets")
                    st.markdown("Hover over the block below and click the **Copy** icon in the top right. Paste directly into cell A1 of a blank spreadsheet.")
                    st.code(final_df.to_csv(sep='\t', index=False), language='plaintext')

                except FileNotFoundError:
                    st.error("R is not installed or not found in your system PATH. Please install R to use this tool.")
                except subprocess.CalledProcessError as e:
                    st.error("An error occurred while executing the R script. Please check your terminal output.")
                    st.code(e.stderr)
                finally:
                    # Robust cleanup: guarantees temporary files are deleted even if errors occur
                    if os.path.exists("generate_design.R"):
                        os.remove("generate_design.R")
                    if os.path.exists("temp_design.csv"):
                        os.remove("temp_design.csv")

# ==========================================
# TOOL 3: DISCRIMINATION TEST
# ==========================================
elif tool == "Discrimination Test":
    st.title("Discrimination Test Analyzer")
    st.markdown("Calculate statistical significance for difference testing.")
    
    method = st.radio("Test Method", ["Triangle", "Tetrad", "Duo-Trio"], horizontal=True)
    
    col1, col2 = st.columns(2)
    with col1:
        tasters = st.number_input("Total Tasters", min_value=1, value=30, step=1)
    with col2:
        correct = st.number_input("Correct Guesses (leave 0 for planning mode)", min_value=0, max_value=tasters, value=0, step=1)
        
    alpha = st.slider("Significance Level (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

    p_guess = 0.5 if method == "Duo-Trio" else (1/3)
    min_correct = int(binom.ppf(1 - alpha, tasters, p_guess)) + 1

    st.divider()
    st.subheader("Results")
    st.metric("Target Correct Guesses Required", min_correct)

    if correct > 0:
        p_value = binom.sf(correct - 1, tasters, p_guess)
        st.markdown("### Post-Test Analysis")
        st.write(f"**Calculated p-value:** {p_value:.5f}")
        
        with st.expander("Interpretation notes"):
            st.markdown(f"""
            * The alpha level is set to **{alpha}**. This represents the acceptable probability of a type I error (false positive).
            * If the calculated p-value is **lower** than {alpha}, the results are statistically significant, indicating the panel detected a difference.
            * If the calculated p-value is **higher** than {alpha}, the results are not statistically significant.
            """)
        
        if p_value < alpha:
            st.success("Statistically significant: The panel detected a difference.")
        else:
            st.error("Not significant: No reliable difference was detected by the panel.")

# ==========================================
# TOOL 4: CORRELATION MATRIX
# ==========================================
elif tool == "Correlation Matrix":
    st.title("Correlation Matrix & Heatmap")
    st.markdown("Analyze relationships between variables using a CSV or public Google Sheet.")

    col_method, col_cmap = st.columns(2)
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
            
            # Build the custom Purple-Yellow diverging colormap
            if "PurpleYellows" in cmap_choice_raw:
                # Uses dark purple -> white -> bright yellow
                cmap_choice = mcolors.LinearSegmentedColormap.from_list(
                    "PurpleYellows", ["#51247A", "#FFFFFF", "#E8E29D"]
                )
            else:
                # Extract standard Matplotlib string code
                cmap_choice = cmap_choice_raw.split(" ")[0]

    st.divider()
    
    col_upload, col_url = st.columns(2)
    with col_upload:
        uploaded_file = st.file_uploader("Upload Data (CSV)", type=["csv"])
    with col_url:
        gsheet_url = st.text_input("OR Paste Public Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")
    
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
                        norm = multivariate_normality(df_numeric, alpha=.05)
                        corr_method = "pearson" if norm.normal else "spearman"
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
            df_corr = corr.stack().reset_index()
            df_corr.columns = ['FEATURE_1', 'FEATURE_2', 'CORRELATION']
            
            mask_dups = (df_corr[['FEATURE_1', 'FEATURE_2']].apply(frozenset, axis=1).duplicated()) | (df_corr['FEATURE_1'] == df_corr['FEATURE_2'])
            df_corr = df_corr[~mask_dups]
            
            sorted_df = df_corr.sort_values(by="CORRELATION", key=abs, ascending=False).reset_index(drop=True)
            st.dataframe(sorted_df, hide_index=True, width='stretch')
            
            with st.expander("Interpretation notes"):
                st.markdown("""
                * **Values approaching 1.0:** Indicate a strong positive correlation (as one variable increases, the other increases).
                * **Values approaching -1.0:** Indicate a strong negative correlation (as one variable increases, the other decreases).
                * **Values approaching 0:** Indicate no apparent linear relationship.
                """)
            
            st.divider()
            st.subheader("Export to Spreadsheets")
            st.code(sorted_df.to_csv(sep='\t', index=False), language='plaintext')

# ==========================================
# TOOL 5: HEDONIC ANALYZER
# ==========================================
elif tool == "Hedonic Analyzer":
    st.title("Hedonic Analyzer (ANOVA & Tukey's HSD)")
    st.markdown("Analyze hedonic scale data to identify statistically significant differences.")
    
    if not STATSMODELS_AVAILABLE:
        st.error("Missing library. Please run `pip install statsmodels` to use this tool.")
        st.stop()

    st.info("Formatting requirement: The dataset should structure samples as columns and tasters as rows (e.g., Column 1: 'Taster', Column 2: 'Sample A', Column 3: 'Sample B').")
    
    apply_zscore = st.checkbox("Standardize data (convert raw scores to z-scores per taster to correct for scale heterogeneity)")

    st.divider()
    
    col_upload, col_url = st.columns(2)
    with col_upload:
        uploaded_file = st.file_uploader("Upload Tasting Scores (CSV)", type=["csv"])
    with col_url:
        gsheet_url = st.text_input("OR Paste Public Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")
    
    df = load_data(uploaded_file, gsheet_url)
    
    if df is not None:
        if 'Taster' in df.columns:
            df = df.drop(columns=['Taster'])
        elif 'taster' in df.columns:
            df = df.drop(columns=['taster'])
            
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty or len(df_numeric.columns) < 2:
            st.error("Error: Could not find at least two numeric columns to compare.")
        else:
            # Apply Z-score transformation if selected
            if apply_zscore:
                df_numeric = df_numeric.apply(lambda row: (row - row.mean()) / row.std(ddof=0) if row.std(ddof=0) > 0 else row - row.mean(), axis=1)

            brands = list(df_numeric.columns)
            df_long = df_numeric.melt(var_name='Brand', value_name='Score').dropna()
            
            st.divider()
            st.subheader("1. Summary Statistics")
            summary_df = df_long.groupby('Brand')['Score'].agg(['mean', 'std', 'count']).reset_index()
            summary_df = summary_df.sort_values(by='mean', ascending=False).round(3)
            summary_df.columns = ['Sample', 'Average Score', 'Standard Deviation', 'Tasters']
            st.dataframe(summary_df, hide_index=True, width='stretch')

            st.divider()
            st.subheader("2. One-way ANOVA Test")
            
            arrays = [df_numeric[col].dropna() for col in brands]
            f_stat, p_val = stats.f_oneway(*arrays)
            
            # Calculate Effect Size (Eta-squared)
            overall_mean = df_long['Score'].mean()
            ss_total = ((df_long['Score'] - overall_mean)**2).sum()
            ss_between = 0
            for brand in brands:
                brand_data = df_long[df_long['Brand'] == brand]['Score']
                ss_between += len(brand_data) * (brand_data.mean() - overall_mean)**2
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write(f"**ANOVA p-value:** {p_val:.5f}")
            with col_res2:
                st.write(f"**Effect size (eta-squared):** {eta_squared:.3f}")
            
            with st.expander("Interpretation notes"):
                st.markdown("""
                * **ANOVA p-value:** Evaluates whether there are any statistically significant differences between the means of the independent groups. If the p-value is greater than the alpha level, it implies no overall significant differences exist among the samples. 
                * **Effect size (eta-squared):** Measures the proportion of variance in the scores associated with the samples rather than random error. General guidelines: ~0.01 is a small effect, ~0.06 is a medium effect, and >0.14 is a large effect.
                """)
            
            if p_val < 0.05:
                st.success("Statistically significant: At least one sample differs significantly. Proceeding to Tukey's HSD test.")
                
                st.divider()
                st.subheader("3. Tukey's HSD Test")
                
                tukey = pairwise_tukeyhsd(endog=df_long['Score'], groups=df_long['Brand'], alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                st.dataframe(tukey_df, hide_index=True, width='stretch')
                
                with st.expander("Interpretation of Tukey's HSD columns"):
                    st.markdown("""
                    * **group1 & group2:** The specific pair of samples being compared.
                    * **meandiff:** The difference between the mean scores of the two groups.
                    * **p-adj:** The adjusted p-value. A value `< 0.05` indicates statistical significance.
                    * **lower & upper:** The 95% confidence interval for the mean difference.
                    * **reject:** Indicates whether to reject the null hypothesis. **True** indicates a statistically significant difference between the two samples.
                    """)
                
            else:
                st.error("Not significant: No statistically significant difference was found among the samples.")

            st.divider()
            st.subheader("4. Data Distribution Plot")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Brand', y='Score', data=df_long, color='white', width=0.5, ax=ax)
            sns.swarmplot(x='Brand', y='Score', data=df_long, size=8, alpha=0.7, ax=ax, palette="husl")
            
            # Adjust y-axis based on whether z-score normalization was applied
            if apply_zscore:
                y_min = math.floor(df_long['Score'].min()) - 0.5
                y_max = math.ceil(df_long['Score'].max()) + 0.5
                ax.set_ylim(y_min, y_max) 
                ax.set_ylabel("Standardized Score (z-score)", fontsize=12)
            else:
                ax.set_ylim(0.5, 9.5) 
                ax.set_ylabel("Score (1-9)", fontsize=12)
                
            ax.set_title("Hedonic Score Distribution", fontsize=14, pad=15)
            ax.set_xlabel("Sample", fontsize=12)
            
            st.pyplot(fig)
