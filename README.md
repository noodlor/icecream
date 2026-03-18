# Sensory Science Suite - User Guide

This suite operates as a command-line interface (CLI) for calculations and as an interactive Streamlit web dashboard for data analysis and experimental design.

---

## Glossary of Statistical Concepts

Understanding the underlying mathematics ensures accurate interpretation of the laboratory results.

### General Terminology
* **Alpha ($\alpha$):** The significance level, typically set to 0.05. It represents the probability of a Type I error (rejecting the null hypothesis when it is actually true). In practical terms, an alpha of 0.05 means you accept a 5% risk of concluding a difference exists when there is no actual difference.
* **p-value:** The probability of obtaining test results at least as extreme as the results actually observed, assuming the null hypothesis is true. If the p-value is less than your alpha ($p < 0.05$), the results are statistically significant.
* **Statistical Power:** The probability that a test will correctly reject a false null hypothesis (avoiding a Type II error). The panel size optimizer aims for 80% power, meaning there is an 80% chance of detecting a true difference if one exists.

### Correlation Methods
* **Pearson Correlation:** A parametric test that measures the linear relationship between two continuous variables. It assumes the data is normally distributed and is highly sensitive to outliers. 
* **Spearman Correlation:** A non-parametric test that measures the monotonic relationship between variables using their rank order rather than raw values. It does not assume normal distribution and is more robust against outliers. 
* **Pingouin Normality Test:** The suite uses the multivariate Shapiro-Wilk test to assess if the data follows a normal distribution. If the data passes, the suite auto-selects Pearson; if it fails (common in small sample sizes), it defaults to Spearman.

### Analysis of Variance (ANOVA) & Post-Hoc Testing
* **One-Way ANOVA:** A test used to determine whether there are any statistically significant differences between the means of three or more independent groups. It acts as an initial gatekeeper; it indicates that *at least one* sample is different, but does not specify which one.
* **Effect Size ($\eta^2$):** Eta-squared measures the proportion of total variance in the data that is associated with the sample differences rather than random noise. A large p-value dictates significance, but $\eta^2$ dictates magnitude. (Guidelines: 0.01 = small effect, 0.06 = medium effect, 0.14+ = large effect).
* **Tukey's HSD (Honestly Significant Difference):** A post-hoc test conducted only after a significant ANOVA result. It performs pairwise comparisons between all samples. Crucially, it controls for the "family-wise error rate"—the mathematical reality that running multiple isolated t-tests increases the likelihood of false positives.
* **Z-Score Normalization:** A mathematical process that converts raw scores into standard deviations from the mean. In sensory science, this corrects for scale heterogeneity (the tendency for some tasters to use the entire 1-9 scale while others only use the middle numbers 4-6). It centers every participant's personal mean at 0, allowing the algorithm to analyze relative preference without being skewed by extreme voters.

---

## The Tools & Interpreting Results

### 1. Panel Size Optimizer
Calculates the number of tasters required to achieve a specific statistical power, or evaluates the resulting power of a fixed panel size.
* **Input:** Number of samples, target point difference, estimated panel standard deviation, and either target statistical power or available tasters.
* **Output:** The minimum number of tasters required, or the estimated statistical power percentage.
* **How to Read the Results:** * **When calculating panel size:** The tool provides a raw minimum and a recommended size rounded to ensure you can evenly distribute serving orders to minimize first-sample bias.
  * **When calculating statistical power:** A standard benchmark is 80%. If your fixed panel yields a lower percentage, your test has an increased risk of failing to detect a real difference between the samples.

### 2. Experimental Block Designer
Generates an optimized, randomized serving schedule (incomplete block design).
* **Input:** Total samples, total tasters, and samples evaluated per taster.
* **Output:** A TSV serving schedule optimized over 150,000 iterations to balance carryover effects.
* **How to Read the Results:** Each row represents one taster. Read left to right to determine the exact serving order for that specific participant.

### 3. Discrimination Test Analyzer
Calculates statistical significance for difference tests (Triangle, Tetrad, Duo-Trio).
* **Input:** Test method, total tasters, and correct guesses.
* **Output:** A calculated p-value based on the binomial distribution.
* **How to Read the Results:** If your p-value is lower than your alpha level (typically 0.05), the result is statistically significant. This indicates the panel detected a reliable difference between the samples.

### 4. Correlation Matrix
Calculates correlations between variables in your dataset.
* **Input:** A CSV of numerical data or a public Google Sheet link.
* **Output:** A correlation heatmap and a table of the strongest positive and negative correlations.
* **How to Read the Results:** * **Scores close to 1.0:** Strong positive correlation (as variable X increases, Y increases).
  * **Scores close to -1.0:** Strong negative correlation (as variable X increases, Y decreases).
  * **Scores close to 0:** No apparent linear relationship.

### 5. Hedonic Analyzer
Analyzes 1-to-9 point hedonic scale data to identify statistically significant preferences.
* **Input:** A CSV or public Google Sheet link where columns represent samples and rows represent tasters.
* **Output:** A one-way ANOVA test, a Tukey's HSD matrix, and a box/swarm plot showing data distribution.
* **How to Read the Tukey's HSD Table:** * `group1` & `group2`: The two samples being compared.
  * `meandiff`: The difference between their average scores.
  * `p-adj`: The adjusted p-value. If this is < 0.05, the difference is statistically significant.
  * `lower` & `upper`: The 95% confidence interval.
  * `reject`: Indicates whether to reject the null hypothesis. **True** indicates a statistically significant difference between the two samples.
