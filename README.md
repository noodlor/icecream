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


# Sensory Science Suite

The Sensory Science Suite is a collection of computational tools designed to help you plan, execute, and analyze product taste tests. It handles the complex mathematics of experimental design so you can focus on the food.

This suite is particularly useful for running "Incomplete Block Designs"—a tasting format where you test many products, but only feed a few products to each person to prevent palate fatigue. 

## Installation and Setup

### 1. Requirements
To run this application, you need Python and R installed on your computer or server.
* **Python Libraries:** You will need `streamlit`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `scipy`, `pingouin`, and `statsmodels`.
* **R Language:** You need the core R language installed on your system. The app will automatically attempt to install the required `AlgDesign` package for you the first time you generate a serving schedule.

### 2. Running Locally
If you are running the app on your own computer, open your terminal or command prompt, navigate to the folder containing these files, and run:
`streamlit run app.py`

### 3. Deploying to Streamlit Community Cloud
If you want to host this dashboard online, you need to provide the cloud server with two configuration files alongside your `app.py` script:
* **`requirements.txt`**: This tells the server which Python packages to install. List the libraries mentioned above, each on a new line.
* **`packages.txt`**: This tells the server to install the R language. This file should contain exactly one line that reads: `r-base`.

---

## Overview of the Tools

### 1. Panel Size Optimizer
This tool helps you plan your experiment before you begin pouring cups. 
You can use it to determine the minimum number of tasters required to achieve a reliable result, or you can input the number of tasters you already have to see how small of a flavor difference they will mathematically be able to detect.

### 2. Experimental Block Designer
This tool assigns serving schedules to your tasters. 
When running an experiment where you have more samples than a single human can comfortably eat (e.g., 10 ice creams, but a taster can only eat 4), you cannot assign the cups randomly. Doing so creates imbalances in the data. 

This tool uses a statistical engine to calculate a "D-optimal" schedule. It ensures that every product is tested an equal number of times, and that every product is compared against every other product fairly. The result is a clean grid showing exactly which cups to put on which taster's tray.

### 3. Discrimination Test Analyzer
Use this tool for simple difference tests like Triangle Tests, Tetrads, or Duo-Trios. You input the total number of people who took the test and how many of them guessed the odd sample correctly. The tool calculates the p-value and tells you if the group detected a genuine difference, or if they were likely just guessing.

### 4. Correlation Matrix
This tool looks for relationships between different variables in your data. If you upload a spreadsheet of characteristics (e.g., sweetness, firmness, overall liking), the matrix will highlight which traits move together. It uses color coding to show positive correlations (as one goes up, the other goes up) and negative correlations (as one goes up, the other goes down).

### 5. Survey Decoder
When you export raw data from survey software (like Google Forms or SurveyMonkey), it is usually in a messy format where tasters' answers are spread across long rows of 3-digit blind codes. 

The Survey Decoder reorganizes that data. You tell it which columns hold the codes and the scores, and provide a master key translating the 3-digit codes back to the real product names. It will automatically reformat the raw data into a clean, analysis-ready table. 

### 6. Hedonic Analyzer
This is the final step in evaluating your taste test. You upload the clean table of scores (like the one generated by the Survey Decoder) to see which product won.

Because human tasters are naturally biased—some are harsh critics and others are very generous—this tool uses a General Linear Model (a Two-Way ANOVA) to separate the true quality of the product from the personal bias of the taster. 

It provides you with "Adjusted Scores," which represent how the product would have performed if all tasters graded on the exact same scale. It also provides a pairwise comparison table to show exactly which products beat which with statistical significance.

# Sensory Science Suite - User Guide & Statistical Glossary

This suite provides the mathematical framework for evaluating product taste tests. It automates experimental design, data formatting, and statistical analysis, allowing you to confidently base decisions on human sensory data.

## I. Definitions

* **Product:** The actual item or recipe being evaluated (e.g., Recipe A, Prototype 2).
* **Serving:** The physical portion of the product placed in front of a taster.
* **Taster:** A human participant evaluating the servings.

## II. Statistical Glossary

* **ANOVA (Analysis of Variance):** A statistical test used to determine if there are mathematically real differences between the average scores of three or more products.
* **P-value:** A probability metric representing the likelihood that your test results occurred by random chance. A lower p-value indicates stronger evidence of a true difference. 
* **Statistically Significant:** In this application, a result is deemed statistically significant if the p-value is less than 0.05. This means there is less than a 5% probability that the observed differences are due to random background noise.
* **Variance & Standard Deviation:** Measures of how spread out your data is. If all tasters give a product a 5, variance is zero. If half give it a 1 and half give it a 9, variance is high.
* **Incomplete Block Design:** A test structure used when you have more products to test than a human palate can handle without fatigue. Each taster only evaluates a subset of the total available products.

---

## III. The Tools: How and Why

### 1. Panel Size Optimizer
**What it does:** Calculates the number of tasters required for an experiment based on the flavor difference you are trying to detect. 
**Why you need it:** If you use too few tasters, you risk a "false negative" (failing to detect a real difference). Use this planning tool to ensure your panel has sufficient statistical power before you begin. 

### 2. Experimental Block Designer
**What it does:** Generates a serving schedule that maps exactly which products should be given to which tasters.
**Why you need it:** When using an Incomplete Block Design, randomly assigning servings creates imbalances (e.g., Product A is served 15 times, but Product B is only served 8 times). This tool creates a mathematical schedule guaranteeing that every product appears an equal number of times and is compared fairly against all other products.

### 3. Discrimination Test Analyzer
**What it does:** Analyzes simple forced-choice tests (Triangle, Tetrad, Duo-Trio). You input the total tasters and correct guesses.
**Why you need it:** In a Triangle test, tasters have a 33% chance of guessing the correct answer just by picking randomly. This tool calculates if the number of correct guesses exceeds the threshold of random chance, proving the panel genuinely tasted a difference.

### 4. Correlation Matrix
**What it does:** Highlights relationships between different variables (e.g., firmness, sweetness, overall liking) within a dataset. 
**Why you need it:** Values near 1.0 indicate a strong positive relationship (as sweetness increases, liking increases). Values near -1.0 indicate a strong negative relationship. Values near 0 indicate no connection.

### 5. Survey Decoder
**What it does:** Converts raw, horizontal survey exports into a clean vertical matrix.
**Why you need it:** Survey platforms format data in long rows. The Hedonic Analyzer requires data structured with one column per product. This tool automates the process of mapping 3-digit blind codes back to product names, stacking the data, and pivoting the columns into an analysis-ready format.

---

### 6. Hedonic Analyzer
**What it does:** Evaluates incomplete or complete tasting data using a Two-Way ANOVA (General Linear Model).
**Why you need it:** Raw averages are highly vulnerable to taster bias. This engine isolates product performance from human subjectivity to reveal the true quality of the product.

#### The Mathematics: Z-Scores vs. Adjusted Means
It is vital to understand that the Analyzer applies two separate, distinct mathematical corrections to your data:

1. **Standardization (The Z-Score Toggle):** This step corrects **scale variance**. It levels the playing field between a "harsh" taster who grades everything between 1 and 4, and a "generous" taster who grades everything between 6 and 9. It perfectly centers each human's grading curve, then back-transforms the numbers so they are still readable on a 9-point scale.
2. **Incomplete Block Correction (The Adjusted Mean):** This step is *always* on. It corrects **schedule imbalance**. If Product A was randomly assigned to a group of naturally tough graders during the test, its raw average will look artificially low. The ANOVA identifies this bad luck and outputs an "Adjusted Mean" (or Least Squares Mean) that accurately reflects the product's true quality as if everyone had tasted it. 

#### Quality Tiers & Fisher's Protected LSD
When the ANOVA detects a statistically significant difference in the panel, the engine immediately performs a pairwise comparison head-to-head matrix. 

This engine uses **Fisher's Protected Least Significant Difference (LSD)**, which is widely considered the gold standard for sensory tests. The algorithm uses these pairwise tests to assign letter grades (A, B, C) to the products. 
* **Statistical Tie:** If Product A and Product B share a letter (e.g., "A" and "AB"), their numerical difference is within the margin of error, and they are statistically tied. 
* **Significant Difference:** If they do not share a letter (e.g., "A" vs "B"), there is a mathematical difference in quality between them.

#### Taster Severity Calibration
At the bottom of the analyzer is a Calibration Table showing exactly how harsh or generous your tasters were compared to the panel average. 
* *Note for Researchers:* To ensure complete transparency, this table is hard-coded to pull its calculations strictly from the **raw, un-standardized** dataset. Even if you apply the Z-score toggle to your final results, this specific table will always expose the panel's true, natural biases.
