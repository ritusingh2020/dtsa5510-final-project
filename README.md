# Global Life Expectancy Analysis using Unsupervised Learning

## Project Overview

This project explores the WHO Global Health Observatory dataset (spanning 2000-2015) to understand the factors influencing life expectancy across different countries. Using unsupervised learning techniques, specifically clustering, the goal is to identify hidden patterns and groupings among nations based on various health and economic indicators, thereby shedding light on the key drivers of life expectancy.

## Dataset

* **Source:** WHO Global Health Observatory Data, obtained via Kaggle: [Life Expectancy (WHO)](https://www.kaggle.com/code/ahmedabbas757/life-expectancy-prediction)
* **Content:** The dataset includes country-level data from 2000 to 2015, containing 22 attributes such as Life Expectancy, Adult Mortality, Infant Deaths, Alcohol Consumption, Healthcare Expenditure, Immunization Rates (Hepatitis B, Polio, Diphtheria), BMI, HIV/AIDS rates, GDP, Population, Schooling, and Income Composition of Resources.

## Methodology

1.  **Data Loading & Cleaning:**
    * Loaded the dataset using `pandas`.
    * Corrected column names with leading/trailing spaces.
    * Handled missing values:
        * Initial imputation using the mean value for each specific country.
        * Rows with remaining missing values after country-level imputation were dropped.
2.  **Exploratory Data Analysis (EDA):**
    * Examined data distributions using histograms.
    * Analyzed feature correlations using a heatmap.
    * Visualized trends in average life expectancy over the years.
    * Investigated relationships between life expectancy and key features (e.g., HIV/AIDS, Schooling, Income Composition) using scatter plots.
    * Checked for skewness and outliers using histograms and box plots.
3.  **Feature Preprocessing:**
    * Scaled numerical features using `sklearn.preprocessing.StandardScaler`.
4.  **Dimensionality Reduction:**
    * Applied Principal Component Analysis (PCA) to reduce the dimensionality of the scaled features to 2 components for visualization and clustering.
    * Analyzed feature importance within the principal components.
5.  **Clustering:**
    * Implemented K-Means clustering on the PCA-transformed data.
    * Used the Elbow Method (WCSS) and Silhouette Score analysis to determine the optimal number of clusters (identified as 4 in the notebook).
6.  **Cluster Analysis & Profiling:**
    * Visualized the resulting clusters on the 2D PCA plot.
    * Calculated the mean values of key features for each cluster.
    * Used box plots to compare the distribution of important variables (Life Expectancy, Income, Schooling, GDP) across the identified clusters.


## Requirements

Key Python libraries used:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn` (StandardScaler, PCA, KMeans, silhouette_score, SimpleImputer, etc.)

## Usage

1.  Ensure you have the required libraries installed (`pip install pandas numpy matplotlib seaborn scikit-learn`).
2.  Download the dataset (`life_expectancy_data.csv`) from the Kaggle link provided above and place it in the appropriate path (e.g., `/content/drive/MyDrive/5510/` as used in the notebook, or update the path in the code).
3.  Run the Jupyter Notebook  cell by cell to replicate the analysis and view the results.

