# Description of Datasets Used in This Research

## Comparison Data
- Provides the maternal mortality rate estimates (and associated 95% confidence intervals) from the existing BMat, CODEm, and GMatH models that served as benchmarks for the predictions of my models.
  - ihme provides the estimates from the CODEm model.
  - un_mmeig_data contains the estimates from the BMat model.
  - mortality_Mean, mortality_LB, and mortality_UB describes the estimates from the GMatH model (here, the lower and upper bounds on the 95% confidence intervals for these estimates were given in separate files).

## split_income_data
- Contains all the train, validation, and test datasets used to develop the base estimators trained for *country-level prediction*.
- These datasets contain *all features*

## split_year_data
- Contains all the train, validation, and test datasets used to develop the base estimators trained to perform *forecasting*.
- These datasets contain *all features*

## fs_fromlit_data
- Contains all the train, validation, and test datasets used to develop the base estimators trained to perform both *country-level prediction* and *forecasting*.
- These datasets contain *features that the literature describes as having a meaningful relationship with MMR*

## fs_corr_income_data
- Contains all the train, validation, and test datasets used to develop the base estimators trained to perform *country-level prediction*.
- These datasets contain *feature subsets containing variables with a pairwise correlation coefficient with MMR greater than or equal to 0.6, 0.7, or 0.8*

## fs_corr_year_data
- Contains all the train, validation, and test datasets used to develop the base estimators trained to perform *forecasting*.
- These datasets contain *feature subsets containing variables with a pairwise correlation coefficient with MMR greater than or equal to 0.6, 0.7, or 0.8*

## data_with_yearcountry
  - Cntains all the train, validation, and test sets used to develop my base estimators.
  - The difference between the data in this folder and the data in the other folders is that the datasets contained here contain the country and year identifier columns.

## sensitivity_analysis_data
- Contains the train, validation, and test sets filtered for income-specific data used to train the sensitivity models.
