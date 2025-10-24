# Overall Structure:
This GitHub repository contains all the code and data used in my honours project, which investigated 'Using Machine Learning Methods to Estimate Maternal Mortality'.

The primary aims of my research were to:
1. To use interpretable machine learning methods to estimate countries’ maternal mortality ratios and assist in global MMR monitoring.
2. To identify important socio-economic and health-related features to inform targeted policies that will most reduce MMR.

The main contributions of my research were that:
- **I developed decision-tree based Random Forest, XGBoost, and LightGBM models that can effectively deal with sparse data to estimate and forecast MMR.** I found that the specific training data used to fit the models had a greater impact on their predictive accuracy than the choice of model type, feature selection strategy, or proportion of missing data in the model’s input dataset.
- **I used stacking and voting ensemble methods to combine predictions from 300 Random Forest, LightGBM, and XGBoost models fit on different training datasets to further improve predictive accuracy.** The best-performing ensemble leveraged patterns learned by the component models from the various training datasets. I found that the Random Forest Stacking Ensemble had the highest overall predictive performance, with higher performance gains observed when the performance of the models being combined was less uniform.
- **I examined the performance of the best-performing ensemble when it was trained on data describing countries from all income levels versus a specific income level.** Estimates of past MMR values were more accurate when informed by trends across all income levels while MMR forecasts were more accurate when based on income-specific data. Generally, the lowest mean-squared error was achieved when predicting the MMR of higher-income countries.
- **I benchmarked my models’ MMR predictions against estimates from existing maternal mortality models in the literature.** While my predictions were broadly similar to the literature’s estimates, they tended to predict lower MMR values due to methodological differences, variation in model variables, and possible underestimation of MMR in my ground truth data.
- I designed the Python code used to implement and evaluate these models. The code is freely available from this repository. Model training and evaluation was performed on the National Computational Infrastructure’s Gadi Supercomputer.
- **I determined the socio-economic and health-related features with the highest predictive power for MMR**, many of which were established risk factors. I used these results and existing causal research to suggest that investment in women’s education, incentives for skilled medical personnel to practice in rural areas, and increased provision of family planning services would reduce MMR by addressing important drivers of maternal mortality.
- **Using my models, I provided alternative MMR estimates for 172 countries between 1985 and 2018.** These estimates can be used to resolve existing disagreement about the true maternal mortality ratios and inform scientific debate about the relative merits of different MMR modelling approaches.
- **I showed that my models achieved comparable MMR predictive accuracy to existing models in the literature without a similarly heavy reliance on domain knowledge.** Therefore, my models have wider applicability in low resource countries where domain knowledge in this field is still developing and/or field experts are in limited supply.

## Project Monitoring:
- The weekly_reports folder contains all reports presented to my supervisors during our weekly meetings. They contain progress updates and document my research process.
- The Presentations folder the Powerpoint presented at the Australian Health Economics Society's Annual Conference in September and the Powerpoint presented in May to my supervisor's lab group as a wid-point progress update.

## Thesis:
- The Report folder contains the various drafts of my thesis, both with and without supervisor feedback.
- The 'Final_Thesis.pdf file is a copy of the final, submitted version of my thesis.
- The visualisations folder contains some of the visualisation used in my final thesis.

## Input Datasets:
- All input datasets were compressed. 
- The component input datasets were merged into my raw, input dataset using the merging2.ipynb file.
- The file beginning with 'maternal-mortality-ratio' provides the ground truth maternal mortality ratio values used to train and test my models.
- health.zip was retrived from the World Bank Group Gender Data Portal, and provides information about various health and socio-econmomic outcomes.
- Similarly, rep_wb.zip describes health determinants related to the environment, employment, education and social protection. It was downloaded from the World Bank Data Catalog.
- rep_ihme_prev.zip contains information about illness incidence and prevalence, sourced from the Institute of Health Metrics and Evaluation.
- rep_swper.zip provides information about women’s empowerment, and was compiled by the WHO Collaborating Center for Health Equity Monitoring.
- rep_gho.zip provided the World Bank's income level categorisations for all countries used in this report.

##  Datasets Used in this Report (Compressed Datasets):
- The datasets used in this report were large, and thus needed to be compressed. My code assumes that the dataset folders are present in the top-most level of the repository. Before running the code, please uncompress the data files and move the unzipped files into the correct folder.
- See the Readme.md inside this folder for a description of the various datasets provided.

## Models:
- All files ending with _models contain information about the models used in this research. More specifically, for each models contained in a folder, there is an associated:
-   JSON file specifying the name of the model, the dataset used to train the model (missing data threshold, validation fold), its best validation loss, and the best hyperparameters of the model.
-   The Optuna study object associated with the model's hyperparameter fine-tuning.
-   A .pkl file containing the best hyperparameters of the fine-tuned model.
- split_income_models: Contains the base estimator models fit on all features and the ensemble models trained to perform country-level prediction.
- split_year_models: Contains the base estimator models fit on all features and the ensemble models trained to perform forecasting.
- feature_selection: Contains the base estimator models fit on the correlation based feature subsets, with separate files for the models trained to perform country-level prediction and forecasting.
- fs_fromlit_models: Contains the base estimator models fit on literature-based feature subset, with separate files for the models trained to perform country-level prediction and forecasting.

## Code
- This file contains the majority of the code written for this project.
- The data_curation file was used to curate the train, validation, and test sets used in this research.
- The fs_comparison file was used to compare the base estimators trained on different feature subsets.
- The .sh files were used to run my code on the Gadi supercomputer. They repeated some lines of code that could be functionalised to allow me to train multiple models in parallel. 
- The byyear_models and income_models contains the code for the models trained to perform country-level level prediction and forecasting, respectively.
  - Each of these folders only contain the code for base models trained on all features.
  - They also contain the code for the ensemble models.
- The 'feature_selection' and 'feature_selection_from_lit' contain the base estimators trained on the correlation and literature-based feature subsets, respectively. They each contain code for to train the models for both country-level prediction and forecasting.
- The 'feature_selection_from_model' contains code to derive permutation importance scores. This was not used in the final thesis and was only included for completeness.
- The original folder contains code from the first iteration of this project, where imputation was used. This was solely included for historical purposes.

## sensitivity_analysis
- This folder contains all the data curation files and code used to train/evaluate base estimators and ensemble models trained on income-level specific data for the sensitivity analysis.

## Additional Note:
This repository was created for this project in July, as the original repository created in February was corrupted. 



