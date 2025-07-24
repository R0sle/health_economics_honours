import numpy as np
from sklearn.impute import KNNImputer
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

test_input_path = sys.argv[1]
method = sys.argv[2]
test_output_path = sys.argv[3]
y_value = sys.argv[4]

test = np.load(test_input_path)

if 'std' in method:
    scaled = True

if method == 'knn':
    imputer_knn = KNNImputer(n_neighbors=5, keep_empty_features=True)
    train_imputed = imputer_knn.fit_transform(train)
    test_imputed = imputer_knn.transform(test)

elif method == 'poly1':

    train_imputed = train.copy()
    
    for feature in range(0, train.shape[1]):
        non_nan_indices = train[feature].dropna().index
        non_nan_values = train[feature].dropna().values
        
        if non_nan_indices.size != 0:
            
        #fit the polynomial coefficients 
            coefficients = np.polyfit(non_nan_indices, non_nan_values, 1)
            poly1 = np.poly1d(coefficients)
            extrapolated_values = poly1(train[feature].index)
            train_imputed[feature] = train_imputed[feature].where(train_imputed[feature].notna(), extrapolated_values)

    test_imputed = test.copy()
    
    for feature in range(0, test.shape[1]):
        #print(feature)
        non_nan_indices = test[feature].dropna().index
        non_nan_values = test[feature].dropna().values
        
        if non_nan_indices.size != 0:
            
        #fit the polynomial coefficients 
            coefficients = np.polyfit(non_nan_indices, non_nan_values, 1)
            poly1 = np.poly1d(coefficients)
            extrapolated_values = poly1(test[feature].index)
            test_imputed[feature] = test_imputed[feature].where(test_imputed[feature].notna(), extrapolated_values)

elif method == 'poly2':

    train_imputed = train.copy()
    
    for feature in range(0, train.shape[1]):
        non_nan_indices = train[feature].dropna().index
        non_nan_values = train[feature].dropna().values
        
        if non_nan_indices.size != 0:
            
        #fit the polynomial coefficients 
            coefficients = np.polyfit(non_nan_indices, non_nan_values, 2)
            poly1 = np.poly1d(coefficients)
            extrapolated_values = poly1(train[feature].index)
            train_imputed[feature] = train_imputed[feature].where(train_imputed[feature].notna(), extrapolated_values)
    
    test_imputed = test.copy()
    
    for feature in range(0, test.shape[1]):
        #print(feature)
        non_nan_indices = test[feature].dropna().index
        non_nan_values = test[feature].dropna().values
        
        if non_nan_indices.size != 0:
            
        #fit the polynomial coefficients 
            coefficients = np.polyfit(non_nan_indices, non_nan_values, 2)
            poly1 = np.poly1d(coefficients)
            extrapolated_values = poly1(test[feature].index)
            test_imputed[feature] = test_imputed[feature].where(test_imputed[feature].notna(), extrapolated_values)

elif method == 'poly3':

    train_imputed = train.copy()
    
    for feature in range(0, train.shape[1]):
        non_nan_indices = train[feature].dropna().index
        non_nan_values = train[feature].dropna().values
        
        if non_nan_indices.size != 0:
            
        #fit the polynomial coefficients 
            coefficients = np.polyfit(non_nan_indices, non_nan_values, 3)
            poly1 = np.poly1d(coefficients)
            extrapolated_values = poly1(train[feature].index)
            train_imputed[feature] = train_imputed[feature].where(train_imputed[feature].notna(), extrapolated_values)

    test_imputed = test.copy()
    
    for feature in range(0, test.shape[1]):
        #print(feature)
        non_nan_indices = test[feature].dropna().index
        non_nan_values = test[feature].dropna().values
        
        if non_nan_indices.size != 0:
            
        #fit the polynomial coefficients 
            coefficients = np.polyfit(non_nan_indices, non_nan_values, 3)
            poly1 = np.poly1d(coefficients)
            extrapolated_values = poly1(test[feature].index)
            test_imputed[feature] = test_imputed[feature].where(test_imputed[feature].notna(), extrapolated_values)

if scaled == True:
    if method == 'no':
        train_imputed = train
        test_imputed = test

    train_input_scaled = StandardScaler()
    test_input_scaled = StandardScaler()

    train_imputed = train_input_scaled.fit_transform(train_imputed)
    test_imputed = test_input_scaled.fit_transform(test_imputed)

# merging the two target feature columns 

def scale_merge_rescale(data_to_merge):
    scaler = StandardScaler()
    scaler.fit(data_to_merge)
    data_scaled = scaler.transform(data_to_merge)
    data_scaled_pd = pd.DataFrame(data_scaled)
    data_merged = np.nanmean(data_scaled_pd, axis=1)
    data_merged_pd = pd.DataFrame(data_merged)
    data_merged_pd.columns = ['Standard MMR Estimate']
    av_median = (np.nanmedian(data_to_merge.iloc[:,0]) + np.nanmedian(data_to_merge.iloc[:,1]))/2
    av_std = (np.std(data_to_merge.iloc[:,0]) + np.std(data_to_merge.iloc[:,1]))/2
    data_merged_pd['MMR Estimate Rescaled'] = (data_merged_pd['Standard MMR Estimate'] * av_std) + av_median
    return data_merged_pd

if y_value == True:

    if method == 'knn':
        scaler = StandardScaler()
        scaler.fit(train_imputed)
        train_scaled = scaler.transform(train_imputed)
        train_scaled_pd = pd.DataFrame(train_scaled)
        train_merged = np.nanmean(train_scaled_pd, axis=1)
        train_merged_pd = pd.DataFrame(train_merged)
        train_merged_pd.columns = ['Standard MMR Estimate']
        av_median = (np.nanmedian(train_imputed[:, 0]) + np.nanmedian(train_imputed[:, 1]))/2
        av_std = (np.std(train_imputed[:,0]) + np.std(train_imputed[:,1]))/2
        train_merged_pd['MMR Estimate Rescaled'] = (train_merged_pd['Standard MMR Estimate'] * av_std) + av_median
        train_imputed = train_merged_pd.drop('Standard MMR Estimate', axis=1)

        scaler = StandardScaler()
        scaler.fit(test_imputed)
        test_scaled = scaler.transform(test_imputed)
        test_scaled_pd = pd.DataFrame(test_scaled)
        test_merged = np.nanmean(test_scaled_pd, axis=1)
        test_merged_pd = pd.DataFrame(test_merged)
        test_merged_pd.columns = ['Standard MMR Estimate']
        test_av_median = (np.nanmedian(test_imputed[:, 0]) + np.nanmedian(test_imputed[:, 1]))/2
        test_av_std = (np.std(test_imputed[:,0]) + np.std(test_imputed[:,1]))/2
        test_merged_pd['MMR Estimate Rescaled'] = (test_merged_pd['Standard MMR Estimate'] * test_av_std) + test_av_median
        test_imputed = test_merged_pd.drop('Standard MMR Estimate', axis=1)

    else:
        scaler = StandardScaler()
        scaler.fit(train_imputed)
        train_scaled = scaler.transform(train_imputed)
        train_scaled_pd = pd.DataFrame(train_scaled)
        train_merged = np.nanmean(train_scaled_pd, axis=1)
        train_merged_pd = pd.DataFrame(train_merged)
        train_merged_pd.columns = ['Standard MMR Estimate']
        av_median = (np.nanmedian(train_imputed.iloc[:,0]) + np.nanmedian(train_imputed.iloc[:,1]))/2
        av_std = (np.std(train_imputed.iloc[:,0]) + np.std(train_imputed.iloc[:,1]))/2
        train_merged_pd['MMR Estimate Rescaled'] = (train_merged_pd['Standard MMR Estimate'] * av_std) + av_median
        train_imputed = train_merged_pd.drop('Standard MMR Estimate', axis=1)
    
        scaler = StandardScaler()
        scaler.fit(test_imputed)
        test_scaled = scaler.transform(test_imputed)
        test_scaled_pd = pd.DataFrame(test_scaled)
        test_merged = np.nanmean(test_scaled_pd, axis=1)
        test_merged_pd = pd.DataFrame(test_merged)
        test_merged_pd.columns = ['Standard MMR Estimate']
        test_av_median = (np.nanmedian(test_imputed.iloc[:,0]) + np.nanmedian(test_imputed.iloc[:,1]))/2
        test_av_std = (np.std(test_imputed.iloc[:,0]) + np.std(test_imputed.iloc[:,1]))/2
        test_merged_pd['MMR Estimate Rescaled'] = (test_merged_pd['Standard MMR Estimate'] * test_av_std) + test_av_median
        test_imputed = test_merged_pd.drop('Standard MMR Estimate', axis=1)

np.save(train_output_path, train_imputed)
np.save(val_output_path, test_imputed)