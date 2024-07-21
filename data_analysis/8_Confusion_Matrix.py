import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import random
    
from ca_plot import ca_plot

import PIL

try:
    import gdal,ogr
except ModuleNotFoundError:
    from osgeo import gdal,ogr

print('Loading libraries is done.')

#read tif files for Bekchiul vicinity test AOI
#file paths 
print('Start reading geotiffs into np array...')
filename_placers =  'placers_aligned.tif'
filename_predicted =  'predicted_one_class_SVM_aligned.tif'
filename_flow_orders =  'flow_orders_aligned.tif'
folder = os.path.join('..','tif','initial_aligned','metrics_for')
filepath_placers = os.path.join(folder,filename_placers)
filepath_predicted = os.path.join(folder,filename_predicted)
filepath_flows = os.path.join(folder,filename_predicted)

#read tiffs into gdal, then read bands from gdal into matrices
gdal_obj_placers = gdal.Open(os.path.join(filepath_placers)) 
gdal_obj_predicted = gdal.Open(os.path.join(filepath_predicted)) 
gdal_obj_flows = gdal.Open(os.path.join(filepath_flows)) 

placers_mat = gdal_obj_placers.GetRasterBand(1).ReadAsArray()
predicted_mat = gdal_obj_predicted.GetRasterBand(1).ReadAsArray()
flows_mat = gdal_obj_flows.GetRasterBand(1).ReadAsArray()
print('Reading geotiffs into np array is done')

predicted_threshold = 0.56

#resolution check
print('Start resolution check...')
print(np.shape(placers_mat))
print(np.shape(predicted_mat))
print(np.shape(flows_mat))
if np.shape(placers_mat) == np.shape(predicted_mat) and np.shape(predicted_mat) == np.shape(flows_mat):
    print('resolutions are ok')
else:
    print('resolutions are DIFFERENT!')

print('resolution check is done')

#creating flows mask from predicted (need to exclude nans and 0's)
#binary index
print('Getting stream indices into array...')
stream_ind = (flows_mat > 0)
print(np.shape(stream_ind))
print(predicted_mat[stream_ind])
#row, column tuple
r,c = np.where(stream_ind == True)
print('Done - Getting stream indices into array!')

TP = 0
FP = 0
TN = 0
FN = 0

for ri,ci in zip(r,c):
    if predicted_mat[ri,ci] >= predicted_threshold and placers_mat[ri,ci] == 1:
        TP+=1
    elif predicted_mat[ri,ci] < predicted_threshold and placers_mat[ri,ci] == 1:
        FN+=1
    elif predicted_mat[ri,ci] >= predicted_threshold and placers_mat[ri,ci] == 0:
        FP+=1
    elif predicted_mat[ri,ci] < predicted_threshold and placers_mat[ri,ci] == 0:
        TN+=1

print('TP=',TP)
print('TN=',TN)
print('FP=',FP)
print('FN=',FN)


# #open datatable
# #filename =  'new_df2_confusion_analysis.csv'
# filename =  'new_df2_confusion_analysis_trends.csv'
# filepath = os.path.join('..','..','csv',filename)

# #dataframe
# df = pd.read_csv(filepath)

# #number of records
# rec_number = len(df.index)
# indeces_all = list(range(rec_number))

# #train test split
# train, test = train_test_split(df, test_size=0.3)

# #keys in a table  'deposit_po', 'f3l9_density', 'l3l9_density', 'f3l9_density', 'l3l9_minkowski'
# print(df.keys())

# #copy df with predictors
# new_df = df[['id','TP', 'TN', 'FP','FN']].copy()
# new_df = new_df.dropna()

# print(new_df)
# print('total number of samples=',len(new_df))

# #compute confusion matrix
# TP = len(new_df[new_df['TP']==1])
# TN = len(new_df[new_df['TN']==1])
# FP = len(new_df[new_df['FP']==1])
# FN = len(new_df[new_df['FN']==1])

out_cm = f''','неперспективный','перспективный';
'неперспективный',{TN}True Negative,{FP} False Positive;
'перспективный',{FN}False Negative, {TP} True Positive
'''

with open('confusion_matrix.csv','w') as ccm:
    ccm.writelines(out_cm)

#точность accuracy

accuracy = (TP + TN)/(TP + TN + FP + FN)
print('accuracy=',round(accuracy,2))

#True positive rate, recall
TPR = TP/(TP + FN)
print('TPR=',round(TPR,2))

#PPV positive predictive value, precision точность 
PPV = TP/(TP + FP)
print('PPV=',round(PPV,2))

#FScore
FScore = 2* TP/(2* TP + FP + FN)
print('FScore=',round(FScore,2))

#специфичность (Specificity) или TNR – True Negative Rate
TNR = TN / (TN + FP)
print('TNR=',round(TNR,2))

#False Positive Rate (FPR, fall-out, false alarm rate):
FPR = 1 - TNR
print('FPR=',round(FPR,2))

#Kappa
'''
Kappa compares the probability of agreement to that expected if 
the ratings are independent. The values of range lie in [− 1, 1] 
with 1 presenting complete agreement and 0 meaning no agreement or independence
https://www.sciencedirect.com/topics/medicine-and-dentistry/kappa-statistics#:~:text=Kappa%20compares%20the%20probability%20of,agreement%20is%20worse%20than%20random.
'''
m = len(r)
accuracy_chance = (TN + FP)/m * (TN + FN)/m + (FN + TP)/m * (TN + FP)/m
kappa = (accuracy - accuracy_chance)/(1 - accuracy_chance)
print('kappa=',round(kappa,2))

#Balanced Accuracy
BA = (TP/(TP+FN) + TN/(TN+FP))*0.5
print('BA=',round(BA,2))

out_metrics= f'''Метрика,Acc,TPR,PPV,FScore,TNR,FPR,Kappa,BA;
Значение,{round(accuracy,2)},{round(TPR,2)},{round(PPV,2)},{round(FScore,2)},{round(TNR,2)},{round(FPR,2)},{round(kappa,2)},{round(BA,2)};
'''

with open('control_metrics_trends.csv','w') as metrics:
    metrics.writelines(out_metrics)
    
#https://www.picsellia.com/post/understanding-the-f1-score-in-machine-learning-the-harmonic-mean-of-precision-and-recall
F1Score = 2 * (PPV * TPR) /(PPV + TPR) 
    
