from osgeo import gdal,ogr #OpenGIS Simple Features Reference Implementation
import numpy as np
from mygdal_functions0_9 import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import time, datetime;
from sklearn.preprocessing import scale
from sklearn import decomposition
import copy;
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix 
import pickle

#one class SVM
from sklearn import svm, metrics
#selecting training and testing set
from sklearn.model_selection import train_test_split, GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler
import random
#from sklearn.inspection import plot_partial_dependence, partial_dependence
#from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.inspection import PartialDependenceDisplay

from mygdal_functions0_9 import *
from configuration import *


"""
Respect to scikit-learn official website:
### https://scikit-learn.org/0.15/modules/generated/sklearn.svm.OneClassSVM.html
Grid search parameters optimization technique:
### https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html    
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    
"""

#1 Settings
#files for processing, input and output directory
#are in configuration.py

#file_model_data_name='model_data_{}{}'.format(pathrowfolder,datefolder);
#file_model_data_name_path=os.path.join(dir_products_path,file_model_data_name);

file_model_data_name_prob='model_data_prob';
file_model_data_name_path_out=os.path.join(dir_products_path,file_model_data_name_prob);

model_filename=os.path.join('..','data','maxent_model_DPCA_OCSVM.p');

#chhosen due to eigenvalues and different signs
#goetite dpc2, kaolinite dpc2, chlorite dpc2, hematite dpc1 

#2 Loading data for training
with open(file_model_data_name_path, 'rb') as f:
    model_data_df = pickle.load(f)
    
#getting resolution from file    
for file in os.listdir(dir_products_path):         #exclude 3band tif files
        if file.lower().endswith("."+fileext.lower()):
            gdal_object = gdal.Open(os.path.join(dir_products_path,file)); #as new gdal_object was created, no more ColMinInd,RowMinInd
            break;                          

row,col=gdal_object.GetRasterBand(1).ReadAsArray().shape;    
    
#3 Sampling to train the MaxEnt model
#find ore column
for key in model_data_df.columns:
    if key.lower().startswith("ore")==True:
        ore_key=key;
        break;
        
#select DPC1 only indexes
dpc1_columns=[];
for col_name in [*model_data_df]:

    # if col_name == 'srtm' or col_name == 'base6' or col_name=='flowacc' or col_name=='magm-dist'\
    #             or col_name == 'base4' or col_name == 'base3'\
    #             or col_name == 'base2'\
    #             or col_name == 'base-23' or col_name == 'base-34':
    #             dpc1_columns.append(col_name);
    if col_name=='flowacc' or col_name=='magm-dist'\
                or col_name == 'base4' or col_name == 'base3'\
                or col_name == 'base2'\
                or col_name == 'base-23' or col_name == 'base-34':
                dpc1_columns.append(col_name);
    # if col_name == 'srtm' or col_name == 'base6'\
    #             or col_name == 'base4' or col_name == 'base3'\
    #             or col_name == 'base2' or col_name == 'flowacc'\
    #             or col_name == 'base-23' or col_name == 'base-34':
    #             dpc1_columns.append(col_name);

#arrange columns in the necessary order into model_np_array
dpc1_columns=sorted(dpc1_columns, reverse=False);

#
all_data_np_array=np.array(np.array(model_data_df['id']).reshape(-1,1));
all_data_np_array=np.hstack([all_data_np_array,np.array(model_data_df['row']).reshape(-1,1),\
                             np.array(model_data_df['col']).reshape(-1,1)]);

for colname in dpc1_columns:
    col_np=np.array(model_data_df[colname]);
    all_data_np_array=np.hstack([all_data_np_array,col_np.reshape(-1,1)]);


#select ores only records
idx_ore=np.array(model_data_df[ore_key]==1);    
ore_data_np_array=all_data_np_array[idx_ore,:];  #select np array with data corresponding to ore only 
non_ore_idx=model_data_df.index[model_data_df[ore_key] == 0].tolist();

roc_auc_list=[];

#test output of ore data
result = ore_data_np_array[:10]
print("First 10 rows of the DataFrame:")
print(result)

#divide into train and test matrix
col_names=['id','row','col']+dpc1_columns; #concatenated lists
params_train,params_test=train_test_split(ore_data_np_array[:,3:], test_size=0.2,random_state=42); #,\
                                         # random_state=42);  #

### https://scikit-learn.org/0.15/modules/generated/sklearn.svm.OneClassSVM.html
#teach OneSVM model

# Standardize features
train_cover_std = StandardScaler().fit_transform(params_train[:,0:]); #training array 

                                                           
#define optimal parameters with GridSearchCV
print('define optimal parameters with GridSearchCV')
if fit_params==True: #by DEFAULT ITS FALSE IN config
    clf = GridSearchCV(estimator=svm.OneClassSVM(),param_grid=parameters,refit=True,scoring='adjusted_rand_score'); #['adjusted_rand_score','r2',]
    #{'gamma': 0.9, 'kernel': 'rbf', 'nu': 0.3} ----  Area under the ROC curve : 0.637760
    bgrd_rand_idx=random.sample(non_ore_idx, int(len(params_train[:,0]))); #select non-occurence data as much as occurence
    non_ore_data_std=StandardScaler().fit_transform(all_data_np_array[bgrd_rand_idx,3:]);
    
    
    #stack occurence/non-occurence observation
    records=np.r_[non_ore_data_std,train_cover_std];
    occurences=np.r_[np.zeros(non_ore_data_std.shape[0]).T,np.ones(train_cover_std.shape[0]).T];
    
    clf.fit(records,occurences);
    
    print(clf.best_params_);    
    gamma=clf.best_params_['gamma'];
    nu=clf.best_params_['nu'];
    kernel=clf.best_params_['kernel'];
    
# Fit OneClassSVM
print(" - fit OneClassSVM ... ", end='')
clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)  #model регрессор svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.4) 

clf.fit(train_cover_std)        #training the model (data formatted: string - parameter, column - cell)
                               
#prediction 
#all_data_std=(all_data_np_array[:,3:] - mean) / std;  
all_data_std=StandardScaler().fit_transform(all_data_np_array[:,3:]);                    
pred = clf.decision_function(all_data_std) #prediction of distribution values

#normalization of the results
pred_norm=(pred-pred.min())/(pred.max()-pred.min());


#Model quality test
# Compute AUC with regards to background points

#selection of predicted values for barren pixels
#non_ore_idx=model_data_df.index[model_data_df[ore_key] == 0].tolist()
bgrd_rand_idx=random.sample(non_ore_idx, int(len(params_test[:,0])));
pred_background = pred[bgrd_rand_idx];


#test
#standardize within each sample


train_cover_std_test=StandardScaler().fit_transform(params_test[:,0:]);
"""
train_cover_std_test = (params_test[:,0:] - mean) / std
"""
all_data_std_train,all_data_std_test=train_test_split(all_data_std, test_size=0.01,random_state=42);
pred_test = clf.decision_function(train_cover_std_test)



scores = np.r_[pred_test, pred_background]
y = np.r_[np.ones(pred_test.shape), np.zeros(pred_background.shape)]
fpr, tpr, thresholds = metrics.roc_curve(y, scores,pos_label=1,drop_intermediate=True)
roc_auc = metrics.auc(fpr, tpr)
#plt.text(-35, -70, "AUC: %.3f" % roc_auc, ha="right")
print("\n Area under the ROC curve : %f" % roc_auc)

fig=plt.figure();
plt.plot(fpr, tpr, label='ROC-curve');
plt.plot([0,1],[0,1],'--')
plt.xlabel('False positive rate');
plt.ylabel('True positive rate');
plt.title('Area under the ROC curve : {0:.{1}f}'.format(roc_auc,4));
plt.legend();
plt.savefig('ROC-curve_prediction_probability.png',dpi=300);
plt.savefig('ROC-curve_prediction_probability.svg',dpi=300);
plt.show();

roc_auc_list.append(roc_auc);    
max_iter=np.where(roc_auc_list==max(roc_auc_list))[0][0]; #step number for max ROC_AUC value
print('Highest ROC-AUC computed so far: %s on iteration(RS)= %s'%(max(roc_auc_list),max_iter));



#stack predicted values into the np array
#render predicted values into raster
predicted_raster=np.empty([row,col],dtype=np.float64);
predicted_raster[:]=np.nan;

for i in range(0,len(all_data_np_array[:,0]),1):
    r,c=all_data_np_array[i,1],all_data_np_array[i,2];
    predicted_raster[int(r),int(c)]=pred_norm[i];

#save geotiff with normalized predicted data
saveGeoTiff(predicted_raster,os.path.join(dir_products_path,\
    "predicted_oneClassSVM{}_".format('')+"_"+".tif"),\
    gdal_object,0,0);

data_output_df=model_data_df.assign(Prob=pred);  
data_output_df=data_output_df.assign(Prob_norm=pred_norm);  

#output of the data frame to the pickle file (since excel capabilities are not enough)
with open(file_model_data_name_path_out, 'wb') as f:
    pickle.dump(data_output_df, f)


#save model into file
if not os.path.exists(os.path.join('..','data')):
    os.mkdir(os.path.join('..','data'));        
    
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f);  

#plot partial dependence plots

feature_names=dpc1_columns;
features = np.arange(0,len(feature_names),1)

fig, ax = plt.subplots(figsize=(12, 6))
#fig, ax = plt.subplots(figsize=(2, 2))
print(all_data_std_test)
print(features)
print(feature_names)
part_dep=PartialDependenceDisplay.from_estimator(clf, all_data_std_test,features,feature_names=feature_names,ax=ax)
#(clf, all_data_std_test,features,feature_names,n_cols=3,\
#                                 grid_resolution=50,ax=ax)
fig = plt.gcf();
plt.setp(ax.get_yticklabels(), fontsize=3);
plt.setp(ax.get_xticklabels(), fontsize=3);
fig.suptitle('Response curves for the target area')
fig.subplots_adjust(hspace=0.7)
#fig.text(0.06, 0.5, 'common ylabel', ha='center', va='center', rotation='vertical')
fig.savefig('Partial_dependence_training.svg',dpi=300);
fig.savefig('Partial_dependence_training.png',dpi=300);    



#save model parameters into model.log file
moment = datetime.datetime.now();
dt_str= moment.strftime("%d/%m/%Y %H:%M:%S");
log_output_str='{} svm.OneClassSVM(nu={}, kernel={}, gamma={})'.format(dt_str,nu,kernel,gamma);

log_file = open("model.log", "a");
log_file.write(log_output_str);
log_file.write('\r\n');
log_file.close();    
    
#save model into file
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f);    
