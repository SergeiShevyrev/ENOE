from osgeo import gdal,ogr #OpenGIS Simple Features Reference Implementation
import numpy as np
from mygdal_functions0_9 import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import time;
from sklearn.preprocessing import scale
from sklearn import decomposition
import copy;
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix
import pickle
import sklearn

from mygdal_functions0_9 import *
from configuration import *
from scipy.signal import argrelextrema


#compute C-A and plot to determine background value
def CA_compute_plot(data_array):
    data_array_flat = np.ndarray.flatten(data_array[data_array > 0]);  # flat MOPM array for >0

    # 3 Compute log-transformed values
    log_MOPM = np.log10(data_array_flat);
    log_MOPM.sort();

    # iterate log_MOPM values, compute areas for them
    quantity_log_MOMP = np.arange(np.min(log_MOPM), np.max(log_MOPM) + step_value, step_value);
    start_value = int(start_value_perc * len(quantity_log_MOMP));  # start value for differentiation

    area_log_array = np.array([]);
    for i in quantity_log_MOMP:
        ind = np.where(log_MOPM >= i);
        area_log_array = np.append(area_log_array, np.log10(len(ind[0])));

    # compute derivatives d_logArea/d_logMOPM
    # first derivative
    d_logArea = (area_log_array[1:] - area_log_array[0:-1]) / step_value;

    # second derivative
    d2_logArea = (d_logArea[1:] - d_logArea[0:-1]) / step_value;

    # find peaks for 2nd derivatives
    minima = np.array(argrelextrema(d2_logArea[start_value:], np.less));
    maxima = np.array(argrelextrema(d2_logArea[start_value:], np.greater));

    # extrema filtration by threshold values
    min_ind = np.abs(d_logArea[start_value:][minima]) > 1
    minima_filtered = minima[min_ind];
    max_ind = np.abs(d_logArea[start_value:][maxima]) > 1
    maxima_filtered = maxima[max_ind];

    # MOPM class boundaries
    class_boundary_ind = np.append(minima_filtered, maxima_filtered);
    class_boundary_ind.sort();

    # 4 draw logarithmic C-A plot
    plt.plot(quantity_log_MOMP[start_value:], area_log_array[start_value:]);
    plt.title('C-A plot for values indexes starting start_value');
    plt.xlabel('Log transform MOPM values');
    plt.ylabel('Log(Area)');
    plt.plot(quantity_log_MOMP[start_value:][class_boundary_ind], area_log_array[start_value:][class_boundary_ind], "xr");
    plt.plot(quantity_log_MOMP[start_value:][class_boundary_ind], area_log_array[start_value:][class_boundary_ind], "xr");
    for ind in class_boundary_ind:
        x = [quantity_log_MOMP[start_value:][ind], quantity_log_MOMP[start_value:][ind]];
        y = [np.nanmin(area_log_array[start_value:][area_log_array[start_value:] != -np.inf]),
             area_log_array[start_value:][ind]];
        plt.plot(x, y, 'r--');
    plt.savefig('C-A_plot.png', dpi=300);
    plt.savefig('C-A_plot.svg', dpi=300);
    plt.show();

    plt.plot(quantity_log_MOMP[start_value:-1], d_logArea[start_value:]);
    plt.title('first derivative');
    plt.plot(quantity_log_MOMP[start_value:-2][minima_filtered], d_logArea[start_value:][minima_filtered], "xr");
    plt.plot(quantity_log_MOMP[start_value:-2][maxima_filtered], d_logArea[start_value:][maxima_filtered], "o");
    plt.show();

    plt.plot(quantity_log_MOMP[start_value:-2], d2_logArea[start_value:]);
    plt.title('second derivative');
    plt.plot(quantity_log_MOMP[start_value:-2][minima_filtered], d2_logArea[start_value:][minima_filtered], "xr");
    plt.plot(quantity_log_MOMP[start_value:-2][maxima_filtered], d2_logArea[start_value:][maxima_filtered], "o");
    plt.show();

    # find and report class boundary values
    class_boundary_values_log = quantity_log_MOMP[start_value:][class_boundary_ind];
    class_boundary_values = 10 ** class_boundary_values_log;
    print('Values of boundary class values:{}'.format(class_boundary_values));
    return class_boundary_values

#1 Settings

#files for processing, input and output directory, selected NDVI classes and DPC fnames
#prefixes are taking from configuration.py
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#2 Data processing
#создание списка файлов для открытия и имя классифицированного изображения NDVI
files_for_processing=[];

try:
    for file in os.listdir(dir_products_path):         #exclude 3band tif files
        #file=file.lower();
        if file.lower().endswith("."+fileext.lower())  \
        and (file.lower().startswith(prefix[0]) or file.lower().startswith(prefix[1])\
             or file.lower().startswith("ore")):

            files_for_processing.append(file);
            print(file+" was added to data collecting queue.");

           
except(FileNotFoundError):
        print("Input image folder doesn\'t exist...");

#3 Открытие файлов 
#создание словаря каналов bands и загрузка туда файлов
raster_data={};  #dictionary storing dpc names and raster values 
for myfile in files_for_processing:
        try:
            try:
                key=myfile.split('_')[0]+'_'+myfile.split('_')[1]; #crop strings before second underline
            except:
                key=myfile.split('.')[0];
            gdal_object = gdal.Open(os.path.join(dir_products_path,myfile)) #as new gdal_object was created, no more ColMinInd,RowMinInd
            raster_data.update({key:gdal_object.GetRasterBand(1).ReadAsArray()});
        except:
            print('Error! Can not read file '+ myfile +' data!')

#show C-A plot
#class_boundary_values=CA_compute_plot(raster_data['flowacc'])

plt.figure();
plt.imshow(raster_data['flowacc']>np.mean(raster_data['flowacc']));
plt.title('Flow accumulation above background');
plt.show();

#4 create data dictionary
#| - поэлементное "или" & - поэлементное "и"
rect_ind_learn= (raster_data['srtm']>0) & (raster_data['flowacc']>np.mean(raster_data['flowacc']))#create bool matrix ИСКЛЮЧАЕМ МОРЕ и сушу вне рек

#learning points coordinates
ri,ci=np.where(rect_ind_learn==True);       #ri[:,None] - single row transposition


plt.figure();
plt.imshow(rect_ind_learn); 
plt.title('Area for the model teaching ({} NDVI classes)'.format(selects_NDVI_classes));
plt.show();

model_data={};

#adding entry for the id/row/columns indexes
model_data.update({'id':[]});
model_data.update({'row':[]});
model_data.update({'col':[]});

#add dictionaries to model data according to opened rasters
for keyval in [*raster_data]:
    model_data.update({keyval:[]});

#обходим rc,ci попарно, выбираем данные в словарь базы данных
id=0;
for rn,cn in zip(ri,ci):
    #print(rn,cn);
    model_data['id'].append(id);
    model_data['row'].append(rn);
    model_data['col'].append(cn);
    id+=1;
    #picking points from every  image
    for keyval in [*raster_data]:
        model_data[keyval].append(raster_data[keyval][rn,cn]);
        
        
#convert mode data to pandas DataFrame
model_data_df=pd.DataFrame(model_data);

#Display the first 10 rows
result = model_data_df.head(10)
print("First 10 rows of the DataFrame:")
print(result)

print('Всего в таблице',len(model_data_df),'записей')

print('column list:')
print([*model_data_df])

#save model data into pickle (cause excel is out of range)
with open(file_model_data_name_path, 'wb') as f:
    pickle.dump(model_data_df, f)
        

