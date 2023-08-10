#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:57:16 2021

@author: geolog
"""

from osgeo import gdal,ogr #OpenGIS Simple Features Reference Implementation
import numpy as np
from mygdal_functions0_9 import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import time;
from mygdal_functions0_9 import *
from sklearn.preprocessing import scale
from sklearn import decomposition
import copy;
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix 
import pickle

###
import numpy as np


from mygdal_functions0_9 import *
from configuration import *
import imageio as io

###

#1 Settings

#files for processing, input and output directory
#from configuration.py
raster_data={}
raster_data.update({'flowacc': io.v2.imread(os.path.join('..','tif','flowacc.tif'))})
raster_data.update({'srtm': io.v2.imread(os.path.join('..','tif','srtm.tif'))})
print(raster_data['flowacc'])
print(raster_data['srtm'])

#2 Load class boundary values
with open(class_boundaries_value_filename, 'rb') as f:
    class_boundaries_values = pickle.load(f);
#round and insert 0 value in front of array
#class_boundaries_values=np.insert(np.round(class_boundaries_values,2),0,0);
#class_boundaries_values=np.append(class_boundaries_values,1);  
class_boundaries_values=np.arange(0,1.1,0.1);  
    
#3 Open geotiff data
#prospectivity    
MOPM_gdal_object = gdal.Open(predicted_file_path)
MOPM_array = MOPM_gdal_object.GetRasterBand(1).ReadAsArray() #rectanrular MOPM array
MOPM_array[np.isnan(MOPM_array)]=0; #replace nan values with 0       
    
#ore objects  
ORE_gdal_object = gdal.Open(raster_path)
ORE_array = ORE_gdal_object.GetRasterBand(1).ReadAsArray() #rectanrular MOPM array
ORE_array[np.isnan(ORE_array)]=0; #replace nan values with 0  

#ndvi mask read 
# NDVI_classes_gdal_object = gdal.Open(ndvi_file_path)
# NDVI_classes_array = NDVI_classes_gdal_object.GetRasterBand(1).ReadAsArray() #rectanrular MOPM array

#create mask
# rect_ind_learn=(np.zeros(np.shape(NDVI_classes_array))==1) #create bool matrix of False
# for cl in selects_NDVI_classes:
#     rect_ind_learn=rect_ind_learn | (NDVI_classes_array==cl);
rect_ind_learn= (raster_data['srtm']>0) & (raster_data['flowacc']>np.mean(raster_data['flowacc']))#create bool matrix ИСКЛЮЧАЕМ МОРЕ и сушу вне рек


#set 0 for non-learning classes,  apply NDVI mask on data
ORE_array[~rect_ind_learn]=0;
MOPM_array[~rect_ind_learn]=0;

#output rasters 
fig, (ax1, ax2,ax3) = plt.subplots(1,3);
ax1.imshow(MOPM_array); ax1.set_title('Prospectivity');
ax2.imshow(ORE_array); ax2.set_title('Points');
ax3.imshow(raster_data['flowacc']>np.mean(raster_data['flowacc'])); ax3.set_title('flowacc>mean');
plt.show();

#4 iterating through class_boundaries_values and defind % of deposits pixels and % study area
total_ore_pixels=np.sum(ORE_array); 

#total_area_pixels=np.size(ORE_array); #for the total area
total_area_pixels=np.sum(rect_ind_learn); #for the only training area

deposits_percent=np.array([]); #perc of deposits
area_percent=np.array([]); # (1 - (perc_of_area))*100

for cl in class_boundaries_values:
    ind=MOPM_array>=cl;
    #
    mineral_points=100*np.sum(ORE_array[ind])/total_ore_pixels;       
    deposits_percent=np.append(deposits_percent,mineral_points);
    #
    ap=(1-(np.sum(MOPM_array>=cl)/total_area_pixels))*100
    area_percent=np.append(area_percent,ap);

#5 draw plots of area/prediction rate
#start of drawing plots
xticks_values=np.arange(0,1,0.1);
yticks_reversed=np.arange(100,-20,-20);    
x, y = intersection(class_boundaries_values, deposits_percent, class_boundaries_values, area_percent)
print(x,y);    
textbox='(%0.2f,%0.2f)'%(x,y);
#fig=plt.figure();
fig, ax1 = plt.subplots();
plt.plot(class_boundaries_values,deposits_percent,'r-',label='Prediction rate');
plt.plot(class_boundaries_values,area_percent,'g-',label='Area');
plt.text(x[0]-0.1, y[0]+7, textbox, bbox=dict(facecolor='white', alpha=0.1));
plt.plot([x[0],x[0]],[0,y[0]],'r--');
plt.plot([0,x[0]],[y[0],y[0]],'r--');
plt.ylim([-1, 102]);
plt.xlim([class_boundaries_values[1]-0.1, 1]);
plt.plot(x,y,'bo');
ax1.tick_params(axis=u'both', which=u'both',length=0)

plt.legend();
plt.grid();
plt.ylabel('Percentage of known mineralization points (%)');

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#plt.xticks(class_boundaries_values);
plt.xticks(xticks_values);
#plt.yticks(yticks_reversed);
plt.xlabel('MOMP (prospectivity) value');
plt.ylabel('Percentage of study area (%)');
plt.ylim([-1, 102]);
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.tick_params(axis=u'both', which=u'both',length=0)

plt.savefig('P-A_plot_bekchiul.svg',dpi=300);
plt.savefig('P-A_plot_bekchiul.png',dpi=300);
plt.show();
#end of drawing plots

    
#print report 
print(('Value of MOMP≥{:.2f} cover {:.2f}% known points\
       on {:.2f}% of the area').format(x[0],y[0],100-y[0]));

    