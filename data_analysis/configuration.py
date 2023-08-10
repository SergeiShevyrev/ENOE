#location of files and local paths
import os


"""
Important note: files to be processed must have a "tif" extension (in any case)
The eighth channel (panchromatic) must be excluded from the analysis, both as its resolution and,
hence the dimension is greater.
You can exclude, for example, by renaming as ".tiff"
"""

#shapefile of cropping extent
shapefilename='AOI_5.shp'; # sea area
foldname='TO_USE';
foldname_full="../"+foldname;
out_dir_name='AST_L1T_'+foldname+'_Unpacked'
metafn='meta.txt';
shpfilepath=os.path.join("..","shp",shapefilename);

fileext="tif"; #extention for files
metafileext="txt"; #extention for meta files
outdir=os.path.join("..","tif",);
dir_cropped="cropped_bands_topo" #dir for AOI cropped
dir_cropped_path=os.path.join(outdir,dir_cropped);
dir_products="products_topo" 
dir_products_path=os.path.join(outdir);
band_number_inname='_b%N%' #%N% - for band number e.g. LC81050292016143LGN00_B6.TIF NOT A CASE SENSITIVE
band_number_inname=band_number_inname.lower();
target_resolution=15; #resolution to be used in bands resize, POSSIBLE values 15,30,90 m

#points
points_shp_filepath=os.path.join("..","shp","points.shp");
ore_point_radius_m=150; #300
radius=ore_point_radius_m/target_resolution; #радиус объекта в пикселях 

#drm for topocorrection
drm_name="n.tif";
drm_folder=os.path.join("..","srtm","dem");
drm_filepath=os.path.join(drm_folder,drm_name);

#points raster
raster_path=os.path.join(dir_products_path,"ore.tif");

#model output and parameters
file_model_data_name='model_data';
file_model_data_name_path=os.path.join(dir_products_path,file_model_data_name);

cov_ratios_dpca_save_iron=os.path.join(dir_products_path,"ratios_dpca_cov_stat_iron.xls");
cov_ratios_dpca_save_clay=os.path.join(dir_products_path,"ratios_dpca_cov_stat_clay.xls");

loadings_filemask='loadings_DPCA_{}.xls';

#selected NDVI classes and component prefixes
total_number_ndvi_classes=6; 
selects_NDVI_classes=[0,1,2,3]; #[0,1,2,3]; #select only open soils and rare vegetation
prefix=[""]; #only two values sare allowed

#model configuration parameters
fit_params=True; #if True, oneClassSVM model uses GridSearch  to fit nu and gamma
nu=0.1; #An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
kernel="rbf"; #kernel function
gamma=0.9; #Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#parameters considered as range for GridSearch fit if dont_fit=True
parameters = {'kernel':['rbf','sigmoid'],
              'nu':[float(nu)/10 for nu in range(1,10)], #10 and 10 before
              'gamma':[float(gamma)/10 for gamma in range(1,10)]
              }

#Configuration for model extrapolation on the new area (need to define folders for 'mineral' images)
#files with "_new" postfix are related to 
filedir_model=os.path.join("..","data");
model_maxent_name='maxent_model_DPCA_OCSVM.p';
pathrowfolder_new="104_029"
datefolder_new="2019_10_16"
dir_products_new="products_topo" 
outdir_new=os.path.join("..","Landsat8_Processed",pathrowfolder_new,datefolder_new);
dir_products_path_new=os.path.join(outdir_new,dir_products_new);
dir_cropped_new="cropped_bands_topo" 
dir_cropped_path_new=os.path.join(outdir_new,dir_cropped_new);

#C-A analysis class values
predicted_file_path=os.path.join(dir_products_path,\
    "predicted_oneClassSVM{}_".format('')+"_"+".tif");
class_boundaries_value_filename='cbv.values'
step_value=0.005;
filter_extrema=1; #absolute threshold of first area derivative for extrema filtering
start_value_perc=0.98; #98% of values will be ignored for differentiation


#P-A analysis
ndvi_file_path=os.path.join(dir_products_path,\
    "ndvi_classes_{}".format('')+".tif");      
