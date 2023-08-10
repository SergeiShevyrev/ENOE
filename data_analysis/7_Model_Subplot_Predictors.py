import matplotlib.pyplot as plt

from mygdal_functions0_9 import *
from configuration import *
import imageio as io


#1 Data processing
#создание списка файлов для открытия и имя классифицированного изображения NDVI
files_for_output={};

try:
    for file in os.listdir(dir_products_path):
        #file=file.lower();
        if file.lower().endswith("."+fileext.lower())  \
        and (file.lower().startswith(prefix[0]) or file.lower().startswith(prefix[1])\
             or file.lower().startswith("ore")):

            if file.lower().startswith('flowacc') or file.lower().startswith('magm-dist') \
                    or file.lower().startswith('base4') or file.lower().startswith('base3') \
                    or file.lower().startswith('base2') or file.lower().startswith('placers') \
                    or file.lower().startswith('base-23') or file.lower().startswith('base-34'):

                    files_for_output.update({file.split('.')[0]:io.v2.imread(os.path.join(dir_products_path,file))});
                    print(file+" was added to data collecting queue.");

except(FileNotFoundError):
        print("Input image folder doesn\'t exist...");

#number of predictors
pred_num=len([*files_for_output])
pred_colums=3;
if pred_num%pred_colums != 0:
    pred_rows=pred_num//pred_colums + 1
else:
    pred_rows = pred_num // pred_colums

subplt_tot=pred_rows*pred_colums


#contrast flowacc image
try:
    files_for_output['flowacc'][files_for_output['flowacc'] <= np.mean(files_for_output['flowacc'])] = 0
    files_for_output['flowacc'][files_for_output['flowacc']>np.mean(files_for_output['flowacc'])]=1
    #open srtm
    srtm = io.v2.imread(os.path.join(dir_products_path,'srtm.tif'))
    files_for_output['flowacc'][srtm<=0]=0
except KeyError:
    print('flowacc key was not found')

#show predictor plot

plt.figure();
for n,key in enumerate([*files_for_output]):
    plt.subplot(pred_rows*100+pred_colums*10+n+1);
    pic=plt.imshow(files_for_output[key],cmap='bwr');
    if key != 'placers':
        plt.colorbar()
    else: #discreet values for placers
        plt.colorbar(pic, ticks=np.arange(np.min(files_for_output[key]), np.max(files_for_output[key]) + 1))
    plt.title(key);
    plt.axis('off');

plt.savefig('predictors_plot.png',dpi=300);
plt.savefig('predictors_plot.svg',dpi=300);
plt.show();


