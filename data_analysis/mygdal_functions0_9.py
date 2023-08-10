# -*- coding: utf-8 -*-

# some functions for the GDAL data processing
#written and composed by Sergei L Shevyrev http://lefa.geologov.net
#v. 0.9 added tool for getting extent from the shp layer

from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import pandas as pd

from scipy.interpolate import griddata 
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from scipy.ndimage.filters import maximum_filter
from scipy import sparse
import skimage.measure as skms #label, regionprops

import skimage.morphology as skm #scikit-image
from skimage.feature import canny
from skimage.transform import resize
from scipy import ndimage as ndi

#machine learning 
from sklearn import datasets,svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

#im max algorithm 
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import time
import os #file system tools

#PCA principal component analysis
from PIL import Image
#из книги: Jan Erik Solem. «Programming Computer Vision with Python». iBooks. 

from sklearn.model_selection import train_test_split
from skimage.draw import disk

#function declaration

def GetExtent(gt,cols,rows):
    """
    srtm_gdal_object.GetGeoTransform()
    
    (329274.50572846865, - left X
     67.87931651487438,  - dX
     0.0,
     4987329.504699751,  - верх Y
     0.0,
     -92.95187590930819) - dY
    """
    #[[влx,влy],[нлx,нлy],[нпx, нпy],[впx, впy]]
    ext=[[gt[0],gt[3]],[gt[0],(gt[3]+gt[5]*rows)],[(gt[0]+gt[1]*cols),(gt[3]+gt[5]*rows)],[(gt[0]+gt[1]*cols),gt[3]]];
    return ext;
    """
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext
    """
    
def GetExtent2(gt,cols,rows):
    #[[влx,влy],[нлx,нлy],[нпx, нпy],[впx, впy]]
    #ext=[[gt[0],gt[3]],[gt[0],(gt[3]+gt[5]*rows)],[(gt[0]+gt[1]*cols),(gt[3]+gt[5]*rows)],[(gt[0]+gt[1]*cols),gt[3]]];
    #x_min, x_max, y_min, y_max
    return gt[0],(gt[0]+gt[1]*cols),gt[3],(gt[3]+gt[5]*rows);



def getCentroidsArea(labels,min_area): #get labels centroids coordinates and area
    if np.max(labels)==0:
        print('no labels has been detected')
        return
    centroids=[]; #list of centroids and area
    for i in range(1,np.max(labels)+1):
       print('cluster {} out of {}'.format(i,np.max(labels))) 
       (h,w)=np.where(labels==i);
       uh=np.min(h); bh=np.max(h);
       lw=np.min(w); rw=np.max(w);
       cx=np.round(lw+(rw-lw)/2); cy=np.round(uh+(bh-uh)/2)
       area=np.sum(labels[labels==i])
       if area>=min_area:
           centroids.append((cy,cx,area));
    return centroids
           

def saveLinesShpFile(lines,filename,gdal_object):
    #https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    print('dummy function for exporting SHP file data')
    multiline = ogr.Geometry(ogr.wkbMultiLineString)
    
    ###
    gt=gdal_object.GetGeoTransform()
    cols = gdal_object.RasterXSize
    rows = gdal_object.RasterYSize
    ext=GetExtent(gt,cols,rows) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    #resolution in meters
    #dpx=(ext[3][0]-ext[0][0])/cols
    #dpy=(ext[0][1]-ext[2][1])/rows
    dpx=np.abs(gt[1]);
    dpy=np.abs(gt[5]);
    
    for line in lines:
        lineout = ogr.Geometry(ogr.wkbLineString)
        lineout.AddPoint(ext[0][0]+dpx*line[0][0], ext[0][1]-dpy*line[0][1])
        lineout.AddPoint(ext[0][0]+dpx*line[1][0], ext[0][1]-dpy*line[1][1])
        multiline.AddGeometry(lineout)
    
    multiline=multiline.ExportToWkt()
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(filename)
    layer = ds.CreateLayer('', None, ogr.wkbLineString)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkt(multiline)
    
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


#working function for exporting of shape file of points    
def savePointsShpFile(points,filename,gdal_object):
    gt=gdal_object.GetGeoTransform()
    cols = gdal_object.RasterXSize
    rows = gdal_object.RasterYSize
    ext=GetExtent(gt,cols,rows) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    #resolution in meters
    dpx=(ext[3][0]-ext[0][0])/cols
    dpy=(ext[0][1]-ext[2][1])/rows
    
    
    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(filename):
        shpDriver.DeleteDataSource(filename)
    outDataSource = shpDriver.CreateDataSource(filename)
    outLayer = outDataSource.CreateLayer(filename, geom_type=ogr.wkbPoint )
    # create a field
    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    AreaField = ogr.FieldDefn('Area', ogr.OFTInteger)
    outLayer.CreateField(idField); outLayer.CreateField(AreaField );
    
    #create point geometry
    for i in range(0,len(points[0])):
        
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(ext[0][0]+dpx*points[0][i], ext[0][1]-dpy*points[1][i])

        # Create the feature and set values
        featureDefn = outLayer.GetLayerDefn()
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(point)
        outFeature.SetField('id', i)
        outFeature.SetField('Area', int(points[2][i]))
        outLayer.CreateFeature(outFeature)
        outFeature = None    
    outDataSource= None

def savePointsShpFile2(points,filename,gdal_object): #nofield
    gt=gdal_object.GetGeoTransform()
    cols = gdal_object.RasterXSize
    rows = gdal_object.RasterYSize
    ext=GetExtent(gt,cols,rows) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    #resolution in meters
    dpx=(ext[3][0]-ext[0][0])/cols
    dpy=(ext[0][1]-ext[2][1])/rows
    
    
    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(filename):
        shpDriver.DeleteDataSource(filename)
    outDataSource = shpDriver.CreateDataSource(filename)
    outLayer = outDataSource.CreateLayer(filename, geom_type=ogr.wkbPoint )
    # create a field
    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    AreaField = ogr.FieldDefn('Area', ogr.OFTInteger)
    outLayer.CreateField(idField); 
    outLayer.CreateField(AreaField );
    
    #create point geometry
    for i in range(0,len(points[0])):
        
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(ext[0][0]+dpx*points[0][i], ext[0][1]-dpy*points[1][i])

        # Create the feature and set values
        featureDefn = outLayer.GetLayerDefn()
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(point)
        outFeature.SetField('id', i)
        outFeature.SetField('Area', int(points[2][i]))
        outLayer.CreateFeature(outFeature)
        outFeature = None    
    outDataSource= None

def saveGeoTiff(raster,filename,gdal_object,ColMinInd,RowMinInd): #ColMinInd,RowMinInd - start row/col for cropped images
    meas=np.shape(raster)
    rows=meas[0]; cols=meas[1]; 
    if(len(meas)==3):
        zs=meas[2];
    else:
        zs=1;
    print("Saving "+filename)    
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(filename, cols, rows, zs, gdal.GDT_Float64)
    (start_x,resx,zerox,start_y,zeroy,resy)=gdal_object.GetGeoTransform()
    outdata.SetGeoTransform((start_x+(resx*ColMinInd),resx,zerox,start_y+(resy*RowMinInd),zeroy,resy));
    #outdata.SetGeoTransform(gdal_object.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(gdal_object.GetProjection())##sets same projection as input
    #write bands
    if zs>1:
        for b in range(0,zs):
            outdata.GetRasterBand(b+1).WriteArray(raster[:,:,b])
            outdata.GetRasterBand(b+1).SetNoDataValue(10000) ##if you want these values transparent
    else:
        outdata.GetRasterBand(1).WriteArray(raster) #write single value raster
    outdata.FlushCache() ##saves

def saveGeoTiffNodata(raster,filename,gdal_object,ColMinInd,RowMinInd,BitMode): #ColMinInd,RowMinInd - start row/col for cropped images
    if BitMode=="float64":
        bitres=gdal.GDT_Float64;
    elif BitMode=="int16":
       bitres=gdal.GDT_Int16; 
    else:
       bitres=gdal.GDT_Int8;  
    meas=np.shape(raster)
    rows=meas[0]; cols=meas[1]; 
    if(len(meas)==3):
        zs=meas[2];
    else:
        zs=1;
    print("Saving "+filename)    
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(filename, cols, rows, zs, bitres)
    (start_x,resx,zerox,start_y,zeroy,resy)=gdal_object.GetGeoTransform()
    outdata.SetGeoTransform((start_x+(resx*ColMinInd),resx,zerox,start_y+(resy*RowMinInd),zeroy,resy));
    #outdata.SetGeoTransform(gdal_object.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(gdal_object.GetProjection())##sets same projection as input
    #write bands
    if zs>1:
        for b in range(0,zs):
            outdata.GetRasterBand(b+1).WriteArray(raster[:,:,b])
            outdata.GetRasterBand(b+1).SetNoDataValue(-32768) ##if you want these values transparent
    else:
        outdata.GetRasterBand(1).WriteArray(raster) #write single value raster
        outdata.GetRasterBand(1).SetNoDataValue(-32768)
    outdata.FlushCache() ##saves 
    

def classifyPoints(varr,numclasses):
    minval,maxval=np.min(varr),np.max(varr);
    vrange=maxval-minval;
    crange=vrange/numclasses;
    varrnew=np.zeros([varr.shape[0]])    
    for i in range(1,numclasses+1,1):
        ind1=varr>=(minval+crange*(i-1))
        ind2=varr<=(minval+crange*(i))
        ind=np.logical_and(ind1,ind2)
        varrnew[ind]=i 
    return varrnew

def interpolate_relief_dissection(SRTM,winsize,resx,resy):
    [h,w]=np.shape(SRTM);
    wpx=round(winsize/resx);  #разрешение окна в пикс
    wpy=round(winsize/resy);
    winx=round(w/wpx); 
    winy=round(h/wpy);
    xlist=[];
    ylist=[];
    hlist=[]; #списки для хранения координат центров окон и значения параметра 
    
    #3.Выбор подматрицы, нахождения значения параметра
    for i in range(0,(winx),1):
        for j in range(0,(winy),1):
            submatrix=SRTM[(wpy*j):(wpy*(j+1)),(wpx*i):(wpx*(i+1))];
            xlist.append(wpx*i);
            ylist.append(wpy*j);
            hlist.append((np.max(submatrix)-np.min(submatrix)));
            
            #getting rid of the marginal point
            if(i<=(winx-1) and j==(winy-1)):
               xlist.append(wpx*i);
               ylist.append(h);
               submatrix=SRTM[(h-int(wpy/2)):h,(wpx*i):(wpx*(i+1))];
               hlist.append((np.max(submatrix)-np.min(submatrix))); 
       
            if(i==(winx-1) and j<=(winy-1)):
               xlist.append(w);
               ylist.append(wpy*j);
               submatrix=SRTM[(wpy*j):(wpy*(j+1)),(w-int(wpx/2)):w];
               hlist.append((np.max(submatrix)-np.min(submatrix))); 
            
            if(i==(winx-1) and j==(winy-1)):
               xlist.append(w);
               ylist.append(h);
               submatrix=SRTM[(h-int(wpy/2)):h,(w-int(wpx/2)):w];
               hlist.append((np.max(submatrix)-np.min(submatrix))); 
         
    #4. Блок интерполяции
    xx=np.arange(0,w,1);
    yy=np.arange(0,h,1);
    XX, YY = np.meshgrid(xx, yy);
    grid = griddata((xlist,ylist),hlist,(XX,YY),method='cubic');
    return grid;


def get_shp_extent(path_to_file):
    driver = ogr.GetDriverByName('ESRI Shapefile');
    data_source = driver.Open(path_to_file,0); # 0 means read-only. 1 means writeable.
    layer = data_source.GetLayer();
    x_min, x_max, y_min, y_max = layer.GetExtent();
    return x_min, x_max, y_min, y_max;


def check_shp_inside_raster(geotiff_extent,shp_extent):
    #Check out if shp AOI has been included in Geotiff's extent
    #returns true if yes, no otherwise
    print("Checking AOI overlay...")
    
    if (shp_extent[0]>geotiff_extent[0][0]) and \
     (shp_extent[1]<geotiff_extent[3][0]) and \
     (shp_extent[2]>geotiff_extent[2][0]) and\
     (shp_extent[3]<geotiff_extent[3][1]):
         print("Ok")
         return True
    else:
        print("AOI position error?")
        return False

def crop_by_shp(shp_extent,geotiff_extent,dpx,dpy,band_array):
    #cropping indexes
     #shp_ext (ЛX,ПХ,НY,ВY)
    ColMinInd=int((shp_extent[0]-geotiff_extent[0][0])/dpx) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    CropWidth=int((shp_extent[1]-shp_extent[0])/dpx)
    RowMinInd=int((geotiff_extent[3][1]-shp_extent[3])/dpy)
    CropHeight=int((shp_extent[3]-shp_extent[2])/dpy)
    
    #extraction of the sample image
    sampleImage=band_array[RowMinInd:(RowMinInd+CropHeight),ColMinInd:(ColMinInd+CropWidth)]
    return sampleImage,ColMinInd,RowMinInd

def normalize_matrix(matrix): #normalize band ratio
    matrix=(matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix));
    return matrix;

def mynormalize16to8(mat):
    minval=np.min(mat);
    maxval=np.max(mat);
    rangeval=maxval-minval;
    mat8=np.uint8((mat/rangeval)*255);
    return mat8;

#create RGB images
def image_stack(red,green,blue,do_norm8,do_show):
    #do_norm8 - if normalize to uint8 
    #do_show - if show image in console
    
    #stacking image
    if(do_norm8==1):
        red=mynormalize16to8(red)
        green=mynormalize16to8(green)
        blue=mynormalize16to8(blue)
    rgb=np.dstack([red,green,blue]);
    if (do_show==1):
        plt.figure()
        if do_norm8==1:
            plt.imshow(rgb)
        else:
            plt.imshow(np.dstack([mynormalize16to8(red),mynormalize16to8(green),mynormalize16to8(blue)]))
        plt.show()
    return rgb

#principal component analysis
#https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
#more on this decomposition parameters
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html    

def pca(X,n_components,m,n): #correct we export components loadings instead of components
  """  Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance
    and mean."""

  #from previously run batch image processing
  #pca = PCA(n_components=n_components, svd_solver='randomized',
  #        whiten=True).fit_transform(X)
  pca=PCA(n_components=n_components);
  mean_X = X.mean(axis=0)
  var_X=np.var(X,axis=0)
  #eigenvalues = pca.components_.reshape((n_components,m,n)) #according to the number of components 
  eigenvalues = pca.fit(X).components_.reshape(n_components,m,n);
  # return the PCA matrix [num,:,:], the variance and the mean
  return pca,eigenvalues,var_X,mean_X
    
def mat4pca(imlist): 
    #первый столбец - минерал, второй - растительность
    #create matrix to store all flattened images
    immatrix = np.array([np.array(im.flatten())
              for im in imlist],'f') #what is the meaning of the 'f' key?
    return immatrix.T;  #rows for values, cols for variables CORRECTED in 06.02.2021!

def get_pca_image(pca_data,m,n,n_components=1):
    #function to show PCA images
    images={};
    for i in range(0,n_components,1):
        try:
            image=np.array(pca_data[:,i].reshape(m,n));
            images.update({str(i+1):image});
        except:
            print('Can not get {} component. Number of components exceeded'.format(i));
        #image=(np.array(image)-np.min(image))/(np.max(image)-np.min(image));
        
    return images;


#https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
def get_comp_loadings2(pca_model,features,columns):
    loadings = pd.DataFrame(pca_model.components_.T, columns=columns,\
                        index=features)
    eigenvalues=pca_model.explained_variance_; #eigenvalues of the components
    eigenvalues_procents=(eigenvalues/sum(eigenvalues))*100;
        
    row = pd.Series({columns[0]:eigenvalues_procents[0],columns[1]:eigenvalues_procents[1]},name='Eigenvalues')
    loadings = loadings.append(row)
    return loadings;

#deprecated function below, consider to remove
def get_comp_loadings(pca_model,model_data,features,columns):
    model_transform=pca_model.transform(model_data); #data projection to PCA axis
    loadings=pd.DataFrame(model_transform,columns=columns,index=features);
    return loadings;
    #loadings_iron_oxides = pd.DataFrame(pca_iron_oxides.transform(flattened_img_matrix1),\
    #                                 columns=['DPC1', 'DPC2'],index=['Band 4 : Band 2','Band 5 : Band 4']);

def get_image_classes(image):
    m,n=image.shape;
    km = KMeans(n_clusters=10)  #10 classes
    km.fit(image.flatten().reshape(-1,1))  #reshape(-1,1) - transpose of the single-dimension array
    km.predict(image.flatten().reshape(-1,1))
    image_classes = km.labels_.reshape(m,n);
    return image_classes;
      

def show_pca_images(eigenv,immean,m,n,savefile): #m,n - size of the images
    #showing mean image and 7 first modes
    pca_fig=plt.figure()
    plt.gray()
    plt.title("PCA components for the image")    
    plt.subplot(2,4,1)
    plt.gca().axes.xaxis.set_ticklabels([]);
    plt.gca().axes.yaxis.set_ticklabels([]);
    plt.title("Mean value")
    plt.imshow(immean.reshape(m,n))
    for i in range(0,7): #SWAP 7 FOR NUMBER OF COMPONENTS!!!! DO LAYOUTING DYNAMICALLY
        plt.subplot(2,4,i+2)
        #V[i][np.isnan(V[i])]=0; #turn nans into 0 "no values"
        plt.imshow(eigenv[i,:,:])
        plt.gca().axes.xaxis.set_ticklabels([]);
        plt.gca().axes.yaxis.set_ticklabels([]);
        plt.title("PC "+str(i+1))    
    plt.savefig(savefile,dpi=300)
    plt.show()
    #return pca_fig;


def show_pca_cumsum(pca,savefile):
    #determening necessary number of components
    cumsum=np.cumsum(pca.explained_variance_ratio_);
    fig_cumsum=plt.figure;    
    plt.plot(cumsum)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig(savefile,dpi=300)
    plt.show()
    
    for i in range(len(cumsum)):
        if cumsum[i]>0.99:
            print("{} components (PCA) should be enough to cover data variance".format(i+1))
            break;
    #return i,fig_cumsum;

def save_pca_variance(pca,savefile):
    vars=pca.explained_variance_ratio_;
    comps_nums=list(range(1,len(vars)+1));
    data=pd.DataFrame({'number':comps_nums,'variance':vars});
    data.to_excel(savefile,index=False);

def save_landsat_bands_stat(bands,savefile):
    out_dict={'':['Minimun','Maximum','Mean','Median','Std.dev.']}
    for key,value in bands.items():
        minval=np.min(value);maxval=np.max(value);
        meanval=np.mean(value);medianval=np.median(value);
        stdval=np.std(value);
        out_dict.update({key:[minval,maxval,meanval,medianval,stdval]});
    df = pd.DataFrame(out_dict)
    df.to_excel(savefile,index=False)
    
def save_landsat_mutual_cor(bands,savefile):
    key_list=list(bands.keys())
    out_dict={' ':key_list}
    #print(out_dict)
    for i in range(0,len(key_list)):
        cur_band_name=key_list[i];
        cur_band_value=np.array(list(bands.values())[i]).flatten();
        cur_band_correlations=[];
        for ii in range(0,len(key_list)):
            next_band_name=key_list[ii];
            next_band_value=np.array(list(bands.values())[ii]).flatten();
            cor_coef_val=np.corrcoef(cur_band_value,next_band_value)[0,1]
            cur_band_correlations.append(cor_coef_val)
        out_dict.update({cur_band_name:cur_band_correlations});
        #print(out_dict)
    df = pd.DataFrame(out_dict)
    df.to_excel(savefile,index=False) 
    
def save_landsat_pca_cov(bands,eigenvalues,savefile):  #eigenvalues = pca components eigenvalues[ii,:,:]
    key_list=list(bands.keys())
    out_dict={' ':key_list}
    for i in range(0,len(eigenvalues[:,1,1])):
        pca_comp_name="PC{}".format(i+1);
        pca_comp_value=eigenvalues[i,:,:].flatten();
        cur_band_cov=[];
        for ii in range(0,len(key_list)):
            #cur_band_name=key_list[ii];
            cur_band_value=np.array(list(bands.values())[ii]).flatten();
            cur_band_cov.append(np.cov(cur_band_value,pca_comp_value)[0,1])
        out_dict.update({pca_comp_name:cur_band_cov});
    df = pd.DataFrame(out_dict)
    df.to_excel(savefile,index=False)    
    
def GetExtentOnGdal(gdal_object):
    gt=gdal_object.GetGeoTransform();
    cols = gdal_object.RasterXSize;
    rows = gdal_object.RasterYSize;
    ext=GetExtent(gt,cols,rows); #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    return ext;

def GetResolutionMeters(gdal_object):
    #get spatial resolution
    gt=gdal_object.GetGeoTransform()
    xsize = gdal_object.RasterXSize
    ysize = gdal_object.RasterYSize #x and y raster size in pixels
    ext=GetExtent(gt,ysize,xsize) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    #resolution in meters
    dpx=(ext[3][0]-ext[0][0])/xsize;
    dpy=(ext[0][1]-ext[2][1])/ysize; #выдает неверный результат!!!!!
    return dpx,dpy;


def getGeotiffParams(gdal_object):
    gt=gdal_object.GetGeoTransform()
    xsize = gdal_object.RasterXSize
    ysize = gdal_object.RasterYSize #x and y raster size in pixels
    ext=GetExtent(gt,ysize,xsize) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    #resolution in meters
    #dpx=(ext[3][0]-ext[0][0])/xsize
    #dpy=(ext[0][1]-ext[2][1])/ysize
    dpx=np.abs(gt[1]);
    dpy=np.abs(gt[5]);
    return gt,xsize,ysize,ext,dpx,dpy

def pic_show(matrix,plot_title):
    plt.figure;
    plt.imshow(matrix);
    plt.title(plot_title);
    plt.colorbar();
    plt.show();

def hist_show(image):
    ax = plt.hist(image.ravel(), bins = 256)
    plt.show()

def plot_show(xrow,yrow,txttitle,txlabel,tylabel):
    plt.figure;
    plt.plot(xrow,yrow,'r+');
    plt.title(txttitle);
    plt.xlabel(txlabel);
    plt.ylabel(tylabel);
    plt.show();    
    
#mutual crop between landsat and srtm
def MutualCropSrtmToLandsat(landsat_gdal_object, srtm_gdal_object,landsat_band_array, srtm_band_array):
    
    ext_srtm=GetExtentOnGdal(srtm_gdal_object);
    ext_landsat=GetExtentOnGdal(landsat_gdal_object);
    
    #print(ext_srtm)
    #print(ext_landsat)  #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    
    #determine resolution landsat
    dpxl,dpyl=GetResolutionMeters(landsat_gdal_object);
    
    #determine resolution srtm
    dpxs,dpys=GetResolutionMeters(srtm_gdal_object);
    
    #determine crop extent and pixel extent crop 
    cropL=[]; cropR=[];
    cropU=[]; cropD=[];
    
    srtm_dxl=0; srtm_dxr=0; srtm_dyu=0; srtm_dyd=0;
    landsat_dxl=0; landsat_dxr=0; landsat_dyu=0; landsat_dyd=0;
    
    if ext_landsat[0][0]>=ext_srtm[0][0]:
        cropL=ext_landsat[0][0];
        srtm_dxl=np.int((ext_landsat[0][0]-ext_srtm[0][0])/dpxs)
    else:
        cropL=ext_srtm[0][0];
        landsat_dxl=np.int((ext_srtm[0][0]-ext_landsat[0][0])/dpxl)
    
    if ext_landsat[0][1]<=ext_srtm[0][1]:
        cropU=ext_landsat[0][1];
        srtm_dyu=np.int((ext_srtm[0][1]-ext_landsat[0][1])/dpxs)
    else:
        cropU=ext_srtm[0][1];
        landsat_dyu=np.int((ext_landsat[0][1]-ext_srtm[0][1])/dpxl)
    
    if ext_landsat[1][1]>=ext_srtm[1][1]:
        cropD=ext_landsat[1][1];
        srtm_dyd=np.int((ext_landsat[1][1]-ext_srtm[1][1])/dpxs);
    else:
        cropD=ext_srtm[1][1];
        landsat_dyd=np.int((ext_srtm[1][1]-ext_landsat[1][1])/dpyl);
        
    if ext_landsat[3][1]<=ext_srtm[3][1]:
        cropR=ext_landsat[3][1];
        srtm_dxr=np.int((ext_srtm[3][1]-ext_landsat[3][1])/dpxs)
    else:
        cropR=ext_srtm[3][1];
        landsat_dyr=np.int((ext_landsat[3][1]-ext_srtm[3][1])/dpyl);
    #
    [hl,wl]=np.shape(landsat_band_array);[hs,ws]=np.shape(srtm_band_array);
    
    landsat_band_cropped=landsat_band_array[landsat_dyu:hl-landsat_dyd,landsat_dxl:wl-landsat_dxr];
    srtm_band_cropped=srtm_band_array[srtm_dyu:hs-srtm_dyd,srtm_dxl:ws-srtm_dxr];
    
    #adjust srtm resulution to landsat8
    [hlc,wlc]=np.shape(landsat_band_cropped);
    srtm_band_cropped=resize(srtm_band_cropped,(hlc,wlc),preserve_range=True,mode="wrap") #it works with scikit-image resize
    
    return landsat_band_cropped,srtm_band_cropped,landsat_dxl,landsat_dyu;    

#create polygon coverage file outputGridfn имя файла грида
def createSHPfileNetwork(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth):
    #was taken from https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html

    # convert sys.argv to float
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridWidth = float(gridWidth)
    gridHeight = float(gridHeight)

    # get rows
    rows = int((ymax-ymin)/gridHeight)
    # get columns
    cols = int((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,geom_type=ogr.wkbPolygon )
    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    for countcols in range(0,cols):
        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0
        
        for countrows in range(0,rows):
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            outFeature = None

            # new envelope for next poly 
            ringYtop = ringYtop - gridHeight  #уменьшение координат верхнего среза
            ringYbottom = ringYbottom - gridHeight #уменьшение координат нижнего среза

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Save and close DataSources
    outDataSource = None        

def createSHPfileNetworkAddData(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth,data_dict):
    #was taken from https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html

    # convert sys.argv to float
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridWidth = float(gridWidth)
    gridHeight = float(gridHeight)

    # get rows
    rows = int((ymax-ymin)/gridHeight)
    # get columns
    cols = int((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,geom_type=ogr.wkbPolygon )
    
    #create attribute fields
    for el in data_dict:
        #print(data_dict[el]);
        if el=='id':
            outLayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger));
        else:
            outLayer.CreateField(ogr.FieldDefn(el, ogr.OFTReal));
  
    
    #feature definition (needed to address attribute able data)
    featureDefn = outLayer.GetLayerDefn()
    
    feature_counter=0;
        
    # create grid cells
    for countcols in range(0,cols):
        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0
        
        for countrows in range(0,rows):
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            #outFeature.SetGeometry(poly)
            #outLayer.CreateFeature(outFeature)
            
            
            # Setting field data
            for el in data_dict:
                outFeature.SetField(el, data_dict[el][feature_counter]);
            
            #creating of feature MUST be AFTER adding the data
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            #
            #outFeature.SetField("id", 0);
            feature_counter=feature_counter+1;
            
            outFeature = None

            # new envelope for next poly 
            ringYtop = ringYtop - gridHeight  #уменьшение координат верхнего среза
            ringYbottom = ringYbottom - gridHeight #уменьшение координат нижнего среза

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Save and close DataSources
    outDataSource = None        

#getting data from averaging windows
def createSHPfromDictionary(outputGridfn,data_dict):
    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,geom_type=ogr.wkbPolygon )
    
    #create attribute fields
    for el in data_dict:
        #print(data_dict[el]);
        if el=='id':
            outLayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger));
        else:
            outLayer.CreateField(ogr.FieldDefn(el, ogr.OFTReal));
  
    
    #feature definition (needed to address attribute able data)
    featureDefn = outLayer.GetLayerDefn()
    
    feature_counter=0;
        
    # create grid cells
    for idx in range(0,len(data_dict[el]),1):
        #data_dict={"id":TAB_id,"X_left":TAB_X_left,"X_right":TAB_X_right,\
        #           "Y_top":TAB_Y_top,"Y_bottom":TAB_Y_bottom,base_filename:TAB_raster_value};
        #data_dict['X_left'][idx]
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_top'][idx])
        ring.AddPoint(data_dict['X_right'][idx], data_dict['Y_top'][idx])
        ring.AddPoint(data_dict['X_right'][idx], data_dict['Y_bottom'][idx])
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_bottom'][idx])
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_top'][idx])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # add new geom to layer
        outFeature = ogr.Feature(featureDefn)
        #outFeature.SetGeometry(poly)
        #outLayer.CreateFeature(outFeature)
        
        # Setting field data
        for el in data_dict:
            outFeature.SetField(el, float(data_dict[el][feature_counter])); #conversion to FLOAT, cause GDAL dislike NUMPY ARRAY
        
        #creating of feature MUST be AFTER adding the data
        outFeature.SetGeometry(poly)
        outLayer.CreateFeature(outFeature)
        #
        #outFeature.SetField("id", 0);
        feature_counter=feature_counter+1;
        
        outFeature = None


    # Save and close DataSources
    outDataSource = None        

#sort data for training or predicting model
def Datasort(data_table): #datatable is pandas dataframe
    column_names=list(data_table);

    #picking up data
    id=[];      #identifiers array
    oredist=[]; #inversed distance to ores
    params={};  #empty dictionary for parameters storage 
    
    #if ('ore' in column_names)==False:
     #   print('ERROR! There is NO ore column in data table!')
    
    for idx in column_names:
        if idx.lower().find('id')!=-1:
            id=list(data_table[idx]);
        if idx.lower().find('ore')!=-1:
            oredist=list(data_table[idx]);
        #gather parameters for correlation
        if idx.lower().find('ha')!=-1 or idx.lower().find('ioa')!=-1 or \
           idx.lower().find('ndvi')!=-1 or idx.lower().find('cm')!=-1 or\
           idx.lower().find('pca')!=-1:
            params.update({idx:list(data_table[idx])});
            #print(list(data_table[idx]))
    
    
    #4 visualize parameters vs ores        
    
    for idx in params:
        xrow=params[idx];
        yrow=oredist;
        txttitle=idx+' vs inversed ore dist';
        txlabel=idx;
        #compute correlation index
        R=np.corrcoef(xrow,yrow);
        print(R);
        #show plot
        plot_show(xrow,yrow,txttitle,txlabel,'Inv Ore Dist');
        
    #5 Select data for the model learning
    #After the book of Jake VanderPlas - Python Data science Handbook...       
    
    #wrap data as scikit-learn demands
    data_names=np.array([*params]) #.transpose()  #names of the datain columns  
    id=np.array(id);
    datain=np.array([np.array(params[idx]) for idx in params]) #.transpose(); #input data for learning, X
    
    #h stack datain and id column for easiest identifying after dividing into train and test sets 
    datain_and_id=np.hstack((id[:,None],datain.transpose())) #stack id's and data horizontally
    
    dataout=np.array(oredist); #output data for learning, Y
    
    return  data_names, datain_and_id, dataout;  

#sort data for training or predicting model
def Datasort2(data_table,show_pictures=1): #datatable is pandas dataframe
    column_names=list(data_table);

    #picking up data
    id=[];      #identifiers array
    oredist=[]; #inversed distance to ores
    params={};  #empty dictionary for parameters storage 
    
    if ('ore' in column_names)==False:
        print('ERROR! There is NO ore column in data table!')
    
    for idx in column_names:
        if idx.lower().find('id')!=-1:
            id=list(data_table[idx]);
        if idx.lower().find('ore')!=-1:
            oredist=list(data_table[idx]);
        #gather parameters for correlation
        if idx.lower().find('ioa')!=-1 or \
           idx.lower().find('ndvi')!=-1 or idx.lower().find('cm')!=-1 or\
           idx.lower().find('pca')!=-1:
            params.update({idx:list(data_table[idx])});
            #print(list(data_table[idx]))
    
    
    #4 visualize parameters vs ores        
    
    for idx in params:
        xrow=params[idx];
        yrow=oredist;
        txttitle=idx+' vs inversed ore dist';
        txlabel=idx;
        #compute correlation index
        R=np.corrcoef(xrow,yrow);
        
        #show plot
        if(show_pictures==1):
            print(R);
            plot_show(xrow,yrow,txttitle,txlabel,'Inv Ore Dist');
        
    #5 Select data for the model learning
    #After the book of Jake VanderPlas - Python Data science Handbook...       
    
    #wrap data as scikit-learn demands
    data_names=np.array([*params]) #.transpose()  #names of the datain columns  
    id=np.array(id);
    datain=np.array([np.array(params[idx]) for idx in params]) #.transpose(); #input data for learning, X
    
    #h stack datain and id column for easiest identifying after dividing into train and test sets 
    datain_and_id=np.hstack((id[:,None],datain.transpose())) #stack id's and data horizontally
    
    dataout=np.array(oredist); #output data for learning, Y
    
    return  data_names, datain_and_id, dataout, params;

def Datasort3(data_table, colname,show_pictures=1): #datatable is pandas dataframe
    column_names=list(data_table);

    #picking up data
    id=[];      #identifiers array
    oredist=[]; #inversed distance to ores
    params={};  #empty dictionary for parameters storage 
    
    if (colname in column_names)==False:
        print('ERROR! There is NO ore column in data table!')
    
    for idx in column_names:
        if idx.lower().find('id')!=-1:
            id=list(data_table[idx]);
        if idx.lower().find('ore')!=-1:
            oredist=list(data_table[idx]);
        #gather parameters for correlation
        if idx.lower().find('ha')!=-1 or idx.lower().find('ioa')!=-1 or \
           idx.lower().find('ndvi')!=-1 or idx.lower().find('cm')!=-1 or\
           idx.lower().find('pca')!=-1:
            params.update({idx:list(data_table[idx])});
            #print(list(data_table[idx]))
    
    
    #4 visualize parameters vs ores        
    
    for idx in params:
        xrow=params[idx];
        yrow=oredist;
        txttitle=idx+' vs inversed ore dist';
        txlabel=idx;
        #compute correlation index
        R=np.corrcoef(xrow,yrow);
        
        #show plot
        if(show_pictures==1):
            print(R);
            plot_show(xrow,yrow,txttitle,txlabel,'Inv Ore Dist');
        
    #5 Select data for the model learning
    #After the book of Jake VanderPlas - Python Data science Handbook...       
    
    #wrap data as scikit-learn demands
    data_names=np.array([*params]) #.transpose()  #names of the datain columns  
    id=np.array(id);
    datain=np.array([np.array(params[idx]) for idx in params]) #.transpose(); #input data for learning, X
    
    #h stack datain and id column for easiest identifying after dividing into train and test sets 
    datain_and_id=np.hstack((id[:,None],datain.transpose())) #stack id's and data horizontally
    
    dataout=np.array(oredist); #output data for learning, Y
    
    return  data_names, datain_and_id, dataout, params;




def mean_shift_sort(img,do_reverse=0):
    #applies MeanShift classification
    #data=np.array([0,1,2,3,45,46,47,100,101,102]).reshape(-1,1);
    shape=np.shape(img);
    data=img.flatten().reshape(-1,1);
    clustering = MeanShift().fit(data)
    
    classes = clustering.labels_
    
    mean_values=np.array([]);
    class_arr=np.arange(0,max(classes)+1);
    #arrange labels according to their values
    for label in class_arr:
        mean_values=np.append(mean_values,np.mean(data[classes==label]));        
    
    #sort values of array classes according to their labels
    if do_reverse==1:
        inds = mean_values[::-1].argsort();    
    else:
        inds = mean_values.argsort();
    new_classes=np.zeros(classes.shape,dtype=np.int8);
    for ind,new_val in zip(inds,range(0,len(inds))):
        new_classes[classes==ind]=new_val;
    
    new_classes=np.reshape(new_classes,shape);
    return new_classes;

def kmeans_sort(img,do_reverse=0,n_clusters=2):
    #applies KMeans classification
    #data=np.array([0,1,2,3,45,46,47,100,101,102]).reshape(-1,1);
    shape=np.shape(img);
    data=img.flatten().reshape(-1,1);
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    
    classes = kmeans.labels_
    
    mean_values=np.array([]);
    class_arr=np.arange(0,max(classes)+1);
    #arrange labels according to their values
    for label in class_arr:
        mean_values=np.append(mean_values,np.mean(data[classes==label]));        
    
    #sort values of array classes according to their labels
    if do_reverse==1:
        inds = mean_values[::-1].argsort();    
    else:
        inds = mean_values.argsort();
    new_classes=np.zeros(classes.shape,dtype=np.int8);
    for ind,new_val in zip(inds,range(0,len(inds))):
        new_classes[classes==ind]=new_val;
    
    new_classes=np.reshape(new_classes,shape);
    return new_classes; 

def parse_MTL(filepath):
    #function for parsing landsat 8 MTL file into python data array
    with open(os.path.join(filepath),'rt') as f:
        txt=f.read();
    
    #split string into lines
    lines=txt.splitlines();
    my_data={};
    for line in lines:
        try:
            param_value=line.split('=');
            try:
                value=float(param_value[1].strip());
            except:
                value=param_value[1].strip().replace('\"','');
            my_data.update({param_value[0].strip():value});
        except:
            print('symbol \"=\" was not found');
        
    f.close(); #close file descriptor
    return my_data; #return parsed data as dictionary 


#0 declaration of point rasterization function
def objects_radius(ic,jc,h,w,rad,pxs,pys):      #ic,jc - локальные координаты точки пикс;
                                        #rad - max radius to point h,w - ширина, высота изображения
    #Пример: расчет Ed (евклидовых расстояний) для точки (ic,jc), pxs,pys - размеры пикселя
    rad_m=rad*np.sqrt(pxs**2+pys**2);
    Ed=np.zeros((h,w),dtype=np.float64);
    for i in range(0,h):
        for j in range(0,w):
            l=np.float64(np.sqrt(((i-ic)*pys)**2+((j-jc)*pxs)**2)); #расстояние до точки (ic,jc) от данной
            if (l<rad_m):
            #if (np.sqrt((i-ic)**2<(rad*pxs/pys))) and (np.sqrt((j-jc)**2<(rad*pys/pxs))):
                Ed[i,j]=1; #object is here
            else:
                Ed[i,j]=0;
    return Ed;

#Draw circle on image
def draw_circle(img,r,c,rad):
    y,x=disk(r, c, rad, shape=None);
    img[y,x]=1;
    return img;



"""
start
Give, two x,y curves this gives intersection points,
autor: Sukhbinder https://github.com/sukhbinder/intersection
5 April 2017
Based on: http://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections
"""


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]

"""
end
Give, two x,y curves this gives intersection points,
autor: Sukhbinder https://github.com/sukhbinder/intersection
"""


####################################
if __name__=='__main__':
    #start=time.time()
   print("GDAL/OGR functions library. Composed by Shevyrev Sergei (http://lefa.geologov.net), 2019. Free to use and share.")
