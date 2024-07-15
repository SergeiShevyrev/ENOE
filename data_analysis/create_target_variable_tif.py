"""
create buffer example
http://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html?highlight=buffer

take resolution from
../geotiffs/base2.tiff

take polyline from
../shp/placers.shp

"""

from osgeo import ogr,gdal
from osgeo import gdalconst
import os
import numpy as np

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

def createBuffer(inputfn, outputBufferfn, bufferDist):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None

def main(inputfn, outputBufferfn, bufferDist):
    createBuffer(inputfn, outputBufferfn, bufferDist)



inputfn = os.path.join('in','placers_ training_utm53.shp')
outputBufferfn = os.path.join('out','testBuffer_training.shp')
bufferDist = 150 #m

main(inputfn, outputBufferfn, bufferDist)

#####растетизация созданного слоя с буферными зонами
#https://gis.stackexchange.com/questions/212795/rasterizing-shapefiles-with-gdal-and-python

ndsm = os.path.join('in','srtm_surroundings_30032017.tif')
shp = os.path.join('out','testBuffer_training.shp')
data = gdal.Open(ndsm, gdalconst.GA_ReadOnly)
geo_transform = data.GetGeoTransform()
#source_layer = data.GetLayer()

#TODO определить знак -pixel_height


x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * data.RasterXSize
y_min = y_max + geo_transform[5] * data.RasterYSize

x_res = data.RasterXSize
y_res = data.RasterYSize
mb_v = ogr.Open(shp)
mb_l = mb_v.GetLayer()
pixel_width = geo_transform[1]
pixel_height = geo_transform[5]
output = os.path.join('out','rasterized_training.tif')
target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -pixel_height))
band = target_ds.GetRasterBand(1)

NoData_value = -999999
band.SetNoDataValue(NoData_value)
band.FlushCache()
#binarization ore presence 0 - no placer, 1 - placer: burn_values=[1]
gdal.RasterizeLayer(target_ds, [1], mb_l,burn_values=[1])

target_ds = None


#####
#растеризация объектов слоя с магматитами
#####https://gis.stackexchange.com/questions/416179/how-to-generate-euclidean-distance-for-each-polygon-in-shapefile
gdal.UseExceptions()

#Create euclidean distance for each polygon and store "Values"
out_raster_template = os.path.join('out','out_{}.tif')
out_proximity_template = os.path.join('out','prox_{}.tif')
shape_file = os.path.join('in',"magmatic_bodies_utm53.shp")

pixel_size = 10
nodata = -9999

#id_field = 'boro_code'
#value_field = 'value'

drv = gdal.GetDriverByName("ESRI Shapefile")

shp_ds = gdal.OpenEx(shape_file, gdal.OF_VECTOR)
lyr = shp_ds.GetLayer()

xmin, xmax, ymin, ymax = lyr.GetExtent()
srs = lyr.GetSpatialRef()

feat_def = lyr.GetLayerDefn()

lyr.ResetReading()
id=0 #feature counter
print('computing proximity for every feature...')
for feat in lyr:
    #id = int(feat.GetField(id_field))
    #val = feat.GetField(value_field)
    id+=1
    val=5553

    tmp_feat = feat.Clone()

    out_raster = out_raster_template.format(id)
    prox_raster = out_proximity_template.format(id)
    tmp_fn = os.path.join('out','tmp.shp')
    tmp_raster = os.path.join('out','tmp.tif')
    tmp_ds = drv.Create(tmp_fn, 0, 0, 0, gdal.GDT_Unknown )
    tmp_lyr = tmp_ds.CreateLayer(tmp_fn, None, feat_def.GetGeomType())
    tmp_lyr.CreateFeature(tmp_feat)
    tmp_feat, tmp_lyr, tmp_ds = None, None, None

    out_ds = gdal.Rasterize(out_raster, tmp_fn,
                            outputType=gdal.GDT_Float32, format='GTIFF', creationOptions=["COMPRESS=DEFLATE"],
                            noData=nodata, initValues=nodata,
                            xRes=pixel_width, yRes=-pixel_height, outputBounds=(x_min, y_min, x_max, y_max), outputSRS=srs,
                            allTouched=True, burnValues=val)

    out_ds = None

    out_ds = gdal.Rasterize(tmp_raster, tmp_fn,
                            outputType=gdal.GDT_Int32, format='GTIFF', creationOptions=["COMPRESS=DEFLATE"],
                            noData=nodata, initValues=nodata,
                            xRes=pixel_width, yRes=-pixel_height, outputBounds=(x_min, y_min, x_max, y_max), outputSRS=srs,
                            allTouched=True, burnValues=id)

    out_ds = None

    gdal.Translate(prox_raster, out_raster, creationOptions=["COMPRESS=DEFLATE"])
    src_ds = gdal.OpenEx(tmp_raster, gdal.OF_RASTER)
    dst_ds = gdal.OpenEx(prox_raster, gdal.OF_UPDATE)

    src_band = src_ds.GetRasterBand(1)
    dst_band = dst_ds.GetRasterBand(1)

    gdal.ComputeProximity(src_band, dst_band, options=[f'VALUES={id}'])

    #add raster to proximity raster list
    #proximity_raster.append(dst_band.ReadAsArray())

    dst_band, src_band, dst_ds, src_ds = None, None, None, None

    drv.Delete(tmp_fn)

#combine intrusions proximity to one layer
proximity_total=os.path.join('out','proximity_intrusions.tif')
raster_data=[]
print('reading rendered files from appropriate directory')
for file in os.listdir('out'):         #exclude 3band tif files
    if file.lower().endswith("."+'tif'.lower()) and file.lower().startswith('prox'):
        gdal_object = gdal.Open(os.path.join('out', file))  # as new gdal_object was created, no more ColMinInd,RowMinInd
        raster_data.append(gdal_object.GetRasterBand(1).ReadAsArray());

r,c = np.shape(raster_data[0])
prox_result=np.zeros([r,c])
print('reading pixel values')
total_pixel_count=r*c
pixel_count=0
for ri in range(0,r):
    for ci in range(0,c):
        pixel_count+=1
        print('progress',int(100*pixel_count/total_pixel_count),'%')
        pix_val_list=[]
        for raster in raster_data:
            pix_val_list.append(raster[ri,ci])
        prox_result[ri,ci]=np.min(pix_val_list)

print('saving results...')
#save resulting proximity raster
saveGeoTiff(prox_result,proximity_total,gdal_object,0,0)

print('Removing temporary files')

for file in os.listdir('out'):         #exclude 3band tif files
    if (file.lower().endswith("."+'tif'.lower()) and file.lower().startswith('prox') and not file==proximity_total) or (file.lower().endswith("."+'tif'.lower()) and file.lower().startswith('out')):
        print('removing ', file)
        os.remove(os.path.join('out',file))
print('Removing temporary files - DONE')

print('done')
