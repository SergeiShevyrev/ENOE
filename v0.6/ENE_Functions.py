# -*- coding: utf-8 -*-
"""
Functions set for ENOE 0.6 software
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.ndimage as nd
import skimage.morphology as skm  # scikit-image
import scipy.ndimage.measurements as scim  # scikit-image
import scipy.ndimage.morphology as scimo
import skimage.measure as skms  # label #regionprops
from scipy.interpolate import griddata
from pysheds.grid import Grid
try:
    import gdal,ogr
except ModuleNotFoundError:
    from osgeo import gdal,ogr

# from skimage.measure import label, regionprops

from sklearn.neighbors import \
    NearestNeighbors  # for arranging points into natural order https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
from networkx import from_scipy_sparse_matrix, dfs_preorder_nodes
from pysheds.sview import Raster

# import interfacefeatures for progress bar window
from PyQt5.QtWidgets import QProgressBar, QWidget, QLabel, qApp, \
    QApplication, QDesktopWidget
from PyQt5.QtCore import Qt


class ProgressBar(QWidget):

    def __init__(self):
        super().__init__()

        # creating progress bar
        self.pbar = QProgressBar(self)

        # create label
        self.label1 = QLabel('Processing...', self)
        self.label1.resize(140,10)
        self.label1.move(30, 25)

        # setting its geometry
        self.pbar.setGeometry(30, 40, 200, 25)
        self.pbar_val=0 #initial value
        # creating push button
        # self.btn = QPushButton('Start', self)

        # changing its position
        # self.btn.move(40, 80)

        # adding action to push button
        # self.btn.clicked.connect(self.doAction)

        # setting window geometry
        self.setGeometry(300, 300, 280, 80)

        # setting window action
        self.setWindowTitle("Line vectorization")
        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # set in the center of screen
        sizeObject = QDesktopWidget().screenGeometry(-1)
        # print(" Screen size : "  + str(sizeObject.height()) + "x"  + str(sizeObject.width()))
        self.move(int(sizeObject.width() / 2) - 140, int(sizeObject.height() / 2) - 40)

        # self.pbar.hide()
        # self.pbar.show()
        print('this is progress bar window!')
        # showing all the widgets
        self.show()
        #self.doAction()

    # when button is pressed this method is being called
    def doAction(self):
        # setting for loop to set value of progress bar
        for i in range(101):
            qApp.processEvents()  # обработка событий
            # slowing down the loop
            time.sleep(0.05)
            # setting value to progress bar
            self.pbar.setValue(i)
            # print(self.pbar.value())
        self.close()

    def doProgress(self, cur_val, max_val):
        #qApp.processEvents()  # обработка событий
        time.sleep(0.01)
        pbar_val = int((cur_val / max_val) * 100)
        self.pbar.setValue(pbar_val)
        # set value for label
        self.label1.setText(f'Processing {pbar_val} %')
        qApp.processEvents()

def read_dem_geotiff(fname):
    grid = Grid.from_raster(fname)
    dem = grid.read_raster(fname)
    # plotting
    srtm_gdal_object = gdal.Open(fname)
    return grid, dem, srtm_gdal_object


def enchance_srtm(grid, dem):
    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(dem)
    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)
    return inflated_dem


def elevation_to_flow(inflated_dem, grid):
    # Elevation to flow direction
    # Determine D8 flow directions from DEM
    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    # Compute flow directions
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    # Calculate flow accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)
    return fdir, acc


def detectFlowOrders(grid, fdir, acc, dem, accuracy='mean'):
    if accuracy == 'max':
        threshold = 0
    if accuracy == 'mean':
        threshold = 50
    if accuracy == 'min':
        threshold = 90
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    stream_order = grid.stream_order(fdir, acc > threshold, dirmap=dirmap)
    # remove rivers within sea level
    stream_order[dem == 0] = 0
    return stream_order


def detectFlowNetwork(srtm, accuracy):
    min_area_streams = 50  # analysis parameters

    if accuracy == 'min':
        flood_steps = 9
    if accuracy == 'mean':
        flood_steps = 10
    if accuracy == 'max':
        flood_steps = 11
    # imshape

    r, c = np.shape(srtm)

    # gradual flood

    flows = np.zeros([r, c], dtype=float)
    mins = np.min(srtm)
    maxs = np.max(srtm)
    grad_flood_step = int((maxs - mins) / flood_steps)

    # ВНИМАНИЕ!!!! ДЛЯ ТОЧНОГО ДЕТЕКТИРОВАНИЯ НИКАКОГО РАЗМЫТИЯ
    # for i in range(mins,maxs,100): #
    for i in range(mins, maxs, grad_flood_step):  # flood relief and skeletize it

        thinned = skm.thin(np.int16(srtm <= i))  # надо использовать истончение, а не скелетизацию

        if (i > mins):
            thinned[srtm < (i - 100)] = 0

        flows = flows + thinned
    flows = np.int16(flows > 0)

    # remove orphan streams (area less than 10)
    flows_label = skms.label(np.uint(flows), background=None, return_num=False,
                             connectivity=2)
    for i in range(1, np.max(flows_label), 1):
        if np.sum(flows_label == i) <= min_area_streams:
            flows[flows_label == i] = 0

    # close and thin to remove small holes)

    strel = skm.disk(1)
    flows = skm.closing(flows, strel)
    flows = np.int16(skm.skeletonize_3d(flows))  # need to convert into int8, cause closing returns BOOL

    return flows


# compute base surfaces
def compute_bs(flow_orders, inflated_dem, n,interp_method='linear'):
    # select z values for order n
    (yn, xn) = np.where(flow_orders == n);
    zn = inflated_dem[yn, xn]
    # interpolation method='nearest'
    siz = np.shape(inflated_dem)
    Y, X = np.mgrid[0:siz[0], 0:siz[1]]
    basen = griddata((xn, yn), zn, (X, Y), method=interp_method)
    return basen


def saveLinesShpFile3(lines, filename, gdal_object):

    qApp.processEvents()  # обработка событий
    # https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    print('dummy function for exporting SHP file data')
    # multiline = ogr.Geometry(ogr.wkbMultiLineString)
    gt = gdal_object.GetGeoTransform()
    cols = gdal_object.RasterXSize
    rows = gdal_object.RasterYSize
    ext = GetExtent(gt, cols, rows)  # [[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    # resolution in meters
    dpx = np.abs(gt[1]);
    dpy = np.abs(gt[5]);

    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(filename)
    layer = ds.CreateLayer('', None, ogr.wkbLineString)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('Order', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    id_count = 0
    print('number of lines', len(lines['lines']))
    pbar_window = ProgressBar()

    for line in lines['lines']:
        pbar_window.doProgress(id_count, len(lines['lines']))
        multiline = ogr.Geometry(ogr.wkbMultiLineString)

        if len(line[0]) > 2:
            # rearrange points in streams https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
            x = line[0];
            y = line[1]
            points = np.c_[x, y]

            clf = NearestNeighbors(n_neighbors=2).fit(points)
            G = clf.kneighbors_graph()  # G is a sparse N x N matrix

            T = from_scipy_sparse_matrix(G)  # use networkx to construct a graph from this sparse matrix

            # Find shortest path from source

            paths = [list(dfs_preorder_nodes(T, i)) for i in range(len(points))]
            mindist = np.inf
            minidx = 0

            for i in range(len(points)):
                p = paths[i]  # order of nodes
                ordered = points[p]  # ordered nodes
                # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
                cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
                if cost < mindist:
                    mindist = cost
                    minidx = i
            opt_order = paths[minidx]

            line[0] = x[opt_order]
            line[1] = y[opt_order]

        # end of line reordering

        lineout = ogr.Geometry(ogr.wkbLineString)
        for pntnum in range(len(line[0])):
            lineout.AddPoint(ext[0][0] + dpx * line[0][pntnum], ext[0][1] - dpy * line[1][pntnum])
        multiline.AddGeometry(lineout)

        multiline = multiline.ExportToWkt()

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', id_count)
        feat.SetField('Order', lines['orders'][id_count])
        id_count += 1

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkt(multiline)
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
    pbar_window.close()
    feat = geom = None  # destroy these


def GetExtent(gt, cols, rows):
    """
    srtm_gdal_object.GetGeoTransform()

    (329274.50572846865, - left X
     67.87931651487438,  - dX
     0.0,
     4987329.504699751,  - верх Y
     0.0,
     -92.95187590930819) - dY
    """
    # [[влx,влy],[нлx,нлy],[нпx, нпy],[впx, впy]]
    ext = [[gt[0], gt[3]], [gt[0], (gt[3] + gt[5] * rows)], [(gt[0] + gt[1] * cols), (gt[3] + gt[5] * rows)],
           [(gt[0] + gt[1] * cols), gt[3]]];
    return ext


def stream_order2lines(stream_order):
    #global pbar_window

    pbar_window = ProgressBar()
    lines_dict = {'lines': [], 'orders': []}  # {'line':[],'order':[]}

    for order in range(1, np.max(np.array(stream_order)) + 1):
        pbar_window.doProgress(order,np.max(np.array(stream_order)))
        # iterate through stream order
        label_image = skms.label(stream_order == order)
        for label_num in range(1, np.max(label_image) + 1):
            # print(label_num)
            # get points for selected stream
            if np.sum(label_image == label_num) > 1:
                ypnts, xpnts = np.where(label_image == label_num)
                lines_dict['lines'].append([xpnts, ypnts])
                lines_dict['orders'].append(order)
    pbar_window.close()
    return lines_dict


if __name__ == '__main__':
    print(__doc__)