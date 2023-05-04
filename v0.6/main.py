import sys, os
import time
import xmltodict  # for language file

from PyQt5.QtWidgets import QApplication, QWidget, QPlainTextEdit, QAction, QActionGroup, QStatusBar, QLabel, \
    QPushButton
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QAbstractItemView, \
    QListWidget, QComboBox, QListWidgetItem, QPushButton, QFileDialog, \
    QDialog, QMessageBox, qApp, QSplashScreen, QProgressBar, QSpacerItem, QHBoxLayout,QTextEdit
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QCursor, QPixmap, QIcon
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter, QPrintPreviewDialog

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # for colorbar management
import matplotlib.cm as cm
from matplotlib.colors import LightSource

from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
try:
    import gdal
except ModuleNotFoundError:
    from osgeo import gdal,ogr

# timedate to check
from datetime import date

# import user function
from ENE_Functions import detectFlowNetwork, detectFlowOrders, compute_bs, \
    read_dem_geotiff, enchance_srtm, elevation_to_flow, \
    stream_order2lines, saveLinesShpFile3, ProgressBar,GetExtent

#
# set proper current directory
current_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(current_dir)

uifile_main = "enoe_main.ui"  # Enter file here.
uifile_map = "enoe_browser.ui"  # Enter file here.
uifile_export = "enoe_export.ui"  # Enter file here.

form_main, base_main = uic.loadUiType(uifile_main)
form_map, base_map = uic.loadUiType(uifile_map)
form_export, base_export = uic.loadUiType(uifile_export)


class Main(base_main, form_main):
    resized = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.layout = QVBoxLayout(self)

        # analysis params
        self.accuracy = 'mean'  # 9 steps flood, for the middle checkBox
        # output param
        self.output = 'raster'  # raster rastercontour 3D
        self.interp_method = 'linear'

        # name of language settings file
        self.localization = 'localization.xml'
        self.settings_lang_file = 'language.sav'  # store saved
        self.selected_language = 'eng'  # default language

        # results of analysis
        self.flows = []
        self.flow_orders = []
        self.flow_acc = []
        self.points_out = []
        self.conn_point = []
        self.points_out = []  # intermittent and start points

        #####app variables for DATA storage
        self.gdal_object = []  # object for storing of gdal
        self.rows = []
        self.cols = []
        self.srtm = []  # raster relief
        self.grid = []  # grid
        self.inflated_dem = []  # dem without sinkholes and flats
        self.gdal_object = []  # gdal object
        self.flow_directions = []  # accumulated flows
        self.flow_orders = []  # flow orders raster dataset
        self.base_surfaces_dict = {}  # dictionary for storing base surfaces
        self.base_surfaces_diff_dict = {}  # dictionary for storing base surface differences
        self.xgrid = []  #
        self.ygrid = []
        self.report_dir=[]
        self.report_dir_image_subdir = 'img'
        self.report_file_ext = '.png'
        self.report_file_name='report.html'
        self.report_window=[]

        # create application splashscreen
        self.splash = QSplashScreen(QPixmap("splash.png"))

        # progress bar in status bar
        self.pbar = QProgressBar()
        self.si = QSpacerItem(100, 25)  # buffer objects
        self.pbar.setGeometry(0, 0, 600, 25)

        # statusbar строка состояния
        self._statusbar_label = QLabel('Processing...')
        self.statusbar.addPermanentWidget(self._statusbar_label)
        self.statusbar.addPermanentWidget(self.pbar)
        # hide pbar widgets
        self.pbar.hide()
        self._statusbar_label.hide()

        # self.pbar.setAlignment(QtCore.Qt.AlignCenter)
        # check if application obsolete
        today = str(date.today())
        print(today)

        # actions for buttons
        self.fileOpenBtn.clicked.connect(self.file_open_dialogue)
        self.fileOpenBtn.setToolTip('Open digital relief model (GeoTiff)')
        self.ProcDRMBtn.clicked.connect(self.on_enchance_srtm)
        self.FlowDirBtn.clicked.connect(self.on_flow_dir)
        self.DetectFlowsBtn.clicked.connect(self.on_flow_order)
        self.BaseSurfBtn.clicked.connect(self.on_compute_bs)
        self.BaseDiffBtn.clicked.connect(self.on_compute_bs_diff)
        self.ExportBtn.clicked.connect(self.on_export)
        self.ReportGenBtn.clicked.connect(self.on_report)

        self.statBut.clicked.connect(self.show_stat)
        self.df = pd.DataFrame()  # global dataframe

        # self menubar строка меню
        self.menubar.setNativeMenuBar(False)  # отключаем вывод меню как в операционной системе

        # actions for checkboxes
        self.AccCheckBox_Min.toggled.connect(lambda: self.BoxChecked('min'))
        self.AccCheckBox_Middle.toggled.connect(lambda: self.BoxChecked('middle'))
        self.AccCheckBox_Max.toggled.connect(lambda: self.BoxChecked('max'))

        # output checkboxes
        self.VisCheckBox_rast.toggled.connect(lambda: self.BoxOutputChecked('raster'))
        self.VisCheckBox_cont.toggled.connect(lambda: self.BoxOutputChecked('contour'))
        self.VisCheckBox_3D.toggled.connect(lambda: self.BoxOutputChecked('3D'))

        # interpolation method checkboxes
        self.IntCheckBox_Linear.toggled.connect(lambda: self.BoxIntChecked('linear'))
        self.IntCheckBox_Nearest.toggled.connect(lambda: self.BoxIntChecked('nearest'))
        self.IntCheckBox_Cubic.toggled.connect(lambda: self.BoxIntChecked('cubic'))

        # browser window
        self.map_browser = MapBrowser(parent=self)
        # self.export_browser = exportMap(parent=self)
        self.win = []
        # self.map_browser.show()

        # show browser window on double click
        self.list1.itemDoubleClicked.connect(self.map_browser.show)
        # set fixed size
        #self.setFixedSize(700, 390)

        # try to load and parse language file
        self.language_dict = []
        try:
            with open(self.localization, 'r', encoding='utf-8') as file:
                my_xml = file.read()
            my_dict = xmltodict.parse(my_xml)
            self.language_dict = my_dict['body']
        except:
            QMessageBox.critical(self, 'Error loading language file.',
                                 'Language file could not be loaded!',
                                 QMessageBox.Ok, QMessageBox.Ok)
            self.close()
        self.lang_actions = {}
        # try to load save language file

        # add language group into settings menu
        language_group = QActionGroup(self)

        # check if language settings file exists
        if not os.path.isfile(self.settings_lang_file):
            print('language file doesnt exist! Saving defaults...')
            with open(self.settings_lang_file, 'w') as f:
                f.write(self.selected_language)
        else:
            print('language file exists! loading...')
            with open(self.settings_lang_file, 'r') as f:
                self.selected_language = f.read()
            #reload language
            self.reset_language()

        # populate language menu items
        for key in [*self.language_dict['languages']]:
            print(self.language_dict['languages'][key])
            menu_element = language_group.addAction(self.language_dict['languages'][key])
            self.menuLanguage.addAction(menu_element)
            menu_element.setCheckable(True)
            if key == self.selected_language:  # if this language is selected
                menu_element.setChecked(True)
            # menu_element.triggered.connect(lambda:self.choose_language(key)) #set action for menu element
            self.lang_actions.update({key: menu_element})  # add lang action to menu

        for key in [*self.lang_actions]:
            self.lang_actions[key].triggered.connect(lambda checked, arg=key: self.choose_language(arg))

        # actions for ALL menu items
        self.actionOpen.triggered.connect(self.file_open_dialogue)
        self.actionExit.triggered.connect(self.exit_dialogue)
        self.actionAbout_ENOE.triggered.connect(self.about_dialogue)



    def choose_language(self, lang):
        # saving default language into self.settings_lang_file
        self.selected_language = lang
        print('select language', lang)
        with open(self.settings_lang_file, 'w') as f: #save to disk
            f.write(self.selected_language)
        self.reset_language()

    def reset_language(self):
        #reset all languages in main window
        self.setWindowTitle(self.language_dict['commands']['maint_title'][self.selected_language])
        self.menuFile.setTitle(self.language_dict['commands']['menu_file'][self.selected_language])
        self.actionOpen.setText(self.language_dict['commands']['menu_fileopen'][self.selected_language])
        self.actionExit.setText(self.language_dict['commands']['menu_fileclose'][self.selected_language])
        self.menuLanguage.setTitle(self.language_dict['commands']['menu_language'][self.selected_language])
        self.menuHelp.setTitle(self.language_dict['commands']['menu_help'][self.selected_language])
        self.actionAbout_ENOE.setText(self.language_dict['commands']['menu_about'][self.selected_language])
        self.label_2.setText(self.language_dict['commands']['main_demfilepath'][self.selected_language])
        self.label_3.setText(self.language_dict['commands']['main_rasterlayers'][self.selected_language])
        self.groupBox_3.setTitle(self.language_dict['commands']['main_processing'][self.selected_language])
        self.groupBox_2.setTitle(self.language_dict['commands']['main_topovis'][self.selected_language])
        self.groupBox.setTitle(self.language_dict['commands']['main_flow_detect_accuracy'][self.selected_language])
        self.groupBox_4.setTitle(self.language_dict['commands']['main_interpolation_method'][self.selected_language])
        self.label_4.setText(self.language_dict['commands']['main_lbl_value'][self.selected_language])
        self.labelX.setText(self.language_dict['commands']['main_lbl_xyz'][self.selected_language])
        self.labelY.setText(self.language_dict['commands']['main_lbl_xyz'][self.selected_language])
        self.labelZ.setText(self.language_dict['commands']['main_lbl_xyz'][self.selected_language])
        self.statBut.setText(self.language_dict['commands']['main_btn_stat'][self.selected_language])
        self.fileOpenBtn.setText(self.language_dict['commands']['main_btn_open_drm'][self.selected_language])
        self.ProcDRMBtn.setText(self.language_dict['commands']['main_btn_enchance_drm'][self.selected_language])
        self.FlowDirBtn.setText(self.language_dict['commands']['main_btn_flowdir'][self.selected_language])
        self.DetectFlowsBtn.setText(self.language_dict['commands']['main_btn_detectflows'][self.selected_language])
        self.BaseSurfBtn.setText(self.language_dict['commands']['main_btn_interpbs'][self.selected_language])
        self.BaseDiffBtn.setText(self.language_dict['commands']['main_btn_bsdiff'][self.selected_language])
        self.ExportBtn.setText(self.language_dict['commands']['main_btn_export'][self.selected_language])
        self.ReportGenBtn.setText(self.language_dict['commands']['main_btn_report'][self.selected_language])

        self.VisCheckBox_rast.setText(self.language_dict['commands']['main_chk_rastermap'][self.selected_language])
        self.VisCheckBox_cont.setText(self.language_dict['commands']['main_chk_contourmap'][self.selected_language])
        self.VisCheckBox_3D.setText(self.language_dict['commands']['main_chk_3d'][self.selected_language])
        self.AccCheckBox_Min.setText(self.language_dict['commands']['main_chk_flow_min'][self.selected_language])
        self.AccCheckBox_Middle.setText(self.language_dict['commands']['main_chk_flow_middle'][self.selected_language])
        self.AccCheckBox_Max.setText(self.language_dict['commands']['main_chk_flow_max'][self.selected_language])
        self.IntCheckBox_Linear.setText(self.language_dict['commands']['main_chk_interp_lin'][self.selected_language])
        self.IntCheckBox_Nearest.setText(self.language_dict['commands']['main_chk_interp_near'][self.selected_language])
        self.IntCheckBox_Cubic.setText(self.language_dict['commands']['main_chk_interp_cubic'][self.selected_language])


    def BoxChecked(self, param):
        print('box ' + param + ' was checked')
        if param == 'min' and self.AccCheckBox_Min.isChecked() == True:
            self.AccCheckBox_Middle.setChecked(False)
            self.AccCheckBox_Max.setChecked(False)
            self.accuracy = 'min'
        if param == 'middle' and self.AccCheckBox_Middle.isChecked() == True:
            self.AccCheckBox_Min.setChecked(False)
            self.AccCheckBox_Max.setChecked(False)
            self.accuracy = 'mean'
        if param == 'max' and self.AccCheckBox_Max.isChecked() == True:
            self.AccCheckBox_Min.setChecked(False)
            self.AccCheckBox_Middle.setChecked(False)
            self.accuracy = 'max'

    def BoxIntChecked(self, param):
        print('box ' + param + ' was checked')
        if param == 'linear' and self.IntCheckBox_Linear.isChecked() == True:
            self.IntCheckBox_Nearest.setChecked(False)
            self.IntCheckBox_Cubic.setChecked(False)
            self.interp_method = 'linear'
        if param == 'nearest' and self.IntCheckBox_Nearest.isChecked() == True:
            self.IntCheckBox_Linear.setChecked(False)
            self.IntCheckBox_Cubic.setChecked(False)
            self.interp_method = 'nearest'
        if param == 'cubic' and self.IntCheckBox_Cubic.isChecked() == True:
            self.IntCheckBox_Linear.setChecked(False)
            self.IntCheckBox_Nearest.setChecked(False)
            self.interp_method = 'cubic'

    def BoxOutputChecked(self, param):
        print('box ' + param + ' was checked')
        if param == 'raster' and self.VisCheckBox_rast.isChecked() == True:
            self.VisCheckBox_3D.setChecked(False)
            if self.VisCheckBox_cont.isChecked() == True:
                self.output = 'rastercontour'
                print('rastercontour')
            else:
                self.output = 'raster'
                print('raster')
        elif param == 'raster' and self.VisCheckBox_rast.isChecked() == False:
            if self.VisCheckBox_cont.isChecked() == True:
                self.output = 'contour'
                print('contour')
        if param == 'contour' and self.VisCheckBox_cont.isChecked() == True:
            self.VisCheckBox_3D.setChecked(False)
            if self.VisCheckBox_rast.isChecked() == True:
                self.output = 'rastercontour'
                print('rastercontour')
            else:
                self.output = 'contour'
                print('contour')
        elif param == 'contour' and self.VisCheckBox_cont.isChecked() == False:
            if self.VisCheckBox_rast.isChecked() == True:
                self.output = 'raster'
                print('raster')
        if param == '3D' and self.VisCheckBox_3D.isChecked() == True:
            self.VisCheckBox_rast.setChecked(False)
            self.VisCheckBox_cont.setChecked(False)
            self.output = '3D'
        elif param == '3D' and self.VisCheckBox_3D.isChecked() == False:
            self.VisCheckBox_rast.setChecked(True)
            self.output = 'raster'
        # if nothing lef checked
        if self.VisCheckBox_rast.isChecked() == False and \
                self.VisCheckBox_3D.isChecked() == False and \
                self.VisCheckBox_cont.isChecked() == False:
            self.VisCheckBox_rast.setChecked(True)
            self.output = 'raster'

    # add item to list 1
    def add_list_item(self, txt_li, list1):
        item1 = QListWidgetItem()  # need to copy theese items twice
        item1.setText(txt_li)
        self.list1.addItem(item1)
        self.list1.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list1.itemSelectionChanged.connect(self.on_change1)

    def file_open_dialogue(self):
        msg=self.language_dict['commands']['app_dialog_open'][self.selected_language]
        fileName = QFileDialog.getOpenFileName(self, ("Open File"), '', ("Tiff (*.tif *.tiff )"))
        # self.label.setText("Hello "+self.lineEdit.text())
        if fileName[0] != '':
            self.list1.clear()  # clear all items
            # add to list items
            item1 = QListWidgetItem()  # need to copy theese items twice
            item1.setText('srtm')
            item1.setToolTip(self.language_dict['commands']['map_window_title_srtm'][self.selected_language])
            self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_srtm'][self.selected_language])
            self.list1.addItem(item1)
            self.list1.setSelectionMode(QAbstractItemView.SingleSelection)
            self.list1.itemSelectionChanged.connect(self.on_change1)
            #

            self.filePath.setPlainText(fileName[0])

            self.grid, self.srtm, self.gdal_object = read_dem_geotiff(fileName[0])
            siz = np.shape(self.srtm)
            self.ygrid, self.xgrid = np.mgrid[0:siz[0], 0:siz[1]]

            [self.cols, self.rows] = self.srtm.shape
            self.show_result([self.srtm])
            #self.map_browser.setWindowTitle('Opened digital relief model')
            self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_srtm'][self.selected_language])

            # button operations
            self.ProcDRMBtn.setEnabled(True)
            self.FlowDirBtn.setEnabled(False)
            self.DetectFlowsBtn.setEnabled(False)
            self.BaseSurfBtn.setEnabled(False)
            self.BaseDiffBtn.setEnabled(False)
            self.ExportBtn.setEnabled(False)
            self.ReportGenBtn.setEnabled(False)

            # set position of map browser window
            widget = self.geometry()
            x = widget.x()
            y = widget.y()
            self.map_browser.move(x + int(widget.width() * 0.5), y - int(widget.height() * 0.25))

            self.map_browser.show()
            self.map_browser.raise_()

    def exit_dialogue(self):
        exit_dialog_title=self.language_dict['commands']['app_dialog_exit_title'][self.selected_language]
        exit_dialog_text=self.language_dict['commands']['app_dialog_exit_text'][self.selected_language]
        msgBox = QMessageBox.question(self, exit_dialog_title, exit_dialog_text,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes:
            app.quit()
        else:
            print('do nothing')

    def about_dialogue(self):
        txt_label=self.language_dict['commands']['app_info_label'][self.selected_language]
        txt_info=self.language_dict['commands']['app_info_title'][self.selected_language]

        print(txt_label)
        msgBox = QMessageBox.information(self, 'Info',
                                         txt_label +
                                         '\n http://lefa.geologov.net',\
                                         QMessageBox.Ok, QMessageBox.Ok)

    def on_enchance_srtm(self):
        self.inflated_dem = enchance_srtm(self.grid, self.srtm)
        self.FlowDirBtn.setEnabled(True)
        self.ExportBtn.setEnabled(True)
        self.ReportGenBtn.setEnabled(True)
        # add to list items
        if 'enchanced srtm' not in self.get_list_items_text(self.list1):
            item1 = QListWidgetItem()  # need to copy theese items twice
            item1.setText('enchanced srtm')
            item1.setToolTip(self.language_dict['commands']['map_window_title_encsrtm'][self.selected_language])
            self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_encsrtm'][self.selected_language])
            self.list1.addItem(item1)

    def on_flow_dir(self):
        self.flow_directions, self.flow_acc = elevation_to_flow(self.inflated_dem, self.grid)
        self.show_result([self.flow_directions])
        self.DetectFlowsBtn.setEnabled(True)
        self.ExportBtn.setEnabled(True)
        # add to list items
        if 'flow directions' not in self.get_list_items_text(self.list1):
            item1 = QListWidgetItem()  # need to copy theese items twice
            item1.setText('flow directions')
            item1.setToolTip(self.language_dict['commands']['map_window_title_flowdir'][self.selected_language])
            #self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_flowdir'][self.selected_language])
            self.list1.addItem(item1)
        if 'flow accumulation' not in self.get_list_items_text(self.list1):
            item1 = QListWidgetItem()  # need to copy theese items twice
            item1.setText('flow accumulation')
            item1.setToolTip(self.language_dict['commands']['map_window_title_flowacc'][self.selected_language])
            #self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_flowdir'][self.selected_language])
            self.list1.addItem(item1)

    def on_flow_order(self):
        self.flow_orders = detectFlowOrders(self.grid, self.flow_directions, self.flow_acc, self.srtm,
                                            accuracy=self.accuracy)
        self.show_result([self.srtm, self.flow_orders])
        self.BaseSurfBtn.setEnabled(True)
        # add to list items if don't exist
        if 'flow orders' not in self.get_list_items_text(self.list1):
            item1 = QListWidgetItem()  # need to copy these items twice
            item1.setText('flow orders')
            item1.setToolTip(self.language_dict['commands']['map_window_title_floworder'][self.selected_language])
            self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_floworder'][self.selected_language])
            self.list1.addItem(item1)

    def detect_flows_event(self):
        app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))  # set cursor
        self.flows = detectFlowNetwork(self.srtm, self.accuracy)
        self._statusbar_label.setText('Flow network is been detecting...')

        self.fileOpenBtn.setEnabled(False)
        self.DetectFlowsBtn.setEnabled(False)
        self.ExportBtn.setEnabled(False)

        self.flow_orders, self.points_out = detectFlowOrders(self.srtm, self.flows)
        self._statusbar_label.setText('Flow orders have been detected.')
        self.show_result([self.srtm, self.flow_orders, self.points_out])
        app.restoreOverrideCursor()  # reset cursor to defaults

        self.fileOpenBtn.setEnabled(True)
        self.DetectFlowsBtn.setEnabled(True)
        self.ExportBtn.setEnabled(True)
        self.ReportGenBtn.setEnabled(True)

        # add to list items
        self.list1.clear()
        item1 = QListWidgetItem()  # need to copy theese items twice
        item1.setText('srtm')
        self.list1.addItem(item1)
        item2 = QListWidgetItem()  # need to copy theese items twice
        item2.setText('points')
        self.list1.addItem(item2)
        item3 = QListWidgetItem()  # need to copy theese items twice
        item3.setText('flows')
        self.list1.addItem(item3)
        self.list1.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list1.setCurrentItem(item3)
        self.list1.itemSelectionChanged.connect(self.on_change1)

    def on_change1(self):
        print('on_change1 was called')
        item_list = [item.text() for item in self.list1.selectedItems()]
        if len(item_list) != 0:
            if (item_list[0] == 'flows'):
                print('want to show flows')
                self.show_result([self.srtm, self.flow_orders])
            elif (item_list[0] == 'points'):
                self.show_result([self.srtm, self.points_out])
            elif (item_list[0] == 'enchanced srtm'):
                #self.map_browser.setWindowTitle('Enchanced SRTM (no sinks and flats)')
                self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_encsrtm'][self.selected_language])

                self.show_result([self.inflated_dem])
            elif (item_list[0] == 'flow directions'):
                #self.map_browser.setWindowTitle('Flow direction pixel map')
                self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_flowdir'][self.selected_language])
                self.show_result([self.flow_directions])
            elif (item_list[0] == 'flow orders'):
                #self.map_browser.setWindowTitle('Detected flow orders')
                self.map_browser.setWindowTitle(self.language_dict['commands']['map_window_title_floworder'][self.selected_language])
                self.show_result([self.srtm, self.flow_orders])
                # self.show_result([self.flow_orders])
            elif ("base" in item_list[0]) and ("diff" not in item_list[0]):
                key = int(item_list[0].split('-')[1])
                txt=self.language_dict['commands']['map_window_base_surface'][self.selected_language]
                txt=txt.replace('#',str(key))
                #self.map_browser.setWindowTitle(f'Base surface of {key} order')
                self.map_browser.setWindowTitle(txt)
                self.show_result([self.base_surfaces_dict[key]])
            elif ("basediff" in item_list[0]):
                key = item_list[0].split('-')[1]
                txt = self.language_dict['commands']['map_window_base_diff'][self.selected_language]
                self.map_browser.setWindowTitle(f'{txt} {key}')
                # print([*self.base_surfaces_diff_dict])
                # print(key)
                self.show_result([self.base_surfaces_diff_dict[key]])
            elif (item_list[0] == 'flow accumulation'):
                self.map_browser.setWindowTitle(
                    self.language_dict['commands']['map_window_title_flowacc'][self.selected_language])
                self.show_result([self.flow_acc])
            else:
                self.map_browser.setWindowTitle('Digital relief model')
                self.show_result([self.srtm])

    def on_compute_bs(self):
        print('BS')
        self.win = ComputeBSWindow(parent=self)
        self.win.setWindowModality(QtCore.Qt.ApplicationModal)
        self.win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.win.move(self.geometry().center() - self.win.rect().center() - QtCore.QPoint(4, 30))
        self.win.show()

    def on_compute_bs_diff(self):
        print('BS Diff')
        self.win = ComputeBSDiffWindow(parent=self)
        self.win.setWindowModality(QtCore.Qt.ApplicationModal)
        self.win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.win.move(self.geometry().center() - self.win.rect().center() - QtCore.QPoint(4, 30))
        self.win.show()

    def on_export(self):
        print('export')
        self.win = exportMap(parent=self)
        self.win.setWindowModality(QtCore.Qt.ApplicationModal)
        self.win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.win.move(self.geometry().center() - self.win.rect().center() - QtCore.QPoint(4, 30))
        self.win.show()

    def on_report(self):
        msg=self.language_dict['commands']['app_dialog_export'][self.selected_language]
        self.report_dir = QFileDialog.getExistingDirectory(self, msg)
        # self.label.setText("Hello "+self.lineEdit.text())
        print(self.report_dir)
        if len(self.report_dir) == 0:
            print('dir was not choosen')
            return
        if len(os.listdir(self.report_dir))!=0:
            print('the directory is not empty')
            msg = self.language_dict['commands']['app_dialog_export_not_empty'][self.selected_language]
            msgBox = QMessageBox.question(self, msg, msg,
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if msgBox == QMessageBox.No:
                return
        print('Staring output procedure')
        pbar_window = ProgressBar()
        layer_list=self.get_list_items_text(self.list1)
        #open report file for output
        report_fn=os.path.join(self.report_dir,self.report_file_name)
        with open(report_fn,'w') as f:
            f.write('<h2>'+self.language_dict['commands']['report_main_title'][self.selected_language]+'</h2>')
            #f.write('<h3>' + self.language_dict['commands']['report_drm_title'][self.selected_language] + '</h3>')
        for i in range(len(layer_list)):
            print(layer_list[i])
            ###start of layer report processing clause
            #check out layers name
            if (layer_list[i] == 'srtm'):
                ititle=self.language_dict['commands']['map_window_title_srtm'][self.selected_language]
                img = self.srtm
                with open(report_fn, 'a') as f:
                    f.write('<h3>' + self.language_dict['commands']['report_drm_title'][self.selected_language] + '</h3>')
                    txt=self.language_dict['commands']['report_drm_text'][self.selected_language]
                    ext=GetExtent(self.gdal_object.GetGeoTransform(), self.cols, self.rows)
                    txt_area=int(abs((ext[0][0]-ext[2][0])*(ext[0][1]-ext[1][1])/1e6)) #km2
                    heightmin=np.min(self.srtm); heightmean=np.mean(self.srtm)
                    heightmax=np.max(self.srtm); txt=txt.replace('#heightmin',str(heightmin))
                    txt=txt.replace('#heightmean',str(int(heightmean)))
                    txt=txt.replace('#heightmax',str(heightmax))
                    txt = txt.replace('#area', str(txt_area))
                    txt='<p>'+txt+'</p>'
                    f.write(txt)
            elif (layer_list[i] == 'enchanced srtm'):
                img = self.inflated_dem
                ititle = self.language_dict['commands']['map_window_title_encsrtm'][self.selected_language]
            elif (layer_list[i] == 'flow directions'):
                img = self.flow_directions
                ititle = self.language_dict['commands']['map_window_title_flowdir'][self.selected_language]
                with open(report_fn, 'a') as f:
                    f.write('<h3>' + self.language_dict['commands']['report_flowdir_title'][self.selected_language] + '</h3>')
                    txt=self.language_dict['commands']['report_flowdir_text'][self.selected_language]
                    txt='<p>'+txt+'</p>'
                    f.write(txt)
            elif (layer_list[i] == 'flow accumulation'):
                img = self.flow_acc
                ititle = self.language_dict['commands']['map_window_title_flowacc'][self.selected_language]
                with open(report_fn, 'a') as f:
                    f.write('<h3>' + self.language_dict['commands']['report_flowacc_title'][self.selected_language] + '</h3>')
                    txt=self.language_dict['commands']['report_flowacc_text'][self.selected_language]
                    txt='<p>'+txt+'</p>'
                    f.write(txt)
            elif (layer_list[i] == 'flow orders'):
                img=self.flow_orders
                ititle = self.language_dict['commands']['report_floworder_title'][self.selected_language]
                with open(report_fn, 'a') as f:
                    f.write('<h3>' + self.language_dict['commands']['report_floworder_title'][self.selected_language] + '</h3>')
                    txt=self.language_dict['commands']['report_floworder_text'][self.selected_language]
                    txt = txt.replace('#order', str(key))
                    txt='<p>'+txt+'</p>'
                    f.write(txt)
            elif ("base" in layer_list[i]) and ("diff" not in layer_list[i]):
                key = int(layer_list[i].split('-')[1])
                img = self.base_surfaces_dict[key]
                ititle = self.language_dict['commands']['map_window_base_surface'][self.selected_language]
                ititle=ititle.replace('#',str(key))
                with open(report_fn, 'a') as f:
                    h3title=self.language_dict['commands']['report_basesurf_title'][self.selected_language]
                    h3title = h3title.replace('#order',str(key))
                    f.write('<h3>' + h3title + '</h3>')
                    txt = self.language_dict['commands']['report_basesurf_text'][self.selected_language]
                    txt=txt.replace('#order',str(key))
                    txt='<p>' + txt + '</p>'
                    f.write(txt)
            elif ("basediff" in layer_list[i]):
                print('diffent surface detected!')
                key = (layer_list[i].split('-')[1])
                img = self.base_surfaces_diff_dict[key]
                ititle = self.language_dict['commands']['map_window_base_diff'][self.selected_language]
                ititle=ititle.replace('#', str(key))
                with open(report_fn, 'a') as f:
                    h3title=self.language_dict['commands']['report_basediffsurf_title'][self.selected_language]
                    h3title=h3title.replace('#order0',key[0])
                    h3title=h3title.replace('#order1',key[1])
                    f.write('<h3>' + h3title + '</h3>')
                    txt = self.language_dict['commands']['report_basediffsurf_text'][self.selected_language]
                    txt = txt.replace('#order', str(key))
                    txt = '<p>' + txt + '</p>'
                    f.write(txt)

            #save plot picture for layer
            file_name = layer_list[i]
            file_name=file_name.replace(' ','_')+self.report_file_ext
            self.create_layer_plot_bgrd(img, str_name=file_name,ititle=ititle)
            with open(report_fn, 'a') as f:
                link = os.path.join(self.report_dir,self.report_dir_image_subdir, file_name)
                img_txt = f'<br><img src="{link}" width="450">'
                f.write(img_txt)
                print(img_txt)
            ###end of layer report processing clause
            pbar_window.doProgress(i, len(layer_list))
        #output reference
        h3title = self.language_dict['commands']['report_reference_title'][self.selected_language]
        ref_text = self.language_dict['commands']['report_reference_text'][self.selected_language]
        ref_text=ref_text.replace('\n','<br>')
        with open(report_fn, 'a') as f:
            f.write('<h3>'+h3title+'</h3>')
            f.write('<p>'+ref_text+'</p>')
        #open report in separate window
        self.report_window=ReportWindow(self)


    def create_layer_plot_bgrd(self,img,str_name='image',ititle='image'):
        #function to create layer plot on background
        #accordin to choosen visualisation plan
        #TODO CHECKING self.output var for building a plot
        if os.path.exists(os.path.join(self.report_dir,self.report_dir_image_subdir))==False:
            os.mkdir(os.path.join(self.report_dir,self.report_dir_image_subdir))

        filename=os.path.join(self.report_dir,self.report_dir_image_subdir,str_name)
        plt.figure()
        #plt.title(str_name)
        ax = plt.gca()
        if self.output == 'contour':
            im = plt.contour(img, cmap="viridis")
            plt.set_aspect('equal', adjustable='box')
            plt.invert_yaxis()
        elif self.output == 'raster':
            im = plt.imshow(img, cmap='terrain', interpolation='nearest')
        elif self.output == 'rastercontour':
            im = plt.imshow(img, cmap='terrain', interpolation='nearest')
            im1 = plt.contour(img, cmap='viridis', interpolation='nearest')
        elif self.output == '3D':
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ls = LightSource(270, 45)
            surf = ax.plot_surface(self.xgrid, self.ygrid, img, cmap=cm.gist_earth,
                                   linewidth=0, antialiased=False)


        plt.title(ititle)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if self.output != '3D':
            plt.colorbar(im, cax=cax,orientation='vertical')
        #plt.colorbar(orientation='vertical')
        plt.savefig(filename,dpi=300)
        plt.close()

    def show_stat(self):
        print('show stat was called')
        try:
            item_list = [item.text() for item in self.list1.selectedItems()]
            if (item_list[0] == 'flow orders'):
                tmp = self.flow_orders
            elif (item_list[0] == 'flow directions'):
                tmp = self.flow_directions
            elif (item_list[0] == 'enchanced srtm'):
                tmp = self.inflated_dem
            else:
                tmp = self.srtm
            txt=self.language_dict['commands']['selected_layer_stat'][self.selected_language]
            txt_min=self.language_dict['commands']['main_chk_flow_min'][self.selected_language]
            txt_mean=self.language_dict['commands']['main_chk_flow_middle'][self.selected_language]
            txt_max=self.language_dict['commands']['main_chk_flow_max'][self.selected_language]
            msgBox = QMessageBox.information(self, txt,
                                             txt_min +':' + str(int(np.min(tmp))) + '\n' + \
                                             txt_mean +':' + str(int(np.mean(tmp))) + '\n' + \
                                             txt_max +':' + str(int(np.max(tmp))) + '\n' + \
                                             'STD:' + str(int(np.std(tmp))), \
                                             QMessageBox.Ok, QMessageBox.Ok)
        except IndexError:
            print('Layer list seems empty')

    def show_result(self, img_arr):  # im_arr in []
        print(self.output)
        self.map_browser.figure.clear()
        ax = self.map_browser.figure.add_subplot(111)
        ax.cla()  # or picture will not be updated
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #
        if len(img_arr) != 1:  # если вывод результата
            im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
            im2 = ax.imshow(img_arr[1], cmap='terrain', alpha=.50, interpolation='nearest')
            self.map_browser.figure.colorbar(im2, ax=ax, orientation='vertical', cax=cax)
            # im3 = ax.imshow(img_arr[2], cmap=plt.cm.viridis, alpha=.95, interpolation='bilinear')
        else:  # если вывод просто картинки
            print('show picture')
            if self.output == 'contour':
                # im1 = ax.contour(img_arr[0], cmap='terrain', interpolation='nearest')
                im1 = ax.contour(img_arr[0], cmap="viridis")
                # levels=list(range(0, 5000, 100)))
                ax.set_aspect('equal', adjustable='box')
                ax.invert_yaxis()
                try:
                    self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)
                except:
                    print('Can not add colorbar to contours')

            elif self.output == 'raster':
                im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
                self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)
            elif self.output == 'rastercontour':
                im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
                im2 = ax.contour(img_arr[0], cmap='viridis', interpolation='nearest')
                self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)
                try:
                    self.map_browser.figure.colorbar(im2, ax=ax, orientation='vertical', cax=cax)
                except:
                    print('Can not add colorbar to contours')
            elif self.output == '3D':
                # determine selected layer
                item_list = [item.text() for item in self.list1.selectedItems()]
                item_text = item_list[0]
                if item_text == 'enchanced srtm' or item_text == 'srtm':
                    ax = self.map_browser.figure.add_subplot(projection="3d")
                    ls = LightSource(270, 45)
                    rgb = ls.shade(img_arr[0], cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
                    surf = ax.plot_surface(self.xgrid, self.ygrid, img_arr[0], rstride=1, cstride=1, facecolors=rgb,
                                           linewidth=0, antialiased=False, shade=False)
                else:
                    print('operation is not supported')
                    msg_title=self.language_dict['commands']['map_window_3d_dialog_title'][self.selected_language]
                    msg_text =self.language_dict['commands']['map_window_3d_dialog_text'][self.selected_language]
                    msgBox = QMessageBox.warning(self, msg_title,
                                                 msg_text,
                                                 QMessageBox.Ok, QMessageBox.Ok)

            else:
                im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
                self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)

        self.map_browser.canvas.draw()
        # self.DetectFlowsBtn.setEnabled(True)
        self.map_browser.show()
        self.map_browser.raise_()

    def export_file(self, title, format, object):
        print('export file bnt was pressed')
        if format == 'tif':
            format_string = "Tiff (*.tif *.tiff )"
        elif format == 'shp':
            format_string = "Shp (*.shp)"
        app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        fileName = QFileDialog.getSaveFileName(self, title, '',
                                               (format_string))
        if format == 'tif':
            if fileName[0] != '':
                driver = gdal.GetDriverByName("GTiff")
                print(self.rows, self.cols)
                outdata = driver.Create(fileName[0], self.rows, self.cols, 1, gdal.GDT_UInt16)
                outdata.SetGeoTransform(self.gdal_object.GetGeoTransform())  ##sets same geotransform as input
                outdata.SetProjection(self.gdal_object.GetProjection())  ##sets same projection as input
                outdata.GetRasterBand(1).WriteArray(object)
                outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
                outdata.FlushCache()  ##saves
            app.restoreOverrideCursor()  # reset cursor to defaults
        elif format == 'shp':
            lines = stream_order2lines(object)
            try:
                saveLinesShpFile3(lines, fileName[0], self.gdal_object)
            except AttributeError:
                print('Error, no filename was provided?')
            app.restoreOverrideCursor()

    def exportFlowOrders(self):
        print('export FLOWS bnt was pressed')
        app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        fileName = QFileDialog.getSaveFileName(self, ("Save flow orders into GeoTiff file..."), '',
                                               ("Tiff (*.tif *.tiff )"))
        # self.flow_orders,self.points_out

        print(self.rows, self.cols)
        print(type(self.rows), type(self.cols))

        if fileName[0] != '':
            driver = gdal.GetDriverByName("GTiff")
            print(self.rows, self.cols)
            outdata = driver.Create(fileName[0], self.rows, self.cols, 1, gdal.GDT_UInt16)
            outdata.SetGeoTransform(self.gdal_object.GetGeoTransform())  ##sets same geotransform as input
            outdata.SetProjection(self.gdal_object.GetProjection())  ##sets same projection as input
            outdata.GetRasterBand(1).WriteArray(self.flow_orders)
            outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
            outdata.FlushCache()  ##saves
        app.restoreOverrideCursor()  # reset cursor to defaults

    def exportPoints(self):
        print('export POINTS bnt was pressed')
        app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        fileName = QFileDialog.getSaveFileName(self, ("Save flow start and intersection points into GeoTiff file..."),
                                               '', ("Tiff (*.tif *.tiff )"))
        # self.flow_orders,self.points_out

        print(self.rows, self.cols)
        print(type(self.rows), type(self.cols))

        if fileName[0] != '':
            driver = gdal.GetDriverByName("GTiff")
            print(self.rows, self.cols)
            outdata = driver.Create(fileName[0], self.rows, self.cols, 1, gdal.GDT_UInt16)
            outdata.SetGeoTransform(self.gdal_object.GetGeoTransform())  ##sets same geotransform as input
            outdata.SetProjection(self.gdal_object.GetProjection())  ##sets same projection as input
            outdata.GetRasterBand(1).WriteArray(self.points_out)
            outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
            outdata.FlushCache()  ##saves
        app.restoreOverrideCursor()  # reset cursor to defaults

    def closeEvent(self, event):
        msg_title = self.language_dict['commands']['app_dialog_exit_title'][self.selected_language]
        msg = self.language_dict['commands']['app_dialog_exit_text'][self.selected_language]
        msgBox = QMessageBox.question(self, msg_title, msg,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes:
            QApplication.quit()
        else:
            print('do nothing')

    # returns list items in text mode
    def get_list_items_text(self, qlist):
        li = []
        for i in range(qlist.count()):
            item_text = qlist.item(i).text()
            li.append(item_text)
        return li

    # show splash screen on process
    def load_data_splash(self, sp):
        for i in range(1, 11):  #
            time.sleep(0.1)  # Что-то загружаем
            sp.showMessage("Processing... {0}%".format(i * 10),
                           QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.black)
            qApp.processEvents()  # Запускаем оборот цикла


class MapBrowser(base_map, form_map):  # map browser
    resized = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        self.setupUi(self)

        self.layout = QVBoxLayout(self)
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.coords = []

        # nav toolbar  https://stackoverflow.com/questions/49057890/matplotlib-navigationtoolbar2-callbacks-e-g-release-zoom-syntax-example
        self.nt = NavigationToolbar(self.canvas, self)  # (1-где разместить, 2-что отследить)
        self.layout.addWidget(self.nt, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(True)
        self.setLayout(self.layout)

        # canvas click
        self.canvas.mpl_connect('button_press_event', self.onClick)
        # resize event
        self.resized.connect(self.onResize)

    def resizeEvent(self, event):
        self.resized.emit()
        return super(MapBrowser, self).resizeEvent(event)

    def onResize(self):
        print('test')
        widget = self.geometry()
        self.scrollArea.setGeometry(0, 0, widget.width(), widget.height())

    def onClick(self, event):  # click on app window
        global ix, iy
        ix, iy = event.xdata, event.ydata

        try:
            # fill window label with values
            self.parent.labelX.setText(str(int(ix)))
            self.parent.labelY.setText(str(int(iy)))
            # if flow orders raster exists, show order, else - srtm
        except TypeError:
            print('Probably click was outside map window?')

        item_list = [item.text() for item in self.parent.list1.selectedItems()]

        # if no items selected select first item
        try:
            if len(self.parent.list1.selectedItems()) == 0:
                self.parent.list1.setCurrentItem(self.parent.list1.item(0))  # set selection to first items
            else:
                if (item_list[0] == 'flow orders'):
                    zvalue = self.parent.flow_orders[int(iy), int(ix)]
                    # print(self.parent.flow_orders[int(iy),int(ix)])
                elif (item_list[0] == 'flow directions'):
                    zvalue = self.parent.flow_directions[int(iy), int(ix)]
                elif (item_list[0] == 'srtm'):
                    zvalue = self.parent.srtm[int(iy), int(ix)]
                    # print(self.parent.srtm[int(iy),int(ix)])
                elif (item_list[0] == 'enchanced srtm'):
                    zvalue = self.parent.inflated_dem[int(iy), int(ix)]
                    print(self.parent.inflated_dem[int(iy), int(ix)])
                elif ('basediff' in item_list[0]):
                    key = (item_list[0].split('-')[1])
                    img = self.parent.base_surfaces_diff_dict[key]
                    zvalue = img[int(iy), int(ix)]
                elif ('base-' in item_list[0]):
                    key = int(item_list[0].split('-')[1])
                    img = self.parent.base_surfaces_dict[key]
                    zvalue = img[int(iy), int(ix)]
                # else:
                #    zvalue=self.parent.points_out[int(iy),int(ix)]
                #    #print(self.parent.points_out[int(iy),int(ix)])
                self.parent.labelZ.setText(str(zvalue))
                self.coords.append((ix, iy))
                if len(self.coords) == 2:
                    return self.parent.coords
        except:
            print('Unknown error, possibly no SRTM was opened')


class ComputeBSWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        txt_title=self.parent.language_dict['commands']['compute_bs_title'][self.parent.selected_language]

        self.setWindowTitle(txt_title)
        self.list1 = QListWidget()
        txt_label=self.parent.language_dict['commands']['compute_bs_label'][self.parent.selected_language]
        self.label1 = QLabel(txt_label)

        self.OKbnt = QPushButton(self.parent.language_dict['commands']['ok_btn'][self.parent.selected_language])
        self.CancelBnt = QPushButton(self.parent.language_dict['commands']['cancel_btn'][self.parent.selected_language])

        # populate list
        self.list1.clear()  # clear all items
        # add to list items
        for i in range(1, np.max(parent.flow_orders) + 1):
            item1 = QListWidgetItem()  # need to copy theese items twice
            item1.setText(f'flow-{i}')
            self.list1.addItem(item1)
            self.list1.setToolTip(self.parent.language_dict['commands']['map_window_title_floworder'][self.parent.selected_language]+f' {i}')
            self.parent.map_browser.setWindowTitle(self.parent.language_dict['commands']['map_window_title_floworder'][self.parent.selected_language]+f' {i}')
        self.list1.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list1.setCurrentItem(self.list1.item(0))  # select first item

        vbox = QVBoxLayout()
        vbox.addWidget(self.list1)
        vbox.addWidget(self.label1)
        vbox.addWidget(self.OKbnt)
        vbox.addWidget(self.CancelBnt)

        self.setLayout(vbox)
        self.resize(200, 50)

        self.CancelBnt.clicked.connect(self.on_cancel)
        self.OKbnt.clicked.connect(self.on_ok)

    def on_cancel(self):
        self.close()

    def on_ok(self):
        item_text = self.list1.selectedItems()[0].text();
        surf_num = int(item_text.split('-')[1])  # number of surface
        txt_question=self.parent.language_dict['commands']['compute_basesurf_dialog'][self.parent.selected_language]
        txt_title = self.parent.language_dict['commands']['compute_basesurf_dialog_title'][self.parent.selected_language]
        msgBox = QMessageBox.question(self, txt_title, txt_question.replace('#',str(surf_num)),
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes:
            # compute surface
            basen = compute_bs(self.parent.flow_orders, self.parent.inflated_dem, surf_num, \
                               interp_method=self.parent.interp_method)
            self.parent.base_surfaces_dict.update({surf_num: basen})
            if f'base-{surf_num}' not in self.parent.get_list_items_text(self.parent.list1):
                # add list item
                self.parent.add_list_item(f'base-{surf_num}', self.parent.list1)
                self.parent.BaseDiffBtn.setEnabled(True);
                self.parent.show_result([self.parent.base_surfaces_dict[surf_num]])
            self.close()
        else:
            self.close()

class ReportWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        txt_title=self.parent.language_dict['commands']['report_main_title'][self.parent.selected_language]
        self.rpt_fn=os.path.join(self.parent.report_dir,self.parent.report_file_name)

        self.setWindowTitle(txt_title)
        self.html_out=QTextEdit('report is not loaded')
        self.html_out.setReadOnly(True)
        self.OKbtn = QPushButton(self.parent.language_dict['commands']['ok_btn'][self.parent.selected_language])
        self.PDFbtn = QPushButton(self.parent.language_dict['commands']['pdf_btn'][self.parent.selected_language])
        self.PRINTbtn = QPushButton(self.parent.language_dict['commands']['print_btn'][self.parent.selected_language])

        vbox = QVBoxLayout()
        vbox.addWidget(self.html_out)
        hbox=QHBoxLayout()
        hbox.addWidget(self.PDFbtn)
        hbox.addWidget(self.PRINTbtn)
        hbox.addWidget(self.OKbtn)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.resize(520, 450)

        self.OKbtn.clicked.connect(self.on_ok)
        self.PDFbtn.clicked.connect(self.on_pdf)
        self.PRINTbtn.clicked.connect(self.on_print)

        #load report
        self.load_report()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.show()
        self.activateWindow()


    def load_report(self):
        print(self.rpt_fn)
        with open(self.rpt_fn,'r') as f:
            html=f.read()
        self.html_out.setHtml(html)

    def on_print(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)
        if dialog.exec_() == QPrintDialog.Accepted:
            self.html_out.print_(printer)

    def on_ok(self):
        self.close()

    def on_pdf(self):
        filename = QFileDialog.getSaveFileName(self, 'Save to PDF','', ("Pdf (*.pdf)"))
        if filename:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setPageSize(QPrinter.A4)
            printer.setColorMode(QPrinter.Color)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(filename[0])
            self.html_out.document().print_(printer)


class ComputeBSDiffWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        txt_title = self.parent.language_dict['commands']['basesurf_diff_window_title'][self.parent.selected_language]
        txt_label1 = self.parent.language_dict['commands']['basesurf_diff_lbl1'][self.parent.selected_language]
        txt_label2 = self.parent.language_dict['commands']['basesurf_diff_lbl2'][self.parent.selected_language]
        txt_yes_btn = self.parent.language_dict['commands']['yes_btn'][self.parent.selected_language]
        txt_no_btn = self.parent.language_dict['commands']['no_btn'][self.parent.selected_language]

        self.setWindowTitle(txt_title)
        self.list1 = QComboBox()
        self.list2 = QComboBox()
        self.label0 = QLabel('DiffBs=BS1-BS2')
        self.label1 = QLabel(txt_label1)
        self.label2 = QLabel(txt_label2)
        self.OKbnt = QPushButton(txt_yes_btn)
        self.CancelBnt = QPushButton(txt_no_btn)

        # populate list
        self.list1.clear()  # clear all items
        # add to list items
        for txt in parent.get_list_items_text(parent.list1):
            if 'base' in txt and not 'diff' in txt:
                self.list1.addItem(txt)
                self.list1.setToolTip(self.parent.language_dict['commands']['map_window_base_surface'][self.parent.selected_language])

        self.list2.clear()  # clear all items
        # add to list items
        for txt in parent.get_list_items_text(parent.list1):
            if 'base' in txt and not 'diff' in txt:
                self.list2.addItem(txt)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label0)
        vbox.addWidget(self.label1)
        vbox.addWidget(self.list1)
        vbox.addWidget(self.label2)
        vbox.addWidget(self.list2)
        vbox.addWidget(self.OKbnt)
        vbox.addWidget(self.CancelBnt)

        self.setLayout(vbox)
        self.resize(200, 50)

        self.CancelBnt.clicked.connect(self.on_cancel)
        self.OKbnt.clicked.connect(self.on_ok)

    def on_cancel(self):
        self.close()

    def on_ok(self):
        item_text = self.list1.currentText()
        # surf_num=int(item_text.split('-')[1]) #number of surface
        txt_title = self.parent.language_dict['commands']['compute_basesurf_diff_dialog_title'][self.parent.selected_language]
        txt_question = self.parent.language_dict['commands']['compute_basesurf_diff_dialog'][self.parent.selected_language]

        msgBox = QMessageBox.question(self, txt_title,
                                      f'{txt_question} {item_text}?',
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes:
            try:
                # self.parent.base_surfaces_dict[key]
                # print(self.list2.currentText())
                key1 = int(self.list1.currentText().split('-')[1])
                base1 = self.parent.base_surfaces_dict[key1]
                key2 = int(self.list2.currentText().split('-')[1])
                base2 = self.parent.base_surfaces_dict[key2]
                base_diff = base1 - base2
                self.parent.base_surfaces_diff_dict.update({str(key1) + str(key2): base_diff})
                print(base_diff)
                if f'basediff-{str(key1) + str(key2)}' not in self.parent.get_list_items_text(self.parent.list1):
                    # add list item
                    self.parent.add_list_item(f'basediff-{str(key1) + str(key2)}', self.parent.list1)
                    # self.parent.BaseDiffBtn.setEnabled(True);
            except KeyError:
                txt_title = self.parent.language_dict['commands']['base_surface_error_dialog_title'][self.parent.selected_language]
                txt_label = self.parent.language_dict['commands']['base_surface_error_dialog_text'][self.parent.selected_language]
                QMessageBox.information(self, txt_title,
                                     txt_label,
                                     QMessageBox.Ok, QMessageBox.Ok)
            self.close()
        else:
            self.close()


class exportMap(base_export, form_export):  # map browser
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setupUi(self)
        self.setWindowTitle("Please, select layer and format to export...")

        # populate list
        self.list1.clear()  # clear all items
        # add to list items
        for txt in parent.get_list_items_text(parent.list1):
            self.list1.addItem(txt)
        # set format accordingly
        self.on_select_layer()  # populate list initially

        self.CancelBnt.clicked.connect(self.on_cancel)
        self.OKbnt.clicked.connect(self.on_ok)
        self.list1.activated.connect(self.on_select_layer)
        self.list2.activated.connect(self.on_select_format)

        #localize interface
        self.groupBox.setTitle(self.parent.language_dict['commands']['formexp_group1'][self.parent.selected_language])
        self.groupBox_2.setTitle(self.parent.language_dict['commands']['formexp_group2'][self.parent.selected_language])
        self.label.setText(self.parent.language_dict['commands']['formexp_label1'][self.parent.selected_language])
        self.label_2.setText(self.parent.language_dict['commands']['formexp_label2'][self.parent.selected_language])
        self.CancelBnt.setText(self.parent.language_dict['commands']['cancel_btn'][self.parent.selected_language])
        self.OKbnt.setText(self.parent.language_dict['commands']['ok_btn'][self.parent.selected_language])

    def on_select_layer(self):
        print('list1 was changed')
        selected_layer = self.list1.currentText()
        # switch layer types and assign formats consequently
        if 'flow orders' in selected_layer:
            output_formats = ['*.shp', '*.geotiff']
        else:
            output_formats = ['*.geotiff']
        # (re)populate list
        self.list2.clear()
        for txt in output_formats:
            self.list2.addItem(txt)

    def on_select_format(self):
        print('on select format')
        selected_format = self.list2.currentText()

    def on_cancel(self):
        self.close()

    def on_ok(self):
        # save output layer name and file to main window attributes
        layer_name = self.list1.currentText()
        if 'shp' in self.list2.currentText():
            format = 'shp'
            # stream_order2lines(self.parent.flow_orders)
        elif 'tif' in self.list2.currentText():
            format = 'tif'
        else:
            format = 'tif'
        # title for export window
        export_title = 'Exporting '
        # get object to be exported
        if (layer_name == 'flows'):
            export_obj = self.parent.flow_orders
        elif (layer_name == 'enchanced srtm'):
            export_title = export_title + 'enchanced SRTM (no sinks and flats)'
            export_obj = self.parent.inflated_dem
        elif (layer_name == 'flow directions'):
            export_title = export_title + 'flow direction pixel map'
            export_obj = self.parent.flow_directions
        elif (layer_name == 'flow orders'):
            export_title = export_title + 'detected flow orders'
            export_obj = self.parent.flow_orders
        elif (layer_name == 'flow accumulation'):
            export_title = export_title + 'detected flow orders'
            export_obj = self.parent.flow_acc
        elif "base-" in layer_name:
            key = int(layer_name.split('-')[1])
            export_title = export_title + f'base surface of {key} order'
            export_obj = self.parent.base_surfaces_dict[key]
        elif "basediff" in layer_name:
            key = layer_name.split('-')[1]
            export_title = export_title + f'base difference surface {key}'
            export_obj = self.parent.base_surfaces_diff_dict[key]

        # export file with dialogue window
        self.parent.export_file(export_title, format, export_obj)
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'ENE_logo_128_WoText2.png')
    app.setWindowIcon(QIcon(path))
    ex = Main()
    ex.show()
    sys.exit(app.exec())
