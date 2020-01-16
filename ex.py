import sys
import numpy as np
import lakeator
import Dialogs
import json
from copy import copy

# from PyQt5 import QtGui
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.pyplot import colormaps


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._load_settings()

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QHBoxLayout(self._main)

        self.setWindowTitle('Locator') 
        self.setWindowIcon(QtGui.QIcon("./kiwi.png"))

        loadAction = QtWidgets.QAction("&Load File", self)
        loadAction.setShortcut("Ctrl+L")
        loadAction.setStatusTip("Load a multichannel .wav file.")
        loadAction.triggered.connect(self.file_open)

        self.saveAction = QtWidgets.QAction("&Save Image", self)
        self.saveAction.setShortcut("Ctrl+S")
        self.saveAction.setStatusTip("Save the current display to a PNG file.")
        self.saveAction.triggered.connect(self.save_display)
        self.saveAction.setDisabled(True)

        self.saveGisAction = QtWidgets.QAction("&Save to GIS", self)
        self.saveGisAction.setShortcut("Ctrl+G")
        self.saveGisAction.setStatusTip("Save the heatmap as a QGIS-readable georeferenced TIFF file.")
        self.saveGisAction.triggered.connect(self.exportGIS)
        self.saveGisAction.setDisabled(True)

        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(loadAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.saveGisAction)


        setArrayDesign = QtWidgets.QAction("&Configure Array Design", self)
        setArrayDesign.setShortcut("Ctrl+A")
        setArrayDesign.setStatusTip("Input relative microphone positions and array bearing.")
        setArrayDesign.triggered.connect(self.get_array_info)

        setGPSCoords = QtWidgets.QAction("&Set GPS Coordinates", self)
        setGPSCoords.setShortcut("Ctrl+C")
        setGPSCoords.setStatusTip("Set the GPS coordinates for the array, and ESPG code for the CRS.")
        setGPSCoords.triggered.connect(self.get_GPS_info)

        arrayMenu = mainMenu.addMenu("&Array")
        arrayMenu.addAction(setArrayDesign)
        arrayMenu.addAction(setGPSCoords)

        setDomain = QtWidgets.QAction("&Set Heatmap Domain", self)
        setDomain.setShortcut("Ctrl+D")
        setDomain.setStatusTip("Configure distances left/right up/down at which to generate the heatmap.")
        setDomain.triggered.connect(self.getBoundsInfo)

        # changeMethod = QtWidgets.QAction("&Change/Configure Method", self)
        # changeMethod.setShortcut("Ctrl+M")
        # changeMethod.setStatusTip("Switch between and configure the AF-MUSIC and GCC algorithms.")
        # changeMethod.triggered.connect(self.file_open)
        # changeMethod.setDisabled(True)

        self.refreshHeatmap = QtWidgets.QAction("&Calculate", self)
        self.refreshHeatmap.setShortcut("Ctrl+H")
        self.refreshHeatmap.setStatusTip("(Re)calculate heatmap.")
        self.refreshHeatmap.triggered.connect(self.generate_heatmap)
        self.refreshHeatmap.setDisabled(True)

        self.refreshView = QtWidgets.QAction("&Recalculate on View", self)
        self.refreshView.setShortcut("Ctrl+R")
        self.refreshView.setStatusTip("Recalculate heatmap at current zoom level.")
        self.refreshView.triggered.connect(self.recalculateOnView)
        self.refreshView.setDisabled(True)

        heatmapMenu = mainMenu.addMenu("&Heatmap")
        heatmapMenu.addAction(setDomain)
        # heatmapMenu.addAction(changeMethod)


        # Initialise canvas
        self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.static_canvas)
        
        # Add a navbar
        navbar = NavigationToolbar(self.static_canvas, self)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, navbar)

        # Override the default mpl save functionality to change default filename
        navbar._actions['save_figure'].disconnect()
        navbar._actions['save_figure'].triggered.connect(self.save_display)

        navbar._actions['home'].triggered.connect(lambda: print("testing"))

        self.img = None

        # Dynamically generate menu full of all available colourmaps. Do not add the inverted ones.
        self.colMenu = heatmapMenu.addMenu("&Choose colour map")
        self.colMenu.setDisabled(True)
        colGroup = QtWidgets.QActionGroup(self)
        for colour in sorted(colormaps(), key=str.casefold):
            if colour[-2:] != "_r":
                cm = self.colMenu.addAction(colour)
                cm.setCheckable(True)
                if colour==self.settings["heatmap"]["cmap"][:-2]:
                    cm.setChecked(True)
                receiver = lambda checked, cmap=colour: self.img.set_cmap(cmap)
                cm.triggered.connect(receiver)
                cm.triggered.connect(self._setcol)
                cm.triggered.connect(self.static_canvas.draw)
                colGroup.addAction(cm)
        
        self.invert = QtWidgets.QAction("&Invert colour map", self)
        self.invert.setShortcut("Ctrl+I")
        self.invert.setStatusTip("Invert the current colourmap.")
        self.invert.triggered.connect(self.invert_heatmap)
        self.invert.setCheckable(True)
        self.invert.setDisabled(True)
        heatmapMenu.addAction(self.invert)

        heatmapMenu.addSeparator()
        heatmapMenu.addAction(self.refreshHeatmap)
        heatmapMenu.addAction(self.refreshView)


        algoMenu = mainMenu.addMenu("Algorithm")
        self.algChoice = algoMenu.addMenu("&Change Algorithm")
        algGroup = QtWidgets.QActionGroup(self)
        for alg in sorted(["GCC", "MUSIC", "AF-MUSIC"], key=str.casefold):
            cm = self.algChoice.addAction(alg)
            cm.setCheckable(True)
            if alg==self.settings["algorithm"]["current"]:
                cm.setChecked(True)
            receiver = lambda checked, al=alg: self.setAlg(al)
            cm.triggered.connect(receiver)
            # cm.triggered.connect(self.static_canvas.draw)
            colGroup.addAction(cm)

        self.params = QtWidgets.QAction("&Algorithm Settings", self)
        self.params.setStatusTip("Alter algorithm-specific settings.")
        self.params.triggered.connect(self.getAlgoInfo)
        algoMenu.addAction(self.params)
        
        # Display a "ready" message
        self.statusBar().showMessage('Ready')

        # Boolean to keep track of whether we have GPS information for the array, and an image
        self._has_GPS = False
        self._has_heatmap = False

        # Keep track of the currently opened file
        self.open_filename = ""
        
        self.loc = lakeator.Lakeator(self.settings["array"]["mic_locations"])
    
    def setAlg(self, alg):
        self.settings["algorithm"]["current"] = alg
        self._save_settings()
    
    def ondraw(self, event):
        """Return the new axis limits when the screen is resized"""
        if self._has_heatmap and (self.settings["heatmap"]["xlim"][0] != self._static_ax.get_xlim()[0] or \
            self.settings["heatmap"]["xlim"][1] != self._static_ax.get_xlim()[1] or \
            self.settings["heatmap"]["ylim"][0] != self._static_ax.get_ylim()[0] or \
            self.settings["heatmap"]["ylim"][1] != self._static_ax.get_ylim()[1]):
            # print("zoom changed: ", self.settings["heatmap"]["xlim"][0], self._static_ax.get_xlim()[0]) 
            self.refreshView.setDisabled(False)
        self.last_zoomed = [self._static_ax.get_xlim(), self._static_ax.get_ylim()]
        return
    
    def recalculateOnView(self):
        if hasattr(self, "last_zoomed"):
            self.settings["heatmap"]["xlim"] = self.last_zoomed[0]
            self.settings["heatmap"]["ylim"] = self.last_zoomed[1]
            self._save_settings()
            self.generate_heatmap()


    def invert_heatmap(self):
        """Adds or removes _r to the current colourmap before setting it and redrawing the canvas"""
        if self.settings["heatmap"]["cmap"][-2:] == "_r":
            self.settings["heatmap"]["cmap"] = self.settings["heatmap"]["cmap"][:-2]
            self.img.set_cmap(self.settings["heatmap"]["cmap"])
            self.static_canvas.draw()
        else:
            try:
                self.img.set_cmap(self.settings["heatmap"]["cmap"] + "_r")
                self.settings["heatmap"]["cmap"] = self.settings["heatmap"]["cmap"] + "_r"
                self.static_canvas.draw()
            except ValueError as inst:
                print(type(inst), inst)
        self._save_settings()

    def _setcol(self, c):
        """Sets the colourmap attribut to the name of the cmap - needed as I'm using strings to set the cmaps rather than cmap objects"""
        self.settings["heatmap"]["cmap"] = self.img.get_cmap().name
        self._save_settings()

    def generate_heatmap(self):
        """Calculate and draw the heatmap"""
        # Initialise the axis on the canvas, refresh the screen
        self.static_canvas.figure.clf()
        self._static_ax = self.static_canvas.figure.subplots()

        cid = self.static_canvas.mpl_connect('draw_event', self.ondraw)

        # Show a loading message while the user waits
        self.statusBar().showMessage('Calculating heatmap...')
        # dom = self.loc.estimate_DOA_heatmap(self.settings["algorithm"]["current"], xrange=self.last_zoomed[0], yrange=self.last_zoomed[1], no_fig=True)
        
        dom = self.loc.estimate_DOA_heatmap(self.settings["algorithm"]["current"], xrange=self.settings["heatmap"]["xlim"], 
                                            yrange=self.settings["heatmap"]["ylim"], no_fig=True, freq=self.settings["algorithm"]["MUSIC"]["freq"], 
                                            AF_freqs=(self.settings["algorithm"]["AF-MUSIC"]["f_min"], self.settings["algorithm"]["AF-MUSIC"]["f_max"]), 
                                            f_0=self.settings["algorithm"]["AF-MUSIC"]["f_0"])

        # Show the image and set axis labels & title      
        self.img = self._static_ax.imshow(dom, cmap=self.settings["heatmap"]["cmap"], interpolation='none', origin='lower',
                   extent=[self.settings["heatmap"]["xlim"][0], self.settings["heatmap"]["xlim"][1], self.settings["heatmap"]["ylim"][0], self.settings["heatmap"]["ylim"][1]])
        self._static_ax.set_xlabel("Horiz. Dist. from Center of Array [m]")
        self._static_ax.set_ylabel("Vert. Dist. from Center of Array [m]")
        self._static_ax.set_title("{}-based Source Location Estimate".format(self.settings["algorithm"]["current"]))

        # Add a colourbar and redraw the screen
        self.static_canvas.figure.colorbar(self.img)
        self.static_canvas.draw()

        # Once there's an image being displayed, you can save it and change the colours
        self.saveAction.setDisabled(False)
        if self._has_GPS:
            self.saveGisAction.setDisabled(False)
        self.statusBar().showMessage('Ready.')
        self.colMenu.setDisabled(False)
        self.invert.setDisabled(False)
        self._has_heatmap = True
        return

    def file_open(self):
        """Let the user pick a file to open, and then calculate the cross-correlations"""
        self.statusBar().showMessage('Loading...')
        name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load .wav file", "./", "Audio *.wav")
        self.loc.load(name)
        self.open_filename = name
        self.refreshHeatmap.setDisabled(False)
        self.statusBar().showMessage('Ready.')
        return
    
    def save_display(self):
        """Save the heatmap and colourbar with a sensible default filename"""
        defaultname = self.open_filename[:-4] + "_" + self.settings["algorithm"]["current"] + "_heatmap.png"
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", defaultname, "PNG files *.png;; All Files *")
        name = name + ".png"
        self.static_canvas.figure.savefig(name)
        return

    def get_GPS_info(self):
        """Listener for the change GPS menu item"""
        self.setGPSInfoDialog = Dialogs.GPSPopUp(coords=self.settings["array"]["GPS"]["coordinates"], 
                                                EPSG=self.settings["array"]["GPS"]["EPSG"]["input"], 
                                                pEPSG=self.settings["array"]["GPS"]["EPSG"]["projected"], 
                                                tEPSG=self.settings["array"]["GPS"]["EPSG"]["target"],)
        self.setGPSInfoDialog.activate.clicked.connect(self.changeGPSInfo)
        self.setGPSInfoDialog.exec()

    def changeGPSInfo(self):
        """ Listener for the change gps info dialog.
        """
        lat, long, EPSG, projEPSG, targetEPSG = self.setGPSInfoDialog.getValues() 
        self.settings["array"]["GPS"]["EPSG"]["input"] = EPSG
        self.settings["array"]["GPS"]["EPSG"]["projected"] = projEPSG
        self.settings["array"]["GPS"]["EPSG"]["target"] = targetEPSG
        self.settings["array"]["GPS"]["coordinates"] = (lat, long)
        self._save_settings()

        self._has_GPS = True
        if self._has_heatmap:
            self.saveGisAction.setDisabled(False)
        self.setGPSInfoDialog.close()

    def get_array_info(self):
        """Listener for the change array info menu item"""
        self.setMicsInfoDialog = Dialogs.MicPositionPopUp(cur_locs=self.settings["array"]["mic_locations"])
        self.setMicsInfoDialog.activate.clicked.connect(self.changeArrayInfo)
        self.setMicsInfoDialog.exec()

    def changeArrayInfo(self):
        """ Listener change array info dialog.
        """
        miclocs = self.setMicsInfoDialog.getValues()
        self.settings["array"]["mic_locations"] = miclocs
        self._save_settings()
        self.loc = lakeator.Lakeator(self.settings["array"]["mic_locations"])
        self.setMicsInfoDialog.close()

    def getBoundsInfo(self):
        """Listener for the change array info menu item"""
        l, r = self.settings["heatmap"]["xlim"]
        d, u = self.settings["heatmap"]["ylim"]
        self.setBoundsInfoDialog = Dialogs.HeatmapBoundsPopUp(l, r, u, d)
        self.setBoundsInfoDialog.activate.clicked.connect(self.changeBoundsInfo)
        self.setBoundsInfoDialog.exec()

    def changeBoundsInfo(self):
        """ Listener change array info dialog.
        """
        l_new, r_new, u_new, d_new = self.setBoundsInfoDialog.getValues()
        self.settings["heatmap"]["xlim"] = [l_new, r_new]
        self.settings["heatmap"]["ylim"] = [d_new, u_new]
        self._save_settings()
        self.generate_heatmap()
        # self.loc = lakeator.Lakeator(self.settings["array"]["mic_locations"])
        self.setBoundsInfoDialog.close()

    def getAlgoInfo(self):
        """Listener for the algorithm settings menu item"""
        self.setAlgoInfoDialog = Dialogs.AlgorithmSettingsPopUp(self.settings["algorithm"])
        self.setAlgoInfoDialog.activate.clicked.connect(self.changeAlgoInfo)
        self.setAlgoInfoDialog.exec()

    def changeAlgoInfo(self):
        """ Listener for the change algorithm settinsg dialog.
        """
        self.settings["algorithm"] = self.setAlgoInfoDialog.getValues() 
        self._save_settings()
        self.setAlgoInfoDialog.close()

    def exportGIS(self):
        defaultname = self.open_filename[:-4] + "_" + self.settings["algorithm"]["current"] + "_heatmap.tif"
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image & GIS Metadata", defaultname, "TIF files *.tif;; All Files *")
        name = name + ".tif"
        self.loc.heatmap_to_GIS(self.settings["array"]["GPS"]["coordinates"], 
                                self.settings["array"]["GPS"]["EPSG"]["input"], 
                                projected_EPSG=self.settings["array"]["GPS"]["EPSG"]["projected"], 
                                target_EPSG=self.settings["array"]["GPS"]["EPSG"]["target"], 
                                filepath=name)
        return

    def _load_settings(self, settings_file="./settings.txt"):
        with open(settings_file, "r") as f:
            self.settings = json.load(f)
    
    def _save_settings(self, settings_file="./settings.txt"):
        with open(settings_file, "w") as f:
            stngsstr = json.dumps(self.settings, sort_keys=True, indent=4)
            f.write(stngsstr)



# Run the thing
if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()