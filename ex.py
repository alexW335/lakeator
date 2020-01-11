import sys
import numpy as np
import lakeator

from copy import copy

# from PyQt5 import QtGui
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.pyplot import colormaps
# from matplotlib.cm import datad


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.saveGisAction.triggered.connect(self.file_open)
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
        setArrayDesign.triggered.connect(self.file_open)

        setGPSCoords = QtWidgets.QAction("&Set GPS Coordinates", self)
        setGPSCoords.setShortcut("Ctrl+C")
        setGPSCoords.setStatusTip("Set the GPS coordinates for the array, and ESPG code for the CRS.")
        setGPSCoords.triggered.connect(self.file_open)

        arrayMenu = mainMenu.addMenu("&Array")
        arrayMenu.addAction(setArrayDesign)
        arrayMenu.addAction(setGPSCoords)

        setDomain = QtWidgets.QAction("&Set Heatmap Domain", self)
        setDomain.setShortcut("Ctrl+D")
        setDomain.setStatusTip("Configure distances left/right up/down at which to generate the heatmap.")
        setDomain.triggered.connect(self.file_open)

        changeMethod = QtWidgets.QAction("&Change/Configure Method", self)
        changeMethod.setShortcut("Ctrl+M")
        changeMethod.setStatusTip("Switch between and configure the AF-MUSIC and GCC algorithms.")
        changeMethod.triggered.connect(self.file_open)

        

        self.refreshHeatmap = QtWidgets.QAction("&Calculate", self)
        self.refreshHeatmap.setShortcut("Ctrl+H")
        self.refreshHeatmap.setStatusTip("(Re)calculate heatmap.")
        self.refreshHeatmap.triggered.connect(self.generate_heatmap)
        self.refreshHeatmap.setDisabled(True)

        heatmapMenu = mainMenu.addMenu("&Heatmap")
        heatmapMenu.addAction(setDomain)
        heatmapMenu.addAction(changeMethod)

        # Set the default colormap
        self.colormap = "bone"
        # colmenu = heatmapMenu.addMenu("Colour")
        # cols = list(datad.keys()).copy()
        # sorted(cols, key=str.casefold)
        # col = {}
        # for c in cols:
        #     pt = copy(c)
        #     col["{}".format(pt)] = "def activate_field_{0}(): self.colormap={0}".format(pt)
        #     print(self.colormap)

        # for key, value in col.items():
        #     colmenu.addAction(key, lambda: exec(value))

        self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.static_canvas)
        navbar = NavigationToolbar(self.static_canvas, self)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, navbar)
        navbar._actions['save_figure'].disconnect()
        navbar._actions['save_figure'].triggered.connect(self.save_display)

        self.img = None

        self.colMenu = heatmapMenu.addMenu("&Choose colour map")
        self.colMenu.setDisabled(True)
        colGroup = QtWidgets.QActionGroup(self)
        for colour in sorted(colormaps(), key=str.casefold):
            cm = self.colMenu.addAction(colour)
            cm.setCheckable(True)
            if colour==self.colormap:
                cm.setChecked(True)
            receiver = lambda checked, cmap=colour: self.img.set_cmap(cmap)
            cm.triggered.connect(receiver)
            cm.triggered.connect(self.static_canvas.draw)
            colGroup.addAction(cm)

        # self.invertcm = specMenu.addAction("Invert colour map",self.invertColourMap)
        # self.invertcm.setCheckable(True)
        # self.invertcm.setChecked(self.config['invertColourMap'])
        
        heatmapMenu.addSeparator()
        heatmapMenu.addAction(self.refreshHeatmap)

        

        # print(navbar._actions['save_figure'])

        # Display a "ready" message
        self.statusBar().showMessage('Ready')

        # Set some lakeator defaults
        # Set default heatmap extent
        self._hm_xrange = [-50, 50]
        self._hm_yrange = [-50, 50]

        

        # Initialise the axis on the canvas
        self._static_ax = self.static_canvas.figure.subplots()
        cid = self.static_canvas.mpl_connect('draw_event', self.ondraw)

        self.last_zoomed = [self._hm_xrange[:], self._hm_yrange[:]]

        # Set the default algorithm
        self.algo = "GCC"

        # Set the default EPSG code
        self._EPSG = 4326 

        # Boolean to keep track of whether we have GPS information for the array
        self._has_GPS = False

        # Keep track of the currently opened file
        self.open_filename = ""

        # Initialise the lakeator object
        self.get_mic_locs()
        self.loc = lakeator.Lakeator(self._micLocs)

    def ondraw(self, event):
        self.last_zoomed = [self._static_ax.get_xlim(), self._static_ax.get_ylim()]
        return

    def _setcol(self, c):
        self.colormap = c

    def generate_heatmap(self):
        self.statusBar().showMessage('Calculating heatmap...')
        # Finish this. Setting it up so that you can zoom and then hit "recalculate"
        # dom = self.loc.estimate_DOA_heatmap(self.algo, xrange=self._hm_xrange, yrange=self._hm_yrange, no_fig=True)
        dom = self.loc.estimate_DOA_heatmap(self.algo, xrange=self.last_zoomed[0], yrange=self.last_zoomed[1], no_fig=True)
        
        self.img = self._static_ax.imshow(dom, cmap=self.colormap, interpolation='none', origin='lower',
                   extent=[self._hm_xrange[0], self._hm_xrange[1], self._hm_yrange[0], self._hm_yrange[1]])
        print(type(self.img))
        self._static_ax.set_xlabel("Horiz. Dist. from Center of Array [m]")
        self._static_ax.set_ylabel("Vert. Dist. from Center of Array [m]")
        self._static_ax.set_title("{}-based Source Location Estimate".format(self.algo))

        self.static_canvas.figure.colorbar(self.img)
        self.static_canvas.draw()

        # Once there's an image being displayed, you can save it
        self.saveAction.setDisabled(False)
        if self._has_GPS:
            self.saveGisAction.setDisabled(False)
        self.statusBar().showMessage('Ready.')
        self.colMenu.setDisabled(False)
        return

    def get_mic_locs(self, filePath="arrayconfig.txt"):
        lt = []
        with open(filePath, 'r') as f:
            ltemp = f.readlines()
        for l in ltemp:
            lt.append([float(x) for x in l.strip().split(',')])
        self._micLocs = np.array(lt)

    def file_open(self):
        self.statusBar().showMessage('Loading...')
        name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load .wav file", "./", "Audio *.wav")
        self.loc.load(name)
        self.open_filename = name
        self.refreshHeatmap.setDisabled(False)
        self.statusBar().showMessage('Ready.')
        print(name)
    
    def save_display(self):
        defaultname = self.open_filename[:-4] + "_" + self.algo + "_heatmap.png"
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", defaultname, "PNG files *.png;; All Files *")
        name = name + ".png"
        self.static_canvas.figure.savefig(name)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()