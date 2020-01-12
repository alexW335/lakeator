import sys
import numpy as np
import lakeator
from copy import copy

# from PyQt5 import QtGui
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.pyplot import colormaps


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

        # Initialise canvas
        self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.static_canvas)
        
        # Add a navbar
        navbar = NavigationToolbar(self.static_canvas, self)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, navbar)

        # Override the default mpl save functionality to change default filename
        navbar._actions['save_figure'].disconnect()
        navbar._actions['save_figure'].triggered.connect(self.save_display)

        self.img = None

        # Dynamically generate menu full of all available colourmaps. Do not add the inverted ones.
        self.colMenu = heatmapMenu.addMenu("&Choose colour map")
        self.colMenu.setDisabled(True)
        colGroup = QtWidgets.QActionGroup(self)
        for colour in sorted(colormaps(), key=str.casefold):
            if colour[-2:] != "_r":
                cm = self.colMenu.addAction(colour)
                cm.setCheckable(True)
                if colour==self.colormap:
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

        # Display a "ready" message
        self.statusBar().showMessage('Ready')

        # Set some lakeator defaults
        # Set default heatmap extent
        self._hm_xrange = [-50, 50]
        self._hm_yrange = [-50, 50]

        # Keep track of the current view window
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
        """Return the new axis limits when the screen is resized"""
        self.last_zoomed = [self._static_ax.get_xlim(), self._static_ax.get_ylim()]
        return

    def invert_heatmap(self):
        """Adds or removes _r to the current colourmap before setting it and redrawing the canvas"""
        print(self.img.get_cmap())
        if self.colormap[-2:] == "_r":
            self.colormap = self.colormap[:-2]
            self.img.set_cmap(self.colormap)
            self.static_canvas.draw()
        else:
            try:
                self.img.set_cmap(self.colormap + "_r")
                self.colormap = self.colormap + "_r"
                self.static_canvas.draw()
            except ValueError as inst:
                print(type(inst), inst)

    def _setcol(self, c):
        """Sets the colourmap attribut to the name of the cmap - needed as I'm using strings to set the cmaps rather than cmap objects"""
        self.colormap = self.img.get_cmap().name

    def generate_heatmap(self):
        """Calculate and draw the heatmap"""
        # Initialise the axis on the canvas, refresh the screen
        self.static_canvas.figure.clf()
        self._static_ax = self.static_canvas.figure.subplots()

        cid = self.static_canvas.mpl_connect('draw_event', self.ondraw)

        # Show a loading message while the user waits
        self.statusBar().showMessage('Calculating heatmap...')
        dom = self.loc.estimate_DOA_heatmap(self.algo, xrange=self.last_zoomed[0], yrange=self.last_zoomed[1], no_fig=True)

        # Show the image and set axis labels & title      
        self.img = self._static_ax.imshow(dom, cmap=self.colormap, interpolation='none', origin='lower',
                   extent=[self.last_zoomed[0][0], self.last_zoomed[0][1], self.last_zoomed[1][0], self.last_zoomed[1][1]])
        print(type(self.img))
        self._static_ax.set_xlabel("Horiz. Dist. from Center of Array [m]")
        self._static_ax.set_ylabel("Vert. Dist. from Center of Array [m]")
        self._static_ax.set_title("{}-based Source Location Estimate".format(self.algo))

        # Add a colourbar and redraw the screen
        self.static_canvas.figure.colorbar(self.img)
        self.static_canvas.draw()

        # Once there's an image being displayed, you can save it and changethe colours
        self.saveAction.setDisabled(False)
        if self._has_GPS:
            self.saveGisAction.setDisabled(False)
        self.statusBar().showMessage('Ready.')
        self.colMenu.setDisabled(False)
        self.invert.setDisabled(False)
        return

    def get_mic_locs(self, filePath="arrayconfig.txt"):
        """Load the microphone locations from disk"""
        lt = []
        with open(filePath, 'r') as f:
            ltemp = f.readlines()
        for l in ltemp:
            lt.append([float(x) for x in l.strip().split(',')])
        self._micLocs = np.array(lt)
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
        defaultname = self.open_filename[:-4] + "_" + self.algo + "_heatmap.png"
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", defaultname, "PNG files *.png;; All Files *")
        name = name + ".png"
        self.static_canvas.figure.savefig(name)
        return

# Run the thing
if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()