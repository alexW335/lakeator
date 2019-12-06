import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.interpolate import interp1d
from numpy import linalg as la
from sympy.utilities.iterables import multiset_combinations
from scipy.signal import fftconvolve
from multiprocessing import Pool
import scipy.linalg as la
from scipy.linalg import eigh
from scipy import signal
from numpy import dot, sqrt, argsort, abs
import pickle
import tqdm
import zipfile
import xml.etree.ElementTree as ET
from motionless import DecoratedMap, LatLonMarker
import requests
import imageio
import matplotlib.image as mpimg



# """
# This messy code is here because I couldn't get pyfftw installed on my Windows OS, but got it fine on my Linux OS.
# Essentially, I wanted the Locator to use pyfftw when it was available, or numpy's fft module otherwise.
# """
wf = False
pftw = False
try:
    raise ImportError
    import pyfftw.interfaces.numpy_fft as fft_pack
    import pyfftw

    pftw = True
    pyfftw.interfaces.cache.enable()
    try:
        with open('pyfftw_wisdom.txt', 'rb') as wizfile:
            pyfftw.import_wisdom(pickle.load(wizfile))
        wf = True
    except:
        pass
except ImportError as e:
    print(e)
    import numpy.fft as fft_pack


class Lakeator:
    """Used to locate the source of a sound in a multi-track .wav file.

    The lakeator class may also be used to generate simulated data by means of the shift_sound method, which
    takes in a mono wav file and a set of coordinates and produces a multi-track wav simulating the data which
    would have been recorded were the signal to have came from the provided location.
    """
    sample_rate: int = None
    """The sample rate of the currently loaded data."""
    data: np.array = None
    """The current data, stored in a numpy array."""
    sound_speed: float = 343.1
    """The speed of sound in air to use in the calculations."""
    _hm_domain_ = None
    _radial_domain_ = None
    _cor_fns_ = {}
    _mic_pairs_ = None
    _GCC_proc_ = ""

    def __init__(self, mic_locations=((0.325, 0.000), (-0.160, 0.248), (-0.146, -0.258), (-0.001, 0.002)),
                 file_path=None):
        """Initialise the Locator. If you pass in a file path "example.wav" here it will call self.load(example.wav).

        Arguments:
            mic_locations (Mx2 tuple): Matrix of microphone coordinates, in meters, relative to the center of the array.
            file_path (None/string): If present, will call self.load() on the given file path with the default load parameters
        """
        self.mics = np.array(mic_locations)
        self._mic_pairs_ = np.array(
            [p for p in multiset_combinations(np.arange(0, self.mics.shape[0], dtype="int16"), n=2)])
        self.maxdist = np.max(np.linalg.norm(self.mics, axis=1))
        self.spatial_nyquist_freq = self.sound_speed/(2*self.maxdist)

        if file_path:
            self.load(file_path)

    def load(self, file_path, normalise: bool=True, GCC_processor="p-PHAT", do_FFTs=True, filterpairs=False):
        """Loads the data from the .wav file, and computes the inter-channel correlations.

        Correlations are computed, interpolated, and then stored within the lakeator
        object for use in the optimisation function or wherever necessary. Pass in a numpy array of data rather than
        loading from a file by setting raw_data = True.

        Arguments:
            file_path (str): The file path of the WAV file to be read.
            normalise (bool): Normalise the data? This is a good idea, hence the truthy default state.
            GCC_processor (str): Which GCC processor to use. Options are: CC, PHAT, Scot, & RIR. See Table 1 of Knapp, C. et. al. (1976) "The Generalised Correlation Method for Estimation of Time Delay"
            do_FFTs (bool): Calculate the cross-correlations? Worth turning off to save time if only MUSIC-based algorithms are to be used.
            filterpairs (bool): If true, filters out all spectral components above the spatial nyquist frequency for each pair of micrphones.
        """
        global wf, pftw
        self._GCC_proc_ = GCC_processor
        if isinstance(file_path, str):
            self.sample_rate, data = wav.read(file_path)
        else:
            data = file_path[1]
            self.sample_rate = file_path[0]

        # Convert from integer array to floating point to allow for computation
        data = data.astype('float64')

        # Normalise the data
        if normalise:
            for i in range(data.shape[1]):
                data[:, i] -= data[:, i].mean()

        # Store appropriately
        self.data = data

        if do_FFTs:
            temp_pad = np.concatenate(
                (data, np.zeros(((2**(np.ceil(np.log2(data.shape[0])))-data.shape[0]).astype('int32'), data.shape[1]))),
                0)
            c = 1
            for prdx in np.arange(0, self._mic_pairs_.shape[0]):
                pr = self._mic_pairs_[prdx, :]
                self._cor_fns_["{}".format(pr)] = self._create_interp_(self.mics[pr[0], :], self.mics[pr[1], :],
                                                                     temp_pad[:, pr[0]], temp_pad[:, pr[1]],
                                                                          filteraliased=filterpairs)
                c += 1
            if pftw and not wf:
                with open('pyfftw_wisdom.txt', 'wb') as f:
                    pickle.dump(pyfftw.export_wisdom(), f)
                wf = True

    def filter_aliased(self):
        """Filters out all frequencies above the spatial Nyquist frequency for the current array confguration.
        This may be questionable.
        """
        fc = self.spatial_nyquist_freq
        w = fc / (self.sample_rate / 2)
        if w >= 1:
            return
        b, a = signal.butter(5, w, 'low')
        output = signal.filtfilt(b, a, self.data, axis=0)
        self.data = output
        return

    def _whiten_signal_(self):

        for idx in np.arange(self.mics.shape[0]):
            t = np.fft.rfft(self.data[:,idx])
            self.data[:, idx] = np.fft.irfft(t/np.abs(t), n=2*len(t)-1)

    def _create_interp_(self, mic1, mic2, mic1data, mic2data, buffer_percent=-10.0, res_scaling=5, filteraliased=False):
        """This function is to create the cubic interpolants for use in the correlation function. Uses GCC.

        Arguments:
            mic1 (int): The index of the first microphone of interest.
            mic2 (int): The index of the second microphone of interest.
            mic1data (np.array): The data corresponding to the first microphone.
            mic2data (np.array): The data corresponding to the second microphone.
            buffer_percent (float): The percent headroom to give the correlation function to avoid out-of-range exceptions.
            res_scaling (float): Scales the resolution
        """
        if filteraliased:
            fc = self.sound_speed / (2 * np.linalg.norm(mic1 - mic2))
            w = fc / (self.sample_rate / 2)
            if w <= 1:
                b, a = signal.butter(5, w, 'low')

                mic1data = signal.filtfilt(b, a, mic1data, axis=0)
                mic2data = signal.filtfilt(b, a, mic2data, axis=0)

        dlen = len(mic1data)
        num_samples = la.norm(mic1-mic2)*(1+buffer_percent/100.0)*(1/self.sound_speed)*self.sample_rate
        num_samples = int(round(num_samples))
        if buffer_percent < 0:
            num_samples = dlen-1

        n = 2*dlen
        X1 = fft_pack.rfft(mic1data, n=n)

        X2 = fft_pack.rfft(mic2data, n=n)
        X2star = np.conj(X2)

        # TODO: Implement more processors (Eckart, ML/HT)
        if self._GCC_proc_== "PHAT":
            corr = fft_pack.irfft(np.exp(1j*np.angle(X1 * X2star)), n=(res_scaling * n))

        elif self._GCC_proc_== "p-CSP" or self._GCC_proc_== "p-PHAT":
            proc = 1.0/(np.abs(X1*X2star)**0.73)
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif self._GCC_proc_== "CC":
            proc = 1.0
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif self._GCC_proc_== "RIR":
            proc = 1.0/(X1*np.conj(X1))
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif self._GCC_proc_== "SCOT":
            proc = 1.0/sqrt((X1*np.conj(X1))*(X2*X2star))
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif self._GCC_proc_== "HB":
            proc = np.abs(X1*np.conj(X2))/(X1*np.conj(X1)*X2*np.conj(X2))
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))
        
        elif self._GCC_proc_[:3].lower() == 'bit':
            # f_weighting = pickle.load("./bitpsd")
            with open("./bitpsd", 'rb') as f:
                f_weighting = pickle.load(f)
            # print(f_weighting, type(f_weighting))
            proc = f_weighting(np.fft.rfftfreq(n)*self.sample_rate)
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))
        
        else:
            # Defaults to regular CC.
            proc = 1.0
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        corr = np.concatenate((corr[-int(res_scaling*n/2):], corr[:int(res_scaling*n/2)+1]))

        corrxs = np.arange(start=(dlen-num_samples)*res_scaling, stop=(dlen+num_samples)*res_scaling, step=1,
                           dtype='int32')

        cInterp = interp1d(x=(corrxs/res_scaling-dlen)+1, y=corr[corrxs], kind='cubic')

        return cInterp

    def _objective_(self, X, Y):
        """This function takes a matrix/vector of each x and y coordinates, and at each location evaluates the sum of the generalised
        cross-correlations between the microphone data as if the signal had come from that location. In this way we can search
        for the point with maximum correlation, which should correspond to the most likely actual source position.

        Args:
            X (np.array): An n by m matrix of x-coordinates of points at which to evaluate the _objective_ function
            Y (np.array): An n by m matrix of Y-coordinates of points at which to evaluate the _objective_ function

        Returns:
            np.array: An n by m matrix of signal correlations corresponding to the source having originated at each point
            generated by the input coordinates
        """

        # Calculate distances
        ds = [sqrt((X-self.mics[i, 0])**2+(Y-self.mics[i, 1])**2) for i in
              np.arange(0, self.mics.shape[0], dtype="int16")]

        # Calculate times, then pass the times into the correlation functions and sum them
        ts = np.array([self._cor_fns_["{}".format(ps)]((ds[ps[0]] - ds[ps[1]]) * self.sample_rate / self.sound_speed) for ps in
                       self._mic_pairs_])

        return np.sum(ts, axis=0)

    def estimate_DOA_heatmap(self, method, xrange=(-50, 50), yrange=(-50, 50), xstep=False, ystep=False, colormap="gist_heat", shw=True,
                        block_run=True, no_fig=False, freq=False, signals=1):
        """Displays a heatmap for visual inspection of correlation-based location estimation.

        Generates a grid of provided dimension/resolution, and evaluates the selected DOA-estimation at each point on the grid.
        Vectorised for fast execution.

        Arguments:
            method (str): One of; "GCC", "MUSIC" or "AF-MUSIC". The method to be used in heatmap generation.
            xrange (float, float): The lower and upper bound in the x-direction.
            yrange (float, float): The lower and upper bound in the y-direction.
            xstep (float): If given, determines the size of the steps in the x-direction. Otherwise defaults to 1000 steps.
            ystep (float): If given, determines the size of the steps in the y-direction. Otherwise defaults to 1000 steps.
            colormap (str): The colour map for the heatmap. See https://matplotlib.org/examples/color/colormaps_reference.html
            shw (bool): If False, return the axis object rather than display.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
            no_fig (bool): If True, return the heatmap grid rather than plot it.
            freq (float): Frequency, in Hz, at which to calculate the MUSIC spectrum.
            signals (int): The number of signals to be localised. Only relevant for MUSIC-based methods.

        Returns:
            np.array: Returns EITHER the current (filled) heatmap domain if no_fig == True, OR a handle to the displayed figure.
        """

        if (xstep and ystep):
            xdom = np.linspace(start=xrange[0], stop=xrange[1], num=int((xrange[1] - xrange[0])//xstep))
            ydom = np.linspace(start=yrange[0], stop=yrange[1], num=int((yrange[1] - yrange[0])//ystep))
        else:
            xdom = np.linspace(start=xrange[0], stop=xrange[1], num=1000)
            ydom = np.linspace(start=yrange[0], stop=yrange[1], num=1000)
        self._hm_domain_ = np.zeros((len(ydom), len(xdom)))

        xdom, ydom = np.meshgrid(xdom, ydom)
        
        if method.upper() == "AF-MUSIC" or method.upper() == "AF_MUSIC":
            self.dataFFT = fft_pack.rfft(self.data, axis=0, n=2*self.data.shape[0])
            self._hm_domain_ = self.AF_MUSIC(xdom, ydom)
        elif method.upper() == "MUSIC":
            assert freq, "Frequency must be provided for MUSIC calculation"
            pos = fft_pack.rfftfreq(2*self.data.shape[0])*self.sample_rate
            actidx = np.argmin(abs(pos-freq))
            self.dataFFT = fft_pack.rfft(self.data, axis=0, n=2*self.data.shape[0])
            self._hm_domain_ = self._MUSIC2D_((pos[actidx], actidx), xdom, ydom, numsignals=signals)
        elif method.upper() == "GCC":
            self._hm_domain_ = self._objective_(xdom, ydom)
        else:
            print("Method not recognised. Defaulting to GCC.")
            self._hm_domain_ = self._objective_(xdom, ydom)

        if no_fig:
            return self._hm_domain_

        plt.imshow(self._hm_domain_, cmap=colormap, interpolation='none', origin='lower',
                   extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
        plt.colorbar()
        plt.xlabel("Horiz. Dist. from Center of Array [m]")
        plt.ylabel("Vert. Dist. from Center of Array [m]")
        plt.title("{}-based Source Location Estimate".format(method))

        if shw:
            plt.show(block=block_run)
            return
        else:
            return plt.imshow(self._hm_domain_, cmap=colormap, interpolation='none', origin='lower',
                              extent=[xrange[0], xrange[1], yrange[0], yrange[1]])

    def _polynom_steervec(self, samples, max_tau=1500):
        """Takes a vector of M desired delays and a maximum lag parameter max_tau, and returns the (M, 1, 2*max_tau+1)
        polynomial steering vector which contains the z-transform of the fractional delay filters necessary to delay each
        channel of a M-channel audio file by the corresponding delay parameter from the input vector. For example,
        polynom_steervec([1.2, 3, np.pi]) may be used to delay the first channel of a three-channel audio clip by 1.2 samples,
        the second channel by 3, and the third channel by approximately pi samples.

        Arguments:
            samples (np.array): A 1D array of the desired delay amounts
            max_tau (int): The maximum lag in either direction for the fractional delay filters. A higher number will be more accurate, but take longer to use.

        Returns:
            np.array: A vector containing the desired fractional delay filters.
        """
        mics = self.mics
        tau = samples
        tau.shape = (tau.shape[0], 1)
        Az = np.sinc((np.tile(np.arange(-max_tau, max_tau + 1), (mics.shape[0], 1)) - tau))
        Az = np.reshape(Az, (mics.shape[0], 1, Az.shape[-1]))
        return Az

    def shift_sound(self, location, inputfile, output_filename, noisescale=0):
        """Creates a multi-track wav file from a mono one, simulating the array recordings had the sound came from the
        provided location and were recorded on the current virtual array's microphone configuration.

        Saves the resultant file in the current working directory with the provided filename, at the same sample rate as the input data.

        Arguments:
            location (float, float): A tuple (x,y) providing the location in meters relative to the array at which to simulate the sound as having came from.
            inputfile (str): File path for the mono wav file to be used.
            output_filename (str): The desired file name for the output file.
            noisescale (float): Adds Gaussian white noise with standard deviation noisescale*(standard deviation of input file)
        """

        [spl, dt] = wav.read(inputfile)
        dt = (dt - np.mean(dt))/np.std(dt)
        loc_dif = self.mics - np.tile(location, (self.mics.shape[0], 1))
        dists = np.linalg.norm(loc_dif, axis=1)
        samples = (dists*spl)/self.sound_speed
        samples -= min(samples)

        fracsamples = samples % 1
        intsamples = samples - fracsamples

        svs = self._polynom_steervec(fracsamples)

        t = np.tile(dt.T, (self.mics.shape[0], 1))
        t.shape = (t.shape[0], 1, t.shape[1])

        xout = []
        for r in np.arange(t.shape[0]):
            xout.append(fftconvolve(t[r, 0, :], svs[r, 0, :]))

        xout = np.array(xout)

        xout = np.hstack((xout, np.zeros((xout.shape[0], int(np.max(intsamples))))))

        for idx in np.arange(xout.shape[0]):
            xout[idx,:] = np.roll(xout[idx,:], int(intsamples[idx]))


        if noisescale != 0:
            xout += np.random.normal(0, noisescale*sqrt(np.var(xout)), size=xout.shape)

        xout *= (2**15-1)/np.max(abs(xout))

        xout = xout.astype('int16')
        wav.write(output_filename, spl, xout.T)

    def _MUSIC1D_(self, freqtup, theta, numsignals=1, SI=None):
        """Vectorised implementation of the Multiple Signal Classification algorithm for DOA eastimation.

        Arguments:
            freqtup (float, int): The frequency at which to evaluate the MUSIC algorithm, and the index at where to find it in the FFT of the data.
            theta (float/np.array): The angle of arrial at which to evaluate the MUSIC algorithm. May be a 1D numpy array.
            numsignals (int): How many signals to localise.
            SI (np.array): The covariance matrix S, if known a priori.
        """
        freq, idx = freqtup

        incidentdir = np.array([-np.cos(theta), -np.sin(theta)])
        tau = dot(self.mics, incidentdir) / self.sound_speed

        # Populate a(theta)
        a = np.exp(-1j * 2 * np.pi * freq * tau)

        # Find variance/covariance matrix S=conj(X.X^H)
        # Where X = FFT(recieved vector x)
        if type(SI) == type(None):
            S = dot(self.dataFFT[idx:idx+1, :].T, np.conj(self.dataFFT[idx:idx+1, :]))
        else:
            S = SI

        # Find eigen-stuff of S
        lam, v = la.eig(S)

        # Should be real as S Hermitian, rounding problems
        # mean imaginary part != 0. Take real part.
        lam = lam.real

        # Find a sorting index list
        xs = argsort(lam).astype("int16")
        # print(lam[xs])

        # Take the Eigenvectors corresponding to the
        # 'numsignals' lowest Eigenvalues
        EN = v[:, xs[:len(xs)-numsignals]]

        # Calculate 1/P_MU
        p = dot(dot(np.conj(a.T), EN), dot(np.conj(EN.T), a))

        # If more than 1D find relevant entries and flatten
        if len(p.shape) > 1:
            p = np.ndarray.flatten(np.diag(p))

        # Return P_MU
        return 1/p.real, lam[xs[-1]]

    def _MUSIC2D_(self, freqtup, X, Y, numsignals=1, SI=None):
        """Vectorised 2D implementation of the Multiple Signal Classification algorithm for DOA eastimation.

        Arguments:
            freqtup (float, int): The frequency at which to evaluate the MUSIC algorithm, and the index at where to find it in the FFT of the data.
            X (np.array): An array of x-locations at which to evaluate. Should be the counterpart to Y, as in np.meshgrid
            Y (np.array): An array of y-locations at which to evaluate. Should be the counterpart to X, as in np.meshgrid
            numsignals (int): How many signals to localise.
            SI (np.array): The covariance matrix S, if known a priori.
        """
        # print(X.shape, Y.shape)
        crds = np.dstack((X, Y))
        crds = np.stack([crds for _ in range(self.mics.shape[0])], 3)
        delm = np.linalg.norm(crds[:, :]-self.mics.T, axis=2)/self.sound_speed
        freq, idx = freqtup

        # Populate a(r, theta)
        a = np.exp(-1j*2*np.pi*freq*delm)

        # Find variance/covariance matrix S=X.X^H
        # Where X = FFT(recieved vector x)
        if type(SI)==type(None):
            S = dot(self.dataFFT[idx:idx+1, :].T, np.conj(self.dataFFT[idx:idx+1, :]))
        else:
            S = SI

        # Find eigen-stuff of S
        lam, v = la.eigh(S)

        # Should be real as S Hermitian, rounding problems
        # mean imaginary part != 0. Take real part.
        lam = lam.real

        # Find a sorting index list
        xs = argsort(lam).astype("int16")

        # Take the Eigenvectors corresponding to the
        # 'numsignals' lowest Eigenvalues
        EN = v[:, xs[:len(xs)-numsignals]]

        # Calculate 1/P_MU
        p = dot(a, np.conj(EN))*dot(np.conj(a), EN)
        p = np.sum(p, axis=-1, keepdims=False)
        # print(p.shape)
        # Return P_MU
        return 1/p.real

    def _transpose_(self, mult):
        """
        """
        assert self.data is not None, "No data loaded yet."
        datatr = np.zeros((self.data.shape[0]*mult, self.data.shape[1]))
        tt = self.data.shape[0]/self.sample_rate
        fc = self.sample_rate/(2*mult)
        w = fc / (self.sample_rate / 2)
        b_bl, a_bl = signal.butter(10, w, 'low')
        for ch in np.arange(self.data.shape[1]):
            sig = interp1d(np.linspace(0, tt, num=self.data.shape[0], endpoint=True),
                           self.data[:,ch], kind="cubic", assume_sorted=True)
            d = sig(np.linspace(0, tt, self.sample_rate*mult*tt, endpoint=True))
            datatr[:, ch] = signal.filtfilt(b_bl, a_bl, d, axis=0)
            # datatr[:, ch] = d
        wav.write(data=datatr.astype('int16'), rate=self.sample_rate, filename="./scarynoise.wav")
        _, dnew = wav.read("./scarynoise.wav")
        return dnew

    def _UfitoRyy_(self, Rxx, f):
        """Returns the covariance matrix of the data at frequency f (Hz), shifted to the focussing frequency f_0. These
        should be summed over all frequencies of interest to create the universally focussed sample covariance matrix
        R_{yy} for the AF-MUSIC algorithm.

        Arguments:
            f (int): The frequency index to work with from the FFT of the data
        """
        df = self.dataFFT[:, f:f+1]
        ta = dot(df, df.conj().T)
        ui, Ufi = eigh(Rxx, check_finite=False)
        ui = ui.real
        sortarg = argsort(abs(ui))[::-1]
        Ufi = Ufi[:, sortarg]
        Tauto = (dot(self.Uf0, Ufi.conj().T)) / sqrt(self.numbins)
        Y = dot(Tauto, df)
        Ryy = dot(Y, Y.conj().T)
        return Ryy*abs(ui[sortarg[0]])

    def AF_MUSIC(self, xdom, ydom, focusing_freq=-1, npoints=1000, signals=1, shw=True, block_run=True, chunks=10):
        """Display a polar plot of estimated DOA using the MUSIC algorithm

        Arguments:
            focusing_freq (float): The frequency (in Hz) at which to perform the calculation. If <0, will default to 0.9*(spatial Nyquist frequency)
            npoints (int): The total number of points around the circle at which to evaluate.
            signals (int): The numbers of signals to locate.
            shw (bool): Show the plot? If False, will return the data that was to be plotted.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
            chunks (int): How many sections to split the data up into. Will split up the data and average the result over the split sections
        """

        if focusing_freq < 0:
            focusing_freq = self.spatial_nyquist_freq*0.9

        # First generate Rxxs to get to T_autos
        # Tauto will go in here
        Tauto = np.zeros((self.mics.shape[0], self.mics.shape[0], self.data.shape[0]//2+1) , dtype="complex128")
        # Rxx will go in here
        Rxx = np.zeros((self.mics.shape[0], self.mics.shape[0], self.data.shape[0]//2+1) , dtype="complex128")
        # Ufi will go in here
        Ufi = np.zeros((self.mics.shape[0], self.mics.shape[0], self.data.shape[0]//2+1) , dtype="complex128")
        # Ryy will go in here
        Ryy = np.zeros((self.mics.shape[0], self.mics.shape[0], self.data.shape[0] // 2 + 1), dtype="complex128")

        # Split the data up into "chunks" sections
        indices = [int(x) for x in np.linspace(0, self.data.shape[0], num=chunks+1, endpoint=True)]

        # Calculate Rxx
        for mark in np.arange(len(indices)-1):
            dcr = self.data[indices[mark]:indices[mark+1], :]
            for chnl in np.arange(dcr.shape[1]):
                dcr[:, chnl] *= np.blackman(dcr.shape[0])

            # dft is RFFT of current data chunk
            dft = fft_pack.rfft(dcr, axis=0, n=self.data.shape[0]).T
            dft.shape = (dft.shape[0], 1, dft.shape[1])
            # print(dft.shape, self.data.shape, Tauto.shape)

            Rxx += np.einsum("jin,iln->jln", dft, np.conj(np.transpose(dft, (1,0,2))))/chunks

        # The frequencies for Tauto and DFT. They all have the same length so this is fine to do outside the loop
        pos = fft_pack.rfftfreq(self.data.shape[0]) * self.sample_rate

        # focusing_freq_index is the index along dft and Tauto to find f_0
        focusing_freq_index = np.argmin(np.abs(pos - focusing_freq))

        eig_f0, v_f0 = np.linalg.eigh(Rxx[:,:,focusing_freq_index])
        Uf0 = v_f0[:, np.argsort(np.abs(eig_f0))[::-1]]

        # Calculate Tautos
        for indx, fi in enumerate(pos):
            eig_fi, v_fi = np.linalg.eigh(Rxx[:, :, indx])
            Ufi[:,:,indx] = v_fi[:, np.argsort(np.abs(eig_fi))[::-1]]
            Tauto[:,:,indx] = dot(Uf0, np.conj(Ufi[:,:,indx].T))/np.sqrt(pos.shape[0])

        # Calculate Ryy
        chunks=1.0
        indices = [int(x) for x in np.linspace(0, self.data.shape[0], num=chunks + 1, endpoint=True)]
        for mark in np.arange(len(indices) - 1):
            dcr = self.data[indices[mark]:indices[mark + 1], :]
            for chnl in np.arange(dcr.shape[1]):
                dcr[:, chnl] *= np.blackman(dcr.shape[0])

            # dft is RFFT of current data chunk
            dft = fft_pack.rfft(dcr, axis=0, n=self.data.shape[0]).T
            dft.shape = (dft.shape[0], 1, dft.shape[1])
            # print(dft.shape, self.data.shape, Tauto.shape)

            Yi = np.einsum("abc,bdc->adc", Tauto, dft)
            Ryy += np.einsum("jin,iln->jln", Yi, np.conj(np.transpose(Yi, (1, 0, 2)))) / chunks

        Rcoh = np.sum(Ryy, axis=-1)/(self.data.shape[0]//2+1)

        rest = self._MUSIC2D_((focusing_freq, focusing_freq_index), xdom, ydom, SI=Rcoh)
        
        return rest

    def _get_path(self, GEarthFile, array_center, draw=True):
        try:
            tree = ET.parse('./data/extr/{}.kmz/doc.kml'.format(GEarthFile))
        except FileNotFoundError as e:
            print("Unzipping .kmz file to ./data/extr/{}.kmz/doc.kml".format(GEarthFile))
            with zipfile.ZipFile("./data/{}.kmz".format(GEarthFile), 'r') as zip_ref:
                zip_ref.extractall("./data/extr/{}.kmz/".format(GEarthFile))
            tree = ET.parse('./data/extr/{}.kmz/doc.kml'.format(GEarthFile))

        root = tree.getroot()
        coords_str = root[0][-1][-1][-1].text.strip()
        coords = list(map(lambda x: x.split(","), coords_str.split(" ")))
        coords = np.array(coords).astype(np.float64)
        coords *= np.pi/180.0

        array_center = np.array(array_center)
        array_center *= np.pi/180.0
                
        xy_coords, xy_array = _stereo_proj(coords, array_center)
        
        if False:
            plt.scatter(xy_coords[:,0], xy_coords[:,1])
            plt.scatter(self.mics[:,0], self.mics[:,1])
            plt.xlim([np.min(xy_coords[:,0]), np.max(xy_coords[:,0])])
            plt.ylim([np.min(xy_coords[:,1]), np.max(xy_coords[:,1])])
            plt.title("{} Stereographic Projection".format(GEarthFile))
            plt.xlabel(r"x [m] East/West")
            plt.ylabel(r"y [m] North/South")
            plt.show()

        xfunct = interp1d(x=np.linspace(0, 1, num=xy_coords.shape[0]), y=xy_coords[:,0], kind="linear")
        yfunct = interp1d(x=np.linspace(0, 1, num=xy_coords.shape[0]), y=xy_coords[:,1], kind="linear")

        pts = lambda x: (xfunct(x), yfunct(x))
        return pts

    def estimate_DOA_path(self, method, path=lambda x: (np.cos(2*np.pi*x), np.sin(2*np.pi*x)), array_GPS=False, npoints=2500, map_zoom=20, map_scale=2, freq=False):
        """Gives an estimate of the source DOA along the `path` provided, otherwise along the unit circle if `path` is not present. 

        Arguments:
            method (str): One of; "GCC", "MUSIC", or "AF-MUSIC". The method to use for DOA estimation.
            path (str/function): A filepath to a saved Google Earth path (in .kmz form), else a function f: [0,1]->R^2 to act as a 
                                 parametrisation of the path at which to evaluate the DOA estimator.
            npoints (int): The number of points along the path to sample.
        """
        pathstr = False
        if isinstance(path, str):
            assert array_GPS
            pathstr = path
            path = self._get_path(path, array_GPS)
        else:
            assert callable(path)
        
        dom = np.array(path(np.linspace(0, 1, npoints)))

        if method.upper() == "GCC":
            eval_dom = self._objective_(dom[0,:], dom[1,:])
        elif method.upper() == "MUSIC":
            assert freq, "Frequency must be provided for MUSIC calculation"
            pos = fft_pack.rfftfreq(2*self.data.shape[0])*self.sample_rate
            actidx = np.argmin(abs(pos-freq))
            self.dataFFT = fft_pack.rfft(self.data, axis=0, n=2*self.data.shape[0])
            eval_dom = self._MUSIC2D_((pos[actidx], actidx), dom[0:1,:].T, dom[1:,:].T).flatten()
        elif method.upper() == "AF-MUSIC" or method.upper() == "AF_MUSIC":
            self.dataFFT = fft_pack.rfft(self.data, axis=0, n=2*self.data.shape[0])
            eval_dom = self.AF_MUSIC(dom[0:1,:].T, dom[1:,:].T).flatten()
        else:
            print("Method not recognised. Defaulting to GCC.")
            eval_dom = self._objective_(dom[0,:], dom[1,:])

        maxidx = np.argmax(eval_dom)
        x_max = dom[0, maxidx] 
        y_max = dom[1, maxidx]
        theta = np.arctan2(y_max, x_max)*180/np.pi

        plt.figure(1)
        if pathstr:
            plt.subplot(121)
        else:
            pass

        p = plt.scatter(dom[0,:], dom[1,:], c=eval_dom)
        for m in np.arange(self.mics.shape[0]):
            plt.scatter(self.mics[m,0], self.mics[m,1], marker='x')
        plt.xlim([np.min(dom[0,:]), np.max(dom[0,:])])
        plt.ylim([np.min(dom[1,:]), np.max(dom[1,:])])
        plt.title(r"{} DOA Estimate; Max at $({:.2f}, {:.2f})$, $\theta={:.1f}^\circ$".format(pathstr if pathstr else "", x_max, y_max, theta))
        plt.xlabel(r"x [m] East/West")
        plt.ylabel(r"y [m] North/South")
        plt.colorbar(p)
    
        if pathstr:
            lat, lon = _inv_proj(dom[:,maxidx:maxidx+1].T, array_GPS)
            # print(lat, lon, '\n', array_GPS)
            with open("./data/apikey", 'r') as f_ap:
                key = f_ap.readline()
            dmap = DecoratedMap(maptype='satellite', key=key, zoom=map_zoom, scale=map_scale)
            dmap.add_marker(LatLonMarker(lat=array_GPS[1], lon=array_GPS[0], label='A'))
            dmap.add_marker(LatLonMarker(lat=lat[0], lon=lon[0], label='B'))
            response = requests.get(dmap.generate_url())
            with open("{}.png".format(pathstr), 'wb') as outfile:
                outfile.write(response.content)
            
            im = mpimg.imread("{}.png".format(pathstr))
            plt.subplot(122)
            plt.imshow(im)
            # plt.axis("off")
            plt.xticks([])
            plt.yticks([])
            plt.title("{} Satellite Imagery".format(pathstr))
            plt.xlabel("A: Array\nB: Bird")

        
        plt.show()
        return

def UCA(n, r, centerpoint=True, show=False):
    """A helper function to easily set up UCAs (uniform circular arrays).

    Arguments:
        n (int): The number of microphones in the array.
        r (float): The radius of the array
        centerpoint (bool): Include a microphone at (0,0)? This will be one of the n points.
        show (bool): If True, shows a scatterplot of the array

    Returns:
        np.array: An n by 2 numpy array containing the x and y positions of the n microphones in the UCA.
    """
    mics = []
    if centerpoint:
        n -= 1
        mics.append([0,0])
    for theta in np.linspace(0, 2*np.pi, n, endpoint=False):
        mics.append([r*np.cos(theta), r*np.sin(theta)])
    mics = np.array(mics)
    if show:
        plt.scatter(mics[:,0], mics[:,1])
        plt.title("Microphone Locations")
        plt.xlabel("Horizontal Distance From Array Center [m]")
        plt.ylabel("Vertical Distance From Array Center [m]")
        plt.show()
    return mics

def UMA8(bearing=0, center=0):
    pixel_dist = 90.0/(1171.0-77.0)
    pixel_dist /= 1000.0
    
    mics = np.array([[624, 546],
            [1147, 546],
            [885, 93],
            [362, 93],
            [101, 546],
            [362, 999], 
            [885, 999]], dtype='float64')
    
    mics -= mics[0,:]
    mics *= pixel_dist
    theta = -bearing*np.pi/180.0
    for idx in np.arange(mics.shape[0]):
        mics[idx, 0] = mics[idx, 0]*np.cos(theta) - mics[idx, 1]*np.sin(theta)
        mics[idx, 1] = mics[idx, 1]*np.cos(theta) + mics[idx, 0]*np.sin(theta)
    mics += center
    return mics

def _r(phi, r_eq=6378137.0, r_pl=6356752.3):
    """Calculates geocentric radius of Earth at latitude $\phi$.
    r_eq is the radius of Earth at the equator
    r_pl is the radius of Earth at the poles. Distances are in meters.
    Calculated using https://en.wikipedia.org/wiki/Earth_radius#Geocentric_radius
    """
    return np.sqrt(((r_eq**2*np.cos(phi))**2+(r_pl**2*np.sin(phi))**2)/((r_eq*np.cos(phi))**2+(r_pl*np.sin(phi))**2))

def _stereo_proj(points, array_coords):
    """Given a set of points (longitude_i, latitude_i, height_i) on the globe,
    calculates their coordinates in the plane under the stereographic projection
    centered about the microphone array center.
    longitude and latitude here are in (signed) radians.
    http://mathworld.wolfram.com/StereographicProjection.html
    """
    lam, phi, _ = np.array(array_coords)
    points = np.vstack((np.array([lam, phi, 0]), points))
    k = 2*_r(phi)/(1+np.sin(phi)*np.sin(points[:,1])+np.cos(phi)*np.cos(points[:,1])*np.cos(points[:,0]-lam))
    x = k*np.cos(points[:,1])*np.sin(points[:,0]-lam)
    y = k*(np.cos(phi)*np.sin(points[:,1])-np.sin(phi)*np.cos(points[:,1])*np.cos(points[:,0]-lam))
    return np.array([x[1:], y[1:]]).T, np.array([x[0], y[0]])

def _inv_proj(points, array_coords):
    # points = points[:,::-1]
    # print("points", points)
    lam, phi, _ = np.array(array_coords)*np.pi/180.0
    # print("lam, phi", lam, phi)
    rho = np.linalg.norm(points, axis=1)
    # print("rho", rho)
    c = 2*np.arctan2(rho, 2*_r(phi))
    # print("c", c)
    lat = (np.arcsin(np.cos(c)*np.sin(phi)+(points[:,1]*np.sin(c)*np.cos(phi))/rho))*180.0/np.pi
    lon = (lam + np.arctan2(points[:,0]*np.sin(c), rho*np.cos(phi)*np.cos(c)-points[:,1]*np.sin(phi)*np.sin(c)))*180.0/np.pi
    return lat, lon

