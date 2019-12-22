#%%
import matplotlib.pyplot as plt
import lakeator
import numpy as np

in_file = "./data/train1exc50seconds.wav"
out_file = "./data/train1exc50seconds_shifted.wav"

# in_file = "./data/noise.wav"
# out_file = "./data/noise_shifted.wav"

l=lakeator.Lakeator(mic_locations=lakeator.UCA(n=4, r=0.7))
# l=lakeator.Lakeator(mic_locations=np.array([[-10,0],[10,0],[-10,-10],[10,-10],[-10,10],[10,10],[-10,20],[10,20]]))
# l=lakeator.Lakeator(mic_locations=lakeator.UMA8())

# l=lakeator.Lakeator(mic_locations=np.array([[-0.3750, -0.2165],[0.0000, 0.4330],[0.3750, -0.2165]]))
# l=lakeator.Lakeator(mic_locations=np.array([[-0.3750, -0.3750],[-0.3750, 0.3750],[0.3750, 0.3750], [0.3750, -0.3750]]))

# print(l.mics, np.linalg.norm(l.mics[1,:]))
# l.mics *= 35
l.shift_sound(location=(-2,3), inputfile=in_file, output_filename=out_file, noisescale=0.2)
# l.load(out_file, filter_f=(100., 250.))
l.load(out_file)
# l.data = l.data[200:-200]
# plt.plot(l.data)
# plt.show()
# l.estimate_DOA_path("GCC")
# l.estimate_DOA_path("MUSIC", freq=120)
# l.estimate_DOA_path("AF-MUSIC")

# l.estimate_DOA_path("GCC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18)
# l.estimate_DOA_path("MUSIC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18, freq=120)
# l.estimate_DOA_path("AF-MUSIC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18)

# l.estimate_DOA_heatmap("GCC")
# l.estimate_DOA_heatmap("MUSIC", freq=120)
# l.estimate_DOA_heatmap("MUSIC", xrange=(-4, 0), yrange=(1, 5), freq=100)
# l.estimate_DOA_heatmap("AF_MUSIC", xrange=(-4, 0), yrange=(1, 5), AF_freqs=(75.0, 225.0))
l.estimate_DOA_heatmap("AF_MUSIC", xrange=(-10, 10), yrange=(-10, 10), AF_freqs=(75.0, 225.0))


# GIS integration
# Combining heatmaps and GEarth imagery
# Verify and test bittern weighting
# Simulate bittern at position 'a' and sparrow/blackbird/fantail at other positions all over the place and see how BIT weighting compares to GCC