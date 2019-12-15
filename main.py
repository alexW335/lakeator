#%%
import matplotlib.pyplot as plt
import lakeator
import numpy as np

in_file = "./data/train1exc50seconds.wav"
out_file = "./data/train1exc50seconds_shifted.wav"

# l=lakeator.Lakeator(mic_locations=lakeator.UCA(n=7, r=0.7))
l=lakeator.Lakeator(mic_locations=np.array([[-10,0],[10,0],[-10,-10],[10,-10],[-10,10],[10,10],[-10,20],[10,20]]))
l.shift_sound(location=(5,15), inputfile=in_file, output_filename=out_file, noisescale=0.1)
l.load(out_file, GCC_processor="bittern")

# l.estimate_DOA_path("GCC")
# l.estimate_DOA_path("MUSIC", freq=120)
# l.estimate_DOA_path("AF-MUSIC")

# l.estimate_DOA_path("GCC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18)
# l.estimate_DOA_path("MUSIC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18, freq=120)
# l.estimate_DOA_path("AF-MUSIC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18)

l.estimate_DOA_heatmap("GCC")
# l.estimate_DOA_heatmap("MUSIC", freq=120)
# l.estimate_DOA_heatmap("AF-MUSIC")

# GIS integration
# Combining heatmaps and GEarth imagery
# Verify and test bittern weighting
# Simulate bittern at position 'a' and sparrow/blackbird/fantail at other positions all over the place and see how BIT weighting compares to GCC