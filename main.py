#%%
import matplotlib.pyplot as plt
import lakeator

in_file = "./data/train1exc50seconds.wav"
out_file = "./data/train1exc50seconds_shifted.wav"

l=lakeator.Lakeator(mic_locations=lakeator.UCA(n=7, r=0.7))
l.shift_sound(location=(-34.415, -39.752), inputfile=in_file, output_filename=out_file, noisescale=0.1)
l.load(out_file, GCC_processor="p-PHAT")

l.estimate_DOA_path("GCC")
l.estimate_DOA_path("MUSIC", freq=120)
l.estimate_DOA_path("AF-MUSIC")

l.estimate_DOA_path("GCC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18)
l.estimate_DOA_path("MUSIC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18, freq=120)
l.estimate_DOA_path("AF-MUSIC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18)

l.estimate_DOA_heatmap("GCC")
l.estimate_DOA_heatmap("MUSIC", freq=120)
l.estimate_DOA_heatmap("AF-MUSIC")