#%%
import matplotlib.pyplot as plt
import lakeator
from locator import locator

in_file = "./data/train1exc50seconds.wav"
out_file = "./data/train1exc50seconds_shifted.wav"

l=lakeator.Lakeator(mic_locations=lakeator.UCA(n=7, r=0.07))
l.shift_sound(location=(-34.415, -39.752), inputfile=in_file, output_filename=out_file, noisescale=0.05)
l.load(out_file, GCC_processor="p-PHAT")
l.estimate_DOA_path("MUSIC", path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18, freq=120)
# l.estimate_DOA(path="Lake", array_GPS=(172.3977859857789,-43.74149174828536,0), map_zoom=18)

# l = locator.Locator(mic_locations=locator.UCA(n=5, r=50))
# l.shift_sound(location=(-34.415, -39.752), inputfile=in_file, output_filename=out_file, noisescale=0.05)
# l.load(out_file, GCC_processor="p-PHAT")
# l.display_heatmap()
# l.IFFT_HM(frequencies=(120,140))