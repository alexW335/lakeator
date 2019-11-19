#%%
import matplotlib.pyplot as plt
import lakeator

in_file = "./data/k.wav"
out_file = "./data/k_shifted.wav"

l=lakeator.Lakeator(mic_locations=lakeator.UCA(n=7, r=1))
l.shift_sound(location=(-26.543, 18.411), inputfile=in_file, output_filename=out_file)
l.load(out_file, GCC_processor="p-PHAT")
l.estimate_DOA(path="Butterfly Pond", array_GPS=(175.611459264361,-40.357194,0), npoints=2500)