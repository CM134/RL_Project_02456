import glob
import numpy as np



scores = glob.glob('./scores/*')

name = 'rate.csv'


print(scores)

for s in scores:
    if s.find(name) != -1:
        print(s)
        print(np.mean(np.loadtxt(s, dtype=float)))
        