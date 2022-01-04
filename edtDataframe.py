# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:43:20 2021

@author: Patrick
"""
# IMPORT LIBRARIES:
import csv
import numpy as np

# SETUP:
#inputs = [32,33,35]
inputs = np.arange(0,93)
targ = 33 #mHandr
#targ = 63 #ball
#targ = 74 #7seg1c

path = "dataframe.txt"
file = open(path, newline='')
f = open("df.txt",'w')
f2 = open("labels.txt",'w')

reader = csv.reader(file)

data = [row for row in reader]

# MAIN LOOP:
for i in range(len(data)-1):
    for n in range(len(inputs)):
        f.write(data[i][1][inputs[n]]+',')
        f2.write(data[i+1][1][inputs[n]]+',')
    for c in range(len(data[i][2])):
        f.write(data[i][2][c]+',')
    #f.write(data[i+1][1][targ]+'\n')
    f.write('\n')
    f2.write('\n')

file.close()
f.close()
f2.close()

print("Done!")
# END
    