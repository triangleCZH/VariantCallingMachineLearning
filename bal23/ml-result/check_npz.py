import sys
import numpy as np
a = np.load(sys.argv[1])
b = a['data']
for line in b:
  print(line)
