import copy
import numpy as np

results = np.zeros(4)

print(results)

class H:
    def __init__(self):
        self.height = 100
        self.name = 'ABC'


a = H()
b = H()
l = [a,b]
k = copy.deepcopy(l)

for o in l:
    print(o.height)

a.height += 1+1
b.height -= 5+1

for o in k:
    print(o.height)

