import pickle
import matplotlib.pyplot as plt
from icecream import ic
import numpy as np

with open('points.pkl','rb') as f:
    points = pickle.load(f)
    ic(points)
    print('Plotting',len(points),'points')
    x, y = zip(*points)
    c = list(range(0, len(x)))

    cmap = plt.cm.magma

    for idx, point in enumerate(points):
        plt.scatter(point[0], point[1], color=cmap(idx / len(points)))

    plt.plot(x,y)
    plt.show()