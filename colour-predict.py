import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

def plot_generated(wid, hei, a_low, a_high, b_low, b_high, X_grid):
    """
    Plot random subset of the L*A*B colour space
    """
    plt.subplot(1, 1, 1)
    plt.title('Generated Colour Space')
    plt.xlabel('A')
    plt.ylabel('B')
    nticks = 5
    plt.xticks(np.linspace(0, wid, nticks), np.linspace(a_low, a_high, nticks))
    plt.yticks(np.linspace(0, hei, nticks), np.linspace(b_low, b_high, nticks))
    plt.imshow(X_grid.reshape((hei, wid, 3)))


def main():
    plt.figure(figsize=(10, 5))
    resolution = 256
    wid, hei = resolution, resolution

    luminescence = 65
    a_low = np.random.randint(-128, 0)
    a_high = np.random.randint(0, 127)
    b_low = np.random.randint(-128, 0)
    b_high = np.random.randint(0, 127)

    ag = np.linspace(a_low, a_high, wid)
    bg = np.linspace(b_low, b_high, hei)

    aa, bb = np.meshgrid(ag, bg)
    ll = luminescence * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)
    X_grid = lab2rgb(lab_grid)

    plot_generated(wid, hei, a_low, a_high, b_low, b_high, X_grid)

    plt.show()
    

if __name__ == '__main__':
    main()