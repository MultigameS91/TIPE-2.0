import numpy as np
import matplotlib.pyplot as plt

def S(x,y):
    return 0

def pic(x, y, center_x=0, center_y=0, amplitude=1, width=1):
    return amplitude * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * width**2))

def interpolate_surface(x, y, t, f1, f2):
    return (1 - t) * f1(x, y) + t * f2(x, y)

SX, SY = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
SZ = S(SX,SY) - pic(SX,SY)

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")
ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 0.5)
ax.plot_wireframe(SX, SY, np.zeros(np.shape(SX)), color="gray", linewidth = 1)

plt.axis('scaled')
plt.show()


