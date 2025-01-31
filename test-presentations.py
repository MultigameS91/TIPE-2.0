import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(x) * np.sin(y)

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=10, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.colorbar(contour, label="Altitude")

plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.3)

plt.show()
