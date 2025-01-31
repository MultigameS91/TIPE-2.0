import matplotlib.pyplot as plt
import numpy as np

import geodesiqueBezier as gb 

point_number=10
grid_size=5
h=2*grid_size/point_number
grid_origin=-5

def create_grid(proba):
    return [[(np.random.choice([True, False], p=[proba,1-proba])) for i in range(point_number)] for i in range(point_number)]

def pic(x, y, center_x=0, center_y=0, amplitude=1, width=1):
    return amplitude * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * width**2))

A = [3, 0]  # Point de départ
B = [-3, 0]  # Point d'arrivée

probas=np.linspace(0,1,10)

longeurs=[]

for p in probas:
    m=0
    for n in range(20):
        grid=create_grid(p)
        def S(x,y):
            s=0
            for i in range(point_number):
                for j in range(point_number):
                    if grid[i][j]:
                        x0,y0=grid_origin+i*h,grid_origin+j*h
                        s+=pic(x,y,center_x=x0,center_y=y0,amplitude=0.5,width=h)
            return s
        geodesique_bezier, L_bezier=gb.descente_gradient_heavy_ball(S,A[0],A[1],B[0],B[1],nombre_points_max=5,N_integral=10,epsilon=1e-2,affichage=False)
        m+=L_bezier
    longeurs.append(m/10)

plt.plot(probas,longeurs)
plt.show()


'''xBezier,yBezier=geodesique_bezier.courbe(500)

res=5*point_number
SX, SY = np.meshgrid(np.linspace(-grid_size, grid_size, res), np.linspace(-grid_size, grid_size, res))
SZ = S(SX,SY)

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")
ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 0.5)

ax.plot(xBezier[0],yBezier[0],S(xBezier,yBezier), color="blue", label='Géodésique Bézier : '+str(np.round(L_bezier,2)), linewidth=2)

ax.scatter(A[0], A[1], S(A[0],A[1]), color='purple', s=50, label='Point A (départ)')
ax.scatter(B[0], B[1], S(B[0],B[1]), color='red', s=50, label='Point B (arrivée)')

plt.axis('scaled')
plt.show()'''