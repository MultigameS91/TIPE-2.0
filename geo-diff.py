import numpy as np
from sympy import symbols, diff, Matrix, lambdify, simplify, cos, ln,exp
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import geodesiqueBezier as gb 

x, y = symbols('x y')
f = 10*exp(-(x**2+y**2))
def S(a,b):
    return 10*np.exp(-(a**2+b**2))

fx = diff(f, x)
fy = diff(f, y)
g11 = 1 + fx**2
g12 = fx * fy
g22 = 1 + fy**2
g = Matrix([[g11, g12], [g12, g22]])  # Tenseur métrique

# Inverser le tenseur métrique
g_inv = simplify(g.inv())

# Calculer les symboles de Christoffel
Gamma = {}
variables = [x, y]
for k in range(2):  # Indices Christoffel
    for i in range(2):
        for j in range(2):
            Gamma[(k, i, j)] = 0
            for l in range(2):  # Somme sur l
                term = diff(g[i, l], variables[j]) + diff(g[j, l], variables[i]) - diff(g[i, j], variables[l])
                Gamma[(k, i, j)] += g_inv[k, l] * term / 2
            Gamma[(k, i, j)] = simplify(Gamma[(k, i, j)])

Gamma_func = {}
for k in range(2):
    for i in range(2):
        for j in range(2):
            Gamma_func[(k, i, j)] = lambdify((x, y), Gamma[(k, i, j)], 'numpy')

# Équations différentielles des géodésiques
def geodesic_equations(t, U):
    x, y, dx_dt, dy_dt = U
    dx2_dt2 = 0
    dy2_dt2 = 0

    for i in range(2):
        for j in range(2):
            dx2_dt2 -= Gamma_func[(0, i, j)](x, y) * U[2 + i] * U[2 + j]
            dy2_dt2 -= Gamma_func[(1, i, j)](x, y) * U[2 + i] * U[2 + j]

    return [dx_dt, dy_dt, dx2_dt2, dy2_dt2]

# Fonction pour minimiser : ajuster les vitesses initiales pour atteindre le point B
def objective(v0):
    """Minimiser l'écart entre le point final de la géodésique et le point B."""
    U0 = [A[0], A[1], v0[0], v0[1]]
    n=500
    sol = solve_ivp(geodesic_equations, [0, 10], U0, t_eval=np.linspace(0, 10, 500))

    return distance_min(sol.y[0],sol.y[1],n,B)[0]

def distance_min(ensembleX,ensembleY,n,p):
    j=0
    dist_min=np.sqrt((p[0]-ensembleX[j])**2+(p[1]-ensembleY[j])**2)
    for i in range(n):
        d=np.sqrt((p[0]-ensembleX[i])**2+(p[1]-ensembleY[i])**2)
        if d<dist_min:
            dist_min=d
            j=i
    return dist_min,j

A = [3, 0]  # Point de départ
B = [-3, 0]  # Point d'arrivée

# Résoudre pour trouver les vitesses initiales optimales
v0_initial = [B[0]-A[0],B[1]-A[1]]
result = minimize(objective, v0_initial)
v0_opt = result.x

U0 = [A[0], A[1], v0_opt[0], v0_opt[1]]
sol = solve_ivp(geodesic_equations, [0, 10], U0, t_eval=np.linspace(0, 10, 500))

# Extraire la solution
x_sol, y_sol = sol.y[0], sol.y[1]
z_sol = S(np.array(x_sol),np.array(y_sol))  # Calculer z pour le chemin géodésique

cheminX,cheminY,cheminZ=[],[],[]
dist_min,j_max=distance_min(x_sol,y_sol,len(x_sol),B)

for i in range(j_max+1):
    cheminX.append(x_sol[i])
    cheminY.append(y_sol[i])
    cheminZ.append(z_sol[i])
print(len(cheminX))
# calcul longueur geo-diff
L_geodiff=0
for i in range(len(cheminX)-1):
    dx=(cheminX[i+1]-cheminX[i])
    dy=(cheminY[i+1]-cheminY[i])
    dz=(cheminZ[i+1]-cheminZ[i])
    L_geodiff+=np.sqrt(dx**2+dy**2+dz**2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

geodesique_bezier, L_bezier=gb.descente_gradient_heavy_ball(S,A[0],A[1],B[0],B[1],nombre_points_max=5,N_integral=10,epsilon=1e-2,affichage=False)
xBezier,yBezier=geodesique_bezier.courbe(500)

print("Longueur GéoDiff", L_geodiff)

X = np.linspace(-5, 5, 500)
Y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(X, Y)
Z = S(X,Y)

ax.plot_wireframe(X, Y, Z, linewidth = 0.5, alpha=0.8, color='orange')

ax.plot(cheminX, cheminY, cheminZ, color='green', label='Géodésique Géo-Diff : '+str(np.round(L_geodiff,2)), linewidth=2)
ax.plot(xBezier[0],yBezier[0],S(xBezier,yBezier), color="blue", label='Géodésique Bézier : '+str(np.round(L_bezier,2)), linewidth=2)

ax.scatter(A[0], A[1], S(A[0],A[1]), color='purple', s=50, label='Point A (départ)')
ax.scatter(B[0], B[1], S(B[0],B[1]), color='red', s=50, label='Point B (arrivée)')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("Géodésique Bézier vs Géo-Diff")
ax.legend()
plt.show()

fig2d, ax2d = plt.subplots()

contour_surface=plt.contour(X,Y,Z, levels=10, cmap='viridis')
plt.clabel(contour_surface, inline=True, fontsize=8)
plt.colorbar(contour_surface, label="Altitude")
plt.scatter(A[0], A[1], color='purple', label='Point A (départ)')
plt.scatter(B[0], B[1], color='red', label='Point B (arrivée)')

plt.plot(cheminX, cheminY, color='green', label='Géodésique Géo-Diff : '+str(np.round(L_geodiff,2)), linewidth=2)
plt.plot(xBezier[0],yBezier[0], color="blue", label='Géodésique Bézier : '+str(np.round(L_bezier,2)), linewidth=2)

ax2d.set_xlabel('x')
ax2d.set_ylabel('y')
ax2d.set_title("Géodésique Bézier vs Géo-Diff")
ax2d.legend()
plt.show()
