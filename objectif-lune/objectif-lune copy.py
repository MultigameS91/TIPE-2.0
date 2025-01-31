import numpy as np
from numpy.linalg import norm
from numpy import array

import time as time

import matplotlib.pyplot as plt

import bezierAPI

def sign(v):
    if v<0:
        return -1
    return 1

G = 6.67e-11

class Planete:
    def __init__(self, _pos, _mass):
        self.pos = _pos
        self.mass = _mass

    def champ_gravitationnel(self, x, y):
        
        rx = x - self.pos[0]
        ry = y - self.pos[1]
        r_squared = rx**2 + ry**2
        
        if np.isscalar(r_squared):
            if r_squared == 0:
                return np.array([0.0, 0.0])
            r_magnitude = np.sqrt(r_squared)
        else:
            r_magnitude = np.sqrt(r_squared)
            r_magnitude[r_magnitude == 0] = np.inf
        

        return -G * self.mass / r_squared * np.stack((rx / r_magnitude, ry / r_magnitude), axis=0)

Soleil = Planete(array([0, 0]), 1.989e30)
Terre = Planete(array([149.6e9, 0]), 5.972e24)
Mars = Planete(array([-227.9e9, 0]), 6.417e23)
planetes = [Soleil, Terre, Mars]

def champ_espace(x, y):
    chp = np.zeros((2, *x.shape))
    for p in planetes:
        chp += p.champ_gravitationnel(x, y)
    return chp

SX, SY = np.meshgrid(np.linspace(-3e11, 3e11, 50), np.linspace(-3e11, 3e11, 50))
chp = champ_espace(SX, SY)

plt.figure(figsize=(10, 10))
plt.quiver(SX, SY, chp[0], chp[1], color="blue", alpha=0.6, label="Champ gravitationnel")
plt.scatter(0, 0, color="orange", label="Soleil", s=500)  # Soleil
plt.scatter(Terre.pos[0], Terre.pos[1], color="green", label="Terre", s=100)  # Terre
plt.scatter(Mars.pos[0], Mars.pos[1], color="red", label="Mars", s=80)  # Mars
plt.xlabel("Position X (m)")
plt.ylabel("Position Y (m)")
plt.axis('equal')
plt.grid()

courbe=bezierAPI.Bezier([Terre.pos[0]+6e6,Mars.pos[0]+3.3e6],[Terre.pos[1]+6e6,Mars.pos[1]+3.3e6])
bezierX, bezierY = courbe.courbe(500)
plt.plot(bezierX[0], bezierY[0], color="red", label="Courbe de Bézier", linewidth=2)
plt.show()


def calcul_temps_courbe(courbe:bezierAPI.Bezier,v0:array):
    dt=1
    duree=0
    L_tot=courbe.longueur(500)
    a=0
    m=5e3
    u=a/L_tot
    x,y=courbe.courbe_point(u)
    v=v0

    cooldown=time.time()
    print("u",u)
    while u<1:

        if norm(array([x,y])-Soleil.pos)<7e8:
            return float('inf')

        if u<0:
            return float('inf')

        if time.time()-cooldown>5:
            cooldown=time.time()
            print("u",u)

        chp=champ_espace(x,y)

        vecteur_tangent=np.array(list(courbe.courbe_derivee(u)))
        vecteur_tangent=(1/np.linalg.norm(vecteur_tangent)) * vecteur_tangent
        
        force_projetee=np.dot(chp,vecteur_tangent)*vecteur_tangent
        
        dvdt=force_projetee
        v=v+dt*dvdt
        a+=norm(dt*v)*sign(np.dot(v,vecteur_tangent))
        
        u=a/L_tot
        x,y=courbe.courbe_point(u)
        duree+=dt

    return duree

#print(calcul_temps_courbe(courbe,array([15e3,0])))

def monte_carlo(x0,y0,xn,yn,v0,nombre_points_max,epsilon,affichage):
    start_time = time.time()
    print("## MONTE CARLO ##")
    print("Starting...")
    duree_min=float('inf')
    geodesique=None
    for n in range(2,nombre_points_max+1):
        for i in range(50):
            print("n, i:",n,i)
            U0=np.random.uniform(-Mars.pos[0],+Mars.pos[0],(2*n-4,))
            geod,duree=descente_gradient_heavy_ball(x0,y0,xn,yn,v0,U0,n,epsilon,affichage=False)
            if duree<duree_min:
                geodesique=geod
                duree_min=duree

    print('==Géodésique==')
    print('Nombre points:', geodesique.n+1)
    print('durée de parcours:', duree_min)
    end_time = time.time()
    print('Elapsed time:', end_time - start_time)

    if affichage:
        print(f"Durée de parcours : {duree:.2f} secondes")
        fig, ax = plt.subplots(figsize=(10, 10))

        SX, SY = np.meshgrid(np.linspace(-3e11, 3e11, 50), np.linspace(-3e11, 3e11, 50))
        chp = champ_espace(SX, SY)

        plt.figure(figsize=(10, 10))
        plt.quiver(SX, SY, chp[0], chp[1], color="blue", alpha=0.6, label="Champ gravitationnel")
        plt.scatter(0, 0, color="orange", label="Soleil", s=500)  # Soleil
        plt.scatter(149.6e9, 0, color="green", label="Terre", s=100)  # Terre
        plt.scatter(227.9e9, 0, color="red", label="Mars", s=80)  # Mars
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Y (m)")
        plt.axis('equal')
        plt.grid()

        bezierX, bezierY = geodesique.courbe(500)
        plt.plot(bezierX[0], bezierY[0], color="red", label="Courbe de Bézier", linewidth=2)
        plt.show()

def descente_gradient_heavy_ball(x0, y0, xn, yn,v0, U0, nombre_points, epsilon, affichage=True):

    def temps_from_points(vars):
        bezier = construire_bezier(x0, y0, xn, yn, vars)
        return calcul_temps_courbe(bezier,v0)

    geodesique = None
    L = float('inf')
    Uk = U0.copy()
    Gradk = gradient_f(temps_from_points, Uk)
    alpha = 0.1
    beta = 0.1
    k = 0
    while np.linalg.norm(Gradk) > epsilon:
        Uk, Up = Uk - alpha * Gradk + beta * (Uk - Up), Uk
        Gradk = gradient_f(temps_from_points, Uk)
        print(f"Iteration {k}: T = {temps_from_points(Uk):.6f}, |grad| = {np.linalg.norm(Gradk):.6f}")
        k += 1

    geodesique = construire_bezier(x0, y0, xn, yn, Uk)
    duree = calcul_temps_courbe(geodesique,v0)

    return geodesique, duree
    
def construire_bezier(x0,y0,xn,yn,vars):
    x=[x0]
    y=[y0]
    for i in range(len(vars)):
        if i%2==0:
            x.append(vars[i])
        else:
            y.append(vars[i])
    x.append(xn)
    y.append(yn)
    return bezierAPI.Bezier(x,y)

def derivee_partielle_i(f,i,vars):
    h=1e-6
    x_moins=vars.copy()
    x_moins[i]=x_moins[i]-h
    x_plus=vars.copy()
    x_plus[i]=x_plus[i]+h
    return (f(x_plus)-f(x_moins))/(2*h)

def gradient_f(f,vars):
    grad=np.zeros(len(vars))
    for i in range(len(vars)):
        grad[i]=derivee_partielle_i(f,i,vars)
    return grad


monte_carlo(Terre.pos[0]+6e6,Terre.pos[1]+6e6,Mars.pos[0]+3.3e6,Mars.pos[1]+3.3e6,array([-12e3,-12e3]),8,1,affichage=True)