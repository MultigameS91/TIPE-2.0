#OBJECTIF : Créer une modélisation des champs de vents terrestres et implémenter
#l'algorithme sur celui-ci.

#----------------------Modules pour plot-------------

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

import plotly.graph_objects as go

#---------------Affichage final de la Terre----------------
'''
fig = go.Figure(go.Scattergeo())
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=300,margin={"r":0,"t":0,"l":0,"b":0})
fig.show()'''

#--------------------- On crée la sphère qui représentera la Terre ----

R=6.4

'''
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("equal")

u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
x = 2*R*np.cos(u)*np.sin(v)
y = 2*R*np.sin(u)*np.sin(v)
z = 2*R*np.cos(v)
ax.plot_wireframe(x, y, z, color="blue",linewidth=0.2)

plt.show()'''

#------------On va travailler sur deux demi-shpères : on crée une demi-sphère---------

def S(x,y):
    return R*np.sin(np.acos(np.sqrt(x**2+y**2)/R))

#----------- FONCTIONS POUR CALCULER LES PROJECTIONS DU VENT----------

def alpha(x,y):
    return np.acos(np.sqrt(x**2+y**2)/R)

def prodcos1(x,y,vx,vy):
    return vx*np.cos(alpha(x,y))

def prodcos2(x,y,vx,vy):
    return vy*np.cos(alpha(x,y))

def prodsin(x,y,vx,vy):
    return -np.sqrt(vx**2+vy**2)*np.sin(alpha(x,y))

#-----------On applique notre algorithme sur une demi-sphère-----------

import time

def vitesse_vent(x,y,t):
    u = (4/100)*np.cos(y+t)*x**2
    v = (4/100)*np.sin(x)*(y+t)**2
    return u, v

binomial_memo={}

def k_parmi_n(k,n):
    if k == 0 :
        return 1
    if (n,k) in binomial_memo:
        return binomial_memo[(n,k)]
    c=(n-k+1)*k_parmi_n(k-1,n)//k
    binomial_memo[(n,k)]=c
    return c

class Bezier:
    # Il y a n+1 points
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
        self.n=np.size(X,0)-1

    def courbe(self,N):
        t=np.linspace(0,1,N)
        xBezier=np.zeros((1,N))
        yBezier=np.zeros((1,N))
        n=self.n
        for k in range(n+1):
            bnk=Bernstein(n,k).calcul(t)
            xBezier=bnk*self.X[k]+xBezier
            yBezier=bnk*self.Y[k]+yBezier
        return xBezier, yBezier
    
    def courbe_point(self,t):
        xBezier=0
        yBezier=0
        n=self.n
        for k in range(n+1):
            bnk=Bernstein(n,k).calcul(t)
            xBezier+=bnk*self.X[k]
            yBezier+=bnk*self.Y[k]
        return xBezier, yBezier
    
    def courbe_derivee(self,t):
        dx=0
        dy=0
        for k in range(self.n+1):
            deriv=Bernstein(self.n,k).calcul_derivee(t)
            dx+=deriv*self.X[k]
            dy+=deriv*self.Y[k]
        return dx,dy

    
    def longueur(self,N):
        S=0
        h=1/N
        for i in range(N):
            dx=0
            dy=0
            for k in range(self.n+1):
                bnk=Bernstein(self.n,k).calcul_derivee(i*h)
                dx+=bnk*self.X[k]
                dy+=bnk*self.Y[k]
            S+=np.sqrt(dx**2+dy**2)
        return S*h


class Bernstein:
    def __init__(self,n,k):
        self.n=n
        self.k=k

    def power(self,t,p):
        if p==0:
            return 1
        else:
            t2=self.power(t,p//2)
            if p%2==0:
                return t2*t2
            return t2*t2*t

    def calcul(self,t):
        n=self.n
        k=self.k
        return k_parmi_n(k,n)*(self.power(t,k))*(self.power(1-t,n-k))
    
    def calcul_derivee(self,t):
        n=self.n
        k=self.k
        d1=k*(self.power(t,k-1))*(self.power(1-t,n-k)) if k>0 else 0
        d2=-(n-k)*(self.power(t,k))*(self.power(1-t,n-k-1)) if n>k else 0
        return k_parmi_n(k,n)*(d1+d2)


def calcul_temps_courbe(bezier,champ_vent,v0):
    duree=0
    L_tot=bezier.longueur(500)
    a=0
    m=12*(10**3)
    S=20         #Surface de la voile
    C=0.5          #Coefficient de trainée du bateau
    p=1.2          #Masse volumique de l'air (on pourra choisir le fluide)
    alpha=0.5*p*S*C
    
    dt=1e-2
    u=a/L_tot
    x,y=bezier.courbe_point(u)
    om=np.array([x,y])
    v=v0

    while u<1:
        #print(duree,u)
        vent_apparent=np.array(list(champ_vent(om[0],om[1],duree)))-v
        force=(alpha*np.linalg.norm(vent_apparent))*vent_apparent
        vecteur_tangent=np.array(list(bezier.courbe_derivee(u)))
        vecteur_tangent=(1/np.linalg.norm(vecteur_tangent)) * vecteur_tangent
        force_projetee=np.dot(force,vecteur_tangent)*vecteur_tangent
        '''if np.dot(force,vecteur_tangent)<0:
            force_projetee=np.array([0,0])'''
        dvdt=(1/m)*force_projetee
        v=v+dt*dvdt
        om=om+dt*v
        a+=np.linalg.norm(dt*v)
        u=a/L_tot
        duree+=dt
    return duree

#calcul_temps_courbe(Bezier([-2,4],[0,4]))

def monte_carlo(champ_vent,x0,y0,xn,yn,v0,nombre_points_max,epsilon,affichage):
    start_time = time.time()
    print("## MONTE CARLO ##")
    print("Starting...")
    duree_min=float('inf')
    geodesique=None
    for n in range(2,nombre_points_max+1):
        for i in range(20):
            print("n, i:",n,i)
            U0=np.random.uniform(-5,5,(2*n-4,))
            geod,duree=descente_gradient_heavy_ball(vitesse_vent,x0,y0,xn,yn,v0,U0,n,epsilon,affichage=False)
            if duree<duree_min:
                geodesique=geod
                duree_min=duree

    print('==Géodésique==')
    print('Nombre points:', nombre_points)
    print('durée de parcours:', duree_min)
    end_time = time.time()
    print('Elapsed time:', end_time - start_time)

    if affichage:
        print(f"Durée de parcours : {duree:.2f} secondes")
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")

        # Courbe de Bézier
        bezierX, bezierY = geodesique.courbe(500)
        ax.plot(bezierX[0], bezierY[0],S(bezierX[0], bezierY[0]), color="red", label="Courbe de Bézier", linewidth=2)

        # Champ de vent
        SX, SY = np.meshgrid(np.linspace(-6.4, 6.4, 50), 
                            np.linspace(-6.4, 6.4, 50))
        VX, VY = champ_vent(SX, SY,t=0)
        SZ=S(SX,SY)
        courbe=ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 1)
        A=alpha(SX,SY)     #Angle pour calculer la projection du vent sur la surface
        NVX=prodcos1(SX,SY,VX,VY)
        NVY=prodcos2(SX,SY,VX,VY)
        NVZ=prodsin(SX,SY,VX,VY)
        VZ=S(VX,VY)
        ax.quiver(SX, SY,SZ, NVX, NVY,NVZ, color="green", label="Champ de vent", alpha=0.6,linewidth=0.4)

        # Points de départ et d'arrivée
        ax.scatter([bezierX[0][0]], [bezierY[0][0]],[S(bezierX[0][0],bezierY[0][0])], color="blue", s=100, label="Départ")
        ax.scatter([bezierX[0][-1]], [bezierY[0][-1]],[S(bezierX[0][-1],bezierY[0][-1])], color="purple", s=100, label="Arrivée")

        # Légende et détails
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.legend()
        ax.axis("equal")
        plt.grid()
        plt.show()
    
                
def descente_gradient_heavy_ball(champ_vent, x0, y0, xn, yn,v0, U0, nombre_points, epsilon, affichage=True):

    def temps_from_points(vars):
        bezier = construire_bezier(x0, y0, xn, yn, vars)
        return calcul_temps_courbe(bezier,champ_vent,v0)

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
    duree = calcul_temps_courbe(geodesique,champ_vent,v0)
    

    '''if affichage:
        print(f"Durée de parcours : {duree:.2f} secondes")
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")

        # Courbe de Bézier
        bezierX, bezierY = geodesique.courbe(500)
        ax.plot(bezierX[0], bezierY[0], color="red", label="Courbe de Bézier", linewidth=2)

        # Champ de vent
        SX, SY = np.meshgrid(np.linspace(min(bezierX[0]) - 2, max(bezierX[0]) + 2, 20), 
                            np.linspace(min(bezierY[0]) - 2, max(bezierY[0]) + 2, 20))
        VX, VY = champ_vent(SX, SY,0)
        ax.quiver(SX, SY, VX, VY, color="green", label="Champ de vent", alpha=0.6)

        # Points de départ et d'arrivée
        ax.scatter([bezierX[0][0]], [bezierY[0][0]],[S(bezierX[0][0],bezierY[0][0])], color="blue", s=100, label="Départ")
        ax.scatter([bezierX[0][-1]], [bezierY[0][-1]],[S(bezierX[0][-1],bezierY[0][-1])], color="purple", s=100, label="Arrivée")

        # Légende et détails
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_zlabel("z", fontsize=14)
        ax.legend()
        ax.axis("equal")
        plt.show()'''
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
    return Bezier(x,y)

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

epsilon=1e-4
nombre_points=4
x0,y0=-2,-5
xn,yn=4,3
#descente_gradient_heavy_ball(vitesse_vent,x0,y0,xn,yn,np.array([0,5]),nombre_points,epsilon)

monte_carlo(vitesse_vent,x0,y0,xn,yn,np.array([0,-1]),nombre_points,epsilon,affichage=True)



