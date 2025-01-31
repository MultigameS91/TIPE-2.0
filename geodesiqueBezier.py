import numpy as np
import matplotlib.pyplot as plt
import time

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
    
def derivee_surface(S,x,y):
    def S_as_array(var):
        return S(var[0],var[1]) 
    return derivee_partielle_i(S_as_array,0,np.array([x,y])), derivee_partielle_i(S_as_array,1,np.array([x,y]))

###########
#methode des rectangles erreur=O(1/N) ou N=nbre de rectangles
def longueur_sur_courbe_rectangles(bezier,N,S):
    h=1/N
    I=0
    for i in range(N):
        t=i*h
        cpx,cpy=bezier.courbe_point(t)
        dcpx,dcpy=bezier.courbe_derivee(t)
        dSx,dSy=derivee_surface(S,x,y)
        z=dcpx*dSx+dcpy*dSy
        x=dcpx
        y=dcpy
        I+=np.sqrt(x**2+y**2+z**2)
    return I*h

def affichage_2d(S,courbe_bezier):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = S(X, Y)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=10, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.colorbar(contour, label="Altitude")
    N_pas=500
    xBezier,yBezier = courbe_bezier.courbe(N_pas)
    x0,y0=xBezier[0,0],yBezier[0,0]
    xn,yn=xBezier[0,-1],yBezier[0,-1]
    plt.scatter([x0,xn],[y0,yn],c='black')
    plt.plot(xBezier[0],yBezier[0],color='red')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)

    plt.show()

#methode de simpson erreur=O(1/N**4) ou N=nbre de rectangles
def longueur_sur_courbe_simpson(bezier,N,S):
    h=1/N
    I=0

    def dl(t):
        cpx,cpy=bezier.courbe_point(t)
        dcpx,dcpy=bezier.courbe_derivee(t)
        dSx,dSy=derivee_surface(S,cpx,cpy)
        z=dcpx*dSx+dcpy*dSy
        x=dcpx
        y=dcpy
        return np.sqrt(x**2+y**2+z**2)

    for k in range(N):
        f1=dl(k*h)
        f2=dl((k+0.5)*h)
        f3=dl((k+1)*h)
        I+=f1+4*f2+f3
    return h/6*I

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

#vars=[x1,y1,x2,y2,...,x(n-1),y(n-1)]
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

def monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max,N_integral):
    CI=None
    L=float('inf')
    for n in range(2,nombre_points_max+1):
        for i in range(20):
            Uk=np.random.uniform(-5,5,(2*n-4,))
            bezier=construire_bezier(x0,y0,xn,yn,Uk)
            l=longueur_sur_courbe_simpson(bezier,N_integral,S)
            if l<L:
                CI=Uk,n
                L=l
    return CI

def descente_gradient(S,x0,y0,xn,yn,nombre_points_max,N_integral,epsilon,affichage=False):
    # fonction à minimiser
    print("## DESCENTE GRADIENT ##")
    print("Starting...")
    start_time=time.time()
    def longueur_from_points(vars):
        bezier=construire_bezier(x0,y0,xn,yn,vars)
        return longueur_sur_courbe_simpson(bezier,N_integral,S)
    geodesique=None
    L=float('inf')
    
    Uk,nombre_points=monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max,N_integral)
    Gradk=gradient_f(longueur_from_points,Uk)
    alpha=0.1
    k=0
    while np.linalg.norm(Gradk)>epsilon:
        Uk=Uk-alpha*Gradk
        Gradk=gradient_f(longueur_from_points,Uk)
        print(f"Iteration {k}: L = {longueur_from_points(Uk):.6f}, |grad| = {np.linalg.norm(Gradk):.6f}")
        k+=1

    geodesique=construire_bezier(x0,y0,xn,yn,Uk)
    L=longueur_sur_courbe_simpson(geodesique,N_integral,S)
        
    print("## DESCENTE GRADIENT ##")
    print('==Géodésique==')
    print('Nombre points:', nombre_points)
    print('Longueur:', L)
    end_time=time.time()
    print('Elapsed time:',end_time-start_time)

    if affichage:
        # Maillage
        N_bezier=500
        xBezier,yBezier = geodesique.courbe(N_bezier)

        SX, SY = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        SZ = S(SX,SY)

        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")
        ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 0.5)
        ax.plot_wireframe(SX, SY, np.zeros(np.shape(SX)), color="gray", linewidth = 1)

        plt.plot(xBezier[0],yBezier[0],S(xBezier,yBezier), linewidth = 3, color='red')
        plt.scatter([x0,xn],[y0,yn],S(np.array([x0,xn]),np.array([y0,yn])),c='black')
        plt.plot(xBezier[0],yBezier[0],np.zeros(np.shape(xBezier)), linewidth = 2, color='blue')

        plt.axis('scaled')
        plt.show()

    return geodesique, L

# Polyak
def descente_gradient_heavy_ball(S,x0,y0,xn,yn,nombre_points_max,N_integral,epsilon,affichage=False):
    # fonction à minimiser
    print("## HEAVY BALL ##")
    print("Starting...")
    start_time=time.time()
    def longueur_from_points(vars):
        bezier=construire_bezier(x0,y0,xn,yn,vars)
        return longueur_sur_courbe_simpson(bezier,N_integral,S)
    geodesique=None
    L=float('inf')

    Up,nombre_points=monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max,N_integral)
    Uk=Up.copy()
    Gradk=gradient_f(longueur_from_points,Uk)
    alpha=0.1
    beta=0.1
    k=0
    while np.linalg.norm(Gradk)>epsilon:
        Uk, Up=Uk-alpha*Gradk+beta*(Uk-Up), Uk
        Gradk=gradient_f(longueur_from_points,Uk)
        print(f"Iteration {k}: L = {longueur_from_points(Uk):.6f}, |grad| = {np.linalg.norm(Gradk):.6f}")
        k+=1

    geodesique=construire_bezier(x0,y0,xn,yn,Uk)
    L=longueur_sur_courbe_simpson(geodesique,N_integral,S)
    print("## HEAVY BALL ##")
    print('==Géodésique==')
    print('Nombre points:', nombre_points)
    print('Longueur:', L)
    end_time=time.time()
    print('Elapsed time:',end_time-start_time)
    if affichage:
        # Maillage
        N_bezier=500
        xBezier,yBezier = geodesique.courbe(N_bezier)

        SX, SY = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        SZ = S(SX,SY)

        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")
        ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 0.5)
        ax.plot_wireframe(SX, SY, np.zeros(np.shape(SX)), color="gray", linewidth = 1)

        plt.plot(xBezier[0],yBezier[0],S(xBezier,yBezier), linewidth = 3, color='red')
        plt.scatter([x0,xn],[y0,yn],S(np.array([x0,xn]),np.array([y0,yn])),c='black')
        plt.plot(xBezier[0],yBezier[0],np.zeros(np.shape(xBezier)), linewidth = 2, color='blue')

        plt.axis('scaled')
        plt.show()

    return geodesique, L

# 
def descente_gradient_nesterov(S,x0,y0,xn,yn,nombre_points_max,N_integral,epsilon,affichage=False):
    # fonction à minimiser
    print("## Nesterov ##")
    print("Starting...")
    start_time=time.time()
    def longueur_from_points(vars):
        bezier=construire_bezier(x0,y0,xn,yn,vars)
        return longueur_sur_courbe_simpson(bezier,N_integral,S)
    geodesique=None
    L=float('inf')
    
    xk,nombre_points=monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max,N_integral)
    yp=xk.copy()
    yk=yp.copy()
    Gradk=gradient_f(longueur_from_points,yk)
    lambdap=0
    lambdak=0
    gammak=1/(0.5*(1+np.sqrt(1+4*lambdak)))
    k=0
    while np.linalg.norm(Gradk)>epsilon:
        yk,yp=xk-1/10*gradient_f(longueur_from_points,xk),yk
        lambdak,lambdap=0.5*(1+np.sqrt(1+4*lambdak)),lambdak
        gammak=(1-lambdap)/lambdak
        xk=(1-gammak)*yk+gammak*yp
        Gradk=gradient_f(longueur_from_points,yk)
        print(f"Iteration {k}: L = {longueur_from_points(yk):.6f}, |grad| = {np.linalg.norm(Gradk):.6f}")
        k+=1

    geodesique=construire_bezier(x0,y0,xn,yn,yk)
    L=longueur_sur_courbe_simpson(geodesique,N_integral,S)
    
    print("## Nesterov ##")
    print('==Géodésique==')
    print('Nombre points:', nombre_points)
    print('Longueur:', L)
    end_time=time.time()
    print('Elapsed time:',end_time-start_time)

    if affichage:
        # Maillage
        N_bezier=500
        xBezier,yBezier = geodesique.courbe(N_bezier)

        SX, SY = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        SZ = S(SX,SY)

        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")
        ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 0.5)
        ax.plot_wireframe(SX, SY, np.zeros(np.shape(SX)), color="gray", linewidth = 1)

        plt.plot(xBezier[0],yBezier[0],S(xBezier,yBezier), linewidth = 3, color='red')
        plt.scatter([x0,xn],[y0,yn],S(np.array([x0,xn]),np.array([y0,yn])),c='black')
        plt.plot(xBezier[0],yBezier[0],np.zeros(np.shape(xBezier)), linewidth = 2, color='blue')

        plt.axis('scaled')
        plt.show()
    
    return geodesique, L

### SURFACE 1 ###
epsilon=1e-2
def S1(x,y):
    return x**2+y**2

x0,y0=-5,-2
xn,yn=2,2
nombre_points=4
#descente_gradient(S1,x0,y0,xn,yn,nombre_points,10,epsilon)

### SURFACE 2 ###
def S2(x,y):
    return np.sin(x)

x0,y0=-4,2
xn,yn=2,-2
nombre_points=4
#descente_gradient_heavy_ball(S2,x0,y0,xn,yn,nombre_points,10,epsilon)
#descente_gradient_nesterov(S2,x0,y0,xn,yn,nombre_points,10,epsilon)
#descente_gradient(S2,x0,y0,xn,yn,nombre_points,10,epsilon)

### SURFACE 0 TEST ###

def S0 (x,y):
    return x

x0,y0=-4,2
xn,yn=2,-2
nombre_points=6
#descente_gradient(S0,x0,y0,xn,yn,nombre_points,10,epsilon)

### SURFACE 3 ###
def S3(x,y):
    return np.sin(1/((x*y)**2+1))

x0,y0=-4,2
xn,yn=2,-2
nombre_points=4
#descente_gradient_heavy_ball(S3,x0,y0,xn,yn,nombre_points,10,epsilon)
#descente_gradient_nesterov(S3,x0,y0,xn,yn,nombre_points,10,epsilon)
#descente_gradient(S3,x0,y0,xn,yn,nombre_points,10,epsilon)


def descente_gradient_bis(S,S_derivee,x0,y0,xn,yn,nombre_points_max,N_integral):
    # fonction à minimiser
    def longueur_from_points(vars):
        bezier=construire_bezier(x0,y0,xn,yn,vars)
        return longueur_sur_courbe_simpson(bezier,N_integral,S)
    geodesique=None
    L=float('inf')
    for nombre_points in range(2,nombre_points_max+1):
        epsilon=0.1
        Uk=np.random.uniform(-5,5,(2*nombre_points-4,))
        Gradk=gradient_f(longueur_from_points,Uk)
        alpha=0.1
        k=0
        print(np.linalg.norm(Gradk))
        while np.linalg.norm(Gradk)>epsilon:
            Uk=Uk-alpha*Gradk
            Gradk=gradient_f(longueur_from_points,Uk)
            #print(f"Iteration {k}: L = {longueur_from_points(Uk):.6f}, |grad| = {np.linalg.norm(Gradk):.6f}")
            k+=1

        bezier=construire_bezier(x0,y0,xn,yn,Uk)
        l=longueur_sur_courbe_simpson(bezier,N_integral,S)
        if l<L:
            geodesique=bezier
            L=l
            nb_points_ideal=nombre_points
    return geodesique

def descente_gradient_nesterov_bis(S,x0,y0,xn,yn,nombre_points_max,N_integral):
    # fonction à minimiser
    def longueur_from_points(vars):
        bezier=construire_bezier(x0,y0,xn,yn,vars)
        return longueur_sur_courbe_simpson(bezier,N_integral,S)
    geodesique=None
    L=float('inf')
    
    xk,nombre_points=monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max,N_integral)
    yp=xk.copy()
    yk=yp.copy()
    Gradk=gradient_f(longueur_from_points,yk)
    lambdap=0
    lambdak=0
    gammak=1/(0.5*(1+np.sqrt(1+4*lambdak)))
    k=0
    while np.linalg.norm(Gradk)>epsilon:
        yk,yp=xk-1/10*gradient_f(longueur_from_points,xk),yk
        lambdak,lambdap=0.5*(1+np.sqrt(1+4*lambdak)),lambdak
        gammak=(1-lambdap)/lambdak
        xk=(1-gammak)*yk+gammak*yp
        Gradk=gradient_f(longueur_from_points,yk)
        print(f"Iteration {k}: L = {longueur_from_points(yk):.6f}, |grad| = {np.linalg.norm(Gradk):.6f}")
        k+=1

    geodesique=construire_bezier(x0,y0,xn,yn,yk)
    #L=longueur_sur_courbe_simpson(geodesique,N_integral,S)
    return geodesique

def surface_mouvante(Tmax,N_pas):
    h=1/N_pas

    T=[]
    for i in range(int(Tmax*N_pas)):
        t=(i%N_pas)*h
        if i//N_pas%2==1:
            t=(1-(i%N_pas)*h)
        T.append(t)
    
    def Surface_t(x,y,t):
        return np.sin(t*x*y)

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection="3d")
    plt.ion()
    x0,y0=0,0
    xn,yn=4,-2
    geodesique=None
    courbe=None
    courbe_geo=None
    cheminX,cheminY,cheminZ=[x0],[y0],[Surface_t(x0,y0,T[0])]
    for t in T:
        SX, SY = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        SZ = Surface_t(SX,SY,t)
        courbe=ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 0.5)
        def S_en_cours(x,y):
            return np.sin(x*y*t)
        def S_en_cours_d(x,y):
            return t*y*np.cos(x*y*t),t*x*np.cos(x*y*t)
    
        plt.scatter([x0,xn],[y0,yn],S_en_cours(np.array([x0,xn]),np.array([y0,yn])),c='black')

        geodesique=descente_gradient_nesterov_bis(S_en_cours,x0,y0,xn,yn,nombre_points,10)
        xBezier,yBezier = geodesique.courbe(N_pas)
        x0,y0=xBezier[0,1],yBezier[0,1]
        cheminX.append(x0)
        cheminY.append(y0)
        cheminZ.append(S_en_cours(x0,y0))
        courbe_geo=ax.plot(xBezier[0],yBezier[0],S_en_cours(xBezier,yBezier), linewidth = 3, color='red')
        print(courbe_geo)
        proj=ax.plot(xBezier[0],yBezier[0],np.zeros(np.shape(xBezier)), linewidth = 2, color='blue')
        plt.draw()
        plt.pause(0.05)
        courbe.remove()
        liste_proj=proj.pop(0)
        liste_proj.remove()
        liste_geo=courbe_geo.pop(0)
        liste_geo.remove()

surface_mouvante(5,100)

#https://math.stackexchange.com/questions/1708146/shortest-path-between-two-points-on-a-surface
        
