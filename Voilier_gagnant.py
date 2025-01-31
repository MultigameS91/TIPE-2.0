import numpy as np
import matplotlib.pyplot as plt
import time

binomial_memo={}

#----------------------------------Propriétés-bateau-air-eau----------------------------

'''PREMIER MODELE ON NE CONSIDERE PAS LE GLISSEMENT DU BATEAU SUR L'EAU ET LA MER EST PLANE'''

M=12*(10**3)
S=20         #Surface de la voile
C=2          #Coefficient de trainée du bateau
p=1          #Masse volumique de l'air (on pourra choisir le fluide)

def S(x,y):
    return 0


#----------------------------------VENT-----------------------------------------

Liste_des_champs=[]         #C'est une liste comportant les vitesses et les directions du vent à chaque endroit à chaque instant. On la suppose connue.

def intensite_champ_vitesses(t,x,y):
    '''renvoie les conditions météos (vitesse du vent à l'instant t au point de coordonnées x,y,z ainsi que l'angle alpha
    formé entre le vent et le bateau) '''
    return Liste_des_champs[x][y][t][0],Liste_des_champs[x][y][t][1]

def vitesse_bateau(t):
    return derivee_x(t),derivee_y(t)

def force_sur_bateau(t,x,y):
    v_bateau=vitesse_bateau(t)
    v_vent,angle=intensite_champ_vitesses(t,x,y)
    v_vent_selon_angle=v_vent[0]*np.sin(angle),v_vent[1]*np.sin(angle)#Il faut considérer l'effet du vent selon l'angle du bateau
    return (1/2)*C*p*S*(v_vent_selon_angle[0]-v_bateau[0])**2,(1/2)*C*p*S*(v_vent_selon_angle[1]-v_bateau[1])**2

dt=0.1 #C'est le temps fictif entre deux points lors du tracé de la courbe DIFFERENT du temps pour faire un chemin dl à la vitesse

def vitesse_bateau_apres(t):
    v_x_dx=vitesse_bateau(t)[0]+(dt/M)*force_sur_bateau(t,x,y)[0]
    v_y_dy=vitesse_bateau(t)[1]+(dt/M)*force_sur_bateau(t,x,y)[1]
    return v_x_dx,v_y_dy


#--------------------------------MINIMISATION DE LA LONGUEUR-------------------------

#--------------------------Calculs de longueurs----------

def Temps_de_courbe_simpson(bezier,N,S):
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

    def v_ap(t):
        vxa,vya=vitesse_bateau_apres(t)
        return np.sqrt(vxa**2+vya**2)

    def dT(t):
        return dl(t)/v_ap(t)


    for k in range(N):
        f1=dT(k*h)
        f2=dT((k+0.5)*h)
        f3=dT((k+1)*h)
        I+=f1+4*f2+f3
    return h/6*I

#vars=[x1,y1,x2,y2,...,x(n-1),y(n-1)]

def monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max,N_integral):
    CI=None
    T=float('inf')
    for n in range(2,nombre_points_max+1):
        for i in range(20):
            Uk=np.random.uniform(-5,5,(2*n-4,))
            bezier=construire_bezier(x0,y0,xn,yn,Uk)
            t=temps_de_courbe_simpson(bezier,N_integral,S)
            if t<T:
                CI=Uk,n
                T=t
    return CI

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

#------------------------Descente de gradient-------------------

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

def descente_gradient_heavy_ball(S,x0,y0,xn,yn,nombre_points_max,N_integral,epsilon,affichage=False):
    # fonction à minimiser
    print("## HEAVY BALL ##")
    print("Starting...")
    start_time=time.time()
    def temps_from_points(vars):
        bezier=construire_bezier(x0,y0,xn,yn,vars)
        return Temps_de_courbe_simpson(bezier,N_integral,S)
    geodesique=None
    L=float('inf')

    Up,nombre_points=monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max,N_integral)
    Uk=Up.copy()
    Gradk=gradient_f(temps_from_points,Uk)
    alpha=0.1
    beta=0.1
    k=0
    while np.linalg.norm(Gradk)>epsilon:
        Uk, Up=Uk-alpha*Gradk+beta*(Uk-Up), Uk
        Gradk=gradient_f(temps_from_points,Uk)
        print(f"Iteration {k}: T = {temps_from_points(Uk):.6f}, |grad| = {np.linalg.norm(Gradk):.6f}")
        k+=1

    geodesique=construire_bezier(x0,y0,xn,yn,Uk)
    T=Temps_sur_courbe_simpson(geodesique,N_integral,S)
    print("## HEAVY BALL ##")
    print('==Géodésique==')
    print('Nombre points:', nombre_points)
    print('TEMPS:', T)
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

    return geodesique, T

#----------------------------------BEZIER---------------------------------------

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
