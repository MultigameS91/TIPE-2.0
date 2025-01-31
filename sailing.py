import numpy as np
import matplotlib.pyplot as plt
import time

def S(x,y):
    return 0*x

def vitesse_vent1(x,y):
    u = 10*np.cos(y)
    v = 10*np.sin(x)
    return u,v

def vitesse_vent(om):
    return np.array(list(vitesse_vent1(om[0],om[1])))

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


def calcul_temps_courbe(bezier):
    duree=0
    L_tot=bezier.longueur(500)
    a=0
    m=12*(10**3)
    S=20         #Surface de la voile
    C=2          #Coefficient de trainée du bateau
    p=1          #Masse volumique de l'air (on pourra choisir le fluide)
    alpha=0.5*p*S*C
    
    dt=1e-2
    u=a/L_tot
    x,y=bezier.courbe_point(u)
    om=np.array([x,y])
    v=np.array([0,-1])

    while u<1:
        #print(duree,u)
        vent_apparent=vitesse_vent(om)-v
        force=(alpha*np.linalg.norm(vent_apparent))*vent_apparent
        vecteur_tangent=np.array(list(bezier.courbe_derivee(u)))
        vecteur_tangent=(1/np.linalg.norm(vecteur_tangent)) * vecteur_tangent
        force_projetee=np.dot(force,vecteur_tangent)*vecteur_tangent
        if np.dot(force,vecteur_tangent)<0:
            force_projetee=np.array([0,0])
        dvdt=(1/m)*force_projetee
        v=v+dt*dvdt
        om=om+dt*v
        a+=np.linalg.norm(dt*v)
        u=a/L_tot
        duree+=dt
    #print(duree)
    bezierX,bezierY=bezier.courbe(500)

    #plt.plot(bezierX[0],bezierY[0],color="red")
    #SX, SY = np.meshgrid(np.linspace(-5, 5, 25), np.linspace(-5, 5, 25))
    #plt.quiver(SX,SY,vitesse_vent1(SX,SY)[0],vitesse_vent1(SX,SY)[1],color="blue")
    #plt.show()
    return duree

#calcul_temps_courbe(Bezier([-2,4],[0,4]))

def monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max):
    CI=None
    T=float('inf')
    for n in range(2,nombre_points_max+1):
        for i in range(20):
            Uk=np.random.uniform(-5,5,(2*n-4,))
            bezier=construire_bezier(x0,y0,xn,yn,Uk)
            t=calcul_temps_courbe(bezier)
            if t<T:
                CI=Uk,n
                T=t
    return CI


def descente_gradient_heavy_ball(S,x0,y0,xn,yn,nombre_points_max,epsilon,affichage=True):
    # fonction à minimiser
    print("## HEAVY BALL ##")
    print("Starting...")
    start_time=time.time()
    def temps_from_points(vars):
        bezier=construire_bezier(x0,y0,xn,yn,vars)
        return calcul_temps_courbe(bezier)
    geodesique=None
    L=float('inf')

    Up,nombre_points=monte_carlo_condition_initiale(S,x0,y0,xn,yn,nombre_points_max)
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
    T=calcul_temps_courbe(geodesique)
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
        plt.xlabel('x')
        plt.ylabel('y')
        ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 0.5)
        ax.plot_wireframe(SX, SY, np.zeros(np.shape(SX)), color="gray", linewidth = 1)

        #plt.plot(xBezier[0],yBezier[0],S(xBezier,yBezier), linewidth = 3, color='red')
        plt.scatter([x0,xn],[y0,yn],S(np.array([x0,xn]),np.array([y0,yn])),c='black')
        plt.plot(xBezier[0],yBezier[0],np.zeros(np.shape(xBezier)), linewidth = 2, color='blue')
        plt.quiver(SX,SY,[0.0 for i in range(len(SX))],vitesse_vent1(SX,SY)[0],vitesse_vent1(SX,SY)[1],[0.0 for i in range(len(SX))],color="green", length = 0.1, normalize = True)
        plt.axis('scaled')
        plt.show()

    return geodesique, T
    
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

epsilon=1e-2
x0,y0=-4,2
xn,yn=2,-2
nombre_points=4
descente_gradient_heavy_ball(S,x0,y0,xn,yn,nombre_points,epsilon)