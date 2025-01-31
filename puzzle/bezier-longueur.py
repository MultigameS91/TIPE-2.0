import numpy as np
import matplotlib.pyplot as plt

class Bezier:
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
            dx+=deriv*X[k]
            dy+=deriv*Y[k]
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

binomial_memo={}

def k_parmi_n(k,n):
    if k == 0 :
        return 1
    if (n,k) in binomial_memo:
        return binomial_memo[(n,k)]
    c=(n-k+1)*k_parmi_n(k-1,n)//k
    binomial_memo[(n,k)]=c
    return c

nbre_points=5
X=np.random.uniform(-5,5,(nbre_points,))
Y=np.random.uniform(-5,5,(nbre_points,))
bezier=Bezier(X,Y)
xBezier,yBezier=bezier.courbe(500)
print("longueur geod",bezier.longueur(500))
#print("longueur sqrt", np.sqrt((X[1]-X[0])**2+(Y[1]-Y[0])**2))
plt.scatter(X,Y,c='black')
plt.plot(xBezier[0],yBezier[0],c="blue")
#plt.axis('scaled')
plt.show()

def S(x,y):
    return x**2+y**2

def S_der(x,y):
    return 2*x,2*y

SX, SY = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
SZ = S(SX,SY)

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")
ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 1)
ax.plot_wireframe(SX, SY, np.zeros(np.shape(SX)), color="gray", linewidth = 1)

#plt.axis('scaled')

def longueur_sur_courbe(bezier,N):
    h=1/N
    S=0
    for i in range(N):
        t=i*h
        cx,cy=bezier.courbe_point(t)
        dcpx,dcpy=bezier.courbe_derivee(t)
        dSx,dSy=S_der(cx,cy)
        z=dcpx*dSx+dcpy*dSy
        x=dcpx
        y=dcpy
        S+=np.sqrt(x**2+y**2+z**2)
    return S*h

def derivee_partielle_i(f,i,vars):
    h=0.01
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

x0,y0=-2,-2
xn,yn=0,0
def longueur_from_points(vars):
    bezier=construire_bezier(x0,y0,xn,yn,vars)
    return longueur_sur_courbe(bezier,500)

Xk=np.random.uniform(-5,5,(2*nbre_points-4,))
epsilon=0.1
Grad_k=gradient_f(longueur_from_points,Xk)
alpha0=0.01
alpha=alpha0
k=0
while np.linalg.norm(Grad_k)>epsilon:
    Xk=Xk-alpha*Grad_k
    Grad_k=gradient_f(longueur_from_points,Xk)
    print(f"Iteration {k}: L = {longueur_from_points(Xk):.6f}, |grad| = {np.linalg.norm(Grad_k):.6f}")
    k+=1

best_bezier=construire_bezier(x0,y0,xn,yn,Xk)
xBezier,yBezier = best_bezier.courbe(500)
ax.plot(xBezier[0],yBezier[0],S(xBezier,yBezier), linewidth = 3, color='red')
ax.scatter([x0,xn],[y0,yn],S(np.array([x0,xn]),np.array([y0,yn])),c='black')
ax.plot(xBezier[0],yBezier[0],np.zeros(np.shape(xBezier)), linewidth = 2, color='blue')
plt.show()

