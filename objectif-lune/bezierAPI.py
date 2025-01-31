import numpy as np

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