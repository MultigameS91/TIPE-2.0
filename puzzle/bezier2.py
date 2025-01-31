import numpy as np
import matplotlib.pyplot as plt

x=np.array([0.4,1])
y=np.array([0.4, 0])

class Bezier:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.n=np.size(x,0)

CELLS = 100
nCPTS = np.size(x,0)
n=nCPTS-1
i=0
t=np.linspace(0,1,CELLS)

xBezier=np.zeros((1,CELLS))
yBezier=np.zeros((1,CELLS))

binomial_memo={}

def k_parmi_n(n,k):
    if k == 0 :
        return 1
    if (n,k) in binomial_memo:
        return binomial_memo[(n,k)]
    c=(n-k+1)*k_parmi_n(n,k-1)//k
    binomial_memo[(n,k)]=c
    return c

class Bernstein():
    def __init__(self,n,i):
        self.n=n
        self.i=i

    def calculate(self,t):
        i=self.i
        n=self.n
        return k_parmi_n(n,i)*(t**i)*(1-t)**(n-i)

    def calculate_derivee(self,t):
        i=self.i
        n=self.n
        if i==0:
            return 0
        return k_parmi_n(n,i)*(i*(t**(i-1))*((1-t)**(n-i)) - (n-i)*(t**i)*((1-t)**(n-i-1)))

for k in range(nCPTS):
    xBezier=Bernstein(n,k).calculate(t)*x[k]+xBezier
    yBezier=Bernstein(n,k).calculate(t)*y[k]+yBezier

fig1=plt.figure(figsize=(4,4))
ax1=fig1.add_subplot(111)
ax1.scatter(x,y,c='black')
ax1.plot(xBezier[0],yBezier[0],c="blue")

def calcul_norme(bezier,t):
    dx=0
    dy=0
    for k in range(bezier.n):
        dx+=Bernstein(n,k).calculate_derivee(t)*x[k]
        dy+=Bernstein(n,k).calculate_derivee(t)*y[k]
    return np.sqrt(
        (dx)**2 +
        (dy)**2
    )

def longueur(bezier,N):
    h=1/N
    S=0
    for i in range(N):
        S+=calcul_norme(bezier,i*h)
    return S*h
print(x,y)
print("old",np.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2))
print("longueur geod",longueur(Bezier(x,y),500))
plt.show()