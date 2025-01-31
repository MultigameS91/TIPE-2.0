import matplotlib.pyplot as plt
import numpy as np

class Polynome:
    # coeff = liste des coeffs
    def __init__(self,coeff):
        self.coeff=coeff

    def degre(self):
        return len(self.coeff)-1

    # ne change pas cette instance mais renvoie une copie
    def somme(self,Q):
        max_deg=max(self.degre(),Q.degre())
        new_coeff=[0 for i in range(max_deg+1)]
        for i in range(self.degre()):
            new_coeff[i] = new_coeff[i]+self.coeff[i]

        for i in range(Q.degre()):
            new_coeff[i] = new_coeff[i]+self.coeff[i]

        return Polynome(new_coeff)

    def multi_scalaire(self,a):
        deg=self.degre()
        new_coeff=[a*self.coeff[i] for i in range(deg+1)]
        return Polynome(new_coeff)

    def multi_uni(self,n):
        deg=self.degre()
        new_coeff=[0 for i in range(n+deg+1)]
        for i in range(n,n+deg+1):
            new_coeff[i]=self.coeff[i-n]
        return Polynome(new_coeff)

class Point:
    def __init__(self,x,y):
        self.x=x
        self.y=y

binomial_memo={}

def k_parmi_n(n,k):
    if k == 0 :
        return 1
    if (n,k) in binomial_memo:
        return binomial_memo[(n,k)]
    c=(n-k+1)*k_parmi_n(n,k-1)//k
    binomial_memo[(n,k)]=c
    return c

def puissance(t,n):
    if n == 0 :
        return 1
    if n == 1 :
        return t
    p=puissance(t,n//2)
    if n%2==0:
        return p*p
    return t*p*p

class Bernstein():
    def __init__(self,n,i):
        self.n=n
        self.i=i

    def calculate(self,t):
        i=self.i
        n=self.n
        return (k_parmi_n(n,i)*(puissance(t,i)*puissance(1-t,n-i)))

    def calculate_derivee(self,t):
        i=self.i
        n=self.n
        tpi=puissance(t,i-1)
        tpni=puissance(1-t,n-i-1)
        return k_parmi_n(n,i)*(i*tpi*tpni*(1-t) + (n-i)*t*tpi*tpni)

class Bezier:
    def __init__(self,ancrages):
        self.ancrages=ancrages
        self.n=len(ancrages)-1

    def position(self,t):
        x=0
        y=0
        for i in range(self.n):
            bni=Bernstein(self.n,i).calculate(t)
            x+=self.ancrages[i][0]*bni
            y+=self.ancrages[i][1]*bni
        return (x,y)

bezier=Bezier([(0,0),(2,5),(1,2),(31,-6)])

plt.plot(np.linspace(0,1,100),Bernstein(3,0).calculate(np.linspace(0,1,100)))
plt.plot(np.linspace(0,1,100),Bernstein(3,1).calculate(np.linspace(0,1,100)))
plt.plot(np.linspace(0,1,100),Bernstein(3,2).calculate(np.linspace(0,1,100)))
plt.plot(np.linspace(0,1,100),Bernstein(3,3).calculate(np.linspace(0,1,100)))
plt.show()

X=[]
Y=[]
h=0.001
t=0
while t<1:
    point=bezier.position(t)
    X.append(point[0])
    Y.append(point[1])
    t+=h

print(bezier.position(1))

plt.plot(X,Y)
plt.show()    
