import matplotlib.pyplot as plt
import numpy as np

class Graphe:
    def __init__(self,_sommets,_aretes):
        self.sommets=_sommets
        self.aretes=_aretes

def f(x,y):
    return np.sin(np.log(abs(x*y/100)+10))

x, y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
z = f(x,y)

fig = plt.figure(figsize=(14, 7))

# Second sous-graphique (surface)
ax=plt.axes(projection="3d")
ax.plot_wireframe(x, y, z,linewidth = 0.2, color='black')
ax.set_title("Graphique en surface")

def est_dedans(x,y):
    return vmin<=x and x<=vmax and vmin<=y and y<=vmax 

h=0.5

def voisins(x,y):
    v=[]
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if est_dedans(x+i*h,y+j*h) and (i,j)!=(0,0):
                v.append((x+i*h,y+j*h))
    return v

def points_sur_graphe(x,y):
    return(x-x%h,y-y%h)

sommets={}
aretes={}
vmax=20
vmin=-20
x=vmin
depart=points_sur_graphe(6.0011,-7.23)
arrivee=points_sur_graphe(vmax,vmax)
while x<=vmax:
    y=vmin
    while y<=vmax:
        z=f(x,y)
        sommets[(x,y)]=z
        aretes[(x,y)]=voisins(x,y)
        y+=h
    x+=h

def vraie_distance(x1,y1,z1,x2,y2,z2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

def dist_min(a_visiter, distances):
    champion=0
    print("taille a visiter", len(a_visiter))
    champion_d=distances[a_visiter[champion]]
    for i in range(len(a_visiter)):
        sommet=a_visiter[i]
        d=distances[sommet]
        if d<champion_d:
            champion=i
            champion_d=d
    return champion

def construire_chemin(depart,arrivee,predecesseurs):
    chemin=[arrivee]
    sommet=arrivee
    while sommet!=depart:
        sommet=predecesseurs[sommet]
        chemin.insert(0,sommet)
    return chemin

G=Graphe(sommets,aretes)

def djikstra(G,depart,arrivee):
    distances={}
    deja_vus={}
    predecesseurs={}        #à un sommet associe son père
    for s in G.sommets:
        distances[s]=float('inf')
        deja_vus[s]=False

    distances[depart]=0
    deja_vus[depart]=True
    file_a_visiter=[depart]

    sommet=depart
    while sommet != arrivee :
        voisins=G.aretes[sommet]
        for v in voisins:
            if not deja_vus[v]:
                deja_vus[v]=True
                dist=vraie_distance(v[0],v[1],G.sommets[v],sommet[0],sommet[1],G.sommets[sommet])+distances[sommet]
                        #Pour un maillage assez fin on peut considérer que c'est la vraie distance
                if distances[v]>dist:
                    distances[v]=dist
                    predecesseurs[v]=sommet
                    if v not in file_a_visiter:
                        file_a_visiter.append(v)
        i=dist_min(file_a_visiter,distances)
        sommet=file_a_visiter[i]
        file_a_visiter.pop(i)

    return construire_chemin(depart,arrivee,predecesseurs)

def fonction_de_tracage(chemin):
    chemin_X=[]
    chemin_Y=[]
    chemin_Z=[]
    for s in chemin:
        chemin_X.append(s[0])
        chemin_Y.append(s[1])
        chemin_Z.append(G.sommets[s])
    ax.plot(chemin_X,chemin_Y,chemin_Z, linewidth = 2, color='red')

chemin=djikstra(G,depart,arrivee)
fonction_de_tracage(chemin)
print(chemin)

plt.show()