import matplotlib.pyplot as plt 
import matplotlib.image as img
import numpy as np
  
class Graphe:
    def __init__(self,_sommets,_aretes):
        self.sommets=_sommets
        self.aretes=_aretes

carte_puzzle = img.imread('C:\\Users\\yopla\\Desktop\\TIPE 2.0\\puzzle\\puzzle1.png') 
  
plt.imshow(carte_puzzle)
plt.show()

n,m,c=np.shape(carte_puzzle)

def est_dedans(x,y):
    return 0<=x and x<m and 0<=y and y<n

def avoir_fils(x,y):
    L=[]
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if (i,j)!=(0,0) and est_dedans(x+i,y+j):
                L.append((x+i,y+j))
    return L

aretes={}
sommets={}

def est_rouge(couleur):
    return couleur[0] == 1. and couleur[1] == 0. and couleur[2] == 0. and couleur[3] == 1.

for x in range(n):
    for y in range(m):
        print(carte_puzzle[x,y])
        fils=avoir_fils(x,y)
        aretes[(x,y)]=fils
        sommets[(x,y)]= est_rouge(carte_puzzle[x,y])

G=Graphe(sommets,aretes)

def distance(x,y,a,b):
    return np.sqrt((x-a)**2 + (y-b)**2)

def poids(x,y,a,b):
    return distance((x,y,a,b))      #Provisoirement, on prend la distance classique 
            #Par la suite, ce sera la distance à vol d'oiseau pondérée par les zones de ralentissement 
            #(éventuellement +inf pour de véritables obstacles)

def endiago(x1,y1,x2,y2):
    return x1%2 != x2%2 and y1%2 != y2%2

def djikstra(G,depart,arrivee):
    distances={}
    deja_vus={}
    chemin=[]
    predecesseurs={}
    for s in G.sommets:
        distances[s]=float('inf')
        deja_vus[s]= G.sommets[s] # est rouge
    
    distances[depart]=0
    deja_vus[depart]=True
    chemin.append(depart)
    file_a_visiter=[depart]

    sommet=depart

    while sommet != arrivee:
        voisins=G.aretes[sommet]
        for v in voisins:
            if not deja_vus[v]:
                deja_vus[v]=True
                dist=1+distances[sommet]
                if endiago(v[0],v[1],sommet[0],sommet[1]):
                    dist=np.sqrt(2)+distances[sommet]
                if distances[v]>dist:
                    distances[v]=dist
                    predecesseurs[v]=sommet
                    if v not in file_a_visiter:
                        file_a_visiter.append(v)
        i=dist_min(file_a_visiter,distances)
        sommet=file_a_visiter[i]
        file_a_visiter.pop(i)

    return construire_chemin(depart,arrivee,predecesseurs)


def construire_chemin(depart,arrivee,predecesseurs):
    chemin=[arrivee]
    sommet=arrivee
    while sommet!=depart:
        sommet=predecesseurs[sommet]
        chemin.insert(0,sommet)
    return chemin



def dist_min(a_visiter, distances):
    champion=0
    champion_d=distances[a_visiter[champion]]
    for i in range(len(a_visiter)):
        sommet=a_visiter[i]
        d=distances[sommet]
        if d<champion_d:
            champion=i
            champion_d=d
    return champion

def fonction_de_trace(chemin,img):
    for (x,y) in chemin:
        img[x, y]=np.array([0.,0.,0.,1.])

chemin=djikstra(G,(0,0),(31,31))
print(chemin)
fonction_de_trace(chemin,carte_puzzle)
plt.imshow(carte_puzzle)
plt.show()
    
