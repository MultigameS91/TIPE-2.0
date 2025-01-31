import numpy as np
import matplotlib.pyplot as plt

N=20
h=1/N
grid_min,grid_max=-5,5
taille_mesh=abs(grid_max-grid_min)

sommets={}
aretes={}

def avoir_voisins(i,j):
    L=[]
    for k in [-1,0,1]:
        for l in [-1,0,1]:
            if (k,l)!=(0,0) and est_dedans(i+k,j+l):
                L.append((i+k,j+l))
    return L

def est_dedans(i,j):
    return 0<=i and i<=taille_mesh*N and 0<=j and j<=taille_mesh*N

for i in range(abs(grid_max-grid_min)*N+1):
    x=grid_min+i*h
    for j in range(abs(grid_max-grid_min)*N+1):
        y=grid_min+j*h
        sommets[(i,j)]=(x,y)
        aretes[(i,j)]=avoir_voisins(i,j)
def entier_to_coord(i,j):
    return (grid_min+i/N,grid_min+j/N)

def f(x,y):
    return 0*np.exp(-((x)**2 + (y)**2))

def S(x,y):
    return x,y,f(x,y)

def distance(P,M):
    return np.sqrt((M[0]-P[0])**2+(M[1]-P[1])**2+(M[2]-P[2])**2)

def dist_min(liste,liste_dist):
    champ=0
    dist_champ=float('inf')
    for i in range(len(liste)):
        point=liste[i]
        if liste_dist[point]<dist_champ:
            dist_champ=liste_dist[point]
            champ=i
    return champ

# depart/arrivee = (i,j)
def djikstra(depart,arrivee):
    distances={}
    deja_vus={}
    predecesseurs={}
    for s in sommets:
        distances[s]=float('inf')
        deja_vus[s]= False
    
    distances[depart]=0
    deja_vus[depart]=True
    file_a_visiter=[depart]
    liste_meilleurs=[]

    sommet=depart
    print(sommet,arrivee)
    while sommet!=arrivee:
        voisins=aretes[sommet]
        xs,ys=entier_to_coord(sommet[0],sommet[1])
        for v in voisins:
            if not deja_vus[v]:
                deja_vus[v]=True
                xv,yv=entier_to_coord(v[0],v[1])
                dist=distances[sommet]+distance(S(xs,ys),S(xv,yv))+np.exp(distance(S(xv,yv),S(arrivee[0],arrivee[1])))
                if dist < distances[v]:
                    liste_meilleurs=[sommet]
                    distances[v]=dist
                    predecesseurs[v]=sommet
                if dist == distances[v]:
                    liste_meilleurs.append(v)
                    if v not in file_a_visiter:
                        file_a_visiter.append(v)
                        
        i=dist_min(file_a_visiter,distances)
        sommet=file_a_visiter[i]
        file_a_visiter.pop(i)
        
    chemin=[entier_to_coord(arrivee[0],arrivee[1])]
    sommet=arrivee
    while sommet!=depart:
        print("yesss")
        sommet=predecesseurs[sommet]
        chemin.insert(0,entier_to_coord(sommet[0],sommet[1]))
    print(chemin)
    return chemin

x0,y0,xn,yn=-2,-1,5,5
depart=(int((x0-grid_min)*N),int((y0-grid_min)*N))
arrivee=(int((xn-grid_min)*N),int((yn-grid_min)*N))

X, Y = np.meshgrid(np.linspace(grid_min,grid_max,N*taille_mesh), np.linspace(grid_min,grid_max,N*taille_mesh))
Z = f(X,Y)

ax = plt.axes(projection="3d")
ax.plot_wireframe(X, Y, Z, color="orange", linewidth = 0.5)
#ax.axis('scaled')
chemin=djikstra(arrivee,depart)
cheminX,cheminY,cheminZ=[],[],[]
for p in chemin:
    cheminX.append(p[0])
    cheminY.append(p[1])
    cheminZ.append(f(p[0],p[1]))
plt.plot(cheminX,cheminY,cheminZ,color='blue')
plt.show()