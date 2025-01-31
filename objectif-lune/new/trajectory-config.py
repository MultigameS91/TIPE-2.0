import numpy as np
from numpy.linalg import norm
from numpy import array
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import shapely as sh
import time as time
import uuid

G = 1e-3

def regularize(vector, max_norm):
    norm_v = norm(vector)
    if norm_v > max_norm:
        return vector / norm_v * max_norm
    return vector

class Planete:
    def __init__(self, _pos, _mass):
        self.pos = _pos
        self.mass = _mass

    def champ_gravitationnel(self, x, y):
        rx = x - self.pos[0]
        ry = y - self.pos[1]
        r_squared = rx**2 + ry**2

        if np.isscalar(r_squared):
            if r_squared == 0:
                return np.array([0.0, 0.0])
            r_magnitude = np.sqrt(r_squared)
        else:
            r_magnitude = np.sqrt(r_squared)
            r_magnitude[r_magnitude == 0] = np.inf

        force = -G * self.mass / r_squared * np.stack((rx / r_magnitude, ry / r_magnitude), axis=0)
        return regularize(force, max_norm=10)

Soleil = Planete(array([0, 0]), 100)
Terre = Planete(array([30, 0]), 10)
Mars = Planete(array([-30, 0]), 5)
planetes = [Soleil, Terre, Mars]

def champ_espace(x, y):
    chp = np.zeros((2, *x.shape))
    for p in planetes:
        chp += p.champ_gravitationnel(x, y)
    return chp

class PolygonPoints:
    def __init__(self, x, y, uid):
        self.x = x
        self.y = y
        self.uid = uid

SX, SY = np.meshgrid(np.linspace(-50, 50, 50), np.linspace(-50, 50, 50))  
chp = champ_espace(SX, SY)

fig, ax = plt.subplots()
ax.quiver(SX, SY, chp[0], chp[1], color="blue", alpha=0.6, label="Champ gravitationnel")
ax.scatter(0, 0, color="orange", label="Soleil", s=500)  
ax.scatter(Terre.pos[0], Terre.pos[1], color="green", label="Terre", s=100)  
ax.scatter(Mars.pos[0], Mars.pos[1], color="red", label="Mars", s=80)  
plt.xlabel("Position X")
plt.ylabel("Position Y")
ax.axis('equal')
ax.grid()

points_out = {}
plt_poly_out = []
points_in = {}
plt_poly_in = []

mode="OUT"

def get_coords_out():
    return [(p.x,p.y) for p in list(points_out.values())]
def get_coords_in():
    return [(p.x,p.y) for p in list(points_in.values())]

def on_click(event):
    global plt_poly  
    if event.button is MouseButton.LEFT:
        ix, iy = event.xdata, event.ydata
        #print(f'x = {ix}, y = {iy}')
        
        if ix is not None and iy is not None:
            

            if mode=="OUT":
                polygonPoint=PolygonPoints(ix,iy,uuid.uuid4())
                artist=ax.plot(polygonPoint.x,polygonPoint.y,'ro',picker=5)[0]
                artist.obj=polygonPoint

                points_out[polygonPoint.uid]=polygonPoint
                
                if len(points_out) >= 3:
                    if plt_poly_out:  
                        plt_poly_out.pop(0).remove()  


                    polygon = sh.convex_hull(sh.MultiPoint(get_coords_out()))
                    x, y = polygon.exterior.xy
                    plt_poly_out.append(ax.plot(x, y, c="red")[0])

                print("Nouveau point ajouté:",f'x = {ix}, y = {iy}')
            else:
                polygonPoint=PolygonPoints(ix,iy,uuid.uuid4())
                artist=ax.plot(polygonPoint.x,polygonPoint.y,'go',picker=5)[0]
                artist.obj=polygonPoint

                points_in[polygonPoint.uid]=polygonPoint
                
                if len(points_in) >= 3:
                    if plt_poly_in:  
                        plt_poly_in.pop(0).remove()  


                    polygon = sh.convex_hull(sh.MultiPoint(get_coords_in()))
                    x, y = polygon.exterior.xy
                    plt_poly_in.append(ax.plot(x, y, c="green")[0])

                print("Nouveau point ajouté:",f'x = {ix}, y = {iy}')

            fig.canvas.draw()

pick_cooldown=[time.time(),0.200]

def on_pick(event):
    #print(time.time()-pick_cooldown[0])
    if event.mouseevent.button==3 and time.time()-pick_cooldown[0]>=pick_cooldown[1]:
        pick_cooldown[0]=time.time()
        polygonPoint=event.artist.obj
        #print(polygonPoint.uid)
        
        if mode=="OUT":
            del points_out[polygonPoint.uid]
            event.artist.remove()
            if len(points_out) >= 3:
                if plt_poly_out:  
                    plt_poly_out.pop(0).remove()  

                polygon = sh.convex_hull(sh.MultiPoint(get_coords_out()))
                x, y = polygon.exterior.xy
                plt_poly_out.append(ax.plot(x, y, c="red")[0])
            elif plt_poly_out: 
                plt_poly_out.pop(0).remove()
                
            print("Point retiré:",f'x = {polygonPoint.x}, y = {polygonPoint.y}')
        else:
            del points_in[polygonPoint.uid]
            event.artist.remove()
            if len(points_in) >= 3:
                if plt_poly_in:  
                    plt_poly_in.pop(0).remove()  

                polygon = sh.convex_hull(sh.MultiPoint(get_coords_in()))
                x, y = polygon.exterior.xy
                plt_poly_in.append(ax.plot(x, y, c="green")[0])
            elif plt_poly_in: 
                plt_poly_in.pop(0).remove()
                
            print("Point retiré:",f'x = {polygonPoint.x}, y = {polygonPoint.y}')

        fig.canvas.draw()

mode_cooldown=[time.time(),0.200]

def on_key_press(event):
    global mode
    #print(event.key)
    key=event.key
    if time.time()-pick_cooldown[0]>=pick_cooldown[1]:
        pick_cooldown[0]=time.time()
        if key=="h":
            if mode=="IN":
                mode="OUT"
            else:
                mode="IN"
            print("#####################")
            print("Passage au mode:",mode)
            print("#####################")

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key_press)

fig.canvas.callbacks.connect('pick_event', on_pick)

plt.show()
