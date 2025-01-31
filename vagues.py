import numpy as np
import matplotlib.pyplot as plt
def S1(x,y,t):
    return np.sin(t)*np.sin(x+10*t)

def S2(x,y,t):
    return 5*np.sqrt(np.exp(-np.sqrt(x**2+y**2)*t/10))*np.sin(t)*np.sin(np.sqrt(x**2+y**2)+10*t)

def S3(x,y,t):
    return 5*np.sqrt(np.exp(-t))*np.sin(np.sqrt(x**2+y**2)-10*t)

def surface_mouvante(Tmax,N_pas):
    h=1/N_pas

    T1=np.sin(np.linspace(0,Tmax,N_pas*Tmax))
    T2=np.linspace(0,Tmax,N_pas*Tmax)
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection="3d")
    plt.ion()
    for t in T2:
        SX, SY = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
        SZ = S3(SX,SY,t)
        courbe=ax.plot_wireframe(SX, SY, SZ, color="orange", linewidth = 0.5)
        plt.draw()
        plt.pause(0.05)
        plt.axis('scaled')
        courbe.remove()

surface_mouvante(20,25)