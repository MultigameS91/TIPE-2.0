import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fonction qui définit la surface (par exemple une colline)
def f(x, y):
    return x+y

# Fonctions utilitaires pour trouver les voisins sur la grille
def get_neighbors(x, y, grid_size):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Gauche, droite, haut, bas
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            neighbors.append((nx, ny))
    return neighbors

# Dijkstra pour trouver le plus court chemin sur la surface
def dijkstra_surface(start, end, grid_size):
    start_x, start_y = start
    end_x, end_y = end

    # Initialisation des distances et du tas de priorité
    distances = {(x, y): float('infinity') for x in range(grid_size) for y in range(grid_size)}
    distances[(start_x, start_y)] = 0
    priority_queue = [(0, start_x, start_y)]  # (distance, x, y)
    parents = {start: None}  # Pour garder une trace du chemin

    while priority_queue:
        current_distance, x, y = heapq.heappop(priority_queue)

        # Si nous sommes arrivés au point final, on arrête
        if (x, y) == (end_x, end_y):
            break

        # Exploration des voisins
        for nx, ny in get_neighbors(x, y, grid_size):
            # Calcul du coût de déplacement vers le voisin
            surface_difference = abs(f(x, y) - f(nx, ny))
            distance = current_distance + surface_difference
            
            # Si une meilleure distance est trouvée, on la met à jour
            if distance < distances[(nx, ny)]:
                distances[(nx, ny)] = distance
                parents[(nx, ny)] = (x, y)  # Garder trace du parent
                heapq.heappush(priority_queue, (distance, nx, ny))

    # Reconstituer le chemin du point final vers le point de départ
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parents.get(node)
    
    return distances[(end_x, end_y)], path[::-1]  # Retourne la distance et le chemin inversé

# Affichage de la surface et du chemin
def plot_surface_and_path(grid_size, path):
    x = np.linspace(0, grid_size - 1, grid_size)
    y = np.linspace(0, grid_size - 1, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Afficher la surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # Extraire les points du chemin pour les tracer
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    path_z = [f(p[0], p[1]) for p in path]

    # Tracer le chemin
    ax.plot(path_x, path_y, path_z, color='r', marker='o', markersize=5, label="Chemin")

    # Labels et légende
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Hauteur f(x, y)')
    ax.legend()
    plt.show()

# Exemple d'utilisation :
start_point = (0, 0)  # Coordonnées du point A
end_point = (4, 4)    # Coordonnées du point B
grid_size = 100         # Taille de la grille (5x5 dans cet exemple)

# Calculer la distance minimale et le chemin
min_distance, path = dijkstra_surface(start_point, end_point, grid_size)
print(f"Distance minimale entre {start_point} et {end_point}: {min_distance}")
print(f"Chemin: {path}")

# Afficher la surface et le chemin trouvé
plot_surface_and_path(grid_size, path)
