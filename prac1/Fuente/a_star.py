import math
from queue import PriorityQueue
from typing import List, Tuple, Optional
from casilla import Casilla
from mapa import Mapa

class a_star:
    def __init__(self, mapa: Mapa, origen: Casilla, destino: Casilla):
        self.mapa = mapa
        self.origen = origen
        self.destino = destino
        self.n = mapa.getAlto()
        self.m = mapa.getAncho()
        # Direcciones de movimiento (8 direcciones posibles)
        self.dx = [-1, -1, 0, 1, 1, 1, 0, -1]
        self.dy = [0, 1, 1, 1, 0, -1, -1, -1]
        # Mapa de movimientos para reconstruir el camino
        self.moves_map = [
            [8, 1, 2],
            [7, 0, 3],
            [6, 5, 4]
        ]

    def calcular_coste_movimiento(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calcula el coste de movimiento entre dos celdas"""
        return 1.0 if x1 == x2 or y1 == y2 else 1.5

    def calcular_calorias(self, x: int, y: int) -> int:
        """Calcula el coste en calorías según el tipo de terreno"""
        tipo_terreno = self.mapa.getCelda(x, y)
        if tipo_terreno == 0:  # Hierba
            return 2
        elif tipo_terreno == 4:  # Agua
            return 4
        elif tipo_terreno == 5:  # Roca
            return 6
        return 0
    
    def heuristica_chebyshev(self, x: int, y: int) -> float:
        """Heurística de distancia Chebyshevs"""
        dx = abs(x - self.destino.getFila())
        dy = abs(y - self.destino.getCol())
        return max(dx, dy)
    
    def heuristica_diagonal(self, x: int, y: int) -> float:
        """Heurística diagonal
        D * min(dx, dy) + 1.0 * (max(dx, dy) - min(dx, dy))
        donde D es el coste de movimiento diagonal"""
        dx = abs(x - self.destino.getFila())
        dy = abs(y - self.destino.getCol())
    
        # Coste movimiento diagonal = 1.5, ortogonal = 1.0
        return 1.5 * min(dx, dy) + 1.0 * abs(dx - dy)
    
    def heuristica_nula(self, x: int, y: int) -> float:
        """Heurística nula (h=0)"""
        return 0.0

    def heuristica_manhattan(self, x: int, y: int) -> float:
        """Heurística de distancia Manhattan"""
        return abs(x - self.destino.getFila()) + abs(y - self.destino.getCol())

    def heuristica_euclidea(self, x: int, y: int) -> float:
        """Heurística de distancia Euclídea"""
        return math.sqrt((x - self.destino.getFila())**2 + (y - self.destino.getCol())**2)

    def es_valido(self, x: int, y: int) -> bool:
        """Verifica si una celda es válida y transitable"""
        return (0 <= x < self.n and 
                0 <= y < self.m and 
                self.mapa.getCelda(x, y) != 1)

    def reconstruir_camino(self, came_from: dict, current: Tuple[int, int]) -> List[List[str]]:
        """Reconstruye el camino encontrado"""
        path = [['.' for _ in range(self.m)] for _ in range(self.n)]
        while current in came_from:
            x, y = current
            path[x][y] = 'x'
            current = came_from[current]
        path[self.origen.getFila()][self.origen.getCol()] = 'x'
        return path

    def buscar(self, heuristica=None) -> Tuple[float, List[List[str]], int]:
        """Implementación del algoritmo A* con trazas de ejecución"""
        if heuristica is None:
            heuristica = self.heuristica_manhattan

        # Inicialización
        pq = PriorityQueue()
        came_from = {}
        g_score = {(self.origen.getFila(), self.origen.getCol()): 0}
        cal_score = {(self.origen.getFila(), self.origen.getCol()): 
                    self.calcular_calorias(self.origen.getFila(), self.origen.getCol())}

        start = (self.origen.getFila(), self.origen.getCol())
        f_score = heuristica(start[0], start[1])
        pq.put((f_score, start))

        print(f"\n=== Iniciando búsqueda A* ===")
        print(f"Punto de inicio: ({start[0]}, {start[1]})")
        print(f"Punto destino: ({self.destino.getFila()}, {self.destino.getCol()})")

        # Para visualizar la frontera de exploración
        explored = set()
        iteration = 0

        while not pq.empty():
            current = pq.get()[1]
            iteration += 1

            print(f"\nIteración {iteration}")
            print(f"Explorando nodo: ({current[0]}, {current[1]})")
            print(f"G actual: {g_score[current]:.2f}")
            print(f"Calorías acumuladas: {cal_score[current]}")

            explored.add(current)

            if current == (self.destino.getFila(), self.destino.getCol()):
                print("\n¡Destino encontrado!")
                path = self.reconstruir_camino(came_from, current)

                # Visualizar el camino final
                print("\nCamino encontrado:")
                for i in range(len(path)):
                    row = ""
                    for j in range(len(path[0])):
                        if (i, j) in explored and path[i][j] != 'x':
                            row += 'o '  # nodos explorados
                        else:
                            row += path[i][j] + ' '
                    print(row)

                print(f"\nCoste total del camino: {g_score[current]:.2f}")
                print(f"Calorías totales consumidas: {cal_score[current]}")
                print(f"Nodos explorados: {len(explored)}")
                return g_score[current], path, cal_score[current]

            # Explorar vecinos
            print("Explorando vecinos:")
            for i in range(8):
                nx, ny = current[0] + self.dx[i], current[1] + self.dy[i]
                if not self.es_valido(nx, ny):
                    continue

                tentative_g = g_score[current] + self.calcular_coste_movimiento(
                    current[0], current[1], nx, ny)
                neighbor = (nx, ny)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    cal_score[neighbor] = cal_score[current] + self.calcular_calorias(nx, ny)
                    f_score = tentative_g + heuristica(nx, ny)
                    pq.put((f_score, neighbor))
                    print(f"  Vecino ({nx}, {ny}): f={f_score:.2f}, g={tentative_g:.2f}, h={heuristica(nx, ny):.2f}")

        print("\nNo se encontró camino al destino")
        return -1, [['.' for _ in range(self.m)] for _ in range(self.n)], 0

    @staticmethod
    def a_star_epsilon(mapa: Mapa, origen: Casilla, destino: Casilla, epsilon: float = 0.5):
        """Implementación del algoritmo A*ε"""
        solver = a_star(mapa, origen, destino)
        
        def focal_heuristic(x: int, y: int) -> float:
            """Heurística focal basada en calorías"""
            return float(solver.calcular_calorias(x, y))

        pq = PriorityQueue()
        focal = PriorityQueue()
        came_from = {}
        g_score = {(origen.getFila(), origen.getCol()): 0}
        cal_score = {(origen.getFila(), origen.getCol()): 
                    solver.calcular_calorias(origen.getFila(), origen.getCol())}
        
        start = (origen.getFila(), origen.getCol())
        f_score = solver.heuristica_manhattan(start[0], start[1])
        pq.put((f_score, start))
        focal.put((focal_heuristic(start[0], start[1]), start))
        
        while not focal.empty():
            current = focal.get()[1]
            
            if current == (destino.getFila(), destino.getCol()):
                path = solver.reconstruir_camino(came_from, current)
                return g_score[current], path, cal_score[current]

            # Actualizar lista focal
            f_min = pq.queue[0][0] if not pq.empty() else float('inf')
            f_threshold = (1 + epsilon) * f_min

            for i in range(8):
                nx, ny = current[0] + solver.dx[i], current[1] + solver.dy[i]
                if not solver.es_valido(nx, ny):
                    continue

                neighbor = (nx, ny)
                tentative_g = g_score[current] + solver.calcular_coste_movimiento(
                    current[0], current[1], nx, ny)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    cal_score[neighbor] = cal_score[current] + solver.calcular_calorias(nx, ny)
                    f_score = tentative_g + solver.heuristica_manhattan(nx, ny)
                    
                    if f_score <= f_threshold:
                        focal.put((focal_heuristic(nx, ny), neighbor))
                    pq.put((f_score, neighbor))

        return -1, [['.' for _ in range(solver.m)] for _ in range(solver.n)], 0