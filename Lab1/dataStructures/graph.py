from dataStructures.PrioQueue import PrioQueue

from dataStructures.Node import Node


class Graph:
    def __init__(self):
        self.start = Node("Start", 10000)

    def doBFS(self):
        visited = set()
        queue = [self.start]
        while len(queue) > 0:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            node.show_data()
            for vertex in node.vertexes:
                if vertex.node not in visited:
                    queue.append(vertex.node)

    def doDFS(self, node=None, visited=None):
        if visited is None:
            visited = set()
        if node is None:
            node = self.start
        if node in visited:
            return
        visited.add(node)
        node.show_data()
        for vertex in node.vertexes:
            if vertex.node not in visited:
                self.doDFS(vertex.node, visited)

    def a_star_search(self, searchItem):
        closed = set()
        fringe = PrioQueue()
        fringe.insert(self.start, 0)
        path = []
        while len(fringe) > 0:
            fringe.print()
            data = fringe.remove()
            node = data["node"]
            cost = data["cost"]
            if node in closed:
                continue
            path.append(node.name)
            closed.add(node)
            print('Removing: ', end='')
            node.show_data()
            if node == searchItem:
                return {"node": node, "cost": cost, "path": path}
            for vertex in node.vertexes:
                if vertex.node not in closed:
                    node_cost = vertex.cost + cost
                    fringe.insert(vertex.node, node_cost)
        return None
