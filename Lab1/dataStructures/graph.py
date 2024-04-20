from dataStructures.PrioQueue import PrioQueue

from dataStructures.Node import Node


class Graph:
    def __init__(self):
        self.start = Node("Start", 10000)

    def doUCS(self, search_term):
        visited = set()
        queue = PrioQueue()
        queue.insert(self.start, 0, "")
        while len(queue) > 0:
            node, cost, path = queue.remove()
            if node in visited:
                continue
            visited.add(node)
            node.show_data()
            path += node.name
            if node == search_term:
                return {"node": node, "path": path, "cost": cost}
            for vertex in node.vertexes:
                if vertex.node not in visited:
                    queue.insert(vertex.node, vertex.cost + cost, path + " -> ")

    def doDFS(self, search_item, node=None, visited=None, path='', cost=0):
        if visited is None:
            visited = set()
        if node is None:
            node = self.start
        if node in visited:
            return
        visited.add(node)
        node.show_data()
        path += node.name
        if node == search_item:
            return {"node": node, "path": path, "cost": cost}
        vertexes = sorted(node.vertexes,
                          key=lambda x:
                          ((x.cost * 1000) +
                           (0 if len(x.node.name) > 1 else ord(x.node.name)
                            )))
        # print([(x.node.name, (x.cost * 1000) + (0 if len(x.node.name) > 1 else ord(x.node.name))) for x in vertexes])
        for vertex in vertexes:
            if vertex.node not in visited:
                result = self.doDFS(search_item, vertex.node, visited, path + " -> ", cost + vertex.cost)
                if result:
                    return result
        return None

    def a_star_search(self, searchItem):
        closed = set()
        fringe = PrioQueue()
        fringe.insert(self.start, 0, "")
        while len(fringe) > 0:
            node, cost, path = fringe.remove()
            if node in closed:
                continue
            path += node.name
            closed.add(node)
            print('Removing: ', end='')
            node.show_data()
            if node == searchItem:
                return {"node": node, "cost": cost, "path": path}
            for vertex in node.vertexes:
                if vertex.node not in closed:
                    node_cost = vertex.cost + cost
                    fringe.insert(vertex.node, node_cost, path + " -> ", heuristic=vertex.node.h)
        return None

    def doBFS(self, search_term):  # Traditional BFS
        visited = set()
        queue = [{"node": self.start, "cost": 0, "path": ""}]
        while len(queue) > 0:
            val = queue.pop(0)
            node, cost, path = val["node"], val["cost"], val["path"]
            if node in visited:
                continue
            visited.add(node)
            node.show_data()
            path += node.name
            if node == search_term:
                return {"node": node, "path": path, "cost": cost}
            vertexes = sorted(node.vertexes,
                              key=lambda x:
                              ((x.cost * 1000) +
                               (0 if len(x.node.name) > 1
                                else ord(x.node.name))))
            for vertex in vertexes:
                if vertex.node not in visited:
                    queue.append({"node": vertex.node, "cost": vertex.cost + cost, "path": path + " -> "})
