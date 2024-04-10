from dataStructures.Vertex import Vertex


class Node:
    def __init__(self, name, h):
        self.name = name
        self.h = h
        self.vertexes = []

    def add_vertex(self, cost, node):
        x = Vertex(cost, node)
        self.vertexes.append(x)
        node.vertexes.append(x)

    def show_data(self):
        print(self.name)
