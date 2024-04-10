from dataStructures import graph
from dataStructures.Node import Node
from dataStructures.graph import Graph


def seedGraph():
    gra = Graph()
    A = Node("A", 2)
    B = Node("B", 5)
    C = Node("C", 2)
    D = Node("D", 1)
    goal = Node("Goal", 0)

    C.add_vertex(2, goal)
    D.add_vertex(5, goal)
    C.add_vertex(1, D)
    B.add_vertex(4, D)
    A.add_vertex(4, C)
    gra.start.add_vertex(2, A)
    gra.start.add_vertex(3, B)
    gra.start.add_vertex(5, D)
    return {"graph": gra, "goal": goal}


if __name__ == '__main__':
    ret = seedGraph()
    g = ret["graph"]
    search = ret["goal"]
    found = g.a_star_search(search)
    if not found:
        print("No Solution found")
    else:
        print("Goal State found with cost: ",
              found["cost"], "and data: ", found["node"].name,
              "The Nodes Expanded are: ", found["path"])
