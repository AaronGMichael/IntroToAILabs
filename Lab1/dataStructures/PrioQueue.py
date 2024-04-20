import bisect


class PrioQueue:
    def __init__(self):
        self.queue = []

    def insert(self, node, cost, path="", heuristic=0):
        item = {"node": node, "cost": cost, "path": path, "heuristic": heuristic}
        bisect.insort(self.queue, item,
                      key=lambda x: (((x["cost"]+x["heuristic"]) * 1000) + (0 if len(x["node"].name) > 1 else ord(x["node"].name))))

    def remove(self):
        val = self.queue.pop(0)
        return val["node"], val["cost"], val["path"]

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return "[" + ",".join([x["node"].name + " - Cost: " +
                               str(x["cost"]) + " (Path: " + x["path"] + x["node"].name
                               + ")" for x in self.queue]) + "]"
