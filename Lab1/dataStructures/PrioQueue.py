import bisect


class PrioQueue:
    def __init__(self):
        self.queue = []

    def insert(self, node, cost):
        item = {"node": node, "cost": cost}
        bisect.insort(self.queue, item, key=lambda x: (x["cost"] + x["node"].h))

    def remove(self):
        return self.queue.pop(0)

    def __len__(self):
        return len(self.queue)

    def print(self):
        print([x["node"].name + " - Cost: " + str(x["cost"] + x["node"].h) for x in self.queue])
