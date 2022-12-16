
class Key:

    def __init__(self, key1, key2):
        self.key1 = key1
        self.key2 = key2
    
    def __lt__(self, nxt):
        if self.key1 == nxt.key1:
            return self.key2 < nxt.key2
        return self.key1 < nxt.key1