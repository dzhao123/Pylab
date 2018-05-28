from random import randint

class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.ds = {}
        self.ls = []


    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.ds:
            self.ls.append(val)
            self.ds[val] = len(self.ls) - 1
            return True
        else:
            return False


    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        temp = 0
        if val not in self.ds:
            return False
        else:
            temp = self.ls[-1]
            self.ls[-1] = val
            self.ls[self.ds[val]] = temp
            self.ds[self.ls[-1]] = self.ds[val]
            del self.ds[val]
            self.ls.pop()

            return True


    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return self.ls[randint(0,len(self.ls) - 1)]


obj = RandomizedSet()
 param_1 = obj.insert(val)
 param_2 = obj.remove(val)
 param_3 = obj.getRandom()


