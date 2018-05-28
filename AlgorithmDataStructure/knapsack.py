class Solution(object):

    def bestValue(self, w, v, index, C):

        if index < 0 or C <= 0:
            return 0


        res = self.bestValue(w, v, index-1, C)
        if w[index] <= C:
            res = max(res, v[index] + self.bestValue(w, v, index-1, C-w[index]))

        return res

    def knapSack01(self, w, v, C):

        index = len(w)

        return self.bestValue(w, v, index-1, C)


class Solution2(object):

    def __init__(self):
        self.memo = []

    def bestValue(self, w, v, index, C):

        if index < 0 or C <= 0:
            return 0


        res = self.bestValue(w, v, index-1, C)
        self.memo[index-1] = res
        if w[index] <= C:
            res = max(res, v[index] + self.bestValue(w, v, index-1, C-w[index]))
            self.memo[index] = res

        return res

    def knapSack01(self, w, v, C):

        index = len(w)

        self.memo = [-1 for _ in range(index)]
        print(self.memo)

        return self.bestValue(w, v, index-1, C)


a = Solution2()
w = [1,2,3,5]
v = [3,4,5,6]
C = 10
print(a.knapSack01(w, v, C))
