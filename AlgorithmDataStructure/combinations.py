class Solution(object):

    def __init__(self):
        self.res = []
        self.table = []
        self.k = 0

    def solve(self, n, start, number):


        if len(number) == self.k:
            self.res.append(number)
            return

        for i in range(start, n - (self.k - len(number)) + 1):
            #if n - start < self.k - len(number):
                #break
            #else:
            self.solve(n, i + 1, number + [i+1])

    def combine(self, n, k):

        if n == None:
            return self.res

        if n < 0 or k < 0 or n < k:
            return self.res

        self.k = k
        self.solve(n, 0, [])

        return self.res


class Solution(object):

    def helper(self, res, arr, path, start, length):
        if len(path) == length:
            res.append(path)
            return

        for i in range(start, len(arr)):
            self.helper(res, arr, path + [arr[i]], i + 1, length)



    def combine(self, n, k):
        res = []
        arr = [i+1 for i in range(n)]
        self.helper(res, arr, [], 0, k)

        return res
