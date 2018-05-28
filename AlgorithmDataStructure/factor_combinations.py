class Solution(object):

    def solve(self, res, arr, n, start):
        if n == 1:
            if len(arr) > 0:
                res.append(arr)
                return

        for i in range(start, n+1):
            if n % i == 0:
                self.solve(res, arr + [i], n//i, i)


    def getFactors(self, n):

        res = []
        self.solve(res, [], n, 2)

        return res[:-1]


class Solution(object):
    def helper(self, res, arr, n, start):
        if n < start:
            if len(arr) > 0:
                return

        for i in range(start, int(n**0.5) + 1):
            if n % i == 0:
                res.append(arr + [i, n//i])
                self.helper(res, arr + [i], n//i, i)

    def getFactors(self, n):

        res = []
        self.helper(res, [], n, 2)

        return res

a = Solution()
print(a.getFactors(12))
