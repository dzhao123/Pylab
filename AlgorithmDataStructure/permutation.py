class Solution(object):
        
    def __init__(self):
        
        self.res = []
        self.table = []
    
    def solve(self, digits, index, number):
        
        if len(digits) == index:
            self.res.append(number)
            return
        
        for i in range(len(digits)):
            if self.table[i] == 0:
                self.table[i] = 1
                self.solve(digits, index+1, number+[digits[i]])
                self.table[i] = 0
                
    def permute(self, digits):
        
        if digits == None:
            return
        self.table = [0 for _ in range(len(digits))]
        self.solve(digits, 0, [])
        
        return self.res




class Solution(object):

    def helper(self, res, digits, path):
        if not digits:
            res.append(path)
            return

        for i in range(len(digits)):
            self.helper(res, digits[:i] + digits[i+1:], path + [digits[i]])


    def permute(self, digits):

        res = []
        self.helper(res, digits, [])

        return res




class Solution(object):

    def permute(self, digits):
        if not digits:
            return [[]]

        res = []
        for idx, digit in enumerate(digits):
            for perm in self.permute(digits[:idx] + digits[idx+1:]):
                res.append([digit] + perm)


        return res


a = Solution()
print(a.permute([1,2,3]))




def perm(arr):

    for s in perm(arr[1:]):
        for i in range():



