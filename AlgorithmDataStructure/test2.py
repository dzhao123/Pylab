import math
class Solution(object):
    def getFactors(self, n):

        if n==1:
            return []
        ret = []
        def helper(n,index,path):
            if n<index:
                return
            for i in range(index,int(math.sqrt(n))+1):
                if n%i == 0:
                    ret.append(path+[i,n//i])
                    helper(n//i,i,path+[i])
        helper(n,2,[])
        return ret
a = Solution()
print(a.getFactors(12))

a = [1,2,3,4,5]
a.append(6)
b = a + [6]
