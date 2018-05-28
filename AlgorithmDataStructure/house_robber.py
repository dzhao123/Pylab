class Solution(object):

    def __init__(self):
        self.memo = []

    def tryRob(self, nums, index):

        if index >= len(nums):
            return 0

        if self.memo[index] != -1:
            return self.memo[index]



        res = 0
        for i in range(index, len(nums)):
            #if self.memo[i] == -1:
            res = max(res, nums[i] + self.tryRob(nums, i+2))
        self.memo[index] = res
            #else:
                #return self.memo[i]
        return res

    def rob(self, nums):

        self.memo = [-1 for _ in range(len(nums))]
        return self.tryRob(nums, 0)





class Solution2(object):

    def rob(self, nums):

        if len(nums) == 0:
            return 0;

        memo = [-1 for _ in range(len(nums))]
        memo[len(nums)-1] = nums[len(nums)-1]
        for i in range(len(nums)-2, -1, -1):
            for j in range(i, len(nums)):
                if j + 2 < len(nums):
                    memo[i] = max(memo[i], nums[j] + memo[j+2])
                else:
                    memo[i] = max(memo[i], nums[j])
        return memo[0]


#b = Solution2()
#print(b.rob([1,2,3]))



class Solution3(object):

    def rob(self, nums):

        if len(nums) == 0:
            return 0;

        memo = [-1 for _ in range(len(nums))]
        #memo[len(nums)-1] = nums[len(nums)-1]
        memo[0] = nums[0]
        for i in range(1, len(nums)):
            for j in range(i+1):
                if j - 2 >= 0:
                    memo[i] = max(memo[i], nums[j] + memo[j-2])
                else:
                    memo[i] = max(memo[i], nums[j])
        return memo[-1]

c = Solution3()
print(c.rob([1,2,3]))









#a = Solution()
#print(a.rob([1,2,1,0]))
#print(a.rob([183,219,57,193,94,233,202,154,65,240,97,234,100,249,186,66,90,238,168,128,177,235,50,81,185,165,217,207,88,80,112,78,135,62,228,247,211]))
