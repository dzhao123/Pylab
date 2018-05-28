class Solution(object):

    def helper(self, res, set, sub, index):

        if index == len(set):
            res.append(sub[:])
            return

        #for i in range(start, len(set)):
        #leaves are solutions
        sub.append(set[index])
        self.helper(res, set, sub, index+1)
        sub.pop()
        self.helper(res, set, sub, index+1)
        #sub.pop()

    def subset(self, set):
        if set is None:
            return

        res = []
        self.helper(res, set, [], 0)

        return res
a = Solution()
print(a.subset([1,2,3]))






class Solution(object):

    def helper(self, res, path, string):

        if not string:
            res.append(path[:])
            return

        for i in range(len(string)):
            path.append(string[i])
            self.helper(res, path, string[:i] + string[i+1:])
            path.pop()

    def permutation(self, string):

        res = []
        path = []
        self.helper(res, path, string)
        return res


#a = Solution()
#print(a.permutation(['a','b','c']))




class Solution(object):

    def helper(self, res, string, sub, index):

        if index == len(string)-1:
            res.append(sub)
            return

        for i in range(index, len(string)-1):
            sub.append(string)
            string[i], string[i+1] = string[i+1], string[i]
            self.helper(res, string, sub, index + 1)
            string[i], string[i+1] = string[i+1], string[i]
            sub.pop()

    def permutation(self, string):

        res = []
        path = []
        self.helper(res, string, [], 0)
        return res

#a = Solution()
#print(a.permutation(['a','b','c']))








class Solution(object):

    def helper(self, res, path, blen, l, r):

        if l == blen/2 and r == blen/2:
            res.append(path[:])
            return
        if l <= blen/2:
            path.append('(')
            self.helper(res, path, blen, l+1, r)
            path.pop()
        if l > r:
            path.append(')')
            self.helper(res, path, blen, l, r+1)
            path.pop()

    def valid_bracket(self, brackets):

        res = []
        path = []
        blen = len(brackets)
        self.helper(res, path, blen, 0, 0)

        for i in range(len(res)):
            res[i] = ''.join(res[i])
        return res


#a = Solution()
#print(a.valid_bracket(['(','(',')',')','(',')']))
