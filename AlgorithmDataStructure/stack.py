class Solution(object):
    def isValid(self, s):
        stack = []
        for item in s:
            if item in ('(','{','['):
                stack.append(item)
            if item == ')':
                if stack[-1] == '(':
                    stack.pop()
                #else:

            if item == '}':
                if stack[-1] == '{':
                    stack.pop()
            if item == ']':
                if stack[-1] == '[':
                    stack.pop()
        if len(stack) == 0:
            return True
        else:
            return False

#a = Solution()
#print(a.isValid(']'))


a = [1,2].pop()
print(a)
