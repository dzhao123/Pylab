class Solution(object):
    stack = []
    def isPalindrome(self, s):
        if not s:
            return True
        i = 0
        j = len(s) - 1
        for k in range(len(s)):
            if s[i].isalnum() and s[j].isalnum():
                if s[i].lower() == s[j].lower():
                    i += 1
                    j -= 1
                else:
                    return False
            if not s[i].isalnum():
                i += 1
            if not s[j].isalnum():
                j -= 1
        return True


a = Solution()
print(a.isPalindrome("a"))
