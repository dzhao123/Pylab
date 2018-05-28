class Solution(object):

    def reverseHelper(self, str, left, right):

        if left >= right:
            return

        while left <= right:
            str[left], str[right] = str[right], str[left]
            left += 1
            right -= 1

    def reverseWords(self, str):

        self.reverseHelper(str, 0, len(str) - 1)

        left = 0
        right = 0
        while right <= len(str) - 1:
        #while True:
            print(right)
            print(str[right])
            if right == len(str) - 1 or str[right+1] == " ":
                self.reverseHelper(str, left, right)
                left = right + 2
            right += 1

a = Solution()
print(a.reverseWords(["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]))
