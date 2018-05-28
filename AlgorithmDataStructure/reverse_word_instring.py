import heapq
class Solution(object):

    def reverseWords(str):

        heap = []
        start = 0
        for end in range(len(str)):
            if str[end] == " ":
                heapq.heappush(heap, [-end+1, str[start:end]])
                heapq.heappush(heap, [-end, " "])
                start = end + 1
            if end == len(str)-1:
                heapq.heappush(heap, [-end, str[start:end+1]])
        print(heap)
        return [heapq.heappop(heap)[1] for _ in range(len(heap))]



a = Solution
print(a.reverseWords("the sky is blue"))
