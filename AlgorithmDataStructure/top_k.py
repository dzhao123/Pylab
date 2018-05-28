from collections import Counter
import heapq

class Solution(object):
    def topKFrequent(self, words, k):
        counter = Counter(words)
        heap = []

        for key, cnt in counter.items():
            #print("heap:", (cnt, key))
            if len(heap) < k:
                heapq.heappush(heap, (cnt, key))
            else:
                if heap[0][0] < cnt:
                    a = heapq.heappop(heap)
                    #print("heap[0][1] < cnt:", a)
                    heapq.heappush(heap, (cnt, key))
                    #print("heappush:", (cnt, key))
                if heap[0][0] == cnt:
                    if heap[0][1] > key:
                        b = heapq.heappop(heap)
                        #print("heap[0][0][0] > key[0]:", b)
                        heapq.heappush(heap, (cnt, key))
                        #print("heappush:", (cnt, key))

        heap.sort(key = lambda item:(-item[0], item[1]))
        #print(heap)
        return [item[1] for item in heap]

words = ["rmrypv","zgsedk","jlmetsplg","wnfbo","hnnftqf","bxlr","sviavwoxss","jdbgvc","zddeno","rxgw","hnnftqf","hdmvplne","rlmdt","jlmetsplg","ous","rmrypv","fwxulnpit","dmhuq","rxgw","ledleb","bxlr","indbvb","ckqqibnx","cub","ijww","ehd","hfhlfqzkcn","sviavwoxss","rxgw","bxjxpfp","mgqj","oic","ptk","fwxulnpit","ijww","sviavwoxss","bgfvfa","zfkgsudxq","alkq","dmhuq","zddeno","rxgw","zgsedk","amarxpg","bgfvfa","wnfbo","sviavwoxss","sviavwoxss","alkq","nmclxk","zgsedk","bwowfvira","ous","bxlr","zddeno","rxgw","ous","wnfbo","rmrypv","sviavwoxss","ehd","zgsedk","jlmetsplg","abxvhyehv","ledleb","wnfbo","bgfvfa","bwowfvira","amarxpg","wnfbo","bwowfvira","dmhuq","ouy","bxlr","rxgw","oic","hnnftqf","ledleb","rlmdt","oldainprua","ous","ckqqibnx","dmhuq","hnnftqf","oic","jlmetsplg","ppn","amarxpg","jlgfgwb","bxlr","bwowfvira","hdmvplne","oic","ledleb","bwowfvira","oic","ptk","dmhuq","abxvhyehv","ckqqibnx","indbvb","ypzfk","rmrypv","bxjxpfp","amarxpg","dmhuq","sviavwoxss","bwowfvira","zfkgsudxq","wnfbo","rxgw","jxkvrmajri","cub","abxvhyehv","bwowfvira","indbvb","ehd","ckqqibnx","oic","ptk","hnnftqf","ouy","oic","zgsedk","abxvhyehv","ypzfk","ptk","sviavwoxss","rmrypv","oic","ous","abxvhyehv","hnnftqf","hfhlfqzkcn","ledleb","cub","ppn","zddeno","indbvb","oic","jlmetsplg","ouy","bwowfvira","bklij","gucayxp","zfkgsudxq","hfhlfqzkcn","zddeno","ledleb","zfkgsudxq","hnnftqf","bgfvfa","jlmetsplg","indbvb","ehd","wnfbo","hnnftqf","rlmdt","bxlr","indbvb","jdbgvc","jlmetsplg","cub","jlgfgwb","ypzfk","indbvb","dmhuq","jlmetsplg","zgsedk","rmrypv","cub","rxgw","ledleb","rxgw","sviavwoxss","ckqqibnx","hdmvplne","dmhuq","wnfbo","jlmetsplg","bxlr","zfkgsudxq","bxjxpfp","ledleb","indbvb","ckqqibnx","ous","ckqqibnx","cub","ous","abxvhyehv","bxlr","hfhlfqzkcn","hfhlfqzkcn","oic","ten","amarxpg","indbvb","cub","alkq","alkq","sviavwoxss","indbvb","bwowfvira","ledleb"]
k = 41
#k = 14
#words = ["i","love","leetcode","coding", "i", "love"]
#k = 1
a = Solution()
print(a.topKFrequent(words, k))
#print(Counter(words))
#["nftk","qkjzgws","qrkgmliewc","nsfspyox","qengse","htpvnmozay","""fqizrrnmif"","glarko",""hdemkfr"","pwqvwmlgri","qggx","zskaqzwo","ijy","zlfiwwb"]
#["indbvb","oic","sviavwoxss","bwowfvira","jlmetsplg","ledleb","rxgw","bxlr","dmhuq","hnnftqf","wnfbo","ckqqibnx","cub","ous","abxvhyehv","rmrypv","zgsedk","amarxpg","hfhlfqzkcn","zddeno","zfkgsudxq","alkq","bgfvfa","ehd","ptk","bxjxpfp","hdmvplne","ouy","rlmdt","ypzfk","fwxulnpit","ijww","jdbgvc","jlgfgwb","ppn","bklij","gucayxp","jxkvrmajri","mgqj","nmclxk","oldainprua"]
#["indbvb","oic","sviavwoxss","bwowfvira","jlmetsplg","ledleb","rxgw","bxlr","dmhuq","hnnftqf","wnfbo","ckqqibnx","cub","ous","abxvhyehv","rmrypv","zgsedk","amarxpg","hfhlfqzkcn","zddeno","zfkgsudxq","alkq","bgfvfa","ehd","ptk","bxjxpfp","hdmvplne","ouy","rlmdt","ypzfk","fwxulnpit","ijww","jdbgvc","jlgfgwb","ppn","bklij","gucayxp","jxkvrmajri","mgqj","nmclxk","oldainprua"]
#["indbvb","oic","sviavwoxss","bwowfvira","jlmetsplg","ledleb","rxgw","bxlr","dmhuq","hnnftqf","wnfbo","ckqqibnx","cub","ous","abxvhyehv","rmrypv","zgsedk","amarxpg","hfhlfqzkcn","zddeno","zfkgsudxq","alkq","bgfvfa","ehd","ptk","bxjxpfp","hdmvplne","ouy","rlmdt","ypzfk","fwxulnpit","ijww","jdbgvc","jlgfgwb","ppn","gucayxp","jxkvrmajri","mgqj","nmclxk","oldainprua","ten"]


import collections
class Solution(object):
    # time = O(nlog n)
    # space = O(n)
    def topKFrequent(self, words, k):

        # Builds dictionary counts which hold word as a key and its counts as val
        counts = collections.Counter(words)
        print(counts)
        heap = []
        for word, count in counts.items():
            heapq.heappush(heap, (count, word))
            if len(heap) > k:
                heapq.heappop(heap)
        print(heapq.heappop(heap))
        #heap.sort(key = lambda item: (-item[0], item[1]))
        #print(heap)
        #return [heap[i][1] for i in range(k)]

#words = ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
#words = ["a","aa","aaa"]
#words = ["i","love","leetcode","coding", "i", "love"]
#k = 1
#a = Solution()
#print(a.topKFrequent(words, k))
#print(Counter(words))

#print("i" < "love")
