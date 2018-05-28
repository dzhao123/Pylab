class node(object):
    def __init__(self):
        self.child = {}
        self.word =False

class prefix(object):
    def __init__(self):
        self.root = node()

    def build_ptree(self, dicts):
        for word in dicts:
            tnode = self.root
            for lttr in word:
                if lttr not in tnode.child:
                    tnode.child[lttr] = node()
                tnode = tnode.child[lttr]
            tnode.word = True

    def search(self, word):
        tnode = self.root
        succ = ''
        for lttr in word:
            if lttr not in tnode.child:
                return word
            else:
                succ += lttr
                tnode = tnode.child[lttr]
                if tnode.word == True:
                    return succ
        return word

class Solution(object):


    def replaceWords(self, dicts, sentence):
        res = []
        tree = prefix()
        tree.build_ptree(dicts)
        ls = sentence.split(' ')
        index = 1
        for word in ls:
            #print(index)
            #print(tree.search(word))
            index += 1
            res.append(tree.search(word))


        return ' '.join(res)

dicts = ["a", "aa", "aaa", "aaaa"]
sentence = "a aa a aaaa aaa aaa aaa aaaaaa bbb baba ababa"
a = Solution()
#a.replaceWords(dicts,sentence)
print(a.replaceWords(dicts,sentence))
