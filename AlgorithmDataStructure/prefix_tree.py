class Trie(object):

    def __init__(self):
        self.root = {}

    def insert(self, word):
        child = self.root
        for i in range(len(word)):
            if word[i] in child:
                child = child[word[i]]
            else:
                if i == len(word)-1:
                    child[word[i]] = {'end'}
                else:
                    child[word[i]] = {}

    def search(self, word):
        child = self.root
        for l in word:
            if l in child:
                child = child[l]
                if 'end' in child:
                    return True
            else:
                return False

    def startsWith(self, prefix):
        for l in prefix:
            if l in self.root:
                if self.root[l] != None:
                    return True
            else:
                return False


a =Trie()
a.insert("something")
print(a.search("something"))




class TrieNode:
        # Initialize your data structure here.
        def __init__(self):
            self.word=False
            self.children={}

class Trie2:

    def __init__(self):
        self.root = TrieNode()

        # @param {string} word
        # @return {void}
        # Inserts a word into the trie.
    def insert(self, word):
        node=self.root
        for i in word:
            if i not in node.children:
                node.children[i]=TrieNode()
            node=node.children[i]
        node.word=True

        # @param {string} word
        # @return {boolean}
        # Returns if the word is in the trie.
    def search(self, word):
        node=self.root
        for i in word:
            if i not in node.children:
                return False
            node=node.children[i]
        return node.word

        # @param {string} prefix
        # @return {boolean}
        # Returns if there is any word in the trie
        # that starts with the given prefix.
    def startsWith(self, prefix):
        node=self.root
        for i in prefix:
            if i not in node.children:
                return False
            node=node.children[i]
        return True



b = Trie2()
b.insert('something')
print(b.search('something'))
