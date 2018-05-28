class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Solution(object):

    def deleteNode(self, root, key):

        if root is None:
            return

        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        if root.val < key:
            root.right = self.deleteNode(root.right, key)
        if root.val == key:
            if root.left and not root.right:
                return root.left
            if root.right and not root.left:
                return root.right
            if root.left and root.right:
                temp = root
                mini = root.val
                while temp.left:
                    temp = temp.left
                root.val = temp.val
                temp.val = mini
                root.right = self.deleteNode(root.right, mini)
            if not root.left and not root.right:
                return None
        return root



if __name__ == "__main__":
    a = Solution()
    root = TreeNode(5)
    root.left = TreeNode(3)
    root.right = TreeNode(6)
    root.left.left = TreeNode(2)
    root.left.right = TreeNode(4)
    root.right.right = TreeNode(7)

    b = a.deleteNode(root, 0)
    print(b.left.val)
