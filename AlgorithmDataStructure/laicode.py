import sys
class Solution(object):
    def __init__(self):
        self.max = -sys.maxint-1

    def _preorder(self, root):

        if root is None:
            return 0
        else:
            left = max(0,self.maxPathSum(root.left))
            right = max(0,self.maxPathSum(root.right))
            self.max = max(self.max, left+root.val+right)

            return root.val+min(left,right)

    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self._preorder(root)
        return self.max
