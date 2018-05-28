class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None




def lca(root, p, q):

    if root is None:
        return None

    if root == p or root == q:
        return root

    left = lca(root.left, p, q)
    right = lca(root.right, p, q)

    if left and right:
        return root
    if left and not right:
        return left
    if right and not left:
        return right


root = TreeNode(0)
root.left = TreeNode(1)
root.left.left = TreeNode(2)
root.left.right = TreeNode(3)
root.left.left.left = TreeNode(4)
root.left.left.right = TreeNode(5)
p = root.left.left.left
q = root.left.right
x = lca(root,p, q)
print(x.val)
