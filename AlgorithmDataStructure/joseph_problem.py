class listNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None

def joseph(m, n):

    if m < 0 or n < 0:
        return -1

    if n == 1:
        return listNode(1)

    root = listNode(1)
    start = root
    for i in range(2, n+1):
        start.next = listNode(i)
        start = start.next
    start.next = root

    while root.next != root:
        for i in range(m - 2):
            root = root.next
        root.next = root.next.next
        root = root.next

    return root.val

print(joseph(4, 10))
