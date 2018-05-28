class listNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None


def reverse_by_stack(head):

    root = head
    stack = []
    while root:
        stack.insert(0, root.val)
        root = root.next

    root = head
    for val in stack:
        root.val = val
        root = root.next

    return head

def reverse_by_iteration(head):

    prev = None
    curr = head
    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
    return prev


def reverse_by_recursion(head):

    if head is None or head.next is None:
        return head

    head_next = head.next
    new_head = reverse_by_recursion(head.next)
    head_next.next = head
    head.next = None
    return new_head


def remove(head,index):

    dummy = listNode(None)
    dummy.next = head
    temp = dummy

    while index > 1:
        temp = temp.next
        index -= 1
        #print(root.val)

    temp.next = temp.next.next

    return dummy.next


def insert(head, node, index):

    dummy = listNode(None)
    dummy.next = head
    temp  = dummy

    while index > 1:
        temp = temp.next
        index -= 1

    node.next = temp.next
    temp.next = node

    return head


a = listNode(1)
a.next = listNode(2)
a.next.next = listNode(3)
a.next.next.next = listNode(4)
a.next.next.next.next = listNode(5)

b = insert(a, listNode(6), 4)
print(b.next.next.next.val)
