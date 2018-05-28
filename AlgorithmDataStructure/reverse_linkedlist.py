class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

def reverse_list_recursion(head):
    if not head or not head.next:
        return head
    next_node = head.next
    new_head = reverse_list_recursion(head.next)
    next_node.next = head
    head.next = None
    return new_head

def reverse_list_in_pair(head):
    if not head or not head.next:
        return head
    next_node = head.next
    head.next = reverse_list_in_pair(head.next.next)
    next_node.next = head
    return next_node


def reverse(head):

    current = head
    previous = None

    while current:
        next = current.next
        current.next = previous
        previous = current
        current = next
    head = previous
