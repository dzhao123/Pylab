def sift_up(arr, i):

    parent = (i-1)//2
    smaller = i

    if parent >= 0 and arr[parent] > arr[smaller]:
        arr[parent], arr[smaller] = arr[smaller], arr[parent]
        sift_up(arr, parent)


arr = [0, 1, 5, 6, 8, -1]
sift_up(arr, len(arr) - 1)
#print(arr)


def sift_down(arr, i):

    lind = i << 1 + 1
    rind = i << 1 + 2

    smaller = i
    if lind < len(arr) and arr[lind] < arr[smaller]:
        smaller = lind
    if rind < len(arr) and arr[rind] < arr[smaller]:
        smaller = rind
    if smaller != i:
        arr[smaller], arr[i] = arr[i], arr[smaller]
        sift_down(arr, smaller)




import heapq
q = [1,42,523,5,23,657,456]
heapq.heapify(q)
while q:
    print(heapq.heappop(q))

