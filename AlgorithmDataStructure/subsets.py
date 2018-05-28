
def helper(res, path, arr, index):

    if index == len(arr):
        res.append(path[:])
        return

    path.append(arr[index])
    helper(res, path, arr, index+1)
    path.pop()
    helper(res, path, arr, index+1)


def subset(arr):

    res = []
    path = []
    index = 0
    helper(res, path, arr, index)
    return res


print(subset([1,2,3]))
