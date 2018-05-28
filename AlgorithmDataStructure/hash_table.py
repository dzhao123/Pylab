import sys
def nearest_repeat(array):
    hash = {}
    dist = sys.maxsize
    for index, val in enumerate(array):
        if val not in hash:
            hash[val] = index
        else:
            dist = min(dist, index - hash[val])
            hash[val] = index

    return dist

def char_removal(string, remove):

    i = 0
    j = 0
    while j < len(string):
        if string[j] not in remove:
            string[j] = string[i]
            i += 1
        j += 1
    return string[:i]




def longest_contained_range(array):

    slow = 0
    maxdis = 0
    hash_set = {}
    for fast in range(len(array)):
        if array[fast] in hash_set:
            if hash_set[array[fast]] >= slow:
                maxdis = max(maxdis, fast - slow)
                slow = hash_set[array[fast]] + 1
                hash_set[array[fast]] = fast

        hash_set[array[fast]] = fast
    maxdis = max(maxdis, len(array) - slow)

    return maxdis
print(longest_contained_range("abba"))
