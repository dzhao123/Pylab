def remove_char(string, *arg):
    l = []
    for item in string:
        if item not in arg:
            l.append(item)
    return ''.join(l)

def remove_char(string, *arg):
    fast = 0
    slow = 0

    while fast < len(string):
        if string[slow] in arg:
            string[slow] = string[fast]
            slow += 1
        fast += 1

    return string[:slow]

print(remove_char('abcsd','b','d'))



def remove_duplicate_space(string):

    fast = 0
    slow = 0
