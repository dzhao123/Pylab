def palindrom(string):

    if len(string) == 0:
        return True

    if 'a'<= string[0] <= 'z' and 'a'<= string[-1] <= 'z' and string[0] == string[-1]:
        return palindrom(string[1:-1])
    if (string[0] < 'a' or string[0] > 'z') and 'a'<= string[-1] <= 'z':
        return palindrom(string[1:])
    if (string[-1] < 'a' or string[-1] > 'z') and 'a' <= string[0] <= 'z':
        return palindrom(string[:-1])
    if (string[0] < 'a' or string[0] > 'z') and (string[-1] < 'a' or string[-1] > 'z'):
        return palindrom(string[1:-1])
    return False

print(palindrom("ASDFGaacbcaaASFASF"))




