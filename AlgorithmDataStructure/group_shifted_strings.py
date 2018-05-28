class Solution(object):
    def groupStrings(self, strings):
        """
        :type strings: List[str]
        :rtype: List[List[str]]
        """
        container = {}
        output = []
        for item in strings:
            dis = ''
            gro = []
            for i in range(1,len(item)):
                temp = ord(item[i]) - ord(item[i-1])
                if temp > 0:
                    dis += str(temp)
                else:
                    dis += str(26 + temp)
            if dis in container:
                container[dis].append(item)
            else:
                container[dis] = [item]

        for key,val in container.items():
            output.append(val)

        return output

if __name__ == '__main__':
    a = Solution()
    string = ["abc","efg","hmi"]
    print(a.groupStrings(string))
