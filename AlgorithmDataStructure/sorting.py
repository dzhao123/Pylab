def selection(inputs):

    for i in range(len(inputs)):
        mindex = i
        for j in range(i+1,len(inputs)):
           if inputs[mindex] > inputs[j]:
               mindex = j
        inputs[mindex], inputs[i] = inputs[i], inputs[mindex]

    return inputs

def bubble(inputs):

    for i in range(len(inputs)-1,-1,-1):
        for j in range(i):
            if inputs[j] > inputs[j+1]:
                inputs[j], inputs[j+1] = inputs[j+1], inputs[j]

    return inputs


def insertion(inputs):

    for i in range(len(inputs)):
        for j in range(i,0,-1):
            if inputs[j] < inputs[j-1]:
                inputs[j], inputs[j-1] = inputs[j-1], inputs[j]
    return inputs




class Solution(object):
    def merge(self, left, right):

        if not len(left) or not len(right):
            return left or right

        res = []
        i, j = 0, 0
        while len(res) < len(left) + len(right):
            if left[i] < right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1

            if i == len(left) or j == len(right):
                res.extend(left[i:] or right[j:])
                break

        return res


    def mergeSort(self, inputs):

        if len(inputs) < 2:
            return inputs

        mid = len(inputs)//2

        left = self.mergeSort(inputs[:mid])
        right = self.mergeSort(inputs[mid:])

        return self.merge(left, right)



class Solution3(object):

    def merge(self, inputs, left, mid, right):

        i = left
        j = mid + 1
        #i = 0
        #j = mid - left + 1
        #amid = (right-i)//2
        aux = inputs[left : right+1]
        for k in range(left, right+1):

            if i > mid:
                inputs[k] = aux[j - left]
                j += 1
            elif j > right:
                inputs[k] = aux[i - left]
                i += 1
            elif aux[i - left] > aux[j - left]:
                inputs[k] = aux[j - left]
                j += 1
            else:
                inputs[k] = aux[i - left]
                i += 1


    def mergeSort(self, inputs, left, right):

        if left >= right:
            return

        mid = (left + right)//2

        self.mergeSort(inputs, left, mid)
        self.mergeSort(inputs, mid+1, right)
        self.merge(inputs, left, mid, right)

        return inputs







class Solution4(object):

    def merge(self, inputs, left, right):

        #i = left
        #j = mid + 1
        i = 0
        mid = (right-left)//2
        j = mid+1
        aux = inputs[left:right+1]
        end = len(aux) - 1
        for k in range(left, right+1):

            if i > mid:
                inputs[k] = aux[j]
                j += 1
            elif j > end:
                inputs[k] = aux[i]
                i += 1
            elif aux[i] > aux[j]:
                inputs[k] = aux[j]
                j += 1
            else:
                inputs[k] = aux[i]
                i += 1


    def mergeSort(self, inputs, left, right):

        if left >= right:
            return

        mid = (left + right)//2

        self.mergeSort(inputs, left, mid)
        self.mergeSort(inputs, mid+1, right)
        self.merge(inputs, left, right)

        return inputs



class Solution5(object):

    def partition(self, arr, left, right):

        v = arr[left]
        j = left
        for i in range(left+1, right+1):
            if arr[i] < v:
                arr[i], arr[j+1] = arr[j+1], arr[i]
                j += 1
        arr[left], arr[j] = arr[j], arr[left]
        return j

    def quickSort(self, arr, left, right):

        if left >= right:
            return

        p = self.partition(arr, left, right)
        self.quickSort(arr, left, p-1)
        self.quickSort(arr, p+1, right)

        return arr





if __name__ == '__main__':
    a = Solution5()
    print(a.quickSort([10,7,4,3,6,9,2,8,1], 0, 8))
