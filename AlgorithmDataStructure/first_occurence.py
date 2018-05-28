def find_first_occurence(nums, target):
    if len(nums) == 0:
        return -1

    left = 0
    right = len(nums) - 1

    while left < right - 1:

        mid = int((left + right)/2)

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid
        elif nums[mid] > target:
            right = mid

    if nums[left] == target:
        return left
    if nums[right] == target:
        return right

    return -1

print(find_first_occurence([1,2,3,4,5,6,7,8,9],5))
