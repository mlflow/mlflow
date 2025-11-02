class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        hashmap = {}  # store number -> index
        for i, num in enumerate(nums):
            diff = target - num
            if diff in hashmap:
                return [hashmap[diff], i]
            hashmap[num] = i
