# LeetCode 日志

## 《计算之魂》

## 《剑指 offer》

## 《Hot 100》

### 哈希表

[1. 两数之和](https://leetcode.cn/problems/two-sum/)

**思路**

遍历的时候使用哈希表记录之前的值和位置，下一步直接查找key中是否存在target - nums[i]这个值，存在则取出target - nums[i]的位置，返回结果。

**代码**

```java
// key: num[i]位置上的值，value：num[i]位置
class Solution {
  
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]), i};
            } else {
                map.put(nums[i], i);
            }
        }
        return new int[]{};
    }
  
}
```

**复杂度**

- 时间复杂度: O(N)
- 空间复杂度: O(N)



[49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/)

**思路**

使用哈希表，将异位词转变成一个相同的key，value是每一个key对应的异位词列表。

**方法一：排序**

**代码**

```Java []

class Solution {

    // 方法一：对每一个字符串排序
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            // 错误：String key = chars.toString();
            String key = new String(chars); // !API
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }
  
}
```
**复杂度**

- 时间复杂度: O(N * Mlog(M))
- 空间复杂度: O(N * M)

**方法二：计数**

**代码**

```Java []

class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {

        // 方法二：对每一个字符串的字符进行计数
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            // 记录每个字符出现的次数
            int[] counts = new int[26]; 
            int len = str.length();
            for (int i = 0; i < len; i++) {
                counts[str.charAt(i) - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            // 将"abca"转换为”a2b1“作为唯一的 key
            for (int i = 0; i < 26; i++) {
                // 判断是否存在这个字符
                if (counts[i] != 0) {
                    sb.append((char)('a' + i));
                    sb.append(counts[i]);
                }
            }
            String key = sb.toString();
            List<String> list = map.getOrDefault(key, new ArrayList<String>()); // !API
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values()); // !API
    }
  
}
```

**复杂度**

- 时间复杂度: O(N * M)
- 空间复杂度: O(N * M)

### 集合

[128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/)

**思路**

使用集合进行去重，然后从第一个元素开始累加 1，直至不连续跳出循环，更新最大值。

**代码**

```Java []

class Solution {
  
    public int longestConsecutive(int[] nums) {
        int maxLen = 0;
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        for (int num : set) {
            // 如果当前值是连续序列的起点，则继续计数
            if (!set.contains(num - 1)) {
                int curLen = 1;
                while (set.contains(++num)) {
                    curLen++;
                }
                maxLen = Math.max(maxLen, curLen);
            }
        }
        return maxLen;
    }
  
}
```

**复杂度**

- 时间复杂度：O(N)
- 空间复杂度：O(N)

### 双指针

[283. 移动零](https://leetcode.cn/problems/move-zeroes/description/)

**思路**

左指针指向已经处理好的序列的尾部的后一个元素，右指针指向待处理序列的头部，遍历一遍交换即可即可。

**代码**

```Java []

class Solution {
  
    public void moveZeroes(int[] nums) {
        int left = 0; // 左指针指向已经处理好的序列的尾部的后一个元素
        int right = 0; // 右指针指向待处理序列的头部
        while (right < nums.length) {
            if (nums[right] != 0) {
                swap(nums, left, right);
                left++;
            }
            right++;
        }
    }

    private void swap(int[] nums, int left, int right) {
        int tmp = nums[left];
        nums[left] = nums[right];
        nums[right] = tmp;
    }
  
}
```

**复杂度**

- 时间复杂度: O(N)
- 空间复杂度: O(1)



[11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/)

**思路**

水的容积取决于短板，双指针进行循环收窄。

<img src="img/1691907552-JmFzQO-image.png" alt="image20231111999.png" style="zoom: 20%;" />

这里使用了双指针进行循环收窄的过程，每次选定两板中的短板，向内收窄一格。

![image.png](img/1691907885-BsANWU-image.png)

**代码**

```Java []

class Solution {
  
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int max = 0;
        while (left < right) {
            max = height[left] < height[right] ?
                Math.max(max, (right - left) * height[left++]):
                Math.max(max, (right - left) * height[right--]);
        }
        return max;
    }
  
}
```

**复杂度**

- 时间复杂度: O(N)
- 空间复杂度: O(1)



[15. 三数之和](https://leetcode.cn/problems/3sum/description/)

**思路**

三指针遍历

![image.png](img/1692166827-KiksBs-image.png)

使用三个指针对数组逐个遍历，注意去重的几种情况：

1. i > 0 && ((nums[i] == nums[i - 1]))

2. 在找到一组和为 0 的数据后，继续去重：
  
    - nums[right] == nums[right - 1]
    
    - nums[left] == nums[left + 1]

**Code**

```Java []
class Solution {
  
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                return res;
            }
            // i 要和之前的元素比较去重
            if (i > 0 && (nums[i] == nums[i - 1])) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum < 0) {
                    left++;
                } else if (sum > 0) {
                    right--;
                } else {
                    res.add(Arrays.asList(nums[i], nums[left], nums[right])); // !API
                    while ((left < right) && (nums[right] == nums[right - 1])) {
                        right--;
                    }
                    while ((left < right) && (nums[left] == nums[left + 1])) {
                        left++;
                    }
                    right--;
                    left++;
                }
            }
        }
        return res;
    }
  
}
```

**复杂度**

- 时间复杂度: O(N^2)
- 空间复杂度: O(N)



[42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/description/)

**思路**

对于每个位置，计算它左侧最大值和右侧最大值中的较小值，然后减去当前位置的高度。这个值表示当前位置能够积累的雨水量（每个桶的容量）。

<img src="img/image-20231112153559570-9774562.png" alt="image-20231112153559570" style="zoom: 67%;" />

**代码**

```Java []

class Solution {

    public int trap(int[] height) {
        int res = 0;
        int left = 0;
        int right = height.length - 1;
        int leftMax = 0;
        int rightMax = 0;
        while (left <= right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);
            res += leftMax < rightMax ? 
                leftMax - height[left++] : rightMax - height[right--];
        }
        return res;
    }

}
```

**复杂度**

- 时间复杂度: O(N)
- 空间复杂度: O(1)

### 滑动窗口

[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)

**思路**

哈希表记录该字符最近一次出现的位置，left，i为不重复的左右窗口边界，每次遍历都更新一次最大值。

**代码**

```Java []
class Solution {
  
    public int lengthOfLongestSubstring(String s) {
        if ((s == null) || (s.length() == 0)) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int max = 0;
        int left = 0; // left是滑动窗口的左边，i是右边
        // map 中已经有了该字符键就更新 left 的值，不包含就更新 i 和 max 的值
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            } 
            map.put(s.charAt(i), i); 
            max = Math.max(max, i - left + 1);
        }
        return max;
    }
  
}
```

**复杂度**

- 时间复杂度: O(N)
- 空间复杂度: O(N)



[438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/)

**思路**

通过两个数组int[26]记录每次滑动窗口内的字符数量是否相等，相等则为字母异位词。注意第一组数据的处理。

在每个循环迭代中，首先更新 sCount 数组，这是通过以下步骤完成的：

1. 减少窗口左侧字符的频次：sCount[s.charAt(i) - 'a']--。因为窗口的左侧将不再包含在滑动窗口内，所以对应字符的频次需要减少。
2. 增加窗口右侧字符的频次：sCount[s.charAt(i + pLen) - 'a']++。新加入窗口的字符需要增加频次。

随后，代码检查更新后的 sCount 是否与 pCount 相等，如果相等，说明窗口内的字符频次与字符串 p 的字符频次相同，也就是窗口内的子串是 p 的字母异位词。此时，将当前窗口的起始索引 i + 1 加入到结果列表中。

**代码**

```Java []

class Solution {
  
    public List<Integer> findAnagrams(String s, String p) {
        int sLen = s.length();
        int pLen = p.length();

        if (sLen < pLen) {
            return new ArrayList<Integer>();
        }

        ArrayList<Integer> res = new ArrayList<Integer>();
        int[] sCount = new int[26];
        int[] pCount = new int[26];

        for (int i = 0; i < pLen; i++) {
            sCount[s.charAt(i) - 'a']++;
            pCount[p.charAt(i) - 'a']++;
        }

        if (Arrays.equals(sCount, pCount)) {
            res.add(0);
        }

        for (int i = 0; i < sLen - pLen; i++) {
            sCount[s.charAt(i) - 'a']--;
            sCount[s.charAt(i + pLen) - 'a']++; // 右窗口移动

            if (Arrays.equals(sCount, pCount)) {
                res.add(i + 1);
            }
        }
        return res;
    }
  
}
```

**复杂度**

- 时间复杂度: O(M + N)
- 空间复杂度: O(1)

### 字串

[560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/)

**思路**

pre[i] − pre[j − 1] == k => pre[j − 1] == pre[i] − k

**代码**

```Java []
class Solution {
  
    // pre[i] − pre[j − 1] == k —> pre[j − 1] == pre[i] − k
    public int subarraySum(int[] nums, int k) {
        int count = 0;
        int pre = 0;
        HashMap<Integer, Integer> map = new HashMap<>(); // 记录前缀和的数量
        map.put(0, 1); // !初始化
        for (int num : nums) {
            pre += num;
            if (map.containsKey(pre - k)) {
                count += map.get(pre - k);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);
        }
        return count;
    }
  
}
```

复杂度

- 时间复杂度: O(N)
- 空间复杂度: O(N)



[239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/description/)

**思路**

使用固定容量的单调队列收集每一个滑动窗口的最大值。

<img src="img/1699862617-fSZnHF-image.png" alt="image.png" style="zoom:50%;" />

**代码**

```Java []
// 单调队列
class Solution {
  
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] res = new int[n - k + 1];
        Deque<Integer> deque = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            // 添加元素时，如果要添加的元素大于队尾处的元素，就将队尾元素弹出
            while (!deque.isEmpty() && (nums[i] >= nums[deque.peekLast()])) {
                deque.pollLast();
            }
            deque.offerLast(i);
            // 如果队首存储的角标就是滑动窗口左边界数值，就移除队首
            if (!deque.isEmpty() && ((i - k) == deque.peekFirst())) {
                deque.pollFirst();
            }
            // 当i增长到第一个窗口右边界时，每滑动一步都将队首角标对应的元素放入到结果数组
            if (i >= k - 1) {
                res[i - k + 1] = nums[deque.peekFirst()];
            }
        }
        return res;
    }
  
}

```

**复杂度**

- 时间复杂度: O(N)
- 空间复杂度: O(N)



```Java []
// 使用优先队列（大根堆）
class Solution {
  
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        // 设定优先队列的排序顺序：
        // 如果 pair1 和 pair2 的第一个元素不相等 (pair1[0] != pair2[0])，则按照第一个元素降序排列，即 pair2[0] - pair1[0]。
        // 如果 pair1 和 pair2 的第一个元素相等，则按照第二个元素降序排列，即 pair2[1] - pair1[1]，保证稳定性。
        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(
            (pair1, pair2) -> pair1[0] != pair2[0] ? pair2[0] - pair1[0] : pair2[1] - pair1[1]);
        for (int i = 0; i < k; i++) {
            priorityQueue.offer(new int[]{nums[i], i});
        }
        int[] res = new int[n - k + 1];
        res[0] = priorityQueue.peek()[0];
        for (int i = k; i < n; i++) {
            // 保持窗口大小，移除超出窗口范围的元素
            priorityQueue.offer(new int[]{nums[i], i});
            while (priorityQueue.peek()[1] <= i - k) {
                priorityQueue.poll();
            }
            res[i - k + 1] = priorityQueue.peek()[0];
        }
        return res;
    }
  
}
```



[76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/description/)

**思路**

使用滑动窗口双指针遍历字符串，使用count标识当次遍历中还需要几个字符才能够满足包含t中所有字符的条件。

步骤一：不断增加j使滑动窗口增大，直到窗口包含了T的所有元素。

步骤二：不断增加i使滑动窗口缩小，因为是要求最小字串，所以将不必要的元素排除在外，使长度减小，直到碰到一个必须包含的元素，这个时候不能再扔了，再扔就不满足条件了，记录此时滑动窗口的长度，并保存最小值。

步骤三：让i再增加一个位置，这个时候滑动窗口肯定不满足条件了，那么继续从步骤一开始执行，寻找新的满足条件的滑动窗口，如此反复，直到j超出了字符串S范围。

<img src="img/1693208998-WWbNFQ-image.png" alt="image.png" style="zoom:67%;" />

**代码**

```Java []
class Solution {
  
    public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0) {
            return "";
        }
        int[] need = new int[128];
        for (int i = 0; i < t.length(); i++) {
            need[t.charAt(i)]++;
        }
        int l = 0, r = 0, size = Integer.MAX_VALUE, count = t.length(), start = 0;
        while (r < s.length()) {
            char c = s.charAt(r);
            if (need[c] > 0) {
                count--;
            }
            need[c]--;
            // count == 0说明当前的窗口已经满足了包含t所需所有字符的条件
            if (count == 0) {
                // 如果左边界这个字符对应的值在need[]数组中小于0，说明他是一个多余元素，不包含在t内
                while (l < r && need[s.charAt(l)] < 0) {
                    need[s.charAt(l)]++;
                    l++; // 左边界向右移，过滤掉这个元素
                }
                // 如果当前的这个窗口值比之前维护的窗口值更小，需要进行更新
                if (r - l + 1 < size) {
                    size = r - l + 1;
                    start = l;
                }
                need[s.charAt(l)]++; // 先将l位置的字符计数重新加1
                // 重新维护左边界值和当前所需字符的值count 
                l++;
                count++;
            }
            r++;
        }
        return size == Integer.MAX_VALUE ? "" : s.substring(start, start + size); // !API
    }
  
}
```

**复杂度**

- 时间复杂度: O(N)
- 空间复杂度: O(1)

### 普通数组

[53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/)

**思路**

动态规划：

![image.png](img/1693213433-psmwwl-image.png)

**代码**

```Java []
class Solution {

    public int maxSubArray(int[] nums) {
        int len = nums.length;
        int[] dp = new int[len];
        dp[0] = nums[0];

        for (int i = 1; i < len; i++) {
            if (dp[i - 1] > 0) {
                dp[i] = dp[i - 1] + nums[i];
            } else {
                dp[i] = nums[i];
            }
        }

        int res = dp[0];
        for (int i = 1; i < len; i++) {
            res = Math.max(res, dp[i]);
        }
        return res;
    }

}
```

**复杂度**

- 时间复杂度: O(N)
- 空间复杂度: O(1)



[56. 合并区间](https://leetcode.cn/problems/merge-intervals/)

**思路**

如果我们按照区间的左端点排序，那么在排完序的列表中，可以合并的区间一定是连续的。如下图所示，标记为蓝色、黄色和绿色的区间分别可以合并成一个大区间，它们在排完序的列表中是连续的：

<img src="img/50417462969bd13230276c0847726c0909873d22135775ef4022e806475d763e-56-2.png" alt="56-2.png" style="zoom: 67%;" />

**代码**

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] interval1, int[] interval2) {
                return interval1[0] - interval2[0];
            }
        });
        List<int[]> merged = new ArrayList<int[]>();
        for (int i = 0; i < intervals.length; ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            // 如果当前区间的左端点比数组 merged 中最后一个区间的右端点大，那么它们不会重合，我们可以直接将这个区间加入数组 merged 的末尾；
            // 否则，它们重合，我们需要用当前区间的右端点更新数组 merged 中最后一个区间的右端点，将其置为二者的较大值。
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                merged.add(new int[]{L, R});
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }
}
```

**复杂度**

- 时间复杂度：O(N * log(N))
- 空间复杂度：O(N * log(N))



[189. 轮转数组](https://leetcode.cn/problems/rotate-array/)

**思路**

![image-20231122222408712](img/image-20231122222408712-0663050.png)

**代码**

```java
class Solution {
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start += 1;
            end -= 1;
        }
    }
}
```

**复杂度**

- 时间复杂度：O(N)
- 空间复杂度：O(1)



[238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)

**思路**

<img src="img/1624619180-vpyyqh-Picture1.png" alt="Picture1.png" style="zoom:48%;" />

**代码**

```java
class Solution {
  
    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;
        if (len == 0) return new int[0];
        int[] ans = new int[len];
        ans[0] = 1;
        int tmp = 1;
        for (int i = 1; i < len; i++) {
            ans[i] = ans[i - 1] * nums[i - 1];
        }
        for (int i = len - 2; i >= 0; i--) {
            tmp *= nums[i + 1];
            ans[i] *= tmp;
        }
        return ans;
    }
  
}
```

**复杂度**

- 时间复杂度：O(N)
- 空间复杂度：O(1)



[73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/)

**思路**

用矩阵的第一行和第一列代替方法一中的两个标记数组，以达到 O(1) 的额外空间。但这样会导致原数组的第一行和第一列被修改，无法记录它们是否原本包含 0。因此我们需要额外使用两个标记变量分别记录第一行和第一列是否原本包含 0。

**代码**

```java
class Solution {
  
    public void setZeroes(int[][] matrix) {
        // 获取矩阵的行数和列数
        int m = matrix.length, n = matrix[0].length;
        
        // 用于标记第一列是否有零
        boolean flagCol0 = false;
        
        // 用于标记第一行是否有零
        boolean flagRow0 = false;
        
        // 检查第一列是否有零，若有则将 flagCol0 设为 true
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                flagCol0 = true;
            }
        }
        
        // 检查第一行是否有零，若有则将 flagRow0 设为 true
        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == 0) {
                flagRow0 = true;
            }
        }
        
        // 遍历矩阵的除第一行和第一列之外的元素，若某元素为零，则将其对应的第一行和第一列的元素设为零
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        
        // 根据第一行和第一列的标记，将对应的行和列的元素置零
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        // 若 flagCol0 为 true，则将第一列的所有元素置零
        if (flagCol0) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
        
        // 若 flagRow0 为 true，则将第一行的所有元素置零
        if (flagRow0) {
            for (int j = 0; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
    }
  
}
```

**复杂度**

- 时间复杂度：O(N)
- 空间复杂度：O(1)



[54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

**思路**

<img src="img/54_fig1.png" alt="fig1" style="zoom: 50%;" />

**代码**

```java
class Solution {
  
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> order = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return order;
        }
        int rows = matrix.length, columns = matrix[0].length;
        int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
        while (left <= right && top <= bottom) {
            for (int column = left; column <= right; column++) {
                order.add(matrix[top][column]);
            }
            for (int row = top + 1; row <= bottom; row++) {
                order.add(matrix[row][right]);
            }
            if (left < right && top < bottom) {
                for (int column = right - 1; column > left; column--) {
                    order.add(matrix[bottom][column]);
                }
                for (int row = bottom; row > top; row--) {
                    order.add(matrix[row][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return order;
    }
  
}
```

**复杂度**

- 时间复杂度：O(M * N)
- 空间复杂度：O(1)



[48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

**思路**

![image-20231123194904543](img/image-20231123194904543-0740146.png)

**代码**

```java
class Solution {
  
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        // 水平翻转
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - i - 1][j];
                matrix[n - i - 1][j] = temp;
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }
  
}
```

**复杂度**

- 时间复杂度：O(N^2)
- 空间复杂度：O(1)



[240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

**思路**

![image-20231123195832174](img/image-20231123195832174-0740713.png)

**代码**

```java
class Solution {
  
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int x = 0, y = n - 1;
        while (x < m && y >= 0) {
            if (matrix[x][y] == target) {
                return true;
            }
            if (matrix[x][y] > target) {
                --y;
            } else {
                ++x;
            }
        }
        return false;
    }
  
}
```

**复杂度**

- 时间复杂度：O(M + N)
- 空间复杂度：O(1)

## 《代码随想录》

## 《编程之美》
