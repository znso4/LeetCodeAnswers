#include "pch.h"
#include "algs.hpp"
#pragma once

using namespace std;

namespace LC{
//2. 两数相加
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

//8. 字符串转换整数 (atoi)
class Automaton {
    string state = "start";
    unordered_map<string, vector<string>> table = {
        {"start", {"start", "signed", "in_number", "end"}},
        {"signed", {"end", "end", "in_number", "end"}},
        {"in_number", {"end", "end", "in_number", "end"}},
        {"end", {"end", "end", "end", "end"}}
    };

    int get_col(char c) {
        if (isspace(c)) return 0;
        if (c == '+' || c == '-') return 1;
        if (isdigit(c)) return 2;
        return 3;
    }
public:
    int sign = 1;
    long long ans = 0;

    void get(char c) {
        state = table[state][get_col(c)];
        if (state == "in_number") {
            ans = ans * 10 + c - '0';
            ans = sign == 1 ? min(ans, (long long)INT_MAX) : min(ans, -(long long)INT_MIN);
        }
        else if (state == "signed")
            sign = c == '+' ? 1 : -1;
    }
};

//104. 二叉树的最大深度
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    //1. 两数之和
    vector<int> twoSum(vector<int>& nums, int target) {
        if (nums.size() < 2) return {};
        unordered_map<int, int> n;
        int comp;
        for (int i = 0; i < nums.size();++i) {
            comp = target - nums[i];
            if (n.find(comp) != n.end()) {
                return { n[comp], i };
            }
            else {
                n[nums[i]] = i;
            }
        }
        return {};
    }

    //2. 两数相加(反向链表)
    ListNode* addTwoNumbers_reverseList(ListNode* l1, ListNode* l2) {
        int carry = 0;
        ListNode* ret = new ListNode(0);
        ListNode* n = ret;
        while (l1 != NULL || l2 != NULL) {
            if (l1 == NULL) {
                n->next = new ListNode(l2->val + carry);
                l2 = l2->next;
            }
            else if (l2 == NULL) {
                n->next = new ListNode(l1->val + carry);
                l1 = l1->next;
            }
            else {
                n->next = new ListNode(l1->val + l2->val + carry);
                l1 = l1->next;
                l2 = l2->next;
            }
            n = n->next;
            carry = n->val / 10;
            n->val %= 10;
        }
        if (carry != 0) {
            n->next = new ListNode(carry);
        }
        n = ret;
        ret = ret->next;
        delete n;
        return ret;
    }
    
    //3. 无重复字符的最长子串
    int lengthOfLongestSubstring(string s) {
        int result = 0, j = 0;
        auto n = s.size();
        bitset<128> hst = { false };
        for (int i = 0; i < n; ++i) {
            if (i != 0) hst[s[i - 1]] = false;
            while (j < n && !hst[s[j]]) {
                hst[s[j++]] = true;
            }
            if (result < j - i) result = j - i;
        }
        return result;
    }

    //4. 寻找两个正序数组的中位数
    int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */
        int m = nums1.size();
        int n = nums2.size();
        int index1 = 0, index2 = 0;
        while (true) {
            // 边界情况
            if (index1 >= m) {
                return nums2[index2 + k - 1];
            }
            if (index2 >= n) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return min(nums1[index1], nums2[index2]);
            }
            // 正常情况
            int newIndex1 = min(index1 + k / 2 - 1, m - 1);
            int newIndex2 = min(index2 + k / 2 - 1, n - 1);
            int pivot1 = nums1[newIndex1];
            int pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= newIndex1 - index1 + 1;
                index1 = newIndex1 + 1;
            }else {
                k -= newIndex2 - index2 + 1;
                index2 = newIndex2 + 1;
            }
        }
    }
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int totalLength = nums1.size() + nums2.size();
        if (totalLength % 2 == 1) {
            return getKthElement(nums1, nums2, (totalLength + 1) / 2);
        }else{
            return (getKthElement(nums1, nums2, totalLength / 2) + getKthElement(nums1, nums2, totalLength / 2 + 1)) / 2.0;
        }
    }

    //5. 最长回文子串
    int expand(string& s, int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            --left; ++right;
        }
        return right - left - 1;
    }
    string longestPalindrome(string s) {
        if (s.size() <= 1) return s;
        int start = 0, maxlen = 0;
        for (int i = 0; i < s.size(); ++i) {
            int len = max(expand(s, i, i), expand(s, i, i + 1));
            if (maxlen < len) {
                maxlen = len;
                start = i - (len - 1) / 2;
            }
        }
        return string(&s[start], maxlen);
    }

    //7. 整数反转
    int reverseInt(int x) {
        long long ret = 0;
        while (x != 0) {
            ret *= 10;
            ret += x % 10;
            x /= 10;
        }
        if (ret > INT_MAX || ret < INT_MIN) ret = 0;
        return static_cast<int>(ret);
    }

    //8. 字符串转换整数 (atoi)
    int myAtoi(string str) {
        Automaton automaton;
        for (char c : str)
            automaton.get(c);
        return static_cast<int>(automaton.sign * automaton.ans);
    }
    
    //9. 回文数
    bool isPalindrome(int x) {
        string s1 = to_string(x);
        string s2(s1.rbegin(), s1.rend());
        return s1==s2;
    }

    //11. 盛最多水的容器
    int maxArea(vector<int>& height) {
        int ret = 0;
        int i = 0;
        int j = static_cast<int>(height.size()) - 1;
        while (i != j) {
            if (height[i] > height[j]) {
                ret = max(ret, (j - i) * height[j]);
                --j;
            }
            else {
                ret = max(ret, (j - i) * height[j]);
                ++i;
            }
        }
        return ret;
    }

    //12. 整数转罗马数字
    string intToRoman(int num) {
        const pair<int, string> table[] = { {1000, "M"},
            {900, "CM"},{500,"D"}, {400, "CD"}, {100, "C"},
            {90,"XC"},{50,"L"}, {40, "XL"},{10,"X"},
            {9,"IX"},{5,"V"},{4,"IV"},{1,"I"}
        };
        string ret;
        int i = 0;
        while (num > 0 && num <= 3999) {
            if (num >= table[i].first) {
                num -= table[i].first;
                ret += table[i].second;
            }
            else {
                ++i;
            }
        }
        return ret;
    }

    //13. 罗马数字转整数
    int romanToInt(string s) {
        const pair<int, string> table[] = { {1000, "M"},
            {900, "CM"},{500,"D"}, {400, "CD"}, {100, "C"},
            {90,"XC"},{50,"L"}, {40, "XL"},{10,"X"},
            {9,"IX"},{5,"V"},{4,"IV"},{1,"I"}
        };
        int ret = 0;
        for (int i = 0; i < 13; ++i) {
            string t = table[i].second;
            while (t == s.substr(0, t.size())) {
                ret += table[i].first;
                s = s.substr(t.size());
            }
        }
        return ret;
    }

    //14. 最长公共前缀
    string longestCommonPrefix(vector<string>& strs) {
        if(strs.size()==0) return "";
        string& s = strs[0];
        int p = s.size();
        for(int i = 1;i<strs.size() && p>0;++i){
            p = min(p, static_cast<int>(strs[i].size()));
            int j = 0;
            while(j < p){
                if(s[j] == strs[i][j]){
                    ++j;
                }else{
                    break;
                }
            }
            p = j;
        }
        return string(s, 0, p);
    }

    //15. 三数之和
    vector<vector<int>> threeSum(vector<int>& nums) {
        if (nums.size() < 3) return {};
        sort(nums.begin(), nums.end());
        vector<vector<int>> ret;
        int i = 0;
        for (i = 0; i < nums.size()-2 && nums[i]<=0; ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int l = i + 1;
            int r = static_cast<int>(nums.size()) - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum == 0) {
                    ret.push_back({ nums[i] , nums[l] , nums[r] });
                    while (l < r && nums[l] == nums[l+1]) ++l;
                    ++l;
                    while (l < r && nums[r] == nums[r-1]) --r;
                    --r;
                }
                else if (sum < 0) { ++l; }
                else { --r; }
            }
        }
        return ret;
    }

    //16. 最接近的三数之和
    int threeSumClosest(vector<int>& nums, int target) {
        if (nums.size() < 3) return {};
        sort(nums.begin(), nums.end());
        int ret = INT_MAX;
        int min_d = INT_MAX;
        for (int i = 0; i < nums.size() - 2; ++i) {
            int l = i + 1;
            int r = static_cast<int>(nums.size()) - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (abs(sum - target) < min_d) {
                    min_d = abs(sum - target);
                    ret = sum;
                }
                if (sum < target) { ++l; }
                else { --r; }
            }
        }
        return ret;
    }

    //20. 有效的括号
    bool isValid(string s) {
        map<char, int> left = { {'{', 0}, {'[', 1}, {'(', 2} };
        map<char, int> right = { {'}', 0}, {']', 1}, {')', 2} };
        //map<string> legal = { "{}", "[]", "()" };
        stack<char> b;
        for (int i = 0; i < s.size(); ++i) {
            if (left.count(s[i])) {
                b.push(s[i]);
            }
            else if (right.count(s[i]) && !b.empty() && right[s[i]] == left[b.top()]) {
                b.pop();
            }
            else {
                return false;
            }
        }
        return b.empty();
    }

    //21. 合并两个有序链表
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* ret = new ListNode(0);
        ListNode* temp = ret;
        while (l1 && l2) {
            if (l1->val > l2->val) {
                swap(l1, l2);
            }
            temp->next = l1;
            l1 = l1->next;
            temp = temp->next;
        }
        temp->next = l1?l1:l2;
        temp = ret;
        ret = ret->next;
        delete temp;
        return ret;
    }

    //23. 合并K个排序链表
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode* ret = lists[0];
        for (int i = 1; i < lists.size(); ++i) {
            ret = mergeTwoLists(ret, lists[i]);
        }
        return ret;
    }

    // TODO: 24. 两两交换链表中的节点
    // ListNode* swapPairs(ListNode* head) {
        
    // }

    //26. 删除排序数组中的重复项
    int removeDuplicates(vector<int>& nums) {
        if (nums.empty()) return 0;
        int i = 0;
        for (int j = 1; j < nums.size(); ++j) {
            if (nums[j] != nums[i]) {
                swap(nums[++i], nums[j]);
            }
        }
        return i+1;
    }

    //28. 实现 strStr()
    int strStr(string haystack, string needle) {
        if (needle.empty()) return 0;
        for (int i = 0; i <= static_cast<int>(haystack.size()) - static_cast<int>(needle.size()); ++i) {
            int j = 0;
            while (j < needle.size()) {
                if (haystack[i] == needle[j]) {
                    ++i; ++j;
                }
                else {
                    break;
                }
            }
            if (j == needle.size()) {
                return i - j;
            }
            else {
                i -= j;
                continue;
            }
        }
        return -1;
    }

    //33. 搜索旋转排序数组
    int maxIndex(vector<int>& nums) {
        if (nums[0] <= nums.back()) return static_cast<int>(nums.size()) - 1;
        int l = 0, r = static_cast<int>(nums.size()) - 1;
        int k = (l + r) / 2;
        for (; nums[k] <= nums[k + 1]; k = (l + r) / 2) {
            if (nums[k] > nums[l]) l = k;
            else if (nums[k] < nums[r]) r = k;
        }
        return k;
    }
    int binarySearch(vector<int>& nums, int target, int l, int r) {
        for (auto k = (l + r) / 2; l <= r; k = (l + r) / 2) {
            if (target == nums[k]) {
                return k;
            }
            else if (target < nums[k]) {
                r = k - 1;
            }
            else {
                l = k + 1;
            }
        }
        return -1;
    }
    int search(vector<int>& nums, int target) {
        if (nums.empty())return -1;
        else if (nums.size() == 1) return (target == nums[0] ? 0 : -1);
        int maxi = maxIndex(nums);
        if (target < nums[0]) {
            return binarySearch(nums, target, maxi + 1, static_cast<int>(nums.size()) - 1);
        }
        else {
            return binarySearch(nums, target, 0, maxi);
        }
    }

    // 34. 在排序数组中查找元素的第一个和最后一个位置
    int extremeInsertionIndex(vector<int>& nums, int target, bool left){
        int lo = 0, hi = nums.size(), mid;
        while(lo < hi){
            mid = (lo + hi) / 2;
            if(nums[mid] > target || (left && nums[mid] == target)){
                hi = mid;
            }else{
                lo = mid + 1;
            }
        }
        return lo;
    }
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> res(2, -1);
        auto leftIdx = extremeInsertionIndex(nums, target, true);
        if(leftIdx != nums.size() && nums[leftIdx] == target){
            res[0] = leftIdx;
            res[1] = extremeInsertionIndex(nums, target, false) - 1;
        }
        return res;
    }

    // 39. 组合总和
    void combinationSum_dfs(vector<vector<int>>& ans, vector<int>& sequence, vector<int>& candidates, int pos, int rest){
        if(rest == 0){
            ans.push_back(sequence);
        }else if(pos < candidates.size() && rest >= candidates[pos]){
            combinationSum_dfs(ans, sequence, candidates, pos+1, rest);
            auto acc = candidates[pos];
            while(acc <= rest){
                sequence.push_back(candidates[pos]);
                combinationSum_dfs(ans, sequence, candidates, pos+1, rest-acc);
                acc+=candidates[pos];
            }
            while(acc > candidates[pos]){
                sequence.pop_back(); acc -= candidates[pos];
            }
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> ans;
        vector<int> sequence;
        sort(candidates.begin(), candidates.end());
        combinationSum_dfs(ans, sequence, candidates, 0, target);
        return ans;
    }

    // 40. 组合总和 II
    void combinationSum2_dfs(vector<pair<int, int>>& freq, vector<vector<int>>& ans, vector<int>& sequence, int pos, int rest){
        if (rest == 0) {
            ans.push_back(sequence);
            return;
        }
        if (pos == freq.size() || rest < freq[pos].first) {
            return;
        }
        combinationSum2_dfs(freq, ans, sequence, pos + 1, rest);
        int most = min(rest / freq[pos].first, freq[pos].second);
        for (int i = 1; i <= most; ++i) {
            sequence.push_back(freq[pos].first);
            combinationSum2_dfs(freq, ans, sequence, pos + 1, rest - i * freq[pos].first);
        }
        for (int i = 1; i <= most; ++i) {
            sequence.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<pair<int, int>> freq;
        {
            map<int, int> m;
            for(auto& i : candidates){
                ++m[i];
            }
            freq.assign(m.begin(), m.end());
        }
        vector<vector<int>> ans;
        vector<int> sequence;
        combinationSum2_dfs(freq, ans, sequence, 0, target);
        return ans;
    }
    
    //43. 字符串相乘
    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") return "0";
        string result(num1.size() + num2.size(), 0);
        for (int i = static_cast<int>(num1.size()) - 1; i >= 0; --i) {
            for (int j = static_cast<int>(num2.size()) - 1; j >= 0; --j) {
                result[i + j + 1] += (num1[i] - '0') * (num2[j] - '0');
                result[i + j] += result[i + j + 1] / 10;
                result[i + j + 1] %= 10;
            }
        }
        if (!result[0]) result.erase(0, 1);
        for_each(result.begin(), result.end(), [](char& c) {c += '0'; });
        return result;
    }

    //46. 全排列
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ret;
        do {
            ret.emplace_back(nums);
        } while (next_permutation(nums.begin(), nums.end()));
        return ret;
    }

    //53. 最大子序和
    int maxSubArray(vector<int>& nums) {
        if (nums.empty()) return INT_MIN;
        int dp = nums[0];
        int sum = dp;
        for (int i = 1; i < nums.size(); ++i) {
            dp = max(0, dp) + nums[i];
            sum = max(sum, dp);
        }
        return sum;
    }

    //54. 螺旋矩阵
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.empty())return {};
        vector<int> ans;
        int u = 0, d = static_cast<int>(matrix.size()) - 1, l = 0, r = static_cast<int>(matrix[0].size()) - 1;
        while (true) {
            for (int i = l; i <= r; ++i) ans.emplace_back(matrix[u][i]);
            if (++u > d)break;
            for (int i = u; i <= d; ++i) ans.emplace_back(matrix[i][r]);
            if (--r < l)break;
            for (int i = r; i >= l; --i)ans.emplace_back(matrix[d][i]);
            if (--d < u)break;
            for (int i = d; i >= u; --i)ans.emplace_back(matrix[i][l]);
            if (++l > r)break;
        }
        return ans;
    }

    //56. 合并区间
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.empty()) return {};
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> ret{ intervals[0] };
        int i = 1, j = 0;
        for (; i < intervals.size(); ++i) {
            if (ret[j][1] >= intervals[i][0]) {
                ret[j][1] = max(intervals[i][1], ret[j][1]);
            }
            else {
                ret.push_back(intervals[i]);
                ++j;
            }
        }
        return ret;
    }

    //59. 螺旋矩阵 II
    vector<vector<int>> generateMatrix(int n) {
        if (n == 0) return{};
        vector<vector<int>> ans(n, vector<int>(n, -1));
        int u = 0, d = n - 1, l = 0, r = n - 1;
        int put = 0;
        while (true) {
            for (int i = l; i <= r; ++i) ans[u][i] = ++put;
            if (++u > d)break;
            for (int i = u; i <= d; ++i) ans[i][r] = ++put;
            if (--r < l)break;
            for (int i = r; i >= l; --i) ans[d][i] = ++put;
            if (--d < u)break;
            for (int i = d; i >= u; --i) ans[i][l] = ++put;
            if (++l > r)break;
        }
        return ans;
    }

    //61. 旋转链表
    int lengthOfList(ListNode* target) {
        int ret = 0;
        while (target != nullptr) {
            target = target->next;
            ++ret;
        }
        return ret;
    }
    ListNode* rotateRight(ListNode* head, int k) {
        int len = lengthOfList(head);
        if (len < 2) return head;
        k = k % len;
        if (k == 0)return head;
        ListNode* ret = head;
        while (--len > k) {
            ret = ret->next;
        }
        ListNode* tail = ret;
        ret = ret->next;
        tail->next = nullptr;
        tail = ret;
        while (tail->next) tail = tail->next;
        tail->next = head;
        return ret;
    }

    // 62. 不同路径
    int uniquePaths(int m, int n) {
        if (m > n) return uniquePaths(n, m);
        double result = 1;
        for (int i = 1; i < m; ++i) {
            result = result * (i + n - 1) / i * 1.0;
        }
        return static_cast<int>(result);
    }

    // 70. 爬楼梯
    int climbStairs(int n) {
        static vector<int> t = {1, 2};
        for(auto i = t.size(); i < n; ++i){
            t.emplace_back(t[i-2] + t[i-1]);
        }
        return t[n-1];
    }

    // 77. 组合
    void combine_dfs(vector<vector<int>>& ans, vector<int>& seq, int n, int k, int i){
        if(k == 0){
            ans.push_back(seq);
        }else if(i <= n){
            seq.push_back(i);
            combine_dfs(ans, seq, n, k-1, i+1);
            seq.pop_back();
            combine_dfs(ans, seq, n, k, i+1);
        }
    }
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> ans;
        vector<int> seq;
        combine_dfs(ans, seq, n, k, 1);
        return ans;
    }

    // 79. 单词搜索
    bool exist_aux(vector<vector<char>>& board, vector<vector<char>>& searched, int i, int j, string& word, int w){
        if(board[i][j] != word[w] || searched[i][j]) return false;
        if(w == word.length()-1) return true;
        searched[i][j] = 1;
        if(i > 0){
            if(exist_aux(board, searched, i-1, j, word, w+1)) return true;
        }
        if(j > 0){
            if(exist_aux(board, searched, i, j-1, word, w+1)) return true;
        }
        if(i < board.size()-1){
            if(exist_aux(board, searched, i+1, j, word, w+1)) return true;
        }
        if(j < board[i].size()-1){
            if(exist_aux(board, searched, i, j+1, word, w+1)) return true;
        }
        searched[i][j] = 0;
        return false;
    }
    bool exist(vector<vector<char>>& board, string word) {
        vector<vector<char>> searched(board.size(), vector<char>(board[0].size(), 0));
        for(int i = 0; i < board.size(); ++i){
            for(int j = 0; j < board[i].size(); ++j){
                if(exist_aux(board, searched, i, j, word, 0)){
                    return true;
                }
            }
        }
        return false;
    }

    // 92. 反转链表 II
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        if(m<=1){
            ListNode* cur = head;
            ListNode* mid = nullptr;
            for(int i = 1; (i <= n) && cur; ++i){
                ListNode* tmp = cur;
                cur = cur->next;
                tmp->next = mid;
                mid = tmp;
            }
            head->next = cur;
            return mid;
        }else{
            ListNode* prev = head;
            ListNode* cur = head->next;
            for(int i = 2; i < m; ++i){
                prev = cur;
                cur = cur->next;
            }
            ListNode* mid = nullptr;
            ListNode* mid_tail = cur;
            for(int i = m; (i <= n) && cur; ++i){
                ListNode* tmp = cur;
                cur = cur->next;
                tmp->next = mid;
                mid = tmp;
            }
            mid_tail->next = cur;
            prev->next = mid;
            return head;
        }
    }

    // 94. 二叉树的中序遍历
    void inorderTraversal_aux(TreeNode* root, vector<int>& ans){
        if(root){
            inorderTraversal_aux(root->left, ans);
            ans.push_back(root->val);
            inorderTraversal_aux(root->right, ans);
        }
    }
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> ans;
        inorderTraversal_aux(root, ans);
        return ans;
    }

    // 96. 不同的二叉搜索树
    int numTrees(int n) {
        vector<int> t = {1,1};
        for(int i=2; i<=n;++i){
            int sum = 0, j=0;
            for(;j<i/2;++j){
                sum+=t[j]*t[i-j-1]*2;
            }
            if(2*j<i) sum+=t[j]*t[j];
            t.push_back(sum);
        }
        return t[n];
    }
    
    // 101. 对称二叉树(循环)
    bool isSymmetric(TreeNode* root) {
        if(root==nullptr) return true;
        TreeNode* L = root->left;
        TreeNode* R = root->right;
        if(L==nullptr && R==nullptr){
            return true;
        }else if(!(L!=nullptr && R!=nullptr && L->val == R->val)){
            return false;
        }
        vector<TreeNode*> lst;
        vector<TreeNode*> rst;
        lst.push_back(L);
        rst.push_back(R);
        while(!lst.empty() && !rst.empty()){
            L = lst.back();lst.pop_back();
            R = rst.back();rst.pop_back();
            if(L->val != R->val) return false;
            if(L->left && R->right){
                lst.push_back(L->left);rst.push_back(R->right);
            }else if(!(L->left==R->right && L->left==nullptr)) return false;
            if(L->right && R->left){
                lst.push_back(L->right);rst.push_back(R->left);
            }else if(!(L->right==R->left && L->right==nullptr)) return false;
        }
        return (lst.empty() && rst.empty());
    }

    // 104. 二叉树的最大深度
    int maxDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        else {
            return max(maxDepth(root->left), maxDepth(root->right)) + 1;
        }
    }
    
    // 110. 平衡二叉树
    int isBalanced_aux(TreeNode* root){
        if(root){
            int l = isBalanced_aux(root->left);
            int r = isBalanced_aux(root->right);
            if(l==-1 || r==-1 || l-r<-1 || l-r>1){
                return -1;
            }else{
                return max(l, r)+1;
            }
        }else{
            return 0;
        }
    }
    bool isBalanced(TreeNode* root) {
        return (isBalanced_aux(root)>=0);
    }

    // 141. 环形链表
    bool hasCycle(ListNode *head) {
        ListNode* fh = head;
        while(head && fh){
            head = head->next;
            fh = fh->next;
            if(fh){
                fh = fh->next;
            }else{
                break;
            }
            if(head == fh) return true;
        }
        return false;
    }

    // 148. 排序链表
    ListNode* sortList(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return head;
        ListNode* l = nullptr;
        ListNode* lt = nullptr;
        ListNode* cur = head->next;
        ListNode* prev = head;
        while(cur){
            if(cur->val < head->val){
                if(l == nullptr){
                    l = cur; lt = cur;
                }else{
                    lt->next = cur; lt = cur;
                }
                cur = cur->next;
                prev->next = cur;
            }else{
                prev = cur;
                cur = cur->next;
            }
        }
        if(lt) lt->next = nullptr;
        head->next = sortList(head->next);
        ListNode* ans = sortList(l);
        lt = ans;
        while(lt && lt->next){
            lt = lt->next;
        }
        if(ans){
            lt->next = head;
            head = ans;
        }
        return head;
    }

    // 151. 翻转字符串里的单词
    string reverseWords(string s) {
        stringstream ss(s);
        stack<string> words;
        string tmp;
        while(ss>>tmp){
            words.push(tmp);
        }
        if(words.empty()) return "";
        string ans = words.top();
        words.pop();
        while(!words.empty()){
            ans.push_back(' ');
            ans.append(words.top());
            words.pop();
        }
        return ans;
        // string ret;
        // int tail = s.size() - 1;
        // for (; tail >= 0; --tail) {
        //     if (s[tail] != ' ') {
        //         int head = tail;
        //         for (; head >= 0 && s[head] != ' '; --head);
        //         for (int cur = head + 1; cur <= tail; ++cur) {
        //             ret += s[cur];
        //         }
        //         tail = head;
        //         ret += ' ';
        //     }
        // }
        // if (!ret.empty()) {
        //     ret.erase(ret.size() - 1, 1);
        // }
        // return ret;
    }

    //198. 打家劫舍
    int rob(vector<int>& nums) {
        if(nums.empty()){
            return 0;
        }else if(nums.size()==1){
            return nums[0];
        }
        vector<int> t = {nums[0], max(nums[0], nums[1])};
        for(int i =2;i<nums.size();++i){
            t.push_back(max(t[i-2]+nums[i], t[i-1]));
        }
        return t.back();
    }
    
    // 201. 数字范围按位与
    int rangeBitwiseAnd(int m, int n) {
        int cnt = 0;
        while(m != n){
            m=m>>1;
            n=n>>1;
            ++cnt;
        }
        return m<<cnt;
    }

    // 206. 反转链表(循环)
    ListNode* reverseList(ListNode* head) {
        ListNode* p = nullptr;
        ListNode* q = nullptr;
        while(head){
            q = head->next;
            head->next = p;
            p = head;
            head = q;
        }
        return p;
    }
    
    // 216. 组合总和 III
    void combinationSum3_dfs(vector<vector<int>>& ans, vector<int>& seq, int k, int rest, int i){
        if(k == 0 && rest == 0){
            ans.push_back(seq);
        }else if(i <= 9 && i <= rest){
            combinationSum3_dfs(ans, seq, k, rest, i+1);
            seq.push_back(i);
            combinationSum3_dfs(ans, seq, k-1, rest-i, i+1);
            seq.pop_back();
        }
    }
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> ans;
        vector<int> seq;
        combinationSum3_dfs(ans, seq, k, n, 1);
        return ans;
    }

    // 226. 翻转二叉树(递归)
    TreeNode* invertTree(TreeNode* root) {
        if(root){
            TreeNode* temp = invertTree(root->right);
            root->right = invertTree(root->left);
            root->left = temp;
        }
        return root;
    }

    // 235. 二叉搜索树的最近公共祖先
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(p->val > q->val) swap(p,q);
        if(root == nullptr) return root;
        if(root->val < p->val){
            return lowestCommonAncestor(root->right, p, q);
        }else if(root->val > q->val){
            return lowestCommonAncestor(root->left, p, q);
        }else{
            return root;
        }
    }
    
    // 264. 丑数 II
    int nthUglyNumber(int n) {
        vector<int> dp(n);
        dp[0] = 1;
        int a = 0, b=0, c=0;
        int n2,n3,n5;
        for(int i = 1; i < n; ++i){
            n2=dp[a]*2, n3=dp[b]*3, n5=dp[c]*5;
            dp[i] = min(min(n2,n3), n5);
            if(dp[i] == n2) ++a;
            if(dp[i] == n3) ++b;
            if(dp[i] == n5) ++c;
        }
        return dp.back();
    }

    //283. 移动零
    void moveZeroes(vector<int>& nums) {
        int i = -1;
        for (int j = 0; j < nums.size(); ++j) {
            if (nums[j] != 0) {
                swap(nums[++i], nums[j]);
            }
        }
    }

    //287. 寻找重复数
    int countBetween(const int& l, const int& r, const vector<int>& a) const {
        int ret = 0;
        for (int i = 0; i < a.size(); ++i) {
            if (l <= a[i] && a[i] <= r) {
                ++ret;
            }
        }
        return ret;
    }
    int findDuplicate(vector<int>& nums) {
        int l = 1, r = static_cast<int>(nums.size()) - 1, m = r / 2;
        while (l < m) {
            (countBetween(l, m, nums) > (m - l + 1)) ? (r = m) : (l = m);
            m = (l + r) / 2;
        }
        return ((countBetween(l, l, nums) > 1) ? l : r);
    }

    // 323. 无向图中连通分量的数目
    int countComponents_find(vector<int>& p, int x){
        while(p[x] != -1) x = p[x];
        return x;
    }
    int countComponents(int n, vector<vector<int>>& edges) {
        vector<int> parents(n, -1);
        for(auto& e : edges){
            auto root1 = countComponents_find(parents, e[0]);
            auto root2 = countComponents_find(parents, e[1]);
            if(root1 != root2){
                parents[root1] = root2; --n;
            } 
        }
        return n;
    }

    // 343. 整数拆分
    int integerBreak(int n) {
        if(n<=3) return n/3+1;
        int res = 1;
        while(n>4){
            n-=3;
            res*=3;
        }
        res*=n;
        return res;
    }

    // 347. 前 K 个高频元素
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> m;
        m.rehash(nums.size()*1.5);
        for(auto& i : nums){
            ++m[i];
        }
        vector<pair<int,int>> f(m.begin(), m.end());
        int l = 0, r = f.size()-1;
        while(true){
            int pivot_i = r;
            int j = l-1;
            for(int i = l; i < r; ++i){
                if(f[i].second > f[pivot_i].second){
                    ++j; swap(f[j], f[i]);
                }
            }
            swap(f[++j], f[r]);
            if(j == k-1){
                break;
            }else if(j < k-1){
                l = j+1;
            }else{
                r =j-1;
            }
        }
        vector<int> ans(k);
        for(int i = 0; i < k;++i){
            ans[i] = f[i].first;
        }
        return ans;
    }

    //365. 水壶问题
    bool canMeasureWater(int x, int y, int z) {
        if(x<0 || y<0 || z<0) return false;
        if(z==0){
            return true;
        }else if(x > y){
            return canMeasureWater(y, x, z);
        }else if(x == 0){
            return (y==z);
        }else{
            return (z%gcd(x, y)==0 && z<=x+y);
        }
    }

    // 377. 组合总和 IV
    int combinationSum4(vector<int>& nums, int target) {
        if(nums.empty()) return 0;
        sort(nums.begin(), nums.end());
        vector<int> dp(target + 1, 0);
        dp[0] = 1;
        for (int i = 0; i <= target; ++i) {
            // auto tail = upper_bound(nums.begin(), nums.end(), i);
            // for_each(nums.begin(), tail, [&](auto& k){dp[i] += dp[i-k];});
            for(auto& k : nums){
                if(k <= i){
                    if(dp[i] >= INT_MAX - dp[i - k]){
                        dp[i] = INT_MAX;break;
                    }else{
                        dp[i] = dp[i] + dp[i - k];
                    }
                }
            }
        }
        return dp[target];
    }
    
    // 394. 字符串解码
    int decodeString_aux(string& s, int& k){
        string res;
        while(s[k] >= '0' && s[k] <= '9'){
            res += s[k];
            ++k;
        }
        if(res.length()){
            return stoi(res);
        }else{
            return -1;
        }
    }
    string decodeString(string s) {
        vector<int> n;
        string str;
        int k = 0;
        while(k<s.length()){
            if(s[k] >= '0' && s[k] <= '9'){
                int n_tmp = decodeString_aux(s, k);
                if(n_tmp > 0){
                    n.push_back(n_tmp);
                }
            }else if((s[k] >= 'a' && s[k] <= 'z') || s[k] == '['){
                str += s[k];
            }else if(s[k] == ']'){
                auto h = str.find_last_of('[');
                string str_tmp(str.begin() + h + 1, str.end());
                str.erase(str.begin() + h, str.end());
                while(n.back()){
                    str.append(str_tmp);
                    --n.back();
                }
                n.pop_back();
            }
            ++k;
        }
        return str;
    }

    // 410. 分割数组的最大值
    int splitArray(vector<int>& nums, int m) {
        int n = nums.size();
        vector<vector<long>> dp(n+1, vector<long>(m+1, numeric_limits<long>::max()));
        vector<long> sub(n+1, 0);
        for(int i = 0; i < n; ++i){
            sub[i+1] = sub[i] + nums[i];
        }
        dp[0][0] = 0;
        for(int i = 1; i <= n; ++i){
            for(int j = 1; j <= min(i, m); ++j){
                for(int k = 0; k < i; ++k){
                    dp[i][j] = min(dp[i][j], max(dp[k][j-1], sub[i] - sub[k]));
                }
            }
        }
        return static_cast<int>(dp[n][m]);
    }

    //415. 字符串相加
    string addStrings(const string& num1, const string& num2) {
        string result = "";
        char sum = 0;
        char carry = 0;
        int i = static_cast<int>(num1.size()), j = static_cast<int>(num2.size());
        while (i > 0 || j > 0) {
            sum = 0;
            if (i > 0) sum += num1[--i] - '0';
            if (j > 0) sum += num2[--j] - '0';
            sum += carry;
            carry = sum / 10;
            sum %= 10;
            result += (sum + '0');
        }
        if (carry)result += (carry + '0');
        return string(result.rbegin(), result.rend());
    }

    // 424. 替换后的最长重复字符
    int characterReplacement(string s, int k) {
        if(s.size() == 0) return 0;
        int l = 0;
        int r = 0;
        int ans = 0;
        size_t maxCount = 0;
        unordered_map<char, size_t> freq;
        while(r < s.size()){
            ++freq[s[r]];
            maxCount = max(maxCount, freq[s[r]]);
            ++r;
            if(r-l-maxCount > k){
                --freq[s[l]];
                ++l;
            }
            ans = max(ans, r-l);
        }
        return ans;
    }

    //445. 两数相加 II(正向链表)
    ListNode* reverseAdd(ListNode* l, ListNode* r) {
        ListNode* reverse_sum=nullptr;
        ListNode* temp=nullptr;
        int head = lengthOfList(l) - lengthOfList(r);
        if (head < 0) {
            head = -head;
            swap(l, r);
        }
        for (int i = 0; i < head; ++i) {
            temp = reverse_sum;
            reverse_sum = new ListNode(l->val);
            reverse_sum->next = temp;
            l = l->next;
        }
        while (l != nullptr) {
            temp = reverse_sum;
            reverse_sum = new ListNode(l->val + r->val);
            reverse_sum->next = temp;
            l = l->next; r = r->next;
        }
        return reverse_sum;
    }
    ListNode* handleCarry(ListNode* l) {
        ListNode* ret = nullptr;
        ListNode* temp = nullptr;
        while (l != nullptr) {
            if (l->val >= 10) {
                if (l->next == nullptr) {
                    l->next = new ListNode(0);
                }
                l->next->val += l->val / 10;
                l->val %= 10;
            }
            temp = l;
            l = l->next;
            temp->next = ret;
            ret = temp;
        }
        return ret;
    }
    ListNode* addTwoNumbers_forwardList(ListNode* l1, ListNode* l2) {
        return handleCarry(reverseAdd(l1, l2));
    }

    // 466. 统计重复个数
    int getMaxRepetitions(string s1, int n1, string s2, int n2) {
        if (n1 == 0) {
            return 0;
        }
        int s1cnt = 0, index = 0, s2cnt = 0;
        // recall 是我们用来找循环节的变量，它是一个哈希映射
        // 我们如何找循环节？假设我们遍历了 s1cnt 个 s1，此时匹配到了第 s2cnt 个 s2 中的第 index 个字符
        // 如果我们之前遍历了 s1cnt' 个 s1 时，匹配到的是第 s2cnt' 个 s2 中同样的第 index 个字符，那么就有循环节了
        // 我们用 (s1cnt', s2cnt', index) 和 (s1cnt, s2cnt, index) 表示两次包含相同 index 的匹配结果
        // 那么哈希映射中的键就是 index，值就是 (s1cnt', s2cnt') 这个二元组
        // 循环节就是；
        //    - 前 s1cnt' 个 s1 包含了 s2cnt' 个 s2
        //    - 以后的每 (s1cnt - s1cnt') 个 s1 包含了 (s2cnt - s2cnt') 个 s2
        // 那么还会剩下 (n1 - s1cnt') % (s1cnt - s1cnt') 个 s1, 我们对这些与 s2 进行暴力匹配
        // 注意 s2 要从第 index 个字符开始匹配
        unordered_map<int, pair<int, int>> recall;
        pair<int, int> pre_loop, in_loop;
        while (true) {
            // 我们多遍历一个 s1，看看能不能找到循环节
            ++s1cnt;
            for (char ch: s1) {
                if (ch == s2[index]) {
                    index += 1;
                    if (index == s2.size()) {
                        ++s2cnt;
                        index = 0;
                    }
                }
            }
            // 还没有找到循环节，所有的 s1 就用完了
            if (s1cnt == n1) {
                return s2cnt / n2;
            }
            // 出现了之前的 index，表示找到了循环节
            if (recall.count(index)) {
                auto [s1cnt_prime, s2cnt_prime] = recall[index];
                // 前 s1cnt' 个 s1 包含了 s2cnt' 个 s2
                pre_loop = {s1cnt_prime, s2cnt_prime};
                // 以后的每 (s1cnt - s1cnt') 个 s1 包含了 (s2cnt - s2cnt') 个 s2
                in_loop = {s1cnt - s1cnt_prime, s2cnt - s2cnt_prime};
                break;
            }
            else {
                recall[index] = {s1cnt, s2cnt};
            }
        }
        // ans 存储的是 S1 包含的 s2 的数量，考虑的之前的 pre_loop 和 in_loop
        int ans = pre_loop.second + (n1 - pre_loop.first) / in_loop.first * in_loop.second;
        // S1 的末尾还剩下一些 s1，我们暴力进行匹配
        int rest = (n1 - pre_loop.first) % in_loop.first;
        for (int i = 0; i < rest; ++i) {
            for (char ch: s1) {
                if (ch == s2[index]) {
                    ++index;
                    if (index == s2.size()) {
                        ++ans;
                        index = 0;
                    }
                }
            }
        }
        // S1 包含 ans 个 s2，那么就包含 ans / n2 个 S2
        return ans / n2;
    }
    
    // 469. 凸多边形
    int signof(int i){
        if(i>0) return 1;
        else if (i<0) return -1;
        else return 0;
    }
    bool isConvex(vector<vector<int>>& points) {
        auto n = points.size();
        int res = 0;
        for(auto i = 0; i < n; ++i){
            int x1 = points[(i+1)%n][0] - points[i][0];
            int x2 = points[(i+2)%n][0] - points[i][0];
            int y1 = points[(i+1)%n][1] - points[i][1];
            int y2 = points[(i+2)%n][1] - points[i][1];
            int pro = x1*y2 - x2*y1;
            if(pro != 0){
                if(res * pro < 0) return false;
                else res = signof(pro);
            }
        }
        return true;
    }

    //475. 供暖器
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        sort(houses.begin(), houses.end());
        sort(heaters.begin(), heaters.end());
        auto l = heaters.cbegin();
        int ret = 0;
        for(auto i = houses.cbegin();i!=houses.cend();++i){
            int distance = INT_MAX;
            for(auto j = l;j!=heaters.cend();++j){
                if(*i>=*j){
                    l = j;
                }
                distance = min(distance, abs(*i-*j));
                if(*i<*j) break;
            }
            ret = max(ret, distance);
        }
        return ret;
    }
    
    //542. 01 矩阵
    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int n = static_cast<int>(matrix.size());
        int m = static_cast<int>(matrix[0].size());
        vector<vector<int>> ret(n, vector<int>(m, INT_MAX / 2));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (matrix[i][j] == 0) ret[i][j] = 0;
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i - 1 >= 0) ret[i][j] = min(ret[i][j], ret[i - 1][j] + 1);
                if (j - 1 >= 0) ret[i][j] = min(ret[i][j], ret[i][j - 1] + 1);
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = m - 1; j >= 0; --j) {
                if (i + 1 < n) ret[i][j] = min(ret[i][j], ret[i + 1][j] + 1);
                if (j + 1 < m) ret[i][j] = min(ret[i][j], ret[i][j + 1] + 1);
            }
        }
        return ret;
    }

    // 637. 二叉树的层平均值
    vector<double> averageOfLevels(TreeNode* root) {
        vector<double> ans;
        double sum = 0;
        deque<vector<TreeNode*>> dp(2);
        if(root) dp[0].push_back(root);
        while(!dp[0].empty()){
            sum = 0;
            for(auto item : dp[0]){
                if(item->left) dp[1].push_back(item->left);
                if(item->right) dp[1].push_back(item->right);
                sum += item->val;
            }
            ans.push_back(sum/dp[0].size());
            dp.pop_front();
            dp.resize(2);
        }
        return ans;
    }

    // 718. 最长重复子数组
    int findLength(vector<int>& A, vector<int>& B) {
        vector<vector<int>> dp(A.size(), vector<int>(B.size(), 0));
        int ans = 0;
        for(int j = 0; j < B.size(); ++j){
            if(A[0] == B[j]) dp[0][j] = 1;
        }
        for(int i = 1; i < A.size(); ++i){
            dp[i][0] = (A[i] == B[0]);
            for(int j = 1; j < B.size(); ++j){
                if(A[i] == B[j]){
                    dp[i][j] = dp[i-1][j-1] + 1;
                    ans = max(ans, dp[i][j]);
                }
            }
        }
        return ans;
    }
    
    // 740. 删除与获得点数
    int deleteAndEarn(vector<int>& nums) {
        if (nums.empty()) return 0;
        map<int, int> t;
        for (auto& i : nums) {
            ++t[i];
        }
        vector<int> p(t.rbegin()->first + 1, 0);
        for (auto& it : t) {
            p[it.first] = it.first * it.second;
        }
        int m = 0, n = 0, k = 0;
        for (int i = 0; i < p.size(); ++i) {
            k = max(m + p[i], n);
            m = n; n = k;
        }
        return n;
    }

    //817. 链表组件
    int numComponents(ListNode* head, vector<int>& G) {
        unordered_set<int> S(G.begin(), G.end());
        int result = 0;
        while(head != NULL){
            if(S.find(head->val) != S.end()
              &&(head->next == NULL || 
                 S.find(head->next->val) == S.end())){
                ++result;
            }
            head = head->next;
        }
        return result;
    }

    // 822. 翻转卡片游戏
    int flipgame(vector<int>& fronts, vector<int>& backs) {
        int sz = fronts.size();
        set<int> map;
        for(int i = 0; i < sz; ++i){
            if(fronts[i] != backs[i]){
                map.insert(fronts[i]);
                map.insert(backs[i]);
            }
        }
        for(auto& item : map){
            int i = 0;
            for(; i < sz && (fronts[i] != item || backs[i] != item); ++i);
            if(i >= sz) return item;
        }
        return 0;
    }

    // 837. 新21点
    double new21Game(int N, int K, int W) {
        if(K == 0) return 1.0;
        vector<double> dp(K+W, 0);
        for(int i = K; i <= N && i < dp.size(); ++i){
            dp[i] = 1.0;
        }
        dp[K-1] = 1.0 * min(N - K + 1, W) / W;
        for (int i = K - 2; i >= 0; --i) {
            dp[i] = dp[i + 1] - (dp[i + W + 1] - dp[i + 1]) / W;
        }
        return dp[0];
    }

    // 869. 重新排序得到 2 的幂
    bool reorderedPowerOf2(int N) {
        if(N <= 0) return false;
        string s = to_string(N);
        sort(s.begin(), s.end());
        int factor = 1;
        for(int i = 0; i < 32; ++i){
            string s_factor = to_string(factor);
            if(s.length() == s_factor.length()){
                sort(s_factor.begin(), s_factor.end());
                if(s == s_factor){
                    return true;
                }
            }else if(s.length() < s_factor.length()){
                break;
            }
            factor <<= 1;
        }
        return false;
    }

    // 887. 鸡蛋掉落
    int superEggDrop(int K, int N) {
        vector<vector<int>> tb;
        vector<int> line(K, 1);
        int t = 1;
        while(t < N && line.back() < N){
            tb.push_back(line);
            line[0] = ++t;
            for(int i = 1; i < K; ++i){
                line[i] = tb.back()[i] + tb.back()[i-1] + 1;
            }
        }
        return t;
    }

    //945. 使数组唯一的最小增量
    int minIncrementForUnique(vector<int>& A) {
        int ret = 0;
        constexpr int blen = 80000;
        int B[blen] = {0};
        int alen=static_cast<int>(A.size());
        for(int i = 0; i<alen; ++i){
            ++B[A[i]];
        }
        for(int j = 0; j<blen; ++j){
            if(B[j]>1){
                ret += B[j]-1;
                B[j+1] += B[j]-1; 
                B[j] = 1;            
            }
        }
        return ret;
    }
    
    // 984. 不含 AAA 或 BBB 的字符串
    string strWithout3a3b(int A, int B) {
        string res;
        while(A > 0 && B > 0){
            if(A>B){
                res.append("aab");
                A-=2; --B;
            }else if(A == B){
                res.append("ab");
                --A; --B;
            }else{
                res.append("abb");
                --A; B-=2;
            }
        }
        if(A){
            string tmp(A, 'a');
            res.append(tmp);
            return res;
        }else{
            string tmp(B, 'b');
            tmp.append(res);
            return tmp;
        }
    }

    //1010. 总持续时间可被 60 整除的歌曲
    int numPairsDivisibleBy60(vector<int>& time) {
        int t[60] = { 0 };
        for_each(time.begin(), time.end(), [&t](int x) {++t[x % 60]; });
        int result = 0;
        if (t[0]) result += (t[0] * (t[0] - 1) / 2);
        if (t[30]) result += (t[30] * (t[30] - 1) / 2);
        for (int i = 1; i < 30; ++i) {
            result += t[i] * t[60 - i];
        }
        return result;
    }

    // 1024. 视频拼接
    int videoStitching(vector<vector<int>>& clips, int T) {
        sort(clips.begin(), clips.end());
        if(clips.front()[0] > 0) return -1;
        int ans = 0;
        int curIdx = 0;
        int tail = 0;
        while(tail < T){
            int maxtail = tail;
            bool found = false;
            for(;curIdx<clips.size() && clips[curIdx][0] <= tail; ++curIdx){
                maxtail = max(maxtail, clips[curIdx][1]); found = true;
            }
            if(found){
                ++ans; tail = maxtail;
            }else{
                return -1;
            }
        }
        return ans;
    }
    
    //1111. 有效括号的嵌套深度
    vector<int> maxDepthAfterSplit(string seq) {
        int counter = 0;
        vector<int> ret;
        for(auto ch : seq){
            if(ch == '('){
                ++counter;
                ret.push_back(counter%2);
            }else if(ch ==')'){
                ret.push_back(counter%2);
                --counter;
            }
        }
        return ret;
    }
    
    //1139. 最大的以 1 为边界的正方形
    int largest1BorderedSquare(vector<vector<int>>& grid) {
        struct CP{
            int up;
            int left;
            //CP():up(0),left(0){};
            CP(int u, int l):up(u),left(l){};
        };
        vector<vector<CP>> t={{}};
        t[0].emplace_back(grid[0][0], grid[0][0]);
        for(int j=1;j<grid[0].size();++j){
            t[0].emplace_back(grid[0][j], grid[0][j] + t[0][j-1].left);
        }
        for(int i=1;i<grid.size();++i){
            t.emplace_back();
            t[i].emplace_back(t[i-1][0].up + grid[i][0], grid[i][0]);
            for(int j=1;j<grid[i].size();++j){
                t[i].emplace_back(grid[i][j] + t[i-1][j].up, grid[i][j] + t[i][j-1].left);
            }
        }
        int maxres = 0;
        for(int i=0;i<t.size();++i){
            for(int j=0; j<t[i].size();++j){
                int res = maxres;
                if(grid[i][j]){
                    while(i+res<t.size() && j+res<t[i].size()){
                        if(t[i][j+res].left - t[i][j].left == res &&
                        t[i+res][j].up - t[i][j].up == res &&
                        t[i+res][j+res].left - t[i+res][j].left == res &&
                        t[i+res][j+res].up - t[i][j+res].up == res){
                            ++res;
                            maxres=max(maxres, res);
                        }else{++res;}
                    }
                }
            }
        }
        return maxres*maxres;
    }

    // 1246. 删除回文子数组
    int minimumMoves(vector<int>& arr) {
        vector<vector<int>> dp(arr.size(), vector<int>(arr.size()));
        for(int i = 0; i < arr.size(); ++i){
            dp[i][i] = 1;
        }
        for(int i = 0; i < dp.size()-1; ++i){
            if(arr[i] == arr[i + 1]){
                dp[i][i + 1] = 1;
            }else{
                dp[i][i + 1] = 2;
            }   
        }
        for(int len = 2; len < arr.size(); ++len){
            for(int i = 0; i < arr.size() - len; ++i){
                dp[i][i + len] = len + 1;
                for(int k = 0; k < len; ++k){
                    dp[i][i + len] = min(dp[i][i + len], dp[i][i + k] + dp[i + k + 1][i + len]);
                }
                if(arr[i] == arr[i + len]){
                    dp[i][i + len] = min(dp[i][i + len], dp[i + 1][i + len - 1]);
                }
            }
        }
        return dp[0].back();
    }

    // 1262. 可被三整除的最大和
    int maxSumDivThree(vector<int>& nums) {
        vector<vector<int>> dp(nums.size()+1, vector<int>(3, 0));
        dp[0][1] = numeric_limits<int>::min();
        dp[0][2] = numeric_limits<int>::min();
        for(int i = 0; i < nums.size(); ++i){
            switch(nums[i] % 3){
            case 0:
                dp[i+1][0] = max(dp[i][0], dp[i][0] + nums[i]);
                dp[i+1][1] = max(dp[i][1], dp[i][1] + nums[i]);
                dp[i+1][2] = max(dp[i][2], dp[i][2] + nums[i]); break;
            case 1:
                dp[i+1][0] = max(dp[i][0], dp[i][2] + nums[i]);
                dp[i+1][1] = max(dp[i][1], dp[i][0] + nums[i]);
                dp[i+1][2] = max(dp[i][2], dp[i][1] + nums[i]); break;
            case 2:
                dp[i+1][0] = max(dp[i][0], dp[i][1] + nums[i]);
                dp[i+1][1] = max(dp[i][1], dp[i][2] + nums[i]);
                dp[i+1][2] = max(dp[i][2], dp[i][0] + nums[i]); break;
            }
        }
        return dp.back()[0];
    }

    // 1420. 生成数组
    int numOfArrays(int n, int m, int k) {
        const int base = 1000000007;
        vector<vector<vector<int>>> dp(n+1, vector<vector<int>>(m+1, vector<int>(k+1)));
        // int dp[51][101][51] = {0};
        for(int i = 1; i <= n; ++i){
            for(int j = 1; j <= m; ++j){
                for(int l = 1; l <= k; ++l){
                    if(i == 1 && l == 1){
                        dp[i][j][l] = 1;
                    }else{
                        dp[i][j][l] = (long)j * dp[i-1][j][l] % base;
                        for(int t = 1; t < j; ++t){
                            dp[i][j][l] = (dp[i][j][l] + dp[i-1][t][l-1]) % base;
                        }
                    }
                }
            }
        }
        int ans = 0;
        for(int j = 1; j <= m; ++j){
            ans = (ans + dp[n][j][k]) % base;
        }
        return ans;
    }

};
}