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

    // TODO:4. 寻找两个正序数组的中位数
    /*
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int d1 = (nums1.size() - 1) / 2;
        int d2 = (nums2.size() - 1) / 2;

    }*/

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
        if (strs.size() == 0) return "";
        string tmp = strs[0];
        string ret = tmp;
        int size = 0;
        for (int i = 1; i < strs.size() && tmp != ""; ++i) {
            ret = "";
            size = static_cast<int>(min(tmp.size(), strs[i].size()));
            for (int j = 0; j < size; ++j) {
                if (tmp[j] == strs[i][j]) {
                    ret += tmp[j];
                }
                else {
                    break;
                }
            }
            tmp = ret;
        }
        return ret;
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

    // TODO: 151. 翻转字符串里的单词
    // string reverseWords(string s) {
    //     string ret;
    //     int tail = s.size() - 1;
    //     for (; tail >= 0; --tail) {
    //         if (s[tail] != ' ') {
    //             int head = tail;
    //             for (; head >= 0 && s[head] != ' '; --head);
    //             for (int cur = head + 1; cur <= tail; ++cur) {
    //                 ret += s[cur];
    //             }
    //             tail = head;
    //             ret += ' ';
    //         }
    //     }
    //     if (!ret.empty()) {
    //         ret.erase(ret.size() - 1, 1);
    //     }
    //     return ret;
    // }

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
    
    //283. 移动零
    void moveZeroes(vector<int>& nums) {
        int i = -1;
        for (int j = 0; j < nums.size(); ++j) {
            if (nums[j] != 0) {
                swap(nums[++i], nums[j]);
            }
        }
    }

    // 206. 反转链表(循环)
    ListNode* reverseList(ListNode* head) {
        ListNode* res = head;
        if(!head){
            return nullptr;
        }
        head=head->next;
        res->next = nullptr;
        ListNode* t = nullptr;
        while(head){
            t=head->next;
            head->next=res;
            res=head;
            head=t;
        }
        return res;
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

};
}