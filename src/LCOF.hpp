#pragma once
#include "pch.h"
#include "algs.hpp"

using namespace std;

namespace LCOF{

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    // 面试题 01.07. 旋转矩阵
    void rotate(vector<vector<int>>& matrix) {
        auto n = matrix.size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
        for (int j = 0; j < n / 2; ++j) {
            for (int i = 0; i < n; ++i) {
                swap(matrix[i][j], matrix[i][n - 1 - j]);
            }
        }
    }
    
    // 面试题03. 数组中重复的数字
    int findRepeatNumber(vector<int>& nums) {
        bitset<100000> t = {false};
        for(auto& k : nums){
            if(t[k]){
                return k;
            }else{
                t[k] = true;
            }
        }
        return -1;
    }

    // 面试题05. 替换空格
    string replaceSpace(string s) {
        string result;
        for(auto& c:s){
            if(c==' '){
                result+="%20";
            }else{
                result.push_back(c);
            }
        }
        return result;
    }

    // 面试题06. 从尾到头打印链表
    vector<int> reversePrint(ListNode* head) {
        vector<int> result;
        while(head!=nullptr){
            result.emplace_back(head->val);
            head=head->next;
        }
        reverse(result.begin(), result.end());
        return result;
    }

    // 面试题07. 重建二叉树
    TreeNode* buildTree(vector<int>& pre, int l1, int r1, 
    vector<int>& ino, int l2, int r2, unordered_map<int, int> m){
        if(l1>r1){
            return nullptr;
        }else if(l1 == r1){
            return new TreeNode(pre[l1]);
        }else{
            TreeNode* root = new TreeNode(pre[l1]);
            int i = m[pre[l1]];
            root->left = buildTree(pre, l1+1, l1+i-l2, ino, l2, i-1, m);
            root->right = buildTree(pre, l1+i-l2+1, r1, ino, i+1, r2, m);
            return root;
        }
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int sz = preorder.size();
        if(!(sz==inorder.size() && sz>0)){
            return nullptr;
        }else{
            unordered_map<int, int> m;
            for(int i = 0; i<sz;++i){
                m[inorder[i]]=i;
            }
            --sz;
            return buildTree(preorder, 0, sz, inorder, 0, sz, m);
        }
    }

    // 面试题10- I. 斐波那契数列
    vector<int> FIB = {0, 1};
    int fib(int N) {
		for (int i = FIB.size(); i <= N; ++i) {
			FIB.push_back((FIB[i - 2] + FIB[i - 1])%1000000007);
		}
		return FIB[N];
	}

    // 面试题10- II. 青蛙跳台阶问题
    const int base = 1000000007;
    int numWays(int n) {
        return fib(n+1);
    }

    // 面试题11. 旋转数组的最小数字
    int minArray(vector<int>& numbers, int l, int r){
        int mid = (l+r)/2;
        if(mid == l){
            return min(numbers[l], numbers[r]);
        }else if(numbers[l]<numbers[mid]){
            return minArray(numbers, mid, r);
        }else if(numbers[l]>numbers[mid]){
            return minArray(numbers, l, mid);
        }else{
            return min(minArray(numbers, l, mid), minArray(numbers, mid, r));
        }
    }
    int minArray(vector<int>& numbers) {
        if(numbers.empty()) return -1;
        if(numbers[0]<numbers.back()) return numbers[0];
        return minArray(numbers, 0, numbers.size()-1);
    }

    // 面试题14- I. 剪绳子
    int cuttingRope(int n) {
        if(n<=3) return n/3+1;
        int res = 1;
        while(n>4){
            n-=3;
            res*=3;
        }
        res*=n;
        return res;
    }

    // 面试题15. 二进制中1的个数
    int hammingWeight(uint32_t n) {
        int res = 0;
        while(n){
            n = n & (n-1);
            ++res;
        }
        return res;
    }

    // 面试题16. 数值的整数次方
    double myPow(double x, long n) {
        if(n<0){
            x=1.0/x;
            n=-n;
        }
        double res = 1;
        while(n){
            if(n&1) res*=x;
            x*=x;
            n>>=1;
        }
        return res;
    }

    // 面试题17. 打印从1到最大的n位数
    vector<int> printNumbers(int n) {
        int ten = 1;
        while(n){
            ten*=10;
            --n;
        }
        vector<int> res;
        while(++n<ten){
            res.emplace_back(n);
        }
        return res;
    }

    // 面试题18. 删除链表的节点
    ListNode* deleteNode(ListNode* head, int val) {
        if(head && head->val == val){
            auto tmp = head;
            head=head->next;
            tmp->next=nullptr;
            return head;
        }else if(!head){
            return head;
        }
        auto pre = head;
        auto cur = head->next;
        while(cur && cur->val != val){
            pre = cur;
            cur = cur->next;
        }
        if(cur){
            pre->next = cur->next;
            cur->next = nullptr;
        }
        return head;
    }

    // 面试题19. 正则表达式匹配
    bool isMatch(string& s, int i, string& p, int j){
        if(i==0 && j%2==0){
            for(int k = 1;k<j;k+=2){
                if(p[k]!='*') return false;
            }
            return true;
        }else if( (i==0 && j%2!=0) || j==0){
            return false;
        }else{
            switch(p[j-1]){
            case '.': return isMatch(s, i-1, p, j-1);
            case '*': 
                if(p[j-2]=='.' || p[j-2]==s[i-1]) return ( isMatch(s, i-1, p, j) || isMatch(s, i, p, j-2) );
                else return isMatch(s, i, p, j-2);
            default:
                if(p[j-1]==s[i-1]) return isMatch(s, i-1, p, j-1);
                else return false;
            }
        }
    }
    bool isMatch(string s, string p) {
        return isMatch(s, s.size(), p, p.size());
    }

    // 面试题20. 表示数值的字符串
    bool isNumber(string s) {
        class DFA{
            vector<vector<int>> transfer_mat;
            int status;
        public:
            DFA(vector<vector<int>> t):transfer_mat(t), status(0){}
            int get_status(){return status;}
            int update(int k){
                status = transfer_mat[status][k];
                return status;
            }
        };
        const unordered_map<char, int> d = {
            {' ', 0},
            {'+', 1},{'-', 1},
            {'0', 2},{'1', 2},{'2', 2},{'3', 2},{'4', 2},{'5', 2},{'6', 2},{'7', 2},{'8', 2},{'9', 2},
            {'.', 3},{'e', 4},
        };
        const vector<vector<int>> mat = {
            {0,1,2,9,-1},
            {-1,-1,2,9,-1},
            {8,-1,2,3,5},
            {8,-1,4,-1,5},
            {8,-1,4,-1,5},
            {-1,6,7,-1,-1},
            {-1,-1,7,-1,-1},
            {8,-1,7,-1,-1},
            {8,-1,-1,-1,-1,},
            {-1,-1,4,-1,-1},
        };
        const unordered_set<int> allowed_status = {2,3,4,7,8};
        DFA dfa(mat);
        for(auto& c : s){
            auto f = d.find(c);
            if(f==d.end() || dfa.update(f->second)<0){
                return false;
            }
        }
        return allowed_status.count(dfa.get_status());
    }

    // 面试题21. 调整数组顺序使奇数位于偶数前面
    vector<int> exchange(vector<int>& nums) {
        int i = -1;
        for(int j=0; j<nums.size(); ++j){
            if(nums[j]%2==1){
                swap(nums[++i], nums[j]);
            }
        }
        return nums;
    }
};
}