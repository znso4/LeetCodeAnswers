#include "pch.h"
#include "algs.hpp"
#pragma once

using namespace std;

namespace LCOF{

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// 剑指Offer 30. 包含min函数的栈
class MinStack {
    vector<int> s = {};
    vector<int> m = {INT_MAX};
public:
    void push(int x) {
        s.emplace_back(x);
        if(x<=m.back()){
            m.emplace_back(x);
        }
    }
    void pop() {
        if(!s.empty()){
            if(s.back() == m.back()){
                m.pop_back();
            }
            s.pop_back();
        }
    }
    int top() {return s.back();}
    int min() {return m.back();}
};

class Solution {
public:
    // 剑指Offer  01.07. 旋转矩阵
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
    
    // 剑指Offer 03. 数组中重复的数字
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

    // 剑指 Offer 04. 二维数组中的查找
    bool baseFind2DArray(vector<vector<int>>& m, int nl, int nr, int ml, int mr, int target){
        if(nl>nr || ml>mr){
            return false;
        }else if(nl==nr && ml==mr){
            return target==m[nl][ml];
        }
        int nm=nl+(nr-nl)/2, mm=ml+(mr-ml)/2;
        if(target == m[nm][mm]){
            return true;
        }else{
            if(target < m[nm][mm]){
                return ( baseFind2DArray(m, nl, nm, ml, mm, target) ||
                baseFind2DArray(m, nm+1, nr, ml, mm, target) ||
                baseFind2DArray(m, nl, nm, ml+1, mr, target) );
            }else{
                return ( baseFind2DArray(m, nm+1, nr, mm+1, mr, target) ||
                baseFind2DArray(m, nm+1, nr, ml, mm, target) ||
                baseFind2DArray(m, nl, nm, ml+1, mr, target) );
            }

        }
    }
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if(!matrix.empty() && !matrix[0].empty()){
            return baseFind2DArray(matrix, 0, matrix.size()-1, 0, matrix[0].size()-1, target);
        }else{
            return false;
        }
    }

    // 剑指Offer 05. 替换空格
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

    // 剑指Offer 06. 从尾到头打印链表
    vector<int> reversePrint(ListNode* head) {
        vector<int> result;
        while(head!=nullptr){
            result.emplace_back(head->val);
            head=head->next;
        }
        reverse(result.begin(), result.end());
        return result;
    }

    // 剑指Offer 07. 重建二叉树
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
        int sz = static_cast<int>(preorder.size());
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

    // 剑指Offer 10- I. 斐波那契数列
    vector<int> FIB = {0, 1};
    int fib(int N) {
        for (int i = static_cast<int>(FIB.size()); i <= N; ++i) {
            FIB.push_back((FIB[i - 2] + FIB[i - 1])%1000000007);
        }
        return FIB[N];
    }

    // 剑指Offer 10- II. 青蛙跳台阶问题
    const int base = 1000000007;
    int numWays(int n) {
        return fib(n+1);
    }

    // 剑指Offer 11. 旋转数组的最小数字
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
        return minArray(numbers, 0, static_cast<int>(numbers.size()-1));
    }

    // 剑指 Offer 12. 矩阵中的路径
    bool dfs(vector<vector<char>>& board, string& word, int i, int j, int k){
        if(i<0||i>=board.size() || j<0||j>=board[0].size() || board[i][j] != word[k]) return false;
        if(++k == word.size()) return true;
        char tmp = board[i][j];
        board[i][j] = '$';
        bool res = (dfs(board, word, i, j+1, k) || dfs(board, word, i+1, j, k) || dfs(board, word, i, j-1, k) || dfs(board, word, i-1, j, k));
        board[i][j] = tmp;
        return res;
    }
    bool exist(vector<vector<char>>& board, string word) {
        for(size_t ii = 0; ii <= board.size(); ++ii){
            for (size_t jj = 0; jj < board[0].size(); jj++){
                if(dfs(board, word, ii, jj, 0)) return true;
            }
        }
        return false;
    }
    /*用递归lambda函数的方法，比较慢
    bool exist(vector<vector<char>>& board, string word) {
        auto n = board.size()-1;
        auto m = board[0].size()-1;
        auto l = word.size()-1;
        function<bool (int, int, int)> dfs = [&](int i, int j, int k)->bool{
            if(i<0||i>n || j<0||j>m || board[i][j] != word[k]) return false;
            if(k == l) return true;
            char tmp = board[i][j];
            board[i][j] = '$';
            ++k;
            bool res = (dfs(i, j+1, k) || dfs(i+1, j, k) || dfs(i, j-1, k) || dfs(i-1, j, k));
            board[i][j] = tmp;
            return res;
        };
        for(size_t ii = 0; ii <= board.size(); ++ii){
            for (size_t jj = 0; jj < board[0].size(); jj++){
                if(dfs(ii, jj, 0)) return true;
            }
        }
        return false;
    }*/

    // 剑指 Offer 13. 机器人的运动范围
    int digitalSum(int x){
        int res = 0;
        while(x){ res += x % 10; x /= 10;}
        return res;
    }
    int movingCount(int m, int n, int k) {
        if (!k) return 1;
        queue<pair<int,int>> q;
        int dx[2] = {1, 0};
        int dy[2] = {0, 1};
        vector<vector<char> > board(m, vector<char>(n, 0));
        q.push(make_pair(0, 0));
        board[0][0] = 1;
        int ans = 1;
        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();
            for (int i = 0; i < 2; ++i) {
                int tx = dx[i] + x;
                int ty = dy[i] + y;
                if (tx >= m || ty >= n || board[tx][ty] || digitalSum(tx) + digitalSum(ty) > k) continue;
                q.push(make_pair(tx, ty));
                board[tx][ty] = 1;
                ++ans;
            }
        }
        return ans;
    }

    // 剑指Offer 14- I. 剪绳子
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

    // 剑指Offer 15. 二进制中1的个数
    int hammingWeight(uint32_t n) {
        int res = 0;
        while(n){
            n = n & (n-1);
            ++res;
        }
        return res;
    }

    // 剑指Offer 16. 数值的整数次方
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

    // 剑指Offer 17. 打印从1到最大的n位数
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

    // 剑指Offer 18. 删除链表的节点
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

    // 剑指Offer 19. 正则表达式匹配
    bool isMatch(string& s, size_t i, string& p, size_t j){
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

    // 剑指Offer 20. 表示数值的字符串
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

    // 剑指Offer 21. 调整数组顺序使奇数位于偶数前面
    vector<int> exchange(vector<int>& nums) {
        int i = -1;
        for(int j=0; j<nums.size(); ++j){
            if(nums[j]%2){
                swap(nums[++i], nums[j]);
            }
        }
        return nums;
    }

    // 剑指Offer 22. 链表中倒数第k个节点
    int lengthOfList(ListNode* target) {
        int ret = 0;
        while (target != nullptr) {
            target = target->next;
            ++ret;
        }
        return ret;
    }
    ListNode* getKthFromEnd(ListNode* head, int k) {
        int l = lengthOfList(head);
        if(k>l){
            return nullptr;
        }else{
            while(l>k){
                head = head->next;
                --l;
            }
            return head;
        }
    }

    // 剑指Offer 24. 反转链表(递归)
    ListNode* reverseList(ListNode* head) {
        if(!head || !(head->next)){ return head;}
        else{
            ListNode* cur = reverseList(head->next);
            head->next->next = head;
            head->next = nullptr;
            return cur;
        }
    }
    
    // 剑指Offer 25. 合并两个排序的链表
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1==nullptr){
            return l2;
        }else if(l2==nullptr){
            return l1;
        }else{
            if(l1->val > l2->val){
                swap(l1, l2);
            }
            ListNode* cur = l1;
            l1=l1->next;
            cur->next = mergeTwoLists(l1, l2);
            return cur;
        }
    }

    // 剑指Offer 26. 树的子结构(递归)
    bool equalTrees(TreeNode* x, TreeNode* y){
        return ((x->val == y->val) &&
        (y->left==nullptr || (x->left && equalTrees(x->left, y->left))) &&
        (y->right==nullptr || (x->right && equalTrees(x->right, y->right))));
    }
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        return (A&&B) && (equalTrees(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right,B));
    }

    // 剑指Offer 27. 二叉树的镜像(循环)
    TreeNode* mirrorTree(TreeNode* root) {
        vector<TreeNode*> st;
        if(root) st.push_back(root);
        TreeNode* node = root;
        TreeNode* temp = nullptr;
        while(!st.empty()){
            root = st.back();
            st.pop_back();
            if(root->left) st.push_back(root->left);
            if(root->right) st.push_back(root->right);
            temp = root->left;
            root->left = root->right;
            root->right = temp;
        }
        return node;
    }

    // 剑指Offer 28. 对称的二叉树(递归)
    bool isSymmetric(TreeNode* L, TreeNode* R){
        if(L==nullptr && R==nullptr){
            return true;
        }else if(L!=nullptr && R!=nullptr){
            return (L->val==R->val) && isSymmetric(L->left, R->right) && isSymmetric(L->right, R->left);
        }else{
            return false;
        }
    }
    bool isSymmetric(TreeNode* root) {
        if(root){
            return isSymmetric(root->left, root->right);
        }else{
            return true;
        }
    }

    // 剑指Offer 29. 顺时针打印矩阵
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if(matrix.empty() || matrix[0].empty()) return {};
        vector<int> res;
        int ii = 0, ij = matrix.size()-1, ji = 0, jj = matrix[0].size()-1;
        while(true){
            if(ii<=ij){
                for(int u = ji; u<=jj; ++u){
                    res.emplace_back(matrix[ii][u]);
                }
                ++ii;
            }else{break;}
            if(ji<=jj){
                for(int u = ii; u<=ij; ++u){
                    res.emplace_back(matrix[u][jj]);
                }
                --jj;
            }else{break;}
            if(ii<=ij){
                for(int u = jj; u>=ji; --u){
                    res.emplace_back(matrix[ij][u]);
                }
                --ij;
            }else{break;}
            if(ji<=jj){
                for(int u = ij; u>=ii; --u){
                    res.emplace_back(matrix[u][ji]);
                }
                ++ji;
            }else{break;}
        }
        return res;
    }

    // 剑指 Offer 31. 栈的压入、弹出序列
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        size_t j = 0;
        vector<int> s;
        for(size_t i = 0; i<pushed.size(); ++i){
            s.emplace_back(pushed[i]);
            while(!s.empty() && j < popped.size() && s.back() == popped[j]){
                s.pop_back();++j;
            }
        }
        return (s.empty() && j == popped.size());
    }


};
}