#include "pch.h"
#include "algs.hpp"
#pragma once

using namespace std;

namespace LCCI{

class Solution {
public:
    // 面试题 17.24. 最大子矩阵
    tuple<int, int, int> getMaxMatrix_aux(vector<int>& nums) {
        if (nums.empty()) return {-1, -1, INT_MIN};
        int dp = nums[0];
        int sum = dp;
        int ans_l = 0, l = 0;
        int ans_r = 0, r = 0;
        for (int i = 1; i < nums.size(); ++i) {
            if(dp > 0){
                dp += nums[i];
            }else{
                dp = nums[i]; l = i;
            }
            if(sum < dp){
                sum = dp; ans_l = l; ans_r = i;
            }
        }
        return {ans_l, ans_r, sum};
    }
    vector<int> getMaxMatrix(vector<vector<int>>& matrix) {
        int N = matrix.size(), M = matrix[0].size();
        vector<vector<int>> sumsInRow(N, vector<int>(M+1));
        for(int i = 0; i < N; ++i){
            int tmp = 0;
            for(int j = 1; j <= M; ++j){
                tmp += matrix[i][j-1];
                sumsInRow[i][j] = tmp;
            }
        }
        int maxsubsum = INT_MIN;
        vector<int> ans(4, -1);
        for(int lo = 0; lo < M; ++lo){
            for(int hi = lo+1; hi <= M; ++hi){
                vector<int> b(N);
                for(int i = 0; i < N; ++i){
                    b[i] = sumsInRow[i][hi] - sumsInRow[i][lo];
                }
                auto res = getMaxMatrix_aux(b);
                if(get<2>(res) > maxsubsum){
                    maxsubsum = get<2>(res);
                    ans = {get<0>(res), lo, get<1>(res), hi-1};
                }
            }
        }
        return ans;
    }
    
};

}