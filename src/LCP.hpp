#include "pch.h"
#include "algs.hpp"
#pragma once

using namespace std;

namespace LCP{

class Solution {
public:
    // LCP 17. 速算机器人
    int calculate(string s) {
        int x = 1, y = 0;
        for(auto c:s){
            if(c=='A'){
                x=2*x+y;
            }else if(c=='B'){
                y=2*y+x;
            }
        }
        return x+y;
    }

    // LCP 18. 早餐组合
    int breakfastNumber(vector<int>& staple, vector<int>& drinks, int x) {
        sort(staple.begin(), staple.end());
        sort(drinks.begin(), drinks.end());
        long long ans = 0;
        int base = 1000000007;
        auto s_end = upper_bound(staple.begin(), staple.end(), x);
        for(auto item = staple.begin(); item != s_end; ++item){
            ans += upper_bound(drinks.begin(), drinks.end(), x-*item) - drinks.begin();
            ans %= base;
        }
        return static_cast<int>(ans);
    }

    // LCP 19. 秋叶收藏集
    int minimumOperations(string leaves) {
        vector<vector<int>> dp(3, vector<int>(leaves.size(), 0));
        for (int i = 0; i < leaves.size(); i++) {
            if (i < 1) {
                dp[0][i] = (leaves[i] != 'r');
            } else {
                dp[0][i] = dp[0][i - 1] + (leaves[i] != 'r');
            }
            if (i < 1) {
                dp[1][i] = dp[0][i];
            } else {
                dp[1][i] = min(dp[0][i - 1] + (leaves[i] != 'y'), dp[1][i - 1] + (leaves[i] != 'y'));
            }
            if (i < 2) {
                dp[2][i] = dp[1][i];
            } else {
                dp[2][i] = min(dp[1][i - 1] + (leaves[i] != 'r'), dp[2][i - 1] + (leaves[i] != 'r'));
            }
        }
        return dp[2].back();
    }

    // LCP 22. 黑白方格画
    int paintingPlan(int n, int k) {
        int ans = 0;
        if(k == n*n) return 1;
        for(int i = 0; i <= n; ++i){
            for(int j = 0; j <= n; ++j){
                if(n * (i+j) - i*j == k){
                    ans += nCr(n, i) * nCr(n, j);
                }
            }
        }
        return ans;
    }
};

}
