#include "pch.h"
#pragma once

using namespace std;

template<typename T>
void print(T t) {
    for (auto k : t) {
        cout << k << " ";
    }
    cout << endl;
}

template<typename T>
int binarySearch(vector<T>& nums, T target) {
    int l = 0, r = nums.size() - 1;
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

//509. 斐波那契数
size_t fib(size_t N) {
    static vector<size_t> FIB = { 0,1 };
    for (size_t i = FIB.size(); i <= N; ++i) {
        FIB.push_back(FIB[i - 2] + FIB[i - 1]);
    }
    return FIB[N];
}

// 最大公约数
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
gcd(T a, T b){
    T c = b%a;
    while(c!=0){
        b=a;
        a=c;
        c=b%a;
    }
    return a;
}

// 阶乘
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
factor(T x){
    static vector<T> FACTOR = {1};
    for(auto i=FACTOR.size();i<=x;++i){
        FACTOR.push_back(i*FACTOR.back());
    }
    return FACTOR[x];
}

template<typename T, T BASE>
typename enable_if<is_integral<T>::value && (BASE!=0), T>::type
factor_base(T x){
    static vector<T> FACTOR = {1};
    for(auto i=FACTOR.size();i<=x;++i){
        FACTOR.push_back(i*FACTOR.back()%BASE);
    }
    return FACTOR[x];
}

// 排列数
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
nPr(T n, T r){
    return factor(n)/factor(n-r);
}

// 组合数
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
nCr(T n, T r){
    return nPr(n,r)/nPr(r,r);
}