#pragma once
#include "pch.h"

using std::vector;
using std::size_t;
using std::cout;
using std::endl;

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
int fib(int N) {
    static vector<int> FIB = { 0,1 };
    for (int i = FIB.size(); i <= N; ++i) {
        FIB.push_back(FIB[i - 2] + FIB[i - 1]);
    }
    return FIB[N];
}

int gcd(int a, int b){
    int c = b%a;
    while(c!=0){
        b=a;
        a=c;
        c=b%a;
    }
    return a;
}

typedef unsigned long long uint64;
uint64 factor(size_t x, size_t base = 0){
    const static size_t BASE = base;
    static vector<uint64> FACTOR = {1};
    if(BASE){
        for(auto i=FACTOR.size();i<=x;++i){
            FACTOR.push_back(i*FACTOR.back()%BASE);
        }
    }else{
        for(auto i=FACTOR.size();i<=x;++i){
            FACTOR.push_back(i*FACTOR.back());
        }
    }
    return FACTOR[x];
}