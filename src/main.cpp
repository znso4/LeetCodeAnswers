#include "pch.h"
#include "LC.hpp"
#include "LCOF.hpp"
#include<sstream>
using namespace std;


int main(){
    LCOF::Solution s;
    vector<tuple<string, string, bool>> strs = {
        {"aa", "a", false},
        {"aa", "a*", true},
        {"acdadsfea", "acd.*a", true},
        {"acdadsfea", "ac.*d.*a", true},
    };
    for(auto i : strs){
        cout<<(s.isMatch(get<0>(i), get<1>(i))==get<2>(i))<<endl;
    }
    string inputstr;
    vector<int> res;
    while(getline(cin, inputstr)){
        stringstream inputss;
        inputss<<inputstr;
        int val;
        while(inputss>>val){
            res.push_back(val);
        }
        print(res);
        res.clear();
    }
    return 0;
}