#include "pch.h"
#include "LC.hpp"
#include "LCOF.hpp"
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
    return 0;
}