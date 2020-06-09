#include "pch.h"
#pragma once
using std::string;
using std::regex;
using std::vector;
using std::smatch;
template<typename Str = string>
vector<Str> split(Str s, const regex& target) {
    vector<Str> ret;
    smatch match;
    while (regex_search(s, match, target)) {
        ret.push_back(match.prefix());
        s = match.suffix();
    }
    ret.push_back(s);
    return ret;
}
template<typename Str = string>
vector<Str> search(Str s, const regex& target) {
    vector<Str> ret;
    smatch match;
    while (regex_search(s, match, target)) {
        ret.push_back(match.str());
        s = match.suffix();
    }
    return ret;
}