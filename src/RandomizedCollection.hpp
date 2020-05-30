#pragma once
#include "pch.h"
using std::cout;
using std::endl;

template<typename ValueType = int, typename IndexType = size_t>
class RandomizedCollection {
    std::vector<ValueType> values;
    std::unordered_map<ValueType, std::unordered_set<IndexType>> indices;
    std::mt19937 rIndex;
public:
    /** Initialize your data structure here. */
    RandomizedCollection() {}

    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    bool insert(ValueType val) {
        bool ret = (indices.find(val) == indices.end());
        indices[val].insert(values.size());
        values.push_back(val);
        return ret;
    }

    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    bool remove(ValueType val) {
        if (indices.find(val) == indices.end()) {
            return false;
        }
        else {
            ValueType tail = values.back();
            if (val != tail) {
                ValueType i = *indices[val].begin();
                values[i] = tail;
                indices[tail].erase(values.size() - 1);
                indices[tail].insert(i);
                indices[val].erase(i);
            }
            else {
                indices[val].erase(values.size() - 1);
            }
            values.pop_back();
            if (indices[val].empty()) indices.erase(val);
            return true;
        }
    }

    /** Get a random element from the collection. */
    IndexType getRandom() {
        return values[rIndex() % values.size()];
    }
    static void test() {
        // 初始化一个空的集合。
        RandomizedCollection<> collection;

        // 向集合中插入 1 。返回 true 表示集合不包含 1 。
        cout << collection.insert(1) << endl;

        // 向集合中插入另一个 1 。返回 false 表示集合包含 1 。集合现在包含 [1,1] 。
        cout << collection.insert(1) << endl;

        // 向集合中插入 2 ，返回 true 。集合现在包含 [1,1,2] 。
        cout << collection.insert(2) << endl;

        // getRandom 应当有 2/3 的概率返回 1 ，1/3 的概率返回 2 。
        int result = 0;
        for (int i = 0; i < 1000; ++i) {
            if (collection.getRandom() == 1) ++result;
        }
        cout << "P(1) == " << result * 1.0 / 1000 << endl;

        // 从集合中删除 1 ，返回 true 。集合现在包含 [1,2] 。
        cout << collection.remove(1) << endl;

        // getRandom 应有相同概率返回 1 和 2 。
        result = 0;
        for (int i = 0; i < 1000; ++i) {
            if (collection.getRandom() == 1) ++result;
        }
        cout << "P(1) == " << result * 1.0 / 1000 << endl;

        cout << collection.remove(100) << endl;
    }
};
