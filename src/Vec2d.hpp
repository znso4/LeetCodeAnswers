#include"pch.h"
template<typename T>
class Vec2d{
    T x;
    T y;
public:
    Vec2d(T _x=0, T _y=0):x(_x), y(_y){}
    Vec2d(const Vec2d& v){ x = v.x; y = v.y; }
    Vec2d(Vec2d&& v){ x = std::move(v.x); y = std::move(v.y); }
    ~Vec2d(){}
    friend Vec2d&& operator+(const Vec2d&, const Vec2d&);
    friend Vec2d&& operator-(const Vec2d&);
    friend Vec2d&& operator-(const Vec2d&, const Vec2d&);
};
template<typename T>
Vec2d<T>&& operator+(const Vec2d<T>& l, const Vec2d<T>& r){
    return Vec2d(l.x + r.x, l.y + r.y);
}
template<typename T>
Vec2d<T>&& operator-(const Vec2d<T>& r){
    return Vec2d(-r.x, -r.y);
}
template<typename T>
Vec2d<T>&& operator-(const Vec2d<T>& l, const Vec2d<T>& r){
    return Vec2d(l.x - r.x, l.y - r.y);
}