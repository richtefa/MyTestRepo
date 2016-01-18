#include <iostream>

using namespace std;

struct MyStruct
{
public:
    void func() const {
        //foo_ = 5;
    }

    void func2() {
        foo_ = 5;
    }

    int foo_;
};

//void test(MyStruct const& m)
void test(MyStruct const* const m)
//void test(const MyStruct* m)
{
//    m.func();
    m->func();
//    m->foo_ = 6;
}

int main() {
    cout << "Hello, World!" << endl;

    MyStruct m;
    test(&m);

    return 0;
}