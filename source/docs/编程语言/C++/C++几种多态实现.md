# C++几种多态实现
&emsp;&emsp;C++多态是C++语言的基础之一，多态包括类多态和函数多态等，这里只谈论类多态实现。C++原生支持了虚函数，但是虚函数的实现需要额外的开销，所以C++也可以通过其他方式实现静态多态。

## 1. 虚函数
&emsp;&emsp;最基础的多态就是通过虚函数实现，虚函数可以在运行时进行绑定来实现运行时多态。C++的虚函数是通过虚函数表实现的，每个类都有一个虚函数表，虚函数表中存放着该类的虚函数地址。当一个类的对象调用虚函数时，会根据对象的虚指针找到虚函数表，然后在虚函数表中找到虚函数的地址，最后调用虚函数。
```cpp
#include <iostream>

class Animal {
public:
    virtual void makeSound() {
        std::cout << "Generic animal sound" << std::endl;
    }
    virtual ~Animal() = default; // 虚析构函数
};

class Dog : public Animal {
public:
    void makeSound() override {
        std::cout << "Woof!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void makeSound() override {
        std::cout << "Meow!" << std::endl;
    }
};

int main() {
    Animal* animal1 = new Dog();
    Animal* animal2 = new Cat();
    animal1->makeSound(); // 输出 "Woof!"
    animal2->makeSound(); // 输出 "Meow!"
    delete animal1;
    delete animal2;
    return 0;
}
```
## 2. std::variant
&emsp;&emsp;```variant```实现多态本身是利用variant的union能力。
```cpp
#include <iostream>
#include <variant>

struct Dog {
    void makeSound() {
        std::cout << "Woof!" << std::endl;
    }
};

struct Cat {
    void makeSound() {
        std::cout << "Meow!" << std::endl;
    }
};

int main() {
    std::variant<Dog, Cat> animal;
    animal = Dog{}; // 存储 Dog 对象
    std::visit([](auto&& arg) {
        arg.makeSound();
    }, animal); // 输出 "Woof!"

    animal = Cat{}; // 存储 Cat 对象
    std::visit([](auto&& arg) {
        arg.makeSound();
    }, animal); // 输出 "Meow!"
    return 0;
}
```

## 3. CRTP
&emsp;&emsp; CRTP 是一种特殊的模板编程技巧，也称为奇异递归模板模式。它通过将派生类作为基类的模板参数来实现静态多态。基类模板可以访问派生类的成员，从而实现不同的行为。
```cpp
#include <iostream>

template <typename Derived>
class Animal {
public:
    void makeSound() {
        static_cast<Derived*>(this)->makeSoundImpl(); // 调用派生类的 makeSoundImpl
    }
};

class Dog : public Animal<Dog> {
private:
    void makeSoundImpl() {
        std::cout << "Woof!" << std::endl;
    }
};

class Cat : public Animal<Cat> {
private:
    void makeSoundImpl() {
        std::cout << "Meow!" << std::endl;
    }
};

int main() {
    Dog dog;
    Cat cat;
    dog.makeSound(); // 输出 "Woof!"
    cat.makeSound(); // 输出 "Meow!"
    return 0;
}
```

## 4 总结
&emsp;&emsp;选择哪种多态实现方式取决于具体的需求。如果需要在运行时根据对象的实际类型来选择执行不同的操作，并且可以接受一定的运行时开销，那么虚函数是合适的选择。如果需要在编译时实现多态，并且对性能要求较高，那么 CRTP 是更好的选择。如果需要在多个不同类型之间进行选择，并且这些类型在编译时已知，那么 std::variant 是一个不错的选择。

| 特性           | 虚函数          | std::variant   | CRTP             |
| -------------- | --------------- | --------------- | ---------------- |
| 多态类型       | 动态多态        | 静态多态        | 静态多态         |
| 实现方式       | 继承 + 虚函数表 | 类型安全的联合体 | 模板 + 静态转换 |
| 运行时开销     | 有              | 无              | 无               |
| 代码可读性     | 高              | 中              | 低               |
| 灵活性         | 高              | 中              | 低               |
| 类型安全性     | 较低            | 高              | 中               |
| 适用场景       | 运行时多态      | 编译时多态      | 编译时多态，高性能 |
