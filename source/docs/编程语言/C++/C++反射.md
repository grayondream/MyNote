# C++反射

## 1 反射
### 1.1 什么是反射
&emsp;&emsp;反射（Reflection）是一种程序在运行时能够获取自身信息（如类型、成员、方法等）并动态操作这些信息的能力。在 C++ 中，标准并未原生支持反射机制，这与 Java、C# 等语言不同。但通过一些技术手段，我们可以在 C++ 中模拟实现反射功能，实现动态类型识别、对象创建、成员访问等操作。

&emsp;&emsp;用python的反射举例，下面的例子是查询类的属性和方法：
```python
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, my name is {self.name}."

# 创建对象
person = Person("Alice")

# 使用反射获取属性和方法
name = getattr(person, 'name')
greeting = getattr(person, 'greet')()

print(name)       # 输出: Alice
print(greeting)   # 输出: Hello, my name is Alice.

# 动态设置属性
setattr(person, 'age', 30)
print(person.age)  # 输出: 30
```

&emsp;&emsp;下面的例子是通过字符串创建对象：
```cpp
class Dog:
    def bark(self):
        return "Woof!"

# 动态获取类名并创建实例
class_name = 'Dog'
dog_class = globals()[class_name]
dog_instance = dog_class()
print(dog_instance.bark())  # 输出: Woof!
```

### 1.2 为什么需要反射
&emsp;&emsp;反射在需要动态性、灵活性和扩展性的场景中不可或缺，比如：
1. 动态类型处理与通用编程：在编写通用组件（如框架、库）时，开发者往往无法预知用户会定义哪些类型。反射允许程序在运行时动态识别类型信息，而不必在编译时硬编码类型相关逻辑。
2. 解耦与灵活性：反射能打破编译时的类型依赖，让代码更灵活，降低模块间的耦合度。
3. 简化重复性工作：许多场景需要对类的成员进行统一操作（如赋值、验证、日志输出），反射可以自动化这些过程，避免手动编写重复代码。
4. 支持动态行为：在需要动态修改程序行为的场景中，反射允许在运行时动态调用函数、修改属性，而无需提前确定具体操作。
5. 跨语言交互与序列化：在多语言协作场景中，反射是数据交换的桥梁。例如，C++ 程序与 Java 程序通信时，通过反射将 C++ 对象转换为通用格式（如 JSON/Protocol Buffers），接收方再通过反射还原为自身语言的对象。

### 1.3 反射分类
&emsp;&emsp;在 C++ 中，反射主要分为动态反射和静态反射两种：
1. 动态反射：在运行时动态获取类型信息、调用函数、访问成员等。这需要使用 C++ 的 RTTI（Run-Time Type Information）机制，如 `typeid`、`dynamic_cast` 等。
2. 静态反射：在编译时确定类型信息、调用函数、访问成员等。这需要使用模板元编程、代码生成等技术，无法在运行时修改对象的行为。

| 特性       | 动态反射                     | 静态反射                     |
|------------|------------------------------|------------------------------|
| 时间       | 运行时                       | 编译时                       |
| 灵活性     | 高，支持动态类型和行为      | 低，类型在编译时确定        |
| 性能       | 较低，存在运行时开销        | 较高，编译时优化            |
| 类型检查   | 运行时检查，可能导致错误    | 编译时检查，错误可早期发现  |
| 使用场景   | 插件系统、动态对象创建等    | 类型安全、性能敏感的应用    |

## 2 动态反射
&emsp;&emsp;C++中经常会出现静态和动态相关的术语，比如静态类型、动态类型、静态绑定、动态绑定等，无一例外动态是指运行时，静态是指编译时。在反射的场景下依然如此。那么对应的动态反射就是在运行时获取函数的元数据，包含成员，函数等等，那么为了获取对应的属性，我们需要利用相关结构保存详细的属性信息。假如我们要实现一个简单的动态反射demo，我们能够考虑到类的属性有：
- 成员变量（FiledInfo）：
  - 名称；
  - 类型；
  - 相对偏移；
- 函数描述（MethodInfo）：
  - 名称；
  - 返回值类型；
  - 参数列表；
- 类型（TypeInfo）：
  - 名称；
  - 成员列表；
  - 函数列表。

&emsp;&emsp;基于以上的推断，我们能够很快写出三个相关的简单类：
```cpp
class FieldInfo {
public:
    FieldInfo()
        : _name(""), _type(typeid(void)) {} 

    FieldInfo(const std::string& name, std::type_index type)
        : _name(name), _type(type) {}

    std::string getName() const { return _name; }
    std::type_index getType() const { return _type; }

private:
    std::string _name;         // 字段名称
    std::type_index _type;     // 字段类型
};

class MethodInfo {
public:
    MethodInfo()
        : _name(""), _returnType(typeid(void)), _paramTypes() {} 

    MethodInfo(const std::string& name, std::type_index returnType, const std::vector<std::type_index>& paramTypes)
        : _name(name), _returnType(returnType), _paramTypes(paramTypes) {}

    std::string getName() const { return _name; }
    std::type_index getReturnType() const { return _returnType; }
    const std::vector<std::type_index>& getParamTypes() const { return _paramTypes; }

private:
    std::string _name;                       // 方法名称
    std::type_index _returnType;             // 返回类型
    std::vector<std::type_index> _paramTypes; // 参数类型列表
};

// 类型信息
class TypeInfo {
public:
    TypeInfo()
        : _name(""), _creator(nullptr) {} // 默认构造函数

    TypeInfo(const std::string& name)
        : _name(name), _creator(nullptr) {}

    void addField(const FieldInfo& field) {
        _fields[field.getName()] = field;
    }

    void addMethod(const MethodInfo& method) {
        _methods[method.getName()] = method;
    }

    void setCreator(std::function<void* (std::vector<void*>)> creator) {
        _creator = creator;
    }

    void* createInstance(std::vector<void*> args) const {
        return _creator ? _creator(args) : nullptr;
    }

    std::string getName() const { return _name; }

    bool hasField(const std::string& fieldName) const {
        return _fields.find(fieldName) != _fields.end();
    }

    bool hasMethod(const std::string& methodName) const {
        return _methods.find(methodName) != _methods.end();
    }

private:
    std::string _name;
    std::unordered_map<std::string, FieldInfo> _fields; // 字段信息
    std::unordered_map<std::string, MethodInfo> _methods; // 方法信息
    std::function<void* (std::vector<void*>)> _creator; // 带参数的创建函数
};
```

&emsp;&emsp;需要注意的是这里展示的属性的基本描述是不全的，很多时候类型还需要考虑继承，属性控制等，而且MethodInfo没有实现函数调用的接口（实际实现起来比较简单）。具体需要什么属性和方法要根据你具体的工程而来。
&emsp;&emsp;有了这些存储类型的基本信息，我们只需要还需要一个全局的注册器注册对应的类型来进行统一管理。
```cpp
class TypeRegistry {
public:
    static TypeRegistry& instance() {
        static TypeRegistry registry;
        return registry;
    }

    void registerType(const std::string& name, const TypeInfo& typeInfo) {
        _types[name] = typeInfo;
    }

    void registerTypeWithCreator(const std::string& name, const TypeInfo& typeInfo, std::function<void* (std::vector<void*>)> creator) {
        _types[name] = typeInfo;
        _types[name].setCreator(creator);
    }

    const TypeInfo* getType(const std::string& name) const {
        auto it = _types.find(name);
        return (it != _types.end()) ? &it->second : nullptr;
    }

private:
    std::unordered_map<std::string, TypeInfo> _types; // 注册的类型信息
};
```

&emsp;&emsp;有了这些信息，我们就能够在运行时根据类型名称获取到对应的类型信息，从而进行相关的操作。比如下面的```ExampleClass```创建时会自动注册当前类的类型信息，当然如果嫌弃使用```registerType```不方便，也可以通过宏来实现，避免重复代码。
```cpp
class ExampleClass {
public:
    ExampleClass(int value) : _value(value) {
        registerType();
    }

    void setValue(int value) {
        _value = value;
    }

    int getValue() const {
        return _value;
    }

    static void* createInstance(std::vector<void*> args) {
        if (args.empty()) return nullptr; // 参数检查
        int value = *static_cast<int*>(args[0]); // 取出参数
        return new ExampleClass(value); // 创建实例
    }

private:
    void registerType() {
        TypeInfo typeInfo("ExampleClass");
        typeInfo.addField(FieldInfo("value", typeid(int)));
        typeInfo.addMethod(MethodInfo("setValue", typeid(void), { typeid(int) }));
        typeInfo.addMethod(MethodInfo("getValue", typeid(int), {}));

        // 注册带参数的创建函数
        typeInfo.setCreator(createInstance); 
        TypeRegistry::instance().registerType("ExampleClass", typeInfo);
    }

    int _value;
};
```

&emsp;&emsp;对应的测试代码如下：
```cpp
TEST(TypeRegistryTest, TypeRegistration) {
    ExampleClass example(42);

    const TypeInfo* typeInfo = TypeRegistry::instance().getType("ExampleClass");
    ASSERT_NE(typeInfo, nullptr);
    
    EXPECT_EQ(typeInfo->getName(), "ExampleClass");
    EXPECT_TRUE(typeInfo->hasField("value"));
    EXPECT_TRUE(typeInfo->hasMethod("setValue"));
    EXPECT_TRUE(typeInfo->hasMethod("getValue"));

    // 测试对象创建
    int value = 42;
    std::vector<void*> args = { &value }; // 创建参数列表
    void* instance = typeInfo->createInstance(args);
    ExampleClass* createdInstance = static_cast<ExampleClass*>(instance);
    EXPECT_EQ(createdInstance->getValue(), 42);
    delete createdInstance; // 清理动态分配的内存
}

TEST(FieldInfoTest, FieldInfoCreation) {
    FieldInfo field("value", typeid(int));
    
    EXPECT_EQ(field.getName(), "value");
    EXPECT_EQ(field.getType(), typeid(int));
}

TEST(MethodInfoTest, MethodInfoCreation) {
    std::vector<std::type_index> paramTypes = { typeid(int) };
    MethodInfo method("setValue", typeid(void), paramTypes);

    EXPECT_EQ(method.getName(), "setValue");
    EXPECT_EQ(method.getReturnType(), typeid(void));
    EXPECT_EQ(method.getParamTypes().size(), 1);
    EXPECT_EQ(method.getParamTypes()[0], typeid(int));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

## 3 静态反射
&emsp;&emsp;静态反射将所有的推断都放在编译期间，比如这里实现一个简单的成员变量的注册模板。
```cpp
template <typename FieldType, const char* Name>
struct Field {
    using Type = FieldType;

    constexpr static const char* name() {
        return Name; // 返回模板参数名称
    }
};

template <typename T, typename... Fields>
struct Reflector {
    using Type = T;
    static constexpr std::array<const char*, sizeof...(Fields)> fieldNames = { Fields::name()... };
};
```

&emsp;&emsp;使用时通过```Reflector```来注册成员变量，比如这里的```Example```类。如果觉得这样还需要预定义成员string麻烦，可以通过宏来替换实现。
```cpp
struct Example {
    int id;
    std::string name; 
};

constexpr const char idName[] = "id";
constexpr const char nameName[] = "name";

// 特化反射信息
template <>
struct Reflector<Example, Field<int, idName>, Field<std::string, nameName>> {
    using Type = Example;
    static constexpr std::array<const char*, 2> fieldNames = { Field<int, idName>::name(), Field<std::string, nameName>::name() };
};
```

&emsp;&emsp;对应的测试代码：
```cpp
TEST(ReflectionTest, ExampleFieldNames) {
    constexpr auto& fieldNames = Reflector<Example, Field<int, idName>, Field<std::string, nameName>>::fieldNames;

    EXPECT_EQ(std::string(fieldNames[0]) == "id", true);
    EXPECT_EQ(std::string(fieldNames[1]) == "name", true);
}

TEST(ReflectionTest, ExampleFieldCount) {
    constexpr auto fieldCount = Reflector<Example, Field<int, idName>, Field<std::string, nameName>>::fieldNames.size();
    EXPECT_EQ(fieldCount, 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

## 4 CodeGen实现反射
&emsp;&emsp;可以结合 Python 的脚本处理能力和 LLVM 的 C++ 代码解析能力。这种方案的核心思路是：使用 LLVM 的 libclang 库（C++ 的 AST 解析器）来分析 C++ 代码结构，提取类型信息，然后生成反射所需的元数据和绑定代码。
```python
import clang.cindex
import os
import sys

def setup_clang_library():
    """根据操作系统自动设置libclang库路径"""
    if sys.platform.startswith('win32'):
        possible_paths = [
            # 检查环境变量中是否有LLVM路径
            os.path.join(os.environ.get('LLVM_HOME', ''), 'bin', 'libclang.dll')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                clang.cindex.Config.set_library_file(path)
                return True
        
        # 如果找不到，提示用户手动设置
        print("找不到libclang.dll，请确保已安装LLVM并设置正确路径")
        print("可从 https://releases.llvm.org/download.html 下载适合的版本")
        return False
    elif sys.platform.startswith('linux'):
        # Linux系统设置
        clang.cindex.Config.set_library_file('/usr/lib/llvm-14/lib/libclang.so')
        return True
    elif sys.platform.startswith('darwin'):
        # macOS系统设置
        clang.cindex.Config.set_library_file('/usr/local/opt/llvm/lib/libclang.dylib')
        return True
    return False

# 初始化Clang索引
if not setup_clang_library():
    # 初始化失败处理
    sys.exit(1)
index = clang.cindex.Index.create()

class CppReflectionGenerator:
    def __init__(self, cpp_files, include_dirs=None):
        self.cpp_files = cpp_files if isinstance(cpp_files, list) else [cpp_files]
        self.include_dirs = include_dirs or []
        self.classes = []
        self.reflection_code = ""
        
        # 准备Clang编译参数
        self.args = []
        for dir in self.include_dirs:
            self.args.append(f'-I{dir}')
        self.args.extend(['-std=c++17', '-Wall'])

    def parse_code(self):
        """使用Clang解析C++代码并提取类信息"""
        for file in self.cpp_files:
            if not os.path.exists(file):
                print(f"文件不存在: {file}")
                continue
                
            # 解析文件
            translation_unit = index.parse(file, args=self.args)
            
            # 遍历AST提取类信息
            for node in translation_unit.cursor.walk_preorder():
                if node.kind == clang.cindex.CursorKind.CLASS_DECL or \
                   node.kind == clang.cindex.CursorKind.STRUCT_DECL:
                    self._parse_class(node)
    
    def _parse_class(self, class_node):
        """解析类节点，提取成员变量和成员函数"""
        class_info = {
            "name": class_node.spelling,
            "display_name": class_node.displayname,
            "location": f"{class_node.location.file.name}:{class_node.location.line}",
            "members": [],
            "methods": []
        }
        
        # 遍历类的子节点
        for child in class_node.get_children():
            # 处理成员变量
            if child.kind == clang.cindex.CursorKind.FIELD_DECL:
                class_info["members"].append({
                    "name": child.spelling,
                    "type": child.type.spelling,
                    "access": self._get_access_specifier(child)
                })
            
            # 处理成员函数
            elif child.kind == clang.cindex.CursorKind.CXX_METHOD:
                # 跳过析构函数
                if child.spelling.startswith('~'):
                    continue
                    
                # 获取函数参数
                params = []
                for param in child.get_children():
                    if param.kind == clang.cindex.CursorKind.PARM_DECL:
                        params.append({
                            "name": param.spelling,
                            "type": param.type.spelling
                        })
                
                class_info["methods"].append({
                    "name": child.spelling,
                    "return_type": child.result_type.spelling,
                    "parameters": params,
                    "access": self._get_access_specifier(child),
                    "is_const": child.is_const_method()
                })
        
        self.classes.append(class_info)
    
    def _get_access_specifier(self, node):
        """获取成员的访问权限（public, private, protected）"""
        access = "private"  # 默认私有
        current = node.semantic_parent
        while current:
            if current.kind in [clang.cindex.CursorKind.CLASS_DECL, 
                               clang.cindex.CursorKind.STRUCT_DECL]:
                break
            if current.kind == clang.cindex.CursorKind.ACCESS_SPEC_DECL:
                if current.access_specifier == clang.cindex.AccessSpecifier.PUBLIC:
                    access = "public"
                elif current.access_specifier == clang.cindex.AccessSpecifier.PROTECTED:
                    access = "protected"
                else:
                    access = "private"
            current = current.semantic_parent
        return access
    
    def generate_reflection_code(self, output_file="reflection_metadata.h"):
        """生成C++反射元数据代码"""
        if not self.classes:
            print("没有找到任何类信息，无法生成反射代码")
            return False
            
        self.reflection_code = "#ifndef REFLECTION_METADATA_H\n"
        self.reflection_code += "#define REFLECTION_METADATA_H\n\n"
        self.reflection_code += "#include <string>\n"
        self.reflection_code += "#include <vector>\n"
        self.reflection_code += "#include <map>\n"
        self.reflection_code += "#include <any>\n\n"
        
        # 定义反射所需的数据结构
        self.reflection_code += "struct ParameterInfo {\n"
        self.reflection_code += "    std::string name;\n"
        self.reflection_code += "    std::string type;\n"
        self.reflection_code += "};\n\n"
        
        self.reflection_code += "struct MemberInfo {\n"
        self.reflection_code += "    std::string name;\n"
        self.reflection_code += "    std::string type;\n"
        self.reflection_code += "    std::string access;\n"
        self.reflection_code += "};\n\n"
        
        self.reflection_code += "struct MethodInfo {\n"
        self.reflection_code += "    std::string name;\n"
        self.reflection_code += "    std::string return_type;\n"
        self.reflection_code += "    std::vector<ParameterInfo> parameters;\n"
        self.reflection_code += "    std::string access;\n"
        self.reflection_code += "    bool is_const;\n"
        self.reflection_code += "};\n\n"
        
        self.reflection_code += "struct ClassInfo {\n"
        self.reflection_code += "    std::string name;\n"
        self.reflection_code += "    std::string display_name;\n"
        self.reflection_code += "    std::string location;\n"
        self.reflection_code += "    std::vector<MemberInfo> members;\n"
        self.reflection_code += "    std::vector<MethodInfo> methods;\n"
        self.reflection_code += "};\n\n"
        
        self.reflection_code += "class ReflectionDatabase {\n"
        self.reflection_code += "private:\n"
        self.reflection_code += "    static std::map<std::string, ClassInfo> classes;\n\n"
        self.reflection_code += "public:\n"
        self.reflection_code += "    static void register_class(const ClassInfo& info) {\n"
        self.reflection_code += "        classes[info.name] = info;\n"
        self.reflection_code += "    }\n\n"
        self.reflection_code += "    static const ClassInfo* get_class_info(const std::string& name) {\n"
        self.reflection_code += "        auto it = classes.find(name);\n"
        self.reflection_code += "        if (it != classes.end()) {\n"
        self.reflection_code += "            return &it->second;\n"
        self.reflection_code += "        }\n"
        self.reflection_code += "        return nullptr;\n"
        self.reflection_code += "    }\n\n"
        self.reflection_code += "    static std::vector<std::string> get_all_class_names() {\n"
        self.reflection_code += "        std::vector<std::string> names;\n"
        self.reflection_code += "        for (const auto& pair : classes) {\n"
        self.reflection_code += "            names.push_back(pair.first);\n"
        self.reflection_code += "        }\n"
        self.reflection_code += "        return names;\n"
        self.reflection_code += "    }\n"
        self.reflection_code += "};\n\n"
        
        # 为每个类生成反射元数据
        for cls in self.classes:
            self.reflection_code += f"// {cls['name']}的反射元数据\n"
            self.reflection_code += f"class {cls['name']}_reflection {{\n"
            self.reflection_code += "public:\n"
            self.reflection_code += f"    {cls['name']}_reflection() {{\n"
            self.reflection_code += f"        ClassInfo info;\n"
            self.reflection_code += f"        info.name = \"{cls['name']}\";\n"
            self.reflection_code += f"        info.display_name = \"{cls['display_name']}\";\n"
            self.reflection_code += f"        info.location = \"{cls['location']}\";\n"
            
            # 添加成员变量信息
            for member in cls['members']:
                self.reflection_code += f"        info.members.push_back({{\"{member['name']}\", \"{member['type']}\", \"{member['access']}\"}});\n"
            
            # 添加成员函数信息
            for method in cls['methods']:
                self.reflection_code += f"        MethodInfo {method['name']}_method;\n"
                self.reflection_code += f"        {method['name']}_method.name = \"{method['name']}\";\n"
                self.reflection_code += f"        {method['name']}_method.return_type = \"{method['return_type']}\";\n"
                self.reflection_code += f"        {method['name']}_method.access = \"{method['access']}\";\n"
                self.reflection_code += f"        {method['name']}_method.is_const = {str(method['is_const']).lower()};\n"
                
                for param in method['parameters']:
                    self.reflection_code += f"        {method['name']}_method.parameters.push_back({{\"{param['name']}\", \"{param['type']}\"}});\n"
                
                self.reflection_code += f"        info.methods.push_back({method['name']}_method);\n"
            
            self.reflection_code += "        ReflectionDatabase::register_class(info);\n"
            self.reflection_code += "    }\n"
            self.reflection_code += "};\n\n"
            self.reflection_code += f"static {cls['name']}_reflection {cls['name']}_reflection_instance;\n\n"
        
        # 定义静态成员
        self.reflection_code += "std::map<std::string, ClassInfo> ReflectionDatabase::classes;\n\n"
        self.reflection_code += "#endif // REFLECTION_METADATA_H\n"
        
        # 保存生成的代码
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(self.reflection_code)
        print(f"已生成反射元数据头文件: {output_file}")
        return True
    
    def run(self, output_file="reflection_metadata.h"):
        """执行解析和代码生成流程"""
        print("开始解析C++代码...")
        self.parse_code()
        
        if not self.classes:
            print("解析完成，但未发现任何类定义")
            return False
            
        print(f"解析完成，发现 {len(self.classes)} 个类")
        return self.generate_reflection_code(output_file)

if __name__ == "__main__":
    # 分离文件和包含目录
    cpp_files = ["Example.cpp"]
    include_dirs = ["."]

    
    if not cpp_files:
        print("请指定至少一个C++文件")
        sys.exit(1)
    
    generator = CppReflectionGenerator(cpp_files, include_dirs)
    generator.run()
    
```

&emsp;&emsp;比如我们用下面的类来解析：
```cpp
//Example.hpp
#pragma once
#include <string>
class Example{
public:
    std::string getName();
    void setName(std::string &name);
    int getAge();
    void setAge(int a);
    
private:
    std::string name;
    int age;
};
//Example.cpp
#include "Example.hpp"

std::string Example::getName(){
    return name;
}

void Example::setName(std::string &n){
    name = n;
}

void Example::setAge(int a){
    age = a;
}

int Example::getAge(){
    return age;
}
```

&emsp;&emsp;通过上面的脚本解析的内容很多，这里只截取部分和`Example`相关的内容，能够看到这里能够准确的拿到所有属性。
```cpp
// Example的反射元数据
class Example_reflection {
public:
    Example_reflection() {
        ClassInfo info;
        info.name = "Example";
        info.display_name = "Example";
        info.location = ".\Example.hpp:3";
        info.members.push_back({"name", "std::string", "private"});
        info.members.push_back({"age", "int", "private"});
        MethodInfo getName_method;
        getName_method.name = "getName";
        getName_method.return_type = "std::string";
        getName_method.access = "private";
        getName_method.is_const = false;
        info.methods.push_back(getName_method);
        MethodInfo setName_method;
        setName_method.name = "setName";
        setName_method.return_type = "void";
        setName_method.access = "private";
        setName_method.is_const = false;
        setName_method.parameters.push_back({"name", "std::string &"});
        info.methods.push_back(setName_method);
        MethodInfo getAge_method;
        getAge_method.name = "getAge";
        getAge_method.return_type = "int";
        getAge_method.access = "private";
        getAge_method.is_const = false;
        info.methods.push_back(getAge_method);
        MethodInfo setAge_method;
        setAge_method.name = "setAge";
        setAge_method.return_type = "void";
        setAge_method.access = "private";
        setAge_method.is_const = false;
        setAge_method.parameters.push_back({"a", "int"});
        info.methods.push_back(setAge_method);
        ReflectionDatabase::register_class(info);
    }
};
```

&emsp;&emsp;这种方案结合了静态代码分析（编译期）和运行时反射访问，既保持了 C++ 的性能优势，又提供了灵活的反射能力。相比纯手动实现，自动化代码生成大大减少了工作量和出错概率。

## 4 总结
&emsp;&emsp;在 C++ 中，反射机制的划分确实并非绝对泾渭分明，静态反射与动态反射之间存在一定的模糊地带，这主要源于 C++ 语言特性的特殊性和反射实现方式的多样性。
- 静态反射通常指在编译期获取类型信息并进行处理，依赖编译期计算和模板元编程，例如通过type_traits获取类型属性、 constexpr 表达式解析类型结构等。其特点是信息处理在编译期完成，不依赖运行时数据。
- 动态反射则是在运行时获取和操作类型信息，比如通过 RTTI（运行时类型识别）的```typeid```和```dynamic_cast```，或者自定义的元数据注册表。它允许程序在运行时动态检查对象类型、调用方法等。

&emsp;&emsp;但在实际实现中，二者的界限会变得模糊。许多反射库会结合静态和动态技术，例如编译期生成元数据（静态），运行时通过这些元数据进行操作（动态）C++20 引入的概念（Concepts）和 C++23 的静态反射提案，进一步模糊了编译期类型检查与运行时行为的界限模板元编程可以生成具有动态行为特征的代码，而运行时反射也可能依赖编译期生成的辅助结构。