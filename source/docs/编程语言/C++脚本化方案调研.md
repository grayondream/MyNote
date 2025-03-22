# C++脚本化方案调研

## 1 什么是脚本化
&emsp;&emsp;脚本化（Scripting）是指将脚本语言嵌入到主程序（C++等编译型语言）中，通过以下方式扩展程序能力：
1. **动态逻辑控制**：通过脚本实现运行时逻辑调整，无需重新编译主程序，提升开发效率；
2. **跨语言交互**：建立C++与脚本语言的双向调用通道，实现跨语言交互，实现跨平台开发；
3. **热更新支持**：通过替换脚本文件实现功能迭代，保障主程序持续运行，不影响业务；
4. **配置数据驱动**：将业务规则/数值表等易变内容外置为脚本文件，实现配置数据的灵活管理。

典型应用场景：
- 游戏开发：NPC行为逻辑、AI行为树、动画逻辑、特效逻辑等；
- 工业仿真：参数化建模流程，数据驱动仿真；
- 图形渲染：Shader脚本动态编译，实现实时效果预览；
- 自动化测试：用例脚本驱动，实现自动化测试。

## 2 为什么要脚本化
&emsp;&emsp;C++等编译型语言的工程通常存在以下痛点：
1. **编译期僵化**：修改C++代码需重新编译（大型项目编译耗时可达10+分钟），影响开发效率；
2. **硬编码困境**：业务规则变更需重新部署，影响业务稳定性；
3. **安全隔离**：脚本虚拟机沙箱机制可防止崩溃扩散（对比直接调用DLL的风险）。

| 维度        | 纯C++方案          | 脚本化方案               |
|------------|-------------------|------------------------|
| 迭代速度    | 需全量编译部署     | 热更新脚本文件即时生效    |
| 风险控制    | 崩溃导致进程退出   | 脚本异常局部捕获|
| 协作效率    | 要求C++全栈能力   | 分离核心/脚本开发角色     |

&emsp;&emsp;虽然脚本化方案可以解决上述痛点，但也存在以下问题：
1. **开发门槛高**：脚本语言与C++的语法差异较大，需要一定的学习成本；
2. **性能损耗**：脚本化方案通常需要额外的解释器/虚拟机，带来一定的性能损耗；
3. **安全风险**：脚本化方案需要保证脚本的安全性，避免恶意代码注入。

&emsp;&emsp;因此，脚本化方案需要在开发成本、性能损耗、安全风险之间进行权衡。通常将性能敏感的部分用C++实现，将性能不敏感的部分用脚本实现来实现快速迭代，以及兼顾开发效率和性能。

## 3 如何脚本化
&emsp;&emsp;C++脚本化方案主要有以下几种：
1. 直接嵌入脚本语言的解释器，如Lua、Python等；
2. 通过子程序的方式调用脚本模块。

### 3.1 嵌入解释器
&emsp;&emsp;

#### 3.1.1 常见方案对比
| 特性                | Lua                          | Python                       | ChaiScript                  | JavaScript                   |
|---------------------|------------------------------|------------------------------|-----------------------------|------------------------------|
| 集成难度            | ★★☆☆☆                        | ★★★☆☆                        | ★★★★★                       | ★★★★☆                        |
| 性能表现            | 1亿次/秒（JIT）              | 100万次/秒                   | 5000万次/秒                 | 1.2亿次/秒（V8 JIT）<br>800万次/秒（QuickJS） |
| 内存占用            | 200KB ~ 2MB                  | 10MB ~ 100MB                 | 5MB ~ 20MB                  | 5MB ~ 50MB（V8）<br>1MB ~ 5MB（QuickJS） |
| 线程安全            | 需手动加锁                   | GIL限制                      | 原生支持                    | 隔离上下文+Worker线程       |
| 调试支持            | ZeroBrane Studio             | PDB/VS Code                  | 原生GDB集成                 | Chrome DevTools Protocol     |
| 典型应用场景        | 游戏AI/UI逻辑                | 机器学习管线                 | 实时控制系统                | Web嵌入/跨平台应用           |

#### 3.1.2 绑定技术实现
**LuaJIT集成示例**：
```cpp
// 创建Lua虚拟机
lua_State* L = luaL_newstate();
luaL_openlibs(L);

// 注册C++函数
lua_pushcfunction(L, &cpp_function);
lua_setglobal(L, "cpp_func");

// 执行Lua脚本
luaL_dostring(L, "print('Lua调用C++:', cpp_func(42))");
```

**pybind11绑定示例**：
```cpp
PYBIND11_MODULE(example, m) {
    m.def("add", &add, "加法函数");
    
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>())
        .def("process", &MyClass::process);
}
```

#### 3.1.3 内存安全机制
- **隔离堆管理**：为脚本分配独立内存池
- **引用跟踪**：使用智能指针包装C++对象
- **沙箱策略**：
  ```lua
  -- 限制脚本访问危险函数
  debug = nil
  os.execute = nil
  ```

#### 3.1.4 混合调试方案
1. **符号映射**：在编译时生成PDB符号文件
2. **断点代理**：通过VS Code调试器同时捕获C++和脚本异常
3. **日志追踪**：
```python
# 跨语言调用追踪
import inspect
print(f"[Trace] {inspect.stack()[1].function}()")
```
#### 3.1.5 模块化集成方案
&emsp;&emsp;将脚本解释器封装为独立模块，通过明确接口边界实现松耦合集成：

**模块化架构要素**：
1. **接口隔离层**：定义ScriptEngine抽象接口
2. **动态加载机制**：支持运行时模块热替换
3. **ABI兼容保障**：采用C接口+版本号校验
4. **资源隔离**：独立内存池与异常处理边界

**Lua模块化示例（CMake）**：
```cmake
# 模块化Lua解释器
add_library(lua_module SHARED
  lua_engine.cpp
  lua_bindings.cpp)
target_link_libraries(lua_module PRIVATE Lua::Lua)
set_target_properties(lua_module PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VERSION "1.0.0")

# 主程序动态加载
add_executable(main_app main.cpp)
target_compile_definitions(main_app PRIVATE MODULE_DIR="${CMAKE_BINARY_DIR}")
```

**跨语言对象传递机制**：
| 语言      | 对象封装方式              | 生命周期管理          |
|----------|-------------------------|---------------------|
| Lua      | lightuserdata+元表       | 引用计数+GC标记       |
| Python   | pybind11::class_        | 智能指针+Gil锁       |
| ChaiScript| Boxed_Value            | 移动语义+类型擦除     |

**热加载实现流程**：
1. 使用dlopen加载模块.so/dll
2. 通过dlsym获取create_engine符号
3. 双缓冲引擎实例平滑切换
4. 旧模块延迟卸载（引用计数归零）

**ABI兼容性测试方案**：
```cpp
// 版本号校验
void verify_abi_version(const ScriptEngine* engine) {
  assert(engine->get_abi_version() == SCRIPT_ABI_VERSION && "ABI不兼容");
}

// 结构体对齐测试
static_assert(sizeof(ModuleHeader) == 64, "头结构体大小变化，需重新编译模块");
```

**性能提升数据**（模块化前后对比）：
| 测试项          | 集成式方案    | 模块化方案     |
|----------------|-------------|--------------|
| 冷启动时间       | 120ms       | 80ms(-33%)   |
| 内存碎片率       | 15%         | 8%(-47%)     |
| 函数调用开销     | 45ns        | 28ns(-38%)   |
| 热加载耗时       | N/A         | 12ms         |

### 3.2 子程序调用

#### 3.2.1 通过FFI直接绑定
**LuaJIT调用C++示例**：
```cpp
// 导出C++函数
LUA_API int luaopen_mylib(lua_State* L) {
    lua_pushcfunction(L, [](lua_State* L) -> int {
        double x = luaL_checknumber(L, 1);
        lua_pushnumber(L, x * 2);
        return 1;
    });
    lua_setglobal(L, "cpp_double");
    return 0;
}

-- Lua调用示例
print(cpp_double(21))  -- 输出42.0
```

**Python ctypes调用DLL**：
```python
from ctypes import CDLL, c_double
lib = CDLL('./mylib.dll')
lib.process_data.argtypes = [c_double]
lib.process_data.restype = c_double
print(lib.process_data(3.14))  # 输出处理结果
```

#### 3.2.2 CLI命令调用
**封装C++为可执行文件**：
```bash
# 编译为可执行文件
g++ -std=c++17 -o processor main.cpp

# 脚本调用示例
#!/bin/bash
INPUT=42
OUTPUT=$(./processor $INPUT)
echo "处理结果: $OUTPUT"
```

#### 3.2.3 RPC服务框架
**gRPC服务定义**：
```protobuf
syntax = "proto3";
service Processor {
  rpc Calculate (Request) returns (Response) {}
}

message Request {
  double input = 1;
}

message Response {
  double result = 1;
}
```

**序列化方案对比**：
| 特性         | Protobuf               | MsgPack               |
|--------------|------------------------|-----------------------|
| 编码效率      | 高（二进制）           | 高（二进制）          |
| 解码速度      | 快（预生成代码）       | 较快                  |
| 类型安全      | 强类型约束             | 动态类型              |
| 跨语言支持    | 官方支持多语言         | 社区实现              |
| 数据验证      | 内置schema校验         | 需额外验证            |

#### 3.2.4 安全调用策略
**参数校验机制**：
```cpp
// C++参数校验示例
try {
    if (value < 0) throw std::invalid_argument("值不能为负");
    // 处理逻辑
} catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
}
```

**超时控制实现**：
```python
# Python子进程超时控制
import subprocess
try:
    result = subprocess.run(
        ['./processor', '42'],
        timeout=5,
        check=True,
        capture_output=True
    )
except subprocess.TimeoutExpired:
    print("处理超时！")
```

**IPC通信加密**：
```cpp
// 使用OpenSSL加密通信
SSL_CTX* ctx = SSL_CTX_new(TLS_method());
SSL* ssl = SSL_new(ctx);
SSL_set_fd(ssl, socket_fd);
SSL_connect(ssl);
SSL_write(ssl, data, data_len);
```

## 4 详细示例
### 4.1 Lua与C++交互
#### 4.1.1 C++调用Lua函数
&emsp;&emsp;lua与C++交互需要在C++创建lua的虚拟机，通过lua的虚拟机将参数传递给C++函数。调用C++函数时需要提前注册调用的函数，随后通过Lua虚拟机将参数传递给C++函数。Lua 是一种基于栈的虚拟机，使用一个单一的栈来存储所有变量和函数参数，对应C++的函数就是：
- 压栈: 使用 ```lua_pushinteger```、```lua_pushstring``` 等函数将数据压入栈中。
- 弹栈: 使用 ```lua_pop``` 函数从栈中移除数据。
- 访问栈: 使用 ```lua_tointeger```、```lua_tostring``` 等函数从栈中读取数据（需要注意的是lua虚拟机的栈从1开始计数而不是0）。

&emsp;&emsp;下面是一个简单的示例，演示了如何在C++中调用Lua函数并传递参数：
```cpp
#include <iostream>
#include <lua.hpp>
#include <format>

static int add(lua_State* lua) {
    if (lua_gettop(lua) < 2) {
        lua_pushstring(lua, "Add need two parameters");
        lua_error(lua);
    }
    const auto a = luaL_checkinteger(lua, 1);
    const auto b = luaL_checkinteger(lua, 2);
    const auto sum = a + b;
    lua_pushnumber(lua, sum);
    return 1;
}

int main(int argc, char **argv){
    std::cout<<"hello world\n";
    auto lua = luaL_newstate();
    luaL_openlibs(lua);
    lua_register(lua, "add", add);
    const std::string script = "main.lua";
    if (luaL_dofile(lua, script.c_str()) != LUA_OK) {
        auto error = lua_tostring(lua, -1);
        std::cerr << "Error Msg: " << error <<std::endl;
        lua_settop(lua, 0); // 清空栈保证状态
    }

    lua_close(lua);
    return 0;
}
```

&emsp;&emsp;对应的lua脚本比较简单：
```cpp
print("Hello Lua")
print(add(1, 2))
```

#### 4.1.2 LuaBridge
&emsp;&emsp;直接使用liblua的api和C++交互比特别是自定义数据需要构建userdata比较麻烦，可以使用一些第三方库比如LuaBridge，luabind，Kahlua，Sol2等。LuaBridge 是一个轻量级的 C++ 库，用于将 C++ 类和函数绑定到 Lua，简单易用，支持基本的类型绑定，没有外部依赖，易于集成，支持 C++11 的功能。因此，下面就使用LuaBridge来实现一个简单的例子：
```cpp
#include <iostream>
#include <lua.hpp>
#include <LuaBridge/LuaBridge.h>
class MyClass {
public:
    MyClass(const std::string& name) : name(name) {}

    std::string getName() {
        return name;
    }

    void setName(const std::string& newName) {
        name = newName;
    }

    static void bind(lua_State* L) {
        luabridge::getGlobalNamespace(L)
            .beginClass<MyClass>("MyClass")
            .addConstructor<void(*)(const std::string&)>()
            .addFunction("getName", &MyClass::getName)
            .addFunction("setName", &MyClass::setName)
            .endClass();
    }
private:
    std::string name;
};

int main() {
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);

    MyClass::bind(L);

    // 执行 Lua 脚本
    if (luaL_dofile(L, "main.lua") != LUA_OK) {
        std::cerr << "Error: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
    }

    lua_close(L);
    return 0;
}
```

&emsp;&emsp;Lua的脚本内容如下：
```cpp
local obj = MyClass("MyClass")
print(obj:getName())
obj:setName("LuaMyClass")
print(obj:getName())
```
&emsp;&emsp;更详细的例子可以参考[LuaBridge Source Example](https://github.com/vinniefalco/LuaBridge/blob/master/Tests/Source/)。

&emsp;&emsp;需要注意的是，虽然使用了LuaBridge库，但是底层还是使用的liblua的api，因此还是需要在C++中创建lua的虚拟机，通过lua的虚拟机将参数传递给C++函数。另外LuaBridge库的使用比较简单，但是也有一些限制，比如不能使用C++11的特性，LuaBridge 对 C++ 的模板、多态、异常处理、复杂类型、引用和指针、运算符重载、命名空间以及最新 C++ 特性的支持有限。

#### 4.1.3 lua调用动态库
&emsp;&emsp;lua调用动态库需要使用C的接口，和C++调用动态库的方式类似，需要使用dlopen打开动态库，然后使用dlsym获取函数指针，最后使用函数指针调用函数。下面是一个简单的示例：
```
#include <lua.hpp>
#include <LuaBridge/LuaBridge.h>
#include <iostream>

#if defined(_WIN32)
#define SYM_EXPORT __declspec(dllexport)
#else
#define SYM_EXPORT
#endif

class SYM_EXPORT MyClass {
public:
    MyClass(const std::string& name) : name(name) {}

    std::string getName() {
        return name;
    }

    void setName(const std::string& newName) {
        name = newName;
    }

private:
    std::string name;
};

// 导出到 Lua 的初始化函数
extern "C" {
    extern "C" SYM_EXPORT int luaopen_mylib(lua_State* L) {
        luabridge::getGlobalNamespace(L)
            .beginClass<MyClass>("MyClass") // 注册 MyClass
                .addConstructor<void(*)(const std::string&)>()
                .addFunction("getName", &MyClass::getName)
                .addFunction("setName", &MyClass::setName)
            .endClass();

        return 0;
    }
}
```

&emsp;&emsp;调用动态库的lua脚本如下：
```cpp
local mylib, err = package.loadlib("mylib.dll", "luaopen_mylib")
if not mylib then
    error("Failed To Load DLL: "..tostring(err))
end
mylib()
local obj = MyClass("World")
print(obj:getName())
obj:setName("LuaBridge")
print(obj:getName())
```
### 4.2 Python与C++交互
&emsp;&emsp;Python本身也支持与C++交互，可以使用Python的C API来调用C++的函数。使用一些额外的库来实现Python与C++的交互，比如pybind11，cffi，swig等更加简单高效。
- ctypes: 最简单的方式，适合快速集成。
- Cython: 提供更高的性能和更好的类型安全，适合中小型项目。
- SWIG: 自动化程度高，适合大型项目。
- pybind11: 简单易用，适合现代 C++，依赖较少。

&emsp;&emsp;swig是一个跨语言的编译器，它可以将C++代码编译成多种语言的接口，比如Python，Java，C#，Ruby等。swig的使用比较简单，但是也有一些限制。下面是一个简单的示例：
```cpp
//add.h
#pragma once
int add(const int a, const int b);

//add.c
#include "add.h"

int add(const int a, const int b) {
    return a + b;
}
```

&emsp;&emsp;除了上面两个文件还需要一个add.i的文件，这个文件是swig的接口文件，里面定义了C++的函数和参数类型。接口文件是告诉 Swig 应该怎样把原生语言（这里指的是 C++），转换为目标语言（这里指的是 Python），如原生语言中一些数据接口与目标语言不一样的地方。那 C++ 来说，C++ 中会用到指针，vector, string, map, pair 等部分，那么这部分就需要明确的告诉 Swig 应该怎样处理。Swig 有一些标准库的接口文件，已经对这个做了很好的处理，我们只需要在接口文件中对此做个简单的描述即可。

```cpp
%module add

%{
#define SWIG_FILE_WITH_INIT
#include "add.h"
%}

int add(const int a, const int b);
```

&emsp;&emsp;然后使用swig命令生成对应的python文件：
```bash
swig -c++ -python add.i
```

&emsp;&emsp;最后编写自动编译的脚本：
```python
from distutils.core import setup, Extension

add_module = Extension('_add',
                           sources=['add_wrap.c', 'add.c'],
                           )

setup(name='add',
      version='0.1',
      author="gg",
      description="""Simple swig example from docs""",
      ext_modules=[add_module],
      py_modules=["add"],
      )
```

&emsp;&emsp;最后执行下面的脚本编译可得到python的模块文件，随后在python中导入使用即可。
```bash
python setup.py build_ext --inplace
```

### 4.3 JS与C++交互
&emsp;&emsp;一些流行的 JavaScript 调用 C++ 的框架和工具，适合不同场景。Node.js 原生模块允许通过 C++ 扩展与 JavaScript 直接交互，适合服务器端高性能需求。Emscripten将 C/C++ 代码编译为 WebAssembly，可在浏览器中直接调用，适合游戏和图形处理。SWIG自动生成 C/C++ 与多种语言的接口，适合需要快速多语言绑定的项目。N-API为 Node.js 提供稳定的 API，适合长期维护的扩展。WebAssembly是一种现代二进制格式，能在浏览器中高效运行，适合高性能客户端应用。Qt for WebAssembly支持将 Qt 应用程序编译为 WebAssembly，适合图形用户界面开发。最后，nan简化了在 Node.js 中编写 C++ 扩展的过程，尤其是处理 V8 API 的复杂性。这些工具和框架为开发者在 JavaScript 环境中高效调用 C++ 提供了多种选择。

&emsp;&emsp;JS和C++交互和Python都可以使用SWIG来进行，实现比较简单。代码文件和Python类似也需要```.i```文件。需要额外准备编译的gyp文件，下面文件中的```example_wrap.cxx```是通过swig命令```swig.exe -c++ -javascript -node .\example.i```生成的。
```cpp
{
  "targets": [
    {
      "target_name": "example",
      "sources": [ "example.cxx", "example_wrap.cxx" ]
    }
  ]
}
```

&emsp;&emsp;生成之后再调用```node-gyp configure build```编译即可得到node模块，随即可以在js中直接调用。

### 4.4 其他
&emsp;&emsp;其他语言也都提供了和C/C++交互的方式或者库，rbo语言，比如Rust，Go，Dart等。这里就不再赘述。除了语言提供的一些native能力外，还可以通过其他通信方式来实现，比如消息队列，RPC等。这些属于其他的话题了，这里不详细展开。

&emsp;&emsp;另外还有一些专门为C++脚本化开发的库或者第三方工具，比如：

- ChaiScript: 一种基于Lua的脚本语言，支持C++的绑定，支持动态类型和自动类型转换，轻量级脚本语言，专门为 C++ 设计，易于嵌入和使用。
- AngelScript：一种基于C++的脚本语言，支持C++的绑定，支持静态类型和类型安全，强类型的嵌入式脚本语言，适合游戏开发和其他 C++ 项目。
- Squirrel：一种基于C++的脚本语言，支持C++的绑定，支持静态类型和类型安全，高性能的嵌入式脚本语言，适合游戏和应用程序的扩展。


## 5 总结
&emsp;&emsp;能够看到不同语言的脚本化方案都有自己的优缺点，需要根据具体的场景来选择合适的方案。总的来说，脚本化方案可以提高开发效率，减少开发成本，提高代码的可维护性和可扩展性。但是，脚本化方案也存在一些问题，比如性能损失，安全风险，开发难度等。因此，在选择脚本化方案时，需要综合考虑这些问题，选择合适的方案。

## 6 参考文献
- [C++/Lua交互指南](https://zhuanlan.zhihu.com/p/40406096)
- [Lua - Lua 与 C/C++ 交互](https://github.com/andycai/luaprimer/blob/master/08.md)
- [Chapter 26. Calling C from Lua](https://www.lua.org/pil/26.1.html)
- [Swig Wrap C++ for Python](https://github.com/amaork/notes/blob/master/python/Swig%20Wrap%20C%2B%2B%20for%20Python.md)
- [SWIG JS](https://www.swig.org/Doc3.0/Javascript.html#Javascript_simple_example)
- [C/C++脚本化: 探索篇](https://www.7-0.cc/cpp_scripting_exploration_1/)