# Linux共享库管理和构建

> &emsp;&emsp;有了共享库那么就存在对库版本的管理问题。
## 1 共享库版本
### 1.1 共享库兼容
&emsp;&emsp;共享库更新时一般会存在两种形式的更新，兼容更新和不兼容更新。这里的兼容不仅仅指接口兼容，也指ABI（Application Binary Interface）兼容，ABI兼容一般会涉及到函数调用的堆栈管理、符号命名规则、参数规则、内存分布等。一般C语言库影响ABI兼容性的几种行为（但不是全部）：
1. 导出符号的行为发生改变；
2. 导出符号被删除；
3. 导出数据结构发生改变；
4. 导出函数的接口发生改变，比如返回值、参数等。

&emsp;&emsp;C语言中为了保证ABI兼容本身就比较麻烦，而C++就更难了，以至于C++标准并没有规定C++的ABI。如果开发C++共享库为了保证ABI兼容，可能的注意事项（即便遵守也无法完全保证）：
- 不要在接口类使用虚函数，如果有添加，不要修改子类中的实现；
- 不要改变类中任何的成员变量的位置或者类型；
- 不要删除非内嵌```public```或者```protected```成员函数；
- 不要将非内嵌的成员函数改变成内嵌成员函数；
- 不要改变成员函数的访问权限；
- 不要在接口中使用模板；
- 不要改变接口的任何部分或者使用C作为共享接口。

### 1.2 共享库版本命名
&emsp;&emsp;Linux下共享库使用版本进行管理，有时候查看一个共享库发现其实是一个软连接，实际的文件后面有一串数字。该数字就是库的版本，类似```libname.x.y.z```：
- name：库名字；
- x——主版本号：表示重大升级，不同主版本号之间不兼容，升级时一般会保留几版本的库以供旧版本的软件使用；
- y——次版本号：表示库的增量升级，即添加了一些符号但是原来的符号不变，向后兼容；
- z——发布版本号：仅仅bug修复、性能改进，并不改变其他内容。

### 1.3 SO-NAME
&emsp;&emsp;一个可执行文件依赖于那个版本的共享库在其文件内有记录，但是对于不同版本的库又该依赖哪个？Linux采用SO-NAME的命名管理来记录共享库的依赖关系，SO-NAME就是一个共享库去除次版本号和发布版本号的名称，比如```libmyso.1.0.0```的SO-NAME是```libmyso.1```。Linux会在共享库的相同目录下创建一个以SO-NAME为名称的软连接。程序中只需要指定依赖的SO-NAME即可，这样能够保证库的兼容性的情况下，即便更新库也不需要重新编译程序。
>&emsp;&emsp;由于历史原因，并不是所有的文件都遵守上面的规则比如ld和glibc。

&emsp;&emsp;另外Linux提供了```ldconfig```，在系统更新一个库时，遍历默认的库目录，比如```lib,/usr/lib```更新相关的软连接。
```cpp
➜  tmp ls -la /usr/lib/libxmlsec1-*
lrwxrwxrwx 1 root root     28 Feb 18  2016 /usr/lib/libxmlsec1-openssl.so.1 -> libxmlsec1-openssl.so.1.2.20
-rw-r--r-- 1 root root 237928 Feb 18  2016 /usr/lib/libxmlsec1-openssl.so.1.2.20
➜  tmp readelf -d /usr/lib/libguestlib.so.0

Dynamic section at offset 0x5df8 contains 26 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libvmtools.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000000e (SONAME)             Library soname: [libguestlib.so.0]
```

>&emsp;&emsp;在链接时，使用```-l```指定的库名称为链接名，链接器会根据连接名寻找最新的库。

## 2 符号版本
&emsp;&emsp;上面仅仅通过次版本号控制加载的共享库版本的方式，会导致依赖高此版本的共享库运行在仅仅包含低版本共享库的系统上无法正常运行。现在系统通过符号版本机制来进行更加精细化的管理。
**基于符号的版本机制**
&emsp;&emsp;次版本号的升级一般会添加新的接口，基于符号版本机制的基本方案是为这些符号添加一个版本修饰，来表明对应的符号所对应的版本。其基本思想就是原来是基于整个库做版本管理变成基于单个符号做版本管理。

**Solaris中的符号版本机制**
&emsp;&emsp;Solaris是最早实现符号版本机制的系统，其基本思路是，定义一些符号的集合，每个集合中包含一些指定的符号，除了可以拥有符号以外，一个集合还可以包含另外一个集合。 也就是说相同集合内的符号在指定的版本上是有效的，反之无效。程序员在链接共享库时可以编写一种符号版本脚本来指定集合与集合之间的关系。同时引入了范围机制，可以对某些符号进行作用域的范围控制，修改某些符号的作用范围。

**Linux中的符号版本**
&emsp;&emsp;Linux系统下共享库的符号版本机制并没有广泛应用，主要使用共享库符号版本机制的是Glibc的几个共享库。
&emsp;&emsp;GCC对Solaris符号版本机制的扩展：
- 除了可以符号版本脚本中指定符号的版本之外，gcc允许使用一个叫做```.symver```的汇编宏指令指定符号版本，比如```asm(".symver add, add@VERS_1.1")```；
- gcc允许多个版本的同一个符号存在于共享库，也就是在链接层面以某种形式提供了符号重载机制。这样做的目的是如果程序仅仅修改了接口定义，也不需要更改主版本号保证兼容性。
```c
asm(".symver old_printf, printf@VERS_1.1")
asm(".symver new_printf, printf@VERS_1.2")
int old_printf(){}
int new_printf(){}
```

&emsp;&emsp;Solaris系统符号版本不支持一个符号有多个版本，Linux支持，即链接器可以根据自身程序的情况选择对应版本的符号。

**Linux系统中符号版本机制实践**
&emsp;&emsp;下面分别是涉及到的文件```pic.c, pic.version, main.c```的内容。
```
#include "pic.h"

static int a;

void bar(){
    a = 1;
}

void foo(){
    bar();
}
```

```
VERS_1.2{
    global: 
        foo;
    local:
        *;
};
```

```
#include "pic.h"

int main(){
    foo();
}
```

&emsp;&emsp;下面是基本的步骤，首先将```pic_version```中的版本号设置为1.1编译出1.1版本的库，然后修改为1.2编译出1.2版本的库，之后利用1.1的库链接为可执行文件，最后将1.2的库重命名为可执行文件需要链接的库名称，运行程序。就能够看到需要VERS_1.1而我们提供的是1.2。
```bash
➜  tmp gcc -shared -fPIC pic.c -Xlinker --version-script pic.version -o libpic.so.1.1
➜  tmp gcc -shared -fPIC pic.c -Xlinker --version-script pic.version -o libpic.so.1.2
➜  tmp cp libpic.so.1.1 libpic.so && gcc main.c ./libpic.so -o main
➜  tmp cp libpic.so.1.2 libpic.so
➜  tmp ./main
./main: ./libpic.so: version `VERS_1.1' not found (required by ./main)
```

## 3 Linux共享库
### 3.1 共享库系统路径
&emsp;&emsp;Linux开元系统遵守FHS，规定一个系统中的相关内容该如何存放。一般一些Linux系统中存放共享库的目录有：
- ```/lib```：存放系统相关的基础库；
- ```/usr/lib```：存放一些运行时非系统的运行关键库；
- ```/ur/local/lib```：存放一些第三方的库。

### 3.2 共享库查找过程
&emsp;&emsp;Linux上寻找共享库的目录根据```/etc/ld.so.conf```中指定的目录搜寻，为了加快速度，linux提供了```ldconfig```程序更新共享库，生成对应的软链接，并将库添加到```/etc/ld.so.cache```中，加速查找的速度，如果这个文件中没有还会搜索```/lib,/usr/lib```两个目录，如果还是没有找到则失败。
&emsp;&emsp;从下面能够看出```ld.so.conf```制定了一系列文件。
```bash
➜  tmp cat /etc/ld.so.conf
include /etc/ld.so.conf.d/*.conf

➜  tmp ls /etc/ld.so.conf.d
fakeroot-x86_64-linux-gnu.conf  ld.wsl.conf  x86_64-linux-gnu.conf     zz_i386-biarch-compat.conf
i386-linux-gnu.conf             libc.conf    x86_64-linux-gnu_GL.conf  zz_x32-biarch-compat.conf
```

### 3.3 库相关的一些环境变量
**LD_LIBRARY_PATH**
&emsp;&emsp;用户可以修改```LD_LIBRARY_PATH```来自定义共享库的搜寻路径，同时也可以通过链接器指定，比如```/lib/ld-liunx.so.2 -library-path <path>```。这种方式临时修改环境变量的搜索路径顺序为：
- 由环境变量```LD_LIBRARY_PATH```指定的路径；
- 由路径缓存```/etc/ld.so.cache```指定的路径；
- 默认的库目录，先```/usr/lib```再```/lib```。

**LD_PRELOAD**
&emsp;&emsp;```LD_PRELOAD```指定加载库之前预先需要加载的一些库，即便程序并不依赖于该库。

**LD_DEBUG**
&emsp;&emsp;```LD_DEBUG```可以打开链接器的调试功能，当设置时，动态链接器会在运行时打印各种有用的信息。
```bash
➜  tmp LD_DEBUG=files ./main
       631:
       631:     WARNING: Unsupported flag value(s) of 0x8000000 in DT_FLAGS_1.
       631:
       631:     file=./libpic.so [0];  needed by ./main [0]
       631:     file=./libpic.so [0];  generating link map
       631:       dynamic: 0x00007f7b12ff0e60  base: 0x00007f7b12df0000   size: 0x0000000000201028
       631:         entry: 0x00007f7b12df0460  phdr: 0x00007f7b12df0040  phnum:                  7
       631:
       631:
       631:     file=libc.so.6 [0];  needed by ./main [0]
       631:     file=libc.so.6 [0];  generating link map
       631:       dynamic: 0x00007f7b12ddab80  base: 0x00007f7b129f0000   size: 0x00000000003f0ae0
       631:         entry: 0x00007f7b12a11d10  phdr: 0x00007f7b129f0040  phnum:                 10
       631:
       631:     ./libpic.so: error: version lookup error: version `VERS_1.1' not found (required by ./main) (continued)
```

## 4 Linux共享库的创建与安装
### 4.1 共享库的创建
&emsp;&emsp;共享库的创建很简单就是编译器参数中添加动态库相关的参数，gcc的话是```-shared```。在创建时需要注意：
- 不需要去处输出共享库中的符合和调试信息；
- 开发过程中，为了测试不影响其他程序，可以使用```LD_LIBRARY_PATH```临时修改搜索路径或者使用```-rpath```参数指定路径；
- 可以指定需要导出的符号来减小库的大小。

### 3.2 清除符号信息
&emsp;&emsp;一般生成的共享库为了方便调试都包含符号信息，但是对于发布的版本可以使用```strip```或者```-Wl,-s```参数符号符号信息减小库的大小。

### 3.3 共享库的安装
&emsp;&emsp;安装共享库可以将库拷贝到系统默认的目录，比如```/lib,/usr/lib```然后运行```ldconfig```。也可以使用```ldconfig -n <path>```指定自定义目录，在编译时指定库的目录。

### 3.4 共享库的构造和析构
&emsp;&emsp;有些时候希望在库加载或者卸载时执行一些初始化或者销毁的操作，可以通过```__attribute__((constructor))```和```__attribute__((destructor))```分别修饰函数声明表明是需要在加载或者卸载库时才执行的代码。构造函数是在```dlopen```返回之前被执行，析构函数是在```dlclose```返回之前被执行。

>&emsp;&emsp;如果希望使用构造和析构函数，则无法使用```-nostartfiles,-nostdlib```，因此二者依赖于系统的标准运行库和启动文件。

&emsp;&emsp;如果定义了多个构造或者析构函数，执行顺序是未定义的，可以通过优先级控制执行的顺序，```__attribute__(constructor(n))```，n为优先级，n小的先执行，析构函数类似。

### 3.5 共享库脚本
&emsp;&emsp;Linux也支持使用脚本将多个库合并作为一个新的库，即库本身类似一个脚本。

