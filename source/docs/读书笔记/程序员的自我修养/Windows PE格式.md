# Windows PE格式

&emsp;&emsp;本章主要将windowsPE格式和ELF格式的区别。
## 1 Windows PE/COFF
&emsp;&emsp;PE是windows引入的一种可执行文件格式，该格式和Linux系统的ELF文件格式同源，都是由COFF格式发展而来。在Windows上可执行文件格式为PE格式，而目标文件的格式为COFF，二者略微不同但是大体上相似。同时32bit的windows和64bit的windows的目标二进制文件格式基本相同，只不过64bit中将原PE中的32bit的字段修改为64bit。
&emsp;&emsp;与ELF文件格式类似，PE采用段管理数据与代码，不同的是采用的段名不同，比如windows上的```.code,.data```（改名字根据编译器的不同而不同）。另外，vc中可以通过预编译处理命令指定目标的变量希望存放的段。

## 2 PE的前身——COFF
&emsp;&emsp;为了描述COFF文件格式，下面使用下面简单的程序测试生成的段。
```cpp
int add(int a, int b){
    return a + b;
}

const char *file = "main.o";
int glob_a = 15;
int glob_b;
//test
int main(){
    static char static_a = 16;
    static char static_b;
    long long a = 3;
    long long b;
    add(a, b);
}
```

&emsp;&emsp;将上面的程序通过下面的命令编译后能够得到一个obj文件。windows上可以通过```dumpbin /SUMMARY main.obj```命令查看简要的短信息和```dumpbin /ALL main.obj```查看所有的详细信息。
```bash
## /Za 参数是为了减少编译器行为对结果的干扰
cl /c /Za main.c
```

&emsp;&emsp;下面是通过```Summary```查看的段的内容：
```cpp
          38 .chks64
          14 .data
          68 .debug$S
          18 .drectve
           C .pdata
          41 .text$mn
           8 .xdata
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/coff.drawio.svg)

&emsp;&emsp;下图为COFF文件的基本格式，其基本格式和ELF类似。
**映像文件头**
&emsp;&emsp;首先是映像文件头，该头描述了当前文件的相关内容，相应的结构定义于```winnt.h```文件中的```_IMAGE_FILE_HADER```，能够看到该头包含了下面吉祥内容：
- 当前机器类型；
- 段的数量；
- 文件创建的时间；
- 指向符号表的指针；
- 符号表中符号的数量；
- Header的大小，该字段仅仅在PE文件中使用。
- 标志位。

&emsp;&emsp;上面通过```dumpbin```命令获取的header内容如下，机器类型8664可以在```winnt.h```文件中搜索```IMAGE_FILE_MACHINE_```前缀，就能找到对应的机器类型的表示，我这里查到是Amd64；包含7个段符合上面通过```SUMMARY```参数查看的段的数量；19个符号，后续我们再看下是哪19个符号，如果仅仅看程序的话貌似我们并未定义这么多的符号。
```cpp
FILE HEADER VALUES
            8664 machine (x64)
               7 number of sections
        61ED5A40 time date stamp Sun Jan 23 21:38:08 2022
             27F file pointer to symbol table
              19 number of symbols
               0 size of optional header
               0 characteristics
```

**段表**
&emsp;&emsp;段表是一个数组，数组的每一个元素都是描述一个段的```IMAGE_SECTION_HEADER```的结构体（该结构体的定义同样能够在```winnt.h```中找到），该段描述的基本内容包括：
- ```Name```：段名；
- ```Misc```：该字段是一个```union```，要么是物理地址，要么是该段被加载到内存的大小；
- ```VirtualAdderss```：段被加载到内存后的虚拟地址；
- ```SizeofRawData```：段在文件中的大小往往和Mics中的```VirtualSize``大小不同，比如内存对齐等原因；
- ```Characteristics```：标志位，描述段的读写权限，对齐方式等。
- 其他字段的含义如其名称所描述。

**链接指令**
&emsp;&emsp;使用```dumpbin```命令查看详细内容能够看到如下段描述，首先是和上面段表项中描述的一致每个字段都有相应的内容，最后的```Linker Directives```是传递给连接器的命令，这里的```/DEFAULTLIB:LIBCMT```表示当前段依赖于```libcmt.lib```这个库。
```cpp
SECTION HEADER #1
.drectve name
       0 physical address
       0 virtual address
      18 size of raw data
     12C file pointer to raw data (0000012C to 00000143)
       0 file pointer to relocation table
       0 file pointer to line numbers
       0 number of relocations
       0 number of line numbers
  100A00 flags
         Info
         Remove
         1 byte align

RAW DATA #1
  00000000: 20 20 20 2F 44 45 46 41 55 4C 54 4C 49 42 3A 22     /DEFAULTLIB:"
  00000010: 4C 49 42 43 4D 54 22 20                          LIBCMT" 

   Linker Directives
   -----------------
   /DEFAULTLIB:LIBCMT
```
**调试信息**
&emsp;&emsp;输出的内容可能包含类似下面的段，以debug开头表示调试信息，后面的内容描述具体的调试类型：
- ```.debug$S```：表示包含符号相关的调试信息；
- ```.debug$P```：表示包含预编译头文件相关的调试信息；
- ```.debug$T```：表示包含类型相关的调试信息。

```cpp
SECTION HEADER #2
.debug$S name
       0 physical address
       0 virtual address
      68 size of raw data
     144 file pointer to raw data (00000144 to 000001AB)
       0 file pointer to relocation table
       0 file pointer to line numbers
       0 number of relocations
       0 number of line numbers
42100040 flags
         Initialized Data
         Discardable
         1 byte align
         Read Only
```

**符号表**
&emsp;&emsp;一个完整的符号表如下：
- 第一列为符号的标号，16进制计数；
- 第二列为符号占用的大小；
- 第三列为符号所在的位置，```SECT```接数字表示当前符号所载的段的序号；
- 第四列为符号的类型，分为```notype```和```notype()```分别表示变量、其他符号和函数；
- 第五列表示符号的可见性，```static```表示局部可见，```external```表示外部可见；
- 第六列为符号的名称，如果为段名的符号如```.drectve```后面会继续跟段的一些基本属性如段长度、重定位数、行号数、校验和。

```cpp
COFF SYMBOL TABLE
000 010469A5 ABS    notype       Static       | @comp.id
001 80000190 ABS    notype       Static       | @feat.00
002 00000000 SECT1  notype       Static       | .drectve
    Section length   18, #relocs    0, #linenums    0, checksum        0
004 00000000 SECT2  notype       Static       | .debug$S
    Section length   68, #relocs    0, #linenums    0, checksum        0
006 00000000 SECT3  notype       Static       | .data
    Section length   14, #relocs    1, #linenums    0, checksum 57950DAC
008 00000000 SECT3  notype       External     | file
009 00000008 SECT3  notype       Static       | $SG4309
00A 00000010 SECT3  notype       External     | glob_a
00B 00000004 UNDEF  notype       External     | glob_b
00C 00000000 SECT4  notype       Static       | .text$mn
    Section length   41, #relocs    1, #linenums    0, checksum FEED5EA9
00E 00000000 SECT4  notype ()    External     | add
00F 00000020 SECT4  notype ()    External     | main
010 00000020 SECT4  notype       Label        | $LN3
011 00000000 SECT5  notype       Static       | .xdata
    Section length    8, #relocs    0, #linenums    0, checksum 37887F31
013 00000000 SECT5  notype       Static       | $unwind$main
014 00000000 SECT6  notype       Static       | .pdata
    Section length    C, #relocs    3, #linenums    0, checksum 35DC62C8
016 00000000 SECT6  notype       Static       | $pdata$main
017 00000000 SECT7  notype       Static       | .chks64
    Section length   38, #relocs    0, #linenums    0, checksum        0

String Table Size = 0x1D bytes
```

## 3 PE文件格式
&emsp;&emsp;PE文件格式是COFF文件格式的扩展，基本结构和COFF类似但是添加了额外的一些描述，下面是```winnt.h```中32bit和64bit的文件头的描述。能够看到其中除了COFF的```IMAGE_FILE_HEADER```还额外包含了```Signature```和```OptinalHeader```描述，前者基本固定标识PE文件，后面的对于DLL文件和可执行文件是必须的。另外对比```IMAGE_OPTIONAL_HEADER64```和```IMAGE_OPTIONAL_HEADER32```就能够发现只有部分字段由```DWORD```改成了```ULONGLONG```基本相同（```IMAGE_OPTIONAL_HEADER32```多了一个```BaseOfData```）。

```cpp
typedef struct _IMAGE_NT_HEADERS64 {
    DWORD Signature;
    IMAGE_FILE_HEADER FileHeader;
    IMAGE_OPTIONAL_HEADER64 OptionalHeader;
} IMAGE_NT_HEADERS64, *PIMAGE_NT_HEADERS64;

typedef struct _IMAGE_NT_HEADERS {
    DWORD Signature;
    IMAGE_FILE_HEADER FileHeader;
    IMAGE_OPTIONAL_HEADER32 OptionalHeader;
} IMAGE_NT_HEADERS32, *PIMAGE_NT_HEADERS32;
```

&emsp;&emsp;关于PE按照书里面的节奏不会解释所有的字段，只会关注静态链接需要的字段。Windows系统装载PE可执行文件时，需要很快找到一些装载所需要的数据结构，比如导入表，导出表，资源，重定位表等。这些常用的数据的位置和长度都被保存在了数据目录DataDirectory成员中。这个成员是```IMAGE_DATA_DIRECTORY```结构的一个数组。而每个```_IMAGE_DATA_DIRECTORY```结构体就是一个地址+大小的模式。

```cpp
typedef struct _IMAGE_OPTIONAL_HEADER {
    //省略...
    IMAGE_DATA_DIRECTORY DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES];
} IMAGE_OPTIONAL_HEADER32, *PIMAGE_OPTIONAL_HEADER32;

typedef struct _IMAGE_DATA_DIRECTORY {
    DWORD   VirtualAddress;
    DWORD   Size;
} IMAGE_DATA_DIRECTORY, *PIMAGE_DATA_DIRECTORY;

#define IMAGE_NUMBEROF_DIRECTORY_ENTRIES    16
```

&emsp;&emsp;而这16个表每个表所表示的内容是固定的，索引所对应的表的含义如下所示。
```cpp
#define IMAGE_DIRECTORY_ENTRY_EXPORT          0   // Export Directory
#define IMAGE_DIRECTORY_ENTRY_IMPORT          1   // Import Directory
#define IMAGE_DIRECTORY_ENTRY_RESOURCE        2   // Resource Directory
#define IMAGE_DIRECTORY_ENTRY_EXCEPTION       3   // Exception Directory
#define IMAGE_DIRECTORY_ENTRY_SECURITY        4   // Security Directory
#define IMAGE_DIRECTORY_ENTRY_BASERELOC       5   // Base Relocation Table
#define IMAGE_DIRECTORY_ENTRY_DEBUG           6   // Debug Directory
//      IMAGE_DIRECTORY_ENTRY_COPYRIGHT       7   // (X86 usage)
#define IMAGE_DIRECTORY_ENTRY_ARCHITECTURE    7   // Architecture Specific Data
#define IMAGE_DIRECTORY_ENTRY_GLOBALPTR       8   // RVA of GP
#define IMAGE_DIRECTORY_ENTRY_TLS             9   // TLS Directory
#define IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG    10   // Load Configuration Directory
#define IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT   11   // Bound Import Directory in headers
#define IMAGE_DIRECTORY_ENTRY_IAT            12   // Import Address Table
#define IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT   13   // Delay Load Import Descriptors
#define IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR 14   // COM Runtime descriptor
```