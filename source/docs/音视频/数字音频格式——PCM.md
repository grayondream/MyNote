# 数字音频格式——PCM
## 1 声音
&emsp;&emsp;我们听到的声音是一种波，俗称声波。声波通常是由物体振动激励周围的空气振动，而带动其周围的空气质点振动不断向外传递能量的现象。波分为纵波和横波，纵波表示质点的振动方向和传播方向相同，横波表示质点的振动方向和传播方向垂直。而空气和液体中的声波只能是纵波，固体中一般既有横波又有纵波。
&emsp;&emsp;波的特性由波函数描述，其中比较重要的三个参数为频率、振幅和相位。声波当然不例外，振幅代表着我们听到的声波的音量的大小，而频率代表着音的高低（比如刺耳的声音通常频率高），而相位代表波相对于正弦波或者余弦波在时间上的偏移。当然波也有传递的方向，代表着波前进的方向。
$$
A(t)=A_0sin(2\pi ft + \theta)
$$

## 2 数字音频数据
&emsp;&emsp;声波是模拟信号，为了将模拟信息号数字化一般需要堆模拟信号进行采样、量化和编码。

&emsp;&emsp;模拟信号是连续的，而数字信号是离散的。为了将连续信号离散化，需要每隔一段时间取一个连续信号的一个时间点上的信号值作为当前的数字音频信号。这个每隔一段时间对音频的模拟数据进行选取的过程就是采样，而1s中采样的次数就是采样率。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/sample.png)

### 2.1 采样率
&emsp;&emsp;**采样率（Sample Rate）** 指一秒钟记录的音频点的数据量，通常用HZ（赫兹）来描述，比如44.1kHz表示1s中采样44100次。而音频的一个数据点就是一个数字，该数字如果用float类型表示的话，44.1kHz1s采样的数据就有44100个float数值。根据奈奎斯特采样定理，只要采样率大于等于波的频率的2倍，即波的一个周期中至少采样两个点就能还原出波的原样。一般而言，英文演讲的最低采样率为8kHz，过低的采样率会导致音频中的信息丢失而无法分清一些字符的发音。一般的采样率取值由8000，,1025，,6000,...,44100,...,22579200Hz。

>&emsp;&emsp;现实环境中音频的频率并无法控制，环境中可能既有高频的声音又有低频的声音，采用固定的采样率会部分高频的声音无法满足奈奎斯特采样定理而出现声音信息丢失出现混叠的效应，影响最终的音质。因此为了防止音频中出现可能污染音频的超高频部分，通常音频到数字转换器之间会有低通滤波器来过滤高频声音。

>&emsp;&emsp;为什么通常的标准采样率为44.1kHz?
&emsp;&emsp;人类能够听到的声音的频率范围为20~20kHz，因此为了保证声音不会有明显的缺陷采样率最小为40kHz。而将采样率提高到44.1kHz是技术原因，44.1kHz的采样率的实现只需要一般的低通滤波器就能避免混叠效应，记录更好的音频。

>&emsp;&emsp;越高的采样率越好？
&emsp;&emsp;完全取决于需求，越高的采样率意味着声音更加丰富，对于一些关注于高音的专业人士缺少高音是一种缺陷，但是对于普通人可能并无法分辨二者的区别。同时，越高的采样率对设备的要求越高，且文件的大小越大。

### 2.2 采样位深、通道数和比特率
&emsp;&emsp;模拟音频是一种连续波，采样时取的一个点是对实际的数据的截断。采样位深就是表示一个采样点的音频数据的bit数，越大的位深数据的精度越高，音质也越高。比如真实数据可能是10.134545，而采样的数据为10，更大的位深可表示的数据可截断到10.134。通常的位深由16bit，24bit，32bit。
&emsp;&emsp;音频可能不止有个音频源，可能由2个或更多的音频源，即不同的通道，不同通道的音频是独立的。
&emsp;&emsp;比特率是由就是通道、采样率和位深的乘积，而一段音频的数据大小就是时间x比特率。
$$
bitrate=samplerate\times channel \times bit depth
$$

## 3 PCM
&emsp;&emsp;PCM(Pulse Code Modulation，脉冲编码调制)音频数据是未经压缩的音频采样数据裸流，它是由模拟信号经过采样、量化、编码转换成的标准数字音频数据。PCM数据除了需要关注音频的采样率、位深和通道数外，数据的存储方式是否有符号、存储的字节序、整形还是浮点数据以及不同通道的存储方式完整的表达了PCM的数据形式。FFmpeg中PCM的数据格式分为两种Packed即声道交错存储（LRLRLRLRLR）和Planner存储LLLLRRRR。

>&emsp;&emsp;PCM数据是脉冲数据索引，PCM的静音数据就是不变的值和值多大没有任何关系。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/planner.drawio.svg)

&emsp;&emsp;下面是FFmpeg支持的PCM数据的格式
```c
enum AVSampleFormat {
    AV_SAMPLE_FMT_NONE = -1,
    AV_SAMPLE_FMT_U8,          ///< unsigned 8 bits
    AV_SAMPLE_FMT_S16,         ///< signed 16 bits
    AV_SAMPLE_FMT_S32,         ///< signed 32 bits
    AV_SAMPLE_FMT_FLT,         ///< float
    AV_SAMPLE_FMT_DBL,         ///< double

    AV_SAMPLE_FMT_U8P,         ///< unsigned 8 bits, planar
    AV_SAMPLE_FMT_S16P,        ///< signed 16 bits, planar
    AV_SAMPLE_FMT_S32P,        ///< signed 32 bits, planar
    AV_SAMPLE_FMT_FLTP,        ///< float, planar
    AV_SAMPLE_FMT_DBLP,        ///< double, planar
    AV_SAMPLE_FMT_S64,         ///< signed 64 bits
    AV_SAMPLE_FMT_S64P,        ///< signed 64 bits, planar

    AV_SAMPLE_FMT_NB           ///< Number of sample formats. DO NOT USE if linking dynamically
};

 DE alaw            PCM A-law
 DE f32be           PCM 32-bit floating-point big-endian
 DE f32le           PCM 32-bit floating-point little-endian
 DE f64be           PCM 64-bit floating-point big-endian
 DE f64le           PCM 64-bit floating-point little-endian
 DE mulaw           PCM mu-law
 DE s16be           PCM signed 16-bit big-endian
 DE s16le           PCM signed 16-bit little-endian
 DE s24be           PCM signed 24-bit big-endian
 DE s24le           PCM signed 24-bit little-endian
 DE s32be           PCM signed 32-bit big-endian
 DE s32le           PCM signed 32-bit little-endian
 DE s8              PCM signed 8-bit
 DE u16be           PCM unsigned 16-bit big-endian
 DE u16le           PCM unsigned 16-bit little-endian
 DE u24be           PCM unsigned 24-bit big-endian
 DE u24le           PCM unsigned 24-bit little-endian
 DE u32be           PCM unsigned 32-bit big-endian
 DE u32le           PCM unsigned 32-bit little-endian
 DE u8              PCM unsigned 8-bit
```

&emsp;&emsp;我们可以写一个简单的44100采样率，双通道，plann存储格式的正弦PCM数据。
```c
void write_sine_pcm_8bit_planner(const char *file, int sample_rate, int seconds, int channel) {
	if (file != nullptr && seconds > 0 && sample_rate > 0 && channel > 0) {
		size_t len = sample_rate * channel * seconds;
		uint8_t *pdata = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * len));
		uint8_t *pcur = pdata;
		for (int c = 0; c < channel; c++) {
			pcur = pdata + c * sample_rate * seconds * sizeof(uint8_t);
			int val = 0;
			for (int i = 0; i < len / channel; i ++) {
				val = sin(i) * 100.0;
				memset(pcur, val,  sizeof(uint8_t));
				pcur += sizeof(uint8_t);
			}
		}

		FILE *fp = fopen(file, "wb");
		if (fp != nullptr) {
			fwrite(pdata, len * sizeof(uint8_t), 1, fp);
			fclose(fp);
		}
	}
}
```

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/pcm.png)