# FFmpeg解码转码播放

## 1 ffmpeg解码大致流程
&emsp;&emsp;下图是ffmpeg解码播放音视频的基本流程：
- 首先是网络媒体解协议，解协议之后得到对应的媒体文件比如mp4，ts等，这些格式是媒体文件的封装格式，也就是将音频，视频，字幕等码流编码后打包到一起的格式；
- 之后就是对容器进行解封装，解封装能够分别得到对应的流的编码流，比如视频可能是h264码流，音频可能是aac码流，这些都是对应的流经过编码后的数据；
- 再然后就是需要将编码的流解码为裸流，视频的裸流一般为YUV或者RGB数据，而音频的裸流一般为PCM数据，这些数据也是播放时时真正用到的数据；
- 因为音频和视频文件大小差距比较大解码速度也不一样，一般音频比较小解码比较快，因此会存在二者不同步的情况。也就是说经过相同时间段得到的音频和视频不同步，如果将解码得到的数据立刻输送到播放设备就会出现任务说话口型不对等不同的问题，因此需要音画同步；
- 最后就是经过同步的数据送到对应的设备中播放。


![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/decode.drawio.svg)

## 2 解复用
### 2.1 解复用音频和视频
&emsp;&emsp;使用ffmpeg解复用相对比较简单，直接利用对应的api从文件中读取一个又一个packet然后写入到文件即可。下面代码解复用的基本流程是：
- 使用```avformat_open_input```打开媒体文件，如果失败的话返回值小于0，ffmpeg中小于0的返回值都表示失败，可以使用相应的api比如```av_strerror```获取失败的信息；
- ```avformat_find_stream_info```用于搜索文件中的码流，比如音频流，视频流，字幕流；
- ```av_dump_format```就是将搜索到的信息输出到标准输出中，可以不适用，写上方便调试；
- ```av_find_best_stream```用于查找当前媒体中对应的流，因为有些时候媒体中可能由多个字幕流或者音频流（看视频时选择不同的音轨或者字幕），当然也可以自己遍历```pformat->streams```搜寻自己需要的流；
- 之后便是不断循环调研```av_read_frame```获取对应的编码的流的数据，每个编码的流的数据存储在```AVPacket```中，解码的时候解码的就是这里的数据。
- 最后程序结束时需要记得释放相关的context。

&emsp;&emsp;下面的程序时解复用视频的内容，解复用音频的相同只是将```av_find_best_stream(pformat, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0)```替换成```av_find_best_stream(pformat, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0)```即可，字幕流同理。

```cpp
#define AVERR_BREAK_RET_LT_ZERO(ret) if((ret) < 0){ break; }

//ffmpeg -i ../video/1.mp4 -codec copy -f h264 output.h264
int demux_video(const char *file, const char *ofile) {
	AVFormatContext *pformat = nullptr;
	int ret = 0;
	AVPacket *packet = nullptr;
	FILE *fp = nullptr;
	do 
	{
		//open file format
		ret = avformat_open_input(&pformat, file, 0, NULL);
		AVERR_BREAK_RET_LT_ZERO(ret);
		
		ret = avformat_find_stream_info(pformat, NULL);
		AVERR_BREAK_RET_LT_ZERO(ret);

		av_dump_format(pformat, 0, file, 0);
		int videid = av_find_best_stream(pformat, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
		AVERR_BREAK_RET_LT_ZERO(videid);
		
		fprintf(stderr, "video id is %d\n", videid);

		fp = fopen(ofile, "wb");
		AVERR_BREAK_IF_TRUE(!fp);

		packet = av_packet_alloc();
		while (true) {
			ret = av_read_frame(pformat, packet);
			AVERR_BREAK_RET_LT_ZERO(ret);
			if (packet->stream_index == videid) {
				int writelen = fwrite(packet->data, 1, packet->size, fp);
				fprintf(stdout, "stream id is %d, read packet pts is %d, packet size is %d, write size is %d\n", packet->stream_index, packet->pts, packet->size, writelen);
			}

			av_packet_unref(packet);
		}

		
	} while (false);

	if (ret < 0) {
		char errorbuff[256] = { 0 };
		av_strerror(ret, errorbuff, 256);
		fprintf(stderr, "error message: %s", errorbuff);
	}

	if (fp) {
		fclose(fp);
	}

	if (packet) {
		av_packet_free(&packet);
		packet = nullptr;
	}

	if (pformat) {
		avformat_close_input(&pformat);
		pformat = nullptr;
	}

	return 0;
}
```

&emsp;&emsp;利用上面的程序解复用出来的aac或者h264或者其他格式的码流是无法直接播放的，或者用命令```ffmpeg -i ../video/1.mp4 -codec copy -f h264 output.h264```和```ffmpeg -i ..\video\1.mp4 -acodec aac output.aac```生成对应的码流会发现自己解复用的数据大小和这里解复用得到的文件大小不同。具体原因可参考，简单的说来就是写入的是编码后的视频或者音频的裸流没有相关的header信息，导致无法播放。

- [使用FFMPEG类库分离出多媒体文件中的音频码流](https://blog.csdn.net/leixiaohua1020/article/details/11800791)
- [使用FFMPEG类库分离出多媒体文件中的H.264码流](https://blog.csdn.net/leixiaohua1020/article/details/11800877)
- [11.FFmpeg学习笔记 - 解复用分离aac码流](https://blog.csdn.net/whoyouare888/article/details/94383440)
- [使用FFMPEG类库分离出多媒体文件中的H.264码流](https://www.cnblogs.com/jiu0821/p/9101985.html)

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/mp42h264.png)

> 现在对h264和aac的格式不是很熟悉后续熟悉了再研究些怎么写可播放的aac文件和264文件。

## 3 音视频解码与转码
&emsp;&emsp;得到解封装后的码流之后就需要对对应的编码数据进行解码，解码后就能得到视频数据的码流。有些时候解码出来的数据流并不能直接被被视频设备或者扬声器所使用的，因此需要将解码出来的数据进行转码。视频数据的话一般转码成YUV420P和对应的宽高，而音频数据一般转码为PCM数据。

### 3.1 视频解码转码
&emsp;&emsp;视频数据转码一般转码成YUV420P和对应的宽高，也就是进行缩放。视频解码和转码的基本流程之前需要堆视频容器进行解封装，解封装的流程和之前完全相同。额外需要做的工作就是打开解码器和将得到的编码的packet解码成frame数据，和最后的刷新帧缓冲。
```cpp
int decode_transcode_video(const char *file, const char *ofile, int owidth, int oheigt) {
    //省略代码....
    do
    {
        //省略代码....
        fprintf(stderr, "video id is %d\n", videid);
        param = pformat->streams[videid]->codecpar;
        ret = open_decoder(param, &pdecoderCtx);
        AVERR_BREAK_RET_LT_ZERO(ret);

        packet = av_packet_alloc();
        while (true) {
            ret = av_read_frame(pformat, packet);
            AVERR_BREAK_RET_LT_ZERO(ret);
            if (packet->stream_index == videid) {
                //decode video to yuv420p
                decode_video(packet, pdecoderCtx, owidth, oheigt, fp);
            }

            av_packet_unref(packet);
        }


    } while (false);

    //flush the buffer
    packet->data = nullptr;
    packet->size = 0;
    decode_video(packet, pdecoderCtx, owidth, oheigt, fp);
    //省略代码....
    return 0;
}
```
&emsp;&emsp;**打开解码器**。打开解码器的过程比较简单，主要是将从stream中获取到的参数```AVCodecParameters```拷贝到解码器的context中，然后利用ffmpeg提供的API将解码器打开即可。
```cpp
int open_decoder(AVCodecParameters *param, AVCodecContext ** pdecoderCtx) {
	if (!param || !pdecoderCtx) {
		return -1;
	}

	AVCodec *pdecoder = avcodec_find_decoder(param->codec_id);
	if (!pdecoder) {
		return -1;
	}

	*pdecoderCtx = avcodec_alloc_context3(pdecoder);
	if (!(*pdecoderCtx)) {
		return -1;
	}

	int ret = avcodec_parameters_to_context(*pdecoderCtx, param);
	if (ret < 0) {
		return ret;
	}

	ret = avcodec_open2(*pdecoderCtx, pdecoder, NULL);
	return ret;
}
```
&emsp;&emsp;**解码帧数据**。解码数据ffmpeg提供了比较好理解的接口```avcodec_send_packet```和```avcodec_receive_frame```从函数名上我们就能很容易明白其作用是什么。另外转码使用的是```sws_scale```进行转码，需要注意的是转码出来的frame的内存即```pbuffer```需要我们自己维护，ffmpeg提供了计算需要内存大小的函数```av_image_get_buffer_size```。其他需要注意的点是：
- 调用```avcodec_send_packet```可能返回错误```EAGAIN```表示资源未准备好，因此需要重复将packet送入解码器直到环境准备好为止；
- 一般来说```avcodec_receive_frame```获取的是一帧的数据，但是有些时候会存在空帧之类的数据，因此需要```while```循环尝试取多帧。对于视频数据一般在刷新缓冲区时```while```会进入多次，一般都是一次。

```cpp
int decode_video(AVPacket *packet, AVCodecContext *pdecoderCtx, int owidth, int oheight, FILE *fp) {
	AVPixelFormat outFormat = AV_PIX_FMT_YUV420P;
	int ret = 0;
	AVFrame *pdecodeFrame = av_frame_alloc();
	AVFrame *ptranscodedFrame = av_frame_alloc();
	SwsContext *psws = nullptr;

	size_t size = av_image_get_buffer_size(outFormat, owidth, oheight, 1);
	uint8_t *pbuffer = (uint8_t *)malloc(size);
	static int count = 0;
	do 
	{
		ret = av_image_fill_arrays(ptranscodedFrame->data, ptranscodedFrame->linesize, pbuffer, outFormat, owidth, oheight, 1);
		AVERR_BREAK_RET_LT_ZERO(ret);

		while (AVERROR(EAGAIN) == (ret = avcodec_send_packet(pdecoderCtx, packet))) {}
		AVERR_BREAK_RET_LT_ZERO(ret);

		while (0 <= (ret = avcodec_receive_frame(pdecoderCtx, pdecodeFrame))) {
			psws = sws_getCachedContext(psws, pdecoderCtx->width, pdecoderCtx->height, pdecoderCtx->pix_fmt, owidth, oheight, outFormat, SWS_BICUBIC, 0, 0, 0);
			AVERR_BREAK_IF_TRUE(!psws);
			ret = sws_scale(psws, pdecodeFrame->data, pdecodeFrame->linesize, 0, pdecodeFrame->height, ptranscodedFrame->data, ptranscodedFrame->linesize);
			AVERR_BREAK_RET_LT_ZERO(ret);
			fprintf(stderr, "%d, transcode frame %d : %d, %d into %d, %d\n", count, pdecodeFrame->pts, pdecoderCtx->width, pdecoderCtx->height, owidth, oheight);

			fwrite(pbuffer, 1, size, fp);
			count++;
		}
	} while (false);

	if (psws) {
		sws_freeContext(psws);
	}

	if (pdecodeFrame) {
		av_frame_free(&pdecodeFrame);
	}

	if (ptranscodedFrame) {
		av_frame_free(&ptranscodedFrame);
	}

	if (pbuffer) {
		free(pbuffer);
	}

	return ret;
}
```
&emsp;&emsp;在解码完成之后，ffmpeg的缓冲区内部可能在存在多帧数据，因此需要刷新缓冲区，刷新的方式就是将packet的数据指针和size设为0然后解码即可。
```cpp
    packet->data = nullptr;
    packet->size = 0;
    decode_video(packet, pdecoderCtx, owidth, oheigt, fp);
```

&emsp;&emsp;解码转码完成之后，我们可以使用下面的命令对比下ffmpeg解码出来的数据大小和我们解码出来的数据大小的区别。
```bash
#通过提取yuv
ffmpeg -i ../video/1.mp4 -s 100x100 -pix_fmt yuv420p ./100x100_yuv420p_f.yuv
#播放yuv数据
 ffplay -pixel_format yuv420p -video_size 100x100  ./100x100_yuv420p.yuv
```
&emsp;&emsp;下面是ffmpeg解码（100x100_yuv420p_f.yuv）、程序解码（100x100_yuv420p.yuv）和未刷新缓冲区（100x100_yuv420p_unflush.yuv）的大小，可以看到未刷新的情况下少了刚好一帧30000bytes。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/yuvdecode.png)

### 3.2 音频解码重采样
&emsp;&emsp;音频解码为了方便播放，一般会进行重采样。音频解码和重采样和视频的解码重采样过程类似，具体步骤不同之处是音频使用的是 ```SwrContext``` 进行重采样。主体代码框架类似。
&emsp;&emsp;
```cpp
//省略代码...
		pswr = swr_alloc_set_opts(pswr, outLayout, oformat, orate, pdecoderCtx->channel_layout, pdecoderCtx->sample_fmt, pdecoderCtx->sample_rate, 0, 0);
		AVERR_BREAK_IF_TRUE(!pswr);

		ret = swr_init(pswr);
		AVERR_BREAK_RET_LT_ZERO(ret);

		packet = av_packet_alloc();
		while (true) {
			ret = av_read_frame(pformat, packet);
			AVERR_BREAK_RET_LT_ZERO(ret);
			if (packet->stream_index == audioid) {
				//decode video to yuv420p
				decode_audio(packet, pdecoderCtx, orate, outLayout, oformat, pswr, fp);
			}

			av_packet_unref(packet);
		}
//省略代码...
//刷新缓冲
```
&emsp;&emsp;解码和重采样的结构和视频类似。唯一需要注意的是音频处理采样前后的各种参数的转换：
- ```av_rescale_rnd```和```swr_get_delay```用来计算音频采样后的sample数量；
- ```av_samples_get_buffer_size```用来计算整块缓冲区的大小也可以自己计算，具体大小为通道数×每个sample的字节数×sample数量；
- ```av_samples_alloc```申请内存；
- ```swr_alloc_set_opts```、```swr_init```分别对重采样的context创建和设置参数、初始化，```swr_convert```进行重采样；
- 需要注意的是写的时候需要使用```swr_convert```的返回值，该返回值返回的是真实的sample的数量，如果使用之前计算的sample数量会出现比较大的噪音。
- ```swr_convert(pswr, ptranscodedFrame->data, dstnbSamples, NULL, 0)```是为了刷新缓冲，详情见[音频重采样swr_convert尾部丢帧的问题处理](https://blog.csdn.net/yzb408/article/details/90378807)。

```cpp
//ffplay -ar 44100 -channels 2 -f f32le -i .\44100_2_s16_pcm.pcm
int decode_audio(AVPacket *packet, AVCodecContext *pdecoderCtx, int outRate, int outLayout, AVSampleFormat outFormat, SwrContext *pswr, FILE *fp) {
	int ret = 0;
	AVFrame *pdecodeFrame = av_frame_alloc();
	AVFrame *ptranscodedFrame = av_frame_alloc();
	int outChanel = av_get_channel_layout_nb_channels(outLayout);
	static int count = 0;
	do
	{
		//ret = avcodec_send_packet(pdecoderCtx, packet);
		while (AVERROR(EAGAIN) == (ret = avcodec_send_packet(pdecoderCtx, packet))) {}
		AVERR_BREAK_RET_LT_ZERO(ret);
		while (0 <= (ret = avcodec_receive_frame(pdecoderCtx, pdecodeFrame))) {
			int dstnbSamples = av_rescale_rnd(swr_get_delay(pswr, pdecoderCtx->sample_rate) + pdecodeFrame->nb_samples, outRate, pdecodeFrame->sample_rate, AV_ROUND_UP);
			AVERR_BREAK_RET_LT_ZERO(dstnbSamples);

			//pbuffer = (uint8_t *)malloc(size * 2);
			//ret = av_samples_fill_arrays(ptranscodedFrame->data, ptranscodedFrame->linesize, pbuffer, outChanel, dstnbSamples, outFormat, 1);
			ret = av_samples_alloc(ptranscodedFrame->data, ptranscodedFrame->linesize, outChanel, dstnbSamples, outFormat, 0);
			AVERR_BREAK_RET_LT_ZERO(ret);

			ret = swr_convert(pswr, ptranscodedFrame->data, dstnbSamples, (const uint8_t **)pdecodeFrame->data, pdecodeFrame->nb_samples);
			AVERR_BREAK_RET_LT_ZERO(ret);

			count++;

			int writingSize = av_samples_get_buffer_size(ptranscodedFrame->linesize, outChanel, ret, outFormat, 1);
			//fwrite(pbuffer, 1, writingSize, fp);
			fwrite(ptranscodedFrame->data[0], 1, writingSize, fp);
			//刷新缓冲
			while (0 < (ret = swr_convert(pswr, ptranscodedFrame->data, dstnbSamples, NULL, 0))) {
				writingSize = av_samples_get_buffer_size(ptranscodedFrame->linesize, outChanel, ret, outFormat, 1);
				fwrite(ptranscodedFrame->data[0], 1, writingSize, fp);
			}
			
			av_freep(ptranscodedFrame->data);
		}
	} while (false);

	av_freep(ptranscodedFrame->data);

	if (pdecodeFrame) {
		av_frame_free(&pdecodeFrame);
	}

	if (ptranscodedFrame) {
		av_frame_free(&ptranscodedFrame);
	}

	if (ret < 0) {
		char errorbuff[256] = { 0 };
		av_strerror(ret, errorbuff, 256);
		//fprintf(stderr, "error message: %s\n", errorbuff);
	}

	return ret;
}
```
&emsp;&emsp;解码完成后可以使用下列命令测试下具体的数据对比。
```bash
#播放pcm数据
ffplay -ar 44100 -channels 2 -f s16le -i .\44100_2_s16_pcm.pcm
#提取pcm数据
ffmpeg -i ../video/1.mp4 -f s16le -ac 2 -ar 44100 44100_2_s16_f.pcm
```

&emsp;&emsp;如果不刷新缓冲区音频文件就和ffmpeg解码出来的大小不一样。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/pcm_resample.png)

## 4 播放音频和视频
### 4.1 使用opengl播放视频YUV数据
&emsp;&emsp;下面是主题框架，基本上和之前的解码YUV数据的流程相同，不同的是需要额外添加初始化OpenGL环境和创建OpenGL窗口的代码，即```gl_window_initialize```和```gl_build_program```两个函数，以及处理键盘事件的```processKeyboard```。另外需要提及的细节是变量声明中窗口大小和视频显示大小不一致，是采用```ratio```这个变量进行控制的，```0.9```表示视频显示大小是窗口大小的90%。
```cpp
int gl_display_file(const char *infile) {
	//省略代码...
	int winWidth = 640, winHeight = 480;
	double ratio = 0.9;
	int owidth = winWidth * ratio, oheight = winHeight * ratio;
	GLFWwindow *pwindow = nullptr;
	unsigned int sharedId = 0, VAO;
	GLuint texs[3] = {0};
	
	do
	{
		ret = gl_window_initialize(&pwindow, winWidth, winHeight);
		AVERR_BREAK_RET_LT_ZERO(ret);

		ret = gl_build_program(pwindow, sharedId, VAO, &texs, ratio);
		AVERR_BREAK_RET_LT_ZERO(ret);

		//open file format
		//省略代码...
		while (true && !glfwWindowShouldClose(pwindow)) {
			processKeyboard(pwindow);
			ret = av_read_frame(pformat, packet);
			AVERR_BREAK_RET_LT_ZERO(ret);
			if (packet->stream_index == videid) {
				//decode video to yuv420p and display it by opengl
				gl_display_frame(pwindow, sharedId,packet, pdecoderCtx, owidth, oheight, VAO, texs);
			}

			av_packet_unref(packet);
		}


	} while (false);

	//flush the buffer
	//省略代码...

	glfwTerminate();
	return 0;
}
```

&emsp;&emsp;```processKeyboard```处理键盘事件，```gl_window_initialize```就是常规的opengl初始化创建窗口函数，基本流程都一致，可以参考opengl learn。
```cpp
void processKeyboard(GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

int gl_window_initialize(GLFWwindow** window, int width, int heigth) {
	// glfw: initialize and configure
		// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	*window = glfwCreateWindow(width, heigth, "Hello FFmpeg", NULL, NULL);
	if (*window == NULL)
	{
		fprintf(stderr, "Failed to create GLFW window\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(*window);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
		fprintf(stderr, "Failed to initialize GLAD\n");
		return -1;
	}

	return 0;
}
```
&emsp;&emsp;```gl_build_program```中创建顶点着色器和片段着色器，用来渲染图像。这里的场景主要是渲染2D图像比较简单，仅仅需要四个顶点也就是两个三角形就能实现，因此verties中定义了两个三角形的坐标顶点。 另外需要简单解释的是顶点着色器就是将输入的图像的坐标传递给对应的点，而片段着色器中将YUV数据转换成了RGB，其中```yuv2r,yuv2g,yuv2b```三个矩阵就是对应的转换矩阵。

```cpp

void gl_uniform_setbool(unsigned int id, const std::string &name, const bool val) {
	glUniform1i(glGetUniformLocation(id, name.c_str()), static_cast<int>(val));
}

void gl_uniform_setint(unsigned int id, const std::string &name, const int val) {
	glUniform1i(glGetUniformLocation(id, name.c_str()), static_cast<int>(val));
}

void gl_uniform_setfloat(unsigned int id, const std::string &name, const float val) {
	glUniform1f(glGetUniformLocation(id, name.c_str()), static_cast<float>(val));
}

int gl_build_program(GLFWwindow *pwindow, unsigned int &sharedID, unsigned int &VAO, GLuint (*texs)[3], double ratio) {
	const char *vertexSource = "#version 330 core\n\
								layout(location = 0) in vec3 aPos;\n\
								layout(location = 1) in vec2 aTexCoord;\n\
								out vec2 TexCoord;\n\
								void main(){\n\
									gl_Position = vec4(aPos, 1.0);\n\
									TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n\
								}\n";

	const char *fragmentSource = "#version 330 core\n\
								out vec4 FragColor;\n\
								in vec2 TexCoord;\n\
								// texture sampler\n\
								uniform sampler2D textureY;\n\
								uniform sampler2D textureU;\n\
								uniform sampler2D textureV;\n\
								void main(){\n\
									vec3 yuv, rgb;\n\
									vec3 yuv2r = vec3(1.164, 0.0, 1.596);\n\
									vec3 yuv2g = vec3(1.164, -0.391, -0.813);\n\
									vec3 yuv2b = vec3(1.164, 2.018, 0.0);\n\
									yuv.x = texture(textureY, TexCoord).r - 0.0625; \n\
									yuv.y = texture(textureU, TexCoord).r - 0.5; \n\
									yuv.z = texture(textureV, TexCoord).r - 0.5; \n\
									rgb.x = dot(yuv, yuv2r); \n\
									rgb.y = dot(yuv, yuv2g); \n\
									rgb.z = dot(yuv, yuv2b); \n\
									FragColor = vec4(rgb, 1.0);\n\
								}";
	//需要显示的矩形区域
	float vertices[] = {
		 ratio,  ratio, 0.0,     1.0, 0.0,
		 ratio, -1 * ratio, 0.0,     1.0, 1.0,
		-1 * ratio, -1 * ratio, 0.0,     0.0, 1.0,
		-1 * ratio,  ratio, 0.0,     0.0, 0.0,
	};

	unsigned int indices[] = {
		0, 1, 3,
		1, 2, 3
	};

	unsigned int VBO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1,
		3,
		GL_FLOAT,
		GL_FALSE,
		5 * sizeof(float),
		(void *)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glGenTextures(3, *texs);
	for (int i = 0; i < 3; i++) {
		glBindTexture(GL_TEXTURE_2D, (*texs)[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	//create and compile shader
	unsigned int vertex, fragment;
	int success;

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vertexSource, nullptr);
	glCompileShader(vertex);
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar infoLog[1024];
		glGetShaderInfoLog(vertex, 1024, nullptr, infoLog);
		std::cerr << "compile fail: " << infoLog << std::endl;
		assert(0);
	}

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fragmentSource, nullptr);
	glCompileShader(fragment);
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar infoLog[1024];
		glGetShaderInfoLog(fragment, 1024, nullptr, infoLog);
		std::cerr << "compile fail: " << infoLog << std::endl;
		assert(0);
	}

	sharedID = glCreateProgram();
	glAttachShader(sharedID, vertex);
	glAttachShader(sharedID, fragment);
	glLinkProgram(sharedID);
	glGetProgramiv(sharedID, GL_LINK_STATUS, &success);
	if (!success) {
		assert(0);
	}

	glDeleteShader(vertex);
	glDeleteShader(fragment);
	glUseProgram(sharedID);

	gl_uniform_setint(sharedID, "textureY", 0);
	gl_uniform_setint(sharedID, "textureU", 1);
	gl_uniform_setint(sharedID, "textureV", 2);

	return 0;
}
```

&emsp;&emsp;最后就是解码读数据将数据传递给opengl以及等待定时器。
```cpp
int gl_display_frame(GLFWwindow *pwindow, unsigned int sharedId, AVPacket *packet, AVCodecContext *pdecoderCtx, int owidth, int oheight, unsigned int VAO, GLuint texs[3]) {
	//省略代码...
	do{
		//省略代码...
		while (0 <= (ret = avcodec_receive_frame(pdecoderCtx, pdecodeFrame))) {
			//省略代码...
			fprintf(stderr, "%d, transcode frame %d : %d, %d into %d, %d\n", count, pdecodeFrame->pts, pdecoderCtx->width, pdecoderCtx->height, owidth, oheight);
			ptranscodedFrame->width = owidth;
			ptranscodedFrame->height = oheight;
			gl_display_frame(pwindow, sharedId, ptranscodedFrame, VAO, texs);
			count++;
			AVRational timebase = pdecoderCtx->framerate;
			gl_wait_for_next(timebase, count);
		}	
	} while (false);

	//省略代码...
```
&emsp;&emsp;送数据到opengl上的代码比较简单就是将YUV三个分量的数据绘制到opengl中，并刷新缓冲区。
```cpp
int gl_display_frame(GLFWwindow *pwindow, unsigned int sharedId, AVFrame *pframe, unsigned int VAO, GLuint texs[3]) {
	//背景色
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texs[0]);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, pframe->linesize[0]);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RED,
		pframe->width,
		pframe->height,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		pframe->data[0]);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texs[1]);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, pframe->linesize[1]);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RED,
		pframe->width / 2,
		pframe->height / 2,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		pframe->data[1]);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, texs[2]);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, pframe->linesize[2]);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RED,
		pframe->width / 2,
		pframe->height / 2,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		pframe->data[2]);

	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	glUseProgram(sharedId);
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glfwSwapBuffers(pwindow);
	glfwPollEvents();
	return 0;
}
```

&emsp;&emsp;等待这里做的比较简单就是根据帧率计算时间差然后等待直到到达足够的时间继续播放。
```cpp
void gl_wait_for_next(AVRational framerate, int pts) {
	static bool first_frame = true;
	if (first_frame) {
		glfwSetTime(0.0);
		first_frame = false;
	}

	double pt_in_seconds = pts * (double)framerate.den / (double)framerate.num;
	while (pt_in_seconds > glfwGetTime()) {
		glfwWaitEventsTimeout(pt_in_seconds - glfwGetTime());
	}
}
```
### 4.2 使用openal播放音频
&emsp;&emsp;音频播放使用OpenAL播放，基本框架和video差不多，额外的工作就是创建OpenAL的context，打开播放设备等工作。
```cpp
int al_play_audio(const char *infile) {
	//省略代码...
	ALCdevice *pdevice = nullptr;
	ALCcontext *pcontext = nullptr;
	ALuint sourceId = 0;
	do
	{
		ret = al_context_create(&pdevice, &pcontext, &sourceId);
		AVERR_BREAK_RET_LT_ZERO(ret);
		//省略代码...
		while (true) {
			ret = av_read_frame(pformat, packet);
			AVERR_BREAK_RET_LT_ZERO(ret);
			if (packet->stream_index == audioid) {
				//decode video to yuv420p
				al_play_audio(packet, pdecoderCtx, orate, outLayout, oformat, pswr, sourceId);
			}

			av_packet_unref(packet);
		}
	} while (false);

	//flush the buffer
	packet->data = nullptr;
	packet->size = 0;
	al_play_audio(packet, pdecoderCtx, orate, outLayout, oformat, pswr, sourceId);
	//flush_audio_sample_buffer(pdecoderCtx, orate, fp);
	al_waitfor_end(sourceId);
	//省略代码...

	alcMakeContextCurrent(NULL);
	alcDestroyContext(pcontext);
	alcCloseDevice(pdevice);
	return 0;
}
```
&emsp;&emsp;创建AL环境和打开设备的代码比较固定，看下就好，另外注意下```AL_LOOPING```设为```False```，具体原因在下面。
```cpp
int al_context_create(ALCdevice **pdevice, ALCcontext **pcontext, ALuint *sourceId) {
	//播放源的位置
	ALfloat position[] = { 0.0f,0.0f,0.0f };
	//播放的速度
	ALfloat velocity[] = { 0.0f,0.0f,0.0f };

	*pdevice = alcOpenDevice(nullptr);
	if (!(*pdevice)) {
		return -1;
	}

	*pcontext = alcCreateContext(*pdevice, NULL);
	if (!(*pcontext)) {
		return -1;
	}

	alcMakeContextCurrent(*pcontext);

	alGenSources(1, sourceId);
	//音高倍数
	alSourcef(*sourceId, AL_PITCH, 1.0f);
	//声音的增益
	alSourcef(*sourceId, AL_GAIN, 1.0f);
	//设置位置
	alSourcefv(*sourceId, AL_POSITION, position);
	//设置声音移动速度
	alSourcefv(*sourceId, AL_VELOCITY, velocity);
	//设置是否循环播放
	alSourcei(*sourceId, AL_LOOPING, AL_FALSE);

	return 0;
}
```
&emsp;&emsp;```al_play_audio```里的代码比较简单就是将之前使用的解码代码中写入文件的函数替换即可。
```cpp
//ffplay -ar 44100 -channels 2 -f f32le -i .\44100_2_s16_pcm.pcm
int al_play_audio(AVPacket *packet, AVCodecContext *pdecoderCtx, int outRate, int outLayout, AVSampleFormat outFormat, SwrContext *pswr, ALuint sourceId) {
	//省略代码...
	do
	{
		//ret = avcodec_send_packet(pdecoderCtx, packet);
		while (AVERROR(EAGAIN) == (ret = avcodec_send_packet(pdecoderCtx, packet))) {}
		AVERR_BREAK_RET_LT_ZERO(ret);
		while (0 <= (ret = avcodec_receive_frame(pdecoderCtx, pdecodeFrame))) {
			//省略代码...
			al_play_raw_pcm(sourceId, AL_FORMAT_STEREO16, pdecoderCtx->sample_rate, ptranscodedFrame->data[0], writingSize);

			while (0 < (ret = swr_convert(pswr, ptranscodedFrame->data, dstnbSamples, NULL, 0))) {
				//省略代码...
				//fwrite(ptranscodedFrame->data[0], 1, writingSize, fp);
				al_play_raw_pcm(sourceId, AL_FORMAT_STEREO16, pdecoderCtx->sample_rate, ptranscodedFrame->data[0], writingSize);
			}
			av_freep(ptranscodedFrame->data);
		}
	} while (false);

	//省略代码...
	return ret;
}
```
&emsp;&emsp;下面就是实际上播放音频的部分，代码也很简单就是将数据attach到OpenAL的buffer上，内部会有一个队列维护，不断播放数据，这里做的比较简单粗暴将每次送入的数据都压入队列。一般音频数据比较小不会很占内存，当然也可以通过```alGetSourcei(source, AL_BUFFERS_QUEUED, &queued)```和```alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);```获取队列中的buffer数量和已经处理的数量，然后通过```alSourceUnqueueBuffers```删除队列中的buffer来控制内存占用。

```cpp
void al_play_raw_pcm(ALuint sourceId, ALenum format, ALsizei rate, uint8_t *pbuffer, int buffersize) {
	ALuint bufferID = 0;
	alGenBuffers(1, &bufferID); // 创建buffer并获取bufferID
	alBufferData(bufferID, AL_FORMAT_STEREO16, pbuffer, buffersize, rate);
	alSourceQueueBuffers(sourceId, 1, &bufferID);// 入队bufferID
	ALint stateVaue;
	alGetSourcei(sourceId, AL_SOURCE_STATE, &stateVaue);//获取状态

	if (stateVaue != AL_PLAYING)
	{
		alSourcePlay(sourceId);
	}
}
```
&emsp;&emsp;最后就是等待播放结束，结束的条件就是队列中和已经处理的buffer数量都为0时。需要注意的是这里又一个bug，当设置```AL_LOOPING```为true时，```AL_BUFFERS_PROCESSED```获取到的值始终为0，详情见[wrapper: AL_BUFFERS_PROCESSED always 0 when AL_LOOPING=AL_TRUE](https://openal-devel.opensource.creative.narkive.com/JUWU7EN9/wrapper-al-buffers-processed-always-0-when-al-looping-al-true)。

```cpp
void al_waitfor_end(ALuint source) {
	ALint processed = -1, queued = -1;
	alGetSourcei(source, AL_BUFFERS_QUEUED, &queued);
	alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);
	int ret = alGetError();
	while (queued > 0 || processed > 0) {
		if (processed > 0) {
			std::vector<ALuint> bufids(processed);
			alSourceUnqueueBuffers(source, 1, bufids.data());
		}

		alGetSourcei(source, AL_BUFFERS_QUEUED, &queued);
		alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);
	}
}
```

## 5 全部代码
&emsp;&emsp;代码中有些内容可能检查的不是很全面。
### 5.1 OpenGL播放视频
```cpp
int open_decoder(AVCodecParameters *param, AVCodecContext ** pdecoderCtx) {
	if (!param || !pdecoderCtx) {
		return -1;
	}

	AVCodec *pdecoder = avcodec_find_decoder(param->codec_id);
	if (!pdecoder) {
		return -1;
	}

	*pdecoderCtx = avcodec_alloc_context3(pdecoder);
	if (!(*pdecoderCtx)) {
		return -1;
	}

	int ret = avcodec_parameters_to_context(*pdecoderCtx, param);
	if (ret < 0) {
		return ret;
	}

	ret = avcodec_open2(*pdecoderCtx, pdecoder, NULL);
	return ret;
}

int gl_display_frame(GLFWwindow *pwindow, unsigned int sharedId, AVFrame *pframe, unsigned int VAO, GLuint texs[3]) {
	//背景色
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texs[0]);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, pframe->linesize[0]);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RED,
		pframe->width,
		pframe->height,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		pframe->data[0]);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texs[1]);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, pframe->linesize[1]);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RED,
		pframe->width / 2,
		pframe->height / 2,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		pframe->data[1]);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, texs[2]);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, pframe->linesize[2]);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RED,
		pframe->width / 2,
		pframe->height / 2,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		pframe->data[2]);

	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	glUseProgram(sharedId);
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glfwSwapBuffers(pwindow);
	glfwPollEvents();
	return 0;
}

void gl_wait_for_next(AVRational framerate, int pts) {
	static bool first_frame = true;
	if (first_frame) {
		glfwSetTime(0.0);
		first_frame = false;
	}

	double pt_in_seconds = pts * (double)framerate.den / (double)framerate.num;
	while (pt_in_seconds > glfwGetTime()) {
		glfwWaitEventsTimeout(pt_in_seconds - glfwGetTime());
	}
}

int gl_display_frame(GLFWwindow *pwindow, unsigned int sharedId, AVPacket *packet, AVCodecContext *pdecoderCtx, int owidth, int oheight, unsigned int VAO, GLuint texs[3]) {
	AVPixelFormat outFormat = AV_PIX_FMT_YUV420P;
	int ret = 0;
	AVFrame *pdecodeFrame = av_frame_alloc();
	AVFrame *ptranscodedFrame = av_frame_alloc();
	SwsContext *psws = nullptr;

	size_t size = av_image_get_buffer_size(outFormat, owidth, oheight, 1);
	uint8_t *pbuffer = (uint8_t *)malloc(size);
	static int count = 0;
	do{
		ret = av_image_fill_arrays(ptranscodedFrame->data, ptranscodedFrame->linesize, pbuffer, outFormat, owidth, oheight, 1);
		AVERR_BREAK_RET_LT_ZERO(ret);

		while (AVERROR(EAGAIN) == (ret = avcodec_send_packet(pdecoderCtx, packet))) {}
		AVERR_BREAK_RET_LT_ZERO(ret);

		while (0 <= (ret = avcodec_receive_frame(pdecoderCtx, pdecodeFrame))) {
			psws = sws_getCachedContext(psws, pdecoderCtx->width, pdecoderCtx->height, pdecoderCtx->pix_fmt, owidth, oheight, outFormat, SWS_BICUBIC, 0, 0, 0);
			AVERR_BREAK_IF_TRUE(!psws);
			ret = sws_scale(psws, pdecodeFrame->data, pdecodeFrame->linesize, 0, pdecodeFrame->height, ptranscodedFrame->data, ptranscodedFrame->linesize);
			AVERR_BREAK_RET_LT_ZERO(ret);
			fprintf(stderr, "%d, transcode frame %d : %d, %d into %d, %d\n", count, pdecodeFrame->pts, pdecoderCtx->width, pdecoderCtx->height, owidth, oheight);
			ptranscodedFrame->width = owidth;
			ptranscodedFrame->height = oheight;
			gl_display_frame(pwindow, sharedId, ptranscodedFrame, VAO, texs);
			count++;
			AVRational timebase = pdecoderCtx->framerate;
			gl_wait_for_next(timebase, count);
		}	
	} while (false);

	if (psws) {
		sws_freeContext(psws);
	}

	if (pdecodeFrame) {
		av_frame_free(&pdecodeFrame);
	}

	if (ptranscodedFrame) {
		av_frame_free(&ptranscodedFrame);
	}

	if (pbuffer) {
		free(pbuffer);
	}

	return ret;
}

void processKeyboard(GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

int gl_window_initialize(GLFWwindow** window, int width, int heigth) {
	// glfw: initialize and configure
		// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	*window = glfwCreateWindow(width, heigth, "Hello FFmpeg", NULL, NULL);
	if (*window == NULL)
	{
		fprintf(stderr, "Failed to create GLFW window\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(*window);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
		fprintf(stderr, "Failed to initialize GLAD\n");
		return -1;
	}

	return 0;
}

void gl_uniform_setbool(unsigned int id, const std::string &name, const bool val) {
	glUniform1i(glGetUniformLocation(id, name.c_str()), static_cast<int>(val));
}

void gl_uniform_setint(unsigned int id, const std::string &name, const int val) {
	glUniform1i(glGetUniformLocation(id, name.c_str()), static_cast<int>(val));
}

void gl_uniform_setfloat(unsigned int id, const std::string &name, const float val) {
	glUniform1f(glGetUniformLocation(id, name.c_str()), static_cast<float>(val));
}

int gl_build_program(GLFWwindow *pwindow, unsigned int &sharedID, unsigned int &VAO, GLuint (*texs)[3], double ratio) {
	const char *vertexSource = "#version 330 core\n\
								layout(location = 0) in vec3 aPos;\n\
								layout(location = 1) in vec2 aTexCoord;\n\
								out vec2 TexCoord;\n\
								void main(){\n\
									gl_Position = vec4(aPos, 1.0);\n\
									TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n\
								}\n";

	const char *fragmentSource = "#version 330 core\n\
								out vec4 FragColor;\n\
								in vec2 TexCoord;\n\
								// texture sampler\n\
								uniform sampler2D textureY;\n\
								uniform sampler2D textureU;\n\
								uniform sampler2D textureV;\n\
								void main(){\n\
									vec3 yuv, rgb;\n\
									vec3 yuv2r = vec3(1.164, 0.0, 1.596);\n\
									vec3 yuv2g = vec3(1.164, -0.391, -0.813);\n\
									vec3 yuv2b = vec3(1.164, 2.018, 0.0);\n\
									yuv.x = texture(textureY, TexCoord).r - 0.0625; \n\
									yuv.y = texture(textureU, TexCoord).r - 0.5; \n\
									yuv.z = texture(textureV, TexCoord).r - 0.5; \n\
									rgb.x = dot(yuv, yuv2r); \n\
									rgb.y = dot(yuv, yuv2g); \n\
									rgb.z = dot(yuv, yuv2b); \n\
									FragColor = vec4(rgb, 1.0);\n\
								}";
	//需要显示的矩形区域
	float vertices[] = {
		 ratio,  ratio, 0.0,     1.0, 0.0,
		 ratio, -1 * ratio, 0.0,     1.0, 1.0,
		-1 * ratio, -1 * ratio, 0.0,     0.0, 1.0,
		-1 * ratio,  ratio, 0.0,     0.0, 0.0,
	};

	unsigned int indices[] = {
		0, 1, 3,
		1, 2, 3
	};

	unsigned int VBO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1,
		3,
		GL_FLOAT,
		GL_FALSE,
		5 * sizeof(float),
		(void *)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glGenTextures(3, *texs);
	for (int i = 0; i < 3; i++) {
		glBindTexture(GL_TEXTURE_2D, (*texs)[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	//create and compile shader
	unsigned int vertex, fragment;
	int success;

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vertexSource, nullptr);
	glCompileShader(vertex);
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar infoLog[1024];
		glGetShaderInfoLog(vertex, 1024, nullptr, infoLog);
		std::cerr << "compile fail: " << infoLog << std::endl;
		assert(0);
	}

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fragmentSource, nullptr);
	glCompileShader(fragment);
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar infoLog[1024];
		glGetShaderInfoLog(fragment, 1024, nullptr, infoLog);
		std::cerr << "compile fail: " << infoLog << std::endl;
		assert(0);
	}

	sharedID = glCreateProgram();
	glAttachShader(sharedID, vertex);
	glAttachShader(sharedID, fragment);
	glLinkProgram(sharedID);
	glGetProgramiv(sharedID, GL_LINK_STATUS, &success);
	if (!success) {
		assert(0);
	}

	glDeleteShader(vertex);
	glDeleteShader(fragment);
	glUseProgram(sharedID);

	gl_uniform_setint(sharedID, "textureY", 0);
	gl_uniform_setint(sharedID, "textureU", 1);
	gl_uniform_setint(sharedID, "textureV", 2);

	return 0;
}

int gl_display_file(const char *infile) {
	AVFormatContext *pformat = nullptr;
	int ret = 0;
	AVPacket *packet = nullptr;
	AVCodecParameters *param = nullptr;
	AVCodecContext *pdecoderCtx = nullptr;
	int winWidth = 640, winHeight = 480;
	double ratio = 0.9;
	int owidth = winWidth * ratio, oheight = winHeight * ratio;
	FILE *fp = nullptr;
	GLFWwindow *pwindow = nullptr;
	unsigned int sharedId = 0, VAO;
	GLuint texs[3] = {0};
	
	do
	{
		ret = gl_window_initialize(&pwindow, winWidth, winHeight);
		AVERR_BREAK_RET_LT_ZERO(ret);

		ret = gl_build_program(pwindow, sharedId, VAO, &texs, ratio);
		AVERR_BREAK_RET_LT_ZERO(ret);

		//open file format
		ret = avformat_open_input(&pformat, infile, 0, NULL);
		AVERR_BREAK_RET_LT_ZERO(ret);

		ret = avformat_find_stream_info(pformat, NULL);
		AVERR_BREAK_RET_LT_ZERO(ret);

		av_dump_format(pformat, 0, infile, 0);
		int videid = av_find_best_stream(pformat, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
		AVERR_BREAK_RET_LT_ZERO(videid);

		fprintf(stderr, "video id is %d\n", videid);

		param = pformat->streams[videid]->codecpar;
		ret = open_decoder(param, &pdecoderCtx);
		AVERR_BREAK_RET_LT_ZERO(ret);

		packet = av_packet_alloc();
		while (true && !glfwWindowShouldClose(pwindow)) {
			processKeyboard(pwindow);
			ret = av_read_frame(pformat, packet);
			AVERR_BREAK_RET_LT_ZERO(ret);
			if (packet->stream_index == videid) {
				//decode video to yuv420p and display it by opengl
				gl_display_frame(pwindow, sharedId,packet, pdecoderCtx, owidth, oheight, VAO, texs);
			}

			av_packet_unref(packet);
		}


	} while (false);

	//flush the buffer
	packet->data = nullptr;
	packet->size = 0;
	gl_display_frame(pwindow, sharedId, packet, pdecoderCtx, owidth, oheight, VAO, texs);

	if (ret < 0) {
		char errorbuff[256] = { 0 };
		av_strerror(ret, errorbuff, 256);
		fprintf(stderr, "error message: %s", errorbuff);
	}

	if (fp) {
		fclose(fp);
	}

	if (packet) {
		av_packet_free(&packet);
		packet = nullptr;
	}

	if (pdecoderCtx) {
		avcodec_free_context(&pdecoderCtx);
	}

	if (pformat) {
		avformat_close_input(&pformat);
		pformat = nullptr;
	}

	glfwTerminate();
	return 0;
}
```
### 5.2 OpenAL播放音频
&emsp;&emsp;```open_decoder```和视频的一样就不列了。
```cpp
void al_play_raw_pcm(ALuint sourceId, ALenum format, ALsizei rate, uint8_t *pbuffer, int buffersize) {
	ALuint bufferID = 0;
	alGenBuffers(1, &bufferID); // 创建buffer并获取bufferID
	alBufferData(bufferID, AL_FORMAT_STEREO16, pbuffer, buffersize, rate);
	alSourceQueueBuffers(sourceId, 1, &bufferID);// 入队bufferID
	ALint stateVaue;
	alGetSourcei(sourceId, AL_SOURCE_STATE, &stateVaue);//获取状态

	if (stateVaue != AL_PLAYING)
	{
		alSourcePlay(sourceId);
	}
}

//ffplay -ar 44100 -channels 2 -f f32le -i .\44100_2_s16_pcm.pcm
int al_play_audio(AVPacket *packet, AVCodecContext *pdecoderCtx, int outRate, int outLayout, AVSampleFormat outFormat, SwrContext *pswr, ALuint sourceId) {
	int ret = 0;
	AVFrame *pdecodeFrame = av_frame_alloc();
	AVFrame *ptranscodedFrame = av_frame_alloc();
	int outChanel = av_get_channel_layout_nb_channels(outLayout);
	static int count = 0;
	do
	{
		//ret = avcodec_send_packet(pdecoderCtx, packet);
		while (AVERROR(EAGAIN) == (ret = avcodec_send_packet(pdecoderCtx, packet))) {}
		AVERR_BREAK_RET_LT_ZERO(ret);
		while (0 <= (ret = avcodec_receive_frame(pdecoderCtx, pdecodeFrame))) {
			int dstnbSamples = av_rescale_rnd(swr_get_delay(pswr, pdecoderCtx->sample_rate) + pdecodeFrame->nb_samples, outRate, pdecodeFrame->sample_rate, AV_ROUND_UP);
			AVERR_BREAK_RET_LT_ZERO(dstnbSamples);

			//pbuffer = (uint8_t *)malloc(size * 2);
			//ret = av_samples_fill_arrays(ptranscodedFrame->data, ptranscodedFrame->linesize, pbuffer, outChanel, dstnbSamples, outFormat, 1);
			ret = av_samples_alloc(ptranscodedFrame->data, ptranscodedFrame->linesize, outChanel, dstnbSamples, outFormat, 0);
			AVERR_BREAK_RET_LT_ZERO(ret);

			ret = swr_convert(pswr, ptranscodedFrame->data, dstnbSamples, (const uint8_t **)pdecodeFrame->data, pdecodeFrame->nb_samples);
			AVERR_BREAK_RET_LT_ZERO(ret);

			count++;

			int writingSize = av_samples_get_buffer_size(ptranscodedFrame->linesize, outChanel, ret, outFormat, 1);
			//fwrite(pbuffer, 1, writingSize, fp);
			//fwrite(ptranscodedFrame->data[0], 1, writingSize, fp);
			al_play_raw_pcm(sourceId, AL_FORMAT_STEREO16, pdecoderCtx->sample_rate, ptranscodedFrame->data[0], writingSize);

			while (0 < (ret = swr_convert(pswr, ptranscodedFrame->data, dstnbSamples, NULL, 0))) {
				writingSize = av_samples_get_buffer_size(ptranscodedFrame->linesize, outChanel, ret, outFormat, 1);
				//fwrite(ptranscodedFrame->data[0], 1, writingSize, fp);
				al_play_raw_pcm(sourceId, AL_FORMAT_STEREO16, pdecoderCtx->sample_rate, ptranscodedFrame->data[0], writingSize);
			}

			av_freep(ptranscodedFrame->data);
		}
	} while (false);

	av_freep(ptranscodedFrame->data);

	if (pdecodeFrame) {
		av_frame_free(&pdecodeFrame);
	}

	if (ptranscodedFrame) {
		av_frame_free(&ptranscodedFrame);
	}

	if (ret < 0) {
		char errorbuff[256] = { 0 };
		av_strerror(ret, errorbuff, 256);
		//fprintf(stderr, "error message: %s\n", errorbuff);
	}

	return ret;
}

void al_waitfor_end(ALuint source) {
	ALint processed = -1, queued = -1;
	alGetSourcei(source, AL_BUFFERS_QUEUED, &queued);
	alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);
	int ret = alGetError();
	while (queued > 0 || processed > 0) {
		if (processed > 0) {
			std::vector<ALuint> bufids(processed);
			alSourceUnqueueBuffers(source, 1, bufids.data());
		}

		alGetSourcei(source, AL_BUFFERS_QUEUED, &queued);
		alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);
	}
}

int al_context_create(ALCdevice **pdevice, ALCcontext **pcontext, ALuint *sourceId) {
	//播放源的位置
	ALfloat position[] = { 0.0f,0.0f,0.0f };
	//播放的速度
	ALfloat velocity[] = { 0.0f,0.0f,0.0f };

	*pdevice = alcOpenDevice(nullptr);
	if (!(*pdevice)) {
		return -1;
	}

	*pcontext = alcCreateContext(*pdevice, NULL);
	if (!(*pcontext)) {
		return -1;
	}

	alcMakeContextCurrent(*pcontext);

	alGenSources(1, sourceId);
	//音高倍数
	alSourcef(*sourceId, AL_PITCH, 1.0f);
	//声音的增益
	alSourcef(*sourceId, AL_GAIN, 1.0f);
	//设置位置
	alSourcefv(*sourceId, AL_POSITION, position);
	//设置声音移动速度
	alSourcefv(*sourceId, AL_VELOCITY, velocity);
	//设置是否循环播放
	alSourcei(*sourceId, AL_LOOPING, AL_FALSE);

	return 0;
}

int al_play_audio(const char *infile) {
	AVSampleFormat oformat = AV_SAMPLE_FMT_S16;
	int outLayout = AV_CH_LAYOUT_STEREO;
	AVFormatContext *pformat = nullptr;
	int ret = 0;
	int orate = 44100;
	AVPacket *packet = nullptr;
	AVCodecParameters *param = nullptr;
	AVCodecContext *pdecoderCtx = nullptr;
	FILE *fp = nullptr;
	SwrContext *pswr = nullptr;
	ALCdevice *pdevice = nullptr;
	ALCcontext *pcontext = nullptr;
	ALuint sourceId = 0;
	do
	{
		ret = al_context_create(&pdevice, &pcontext, &sourceId);
		AVERR_BREAK_RET_LT_ZERO(ret);

		//open file format
		ret = avformat_open_input(&pformat, infile, 0, NULL);
		AVERR_BREAK_RET_LT_ZERO(ret);

		ret = avformat_find_stream_info(pformat, NULL);
		AVERR_BREAK_RET_LT_ZERO(ret);

		av_dump_format(pformat, 0, infile, 0);
		int audioid = av_find_best_stream(pformat, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
		AVERR_BREAK_RET_LT_ZERO(audioid);

		fprintf(stderr, "video id is %d\n", audioid);

		param = pformat->streams[audioid]->codecpar;
		ret = open_decoder(param, &pdecoderCtx);
		AVERR_BREAK_RET_LT_ZERO(ret);

		pswr = swr_alloc_set_opts(pswr, outLayout, oformat, orate, pdecoderCtx->channel_layout, pdecoderCtx->sample_fmt, pdecoderCtx->sample_rate, 0, 0);
		AVERR_BREAK_IF_TRUE(!pswr);

		ret = swr_init(pswr);
		AVERR_BREAK_RET_LT_ZERO(ret);

		packet = av_packet_alloc();
		while (true) {
			ret = av_read_frame(pformat, packet);
			AVERR_BREAK_RET_LT_ZERO(ret);
			if (packet->stream_index == audioid) {
				//decode video to yuv420p
				al_play_audio(packet, pdecoderCtx, orate, outLayout, oformat, pswr, sourceId);
			}

			av_packet_unref(packet);
		}


	} while (false);

	//flush the buffer
	packet->data = nullptr;
	packet->size = 0;
	al_play_audio(packet, pdecoderCtx, orate, outLayout, oformat, pswr, sourceId);
	//flush_audio_sample_buffer(pdecoderCtx, orate, fp);

	al_waitfor_end(sourceId);

	if (ret < 0) {
		char errorbuff[256] = { 0 };
		av_strerror(ret, errorbuff, 256);
		fprintf(stderr, "error message: %s", errorbuff);
	}

	if (fp) {
		fclose(fp);
	}

	if (pswr) {
		swr_free(&pswr);
	}

	if (packet) {
		av_packet_free(&packet);
		packet = nullptr;
	}

	if (pdecoderCtx) {
		avcodec_free_context(&pdecoderCtx);
	}

	if (pformat) {
		avformat_close_input(&pformat);
		pformat = nullptr;
	}

	alcMakeContextCurrent(NULL);
	alcDestroyContext(pcontext);
	alcCloseDevice(pdevice);

	return 0;
}
```
## 6 参考
- [音频重采样swr_convert尾部丢帧的问题处理](https://blog.csdn.net/yzb408/article/details/90378807)
- [你好，三角形](https://learnopengl-cn.github.io/01%20Getting%20started/04%20Hello%20Triangle/)
- [yuv格式介绍与opengl 显示 yuv数据](https://blog.csdn.net/zhangpengzp/article/details/89532590)
- [桌面端FFmpeg解析视频及OpenGL渲染](https://zhuanlan.zhihu.com/p/376340749)
- [FFMPEG to OpenGL Texture](https://stackoverflow.com/questions/14821219/ffmpeg-to-opengl-texture)