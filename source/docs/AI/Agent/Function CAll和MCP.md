# Function CAll和MCP
## 1 Function Call
### 1.1 基础概念

&emsp;&emsp;早期的LLMs工作流程基本上是用户通过输入文本与模型进行交互，模型根据输入文本生成响应文本。但是这样的模型有个问题，模型根据用户输入生成对应的响应输出文本，只能利用模型本身的能力。为了突破这层壁垒，将现有大量的软件生态能力接入到LLMs中，让LLMs更具生命力，能够与外部软件生态进行交互。

&emsp;&emsp;GPT提供了根据用户输入文本解析用户意图，生成对应函数调用响应的能力。这种能够根据用户输入生成对应函数调用信息的能力，就是Function Call。需要明确的是，Function Call是建立在模型推理能力和外部软件能力的基础上的，是二者共同交互协作的产物。LLMs本身只能进行推理，无法完成具体的任务，Function Call就是将传统文本推理能力进一步细化，调整成推理函数调用信息。因此，可以看到LLMs在任务中的角色从来没有改变，一直是扮演任务推理的角色只是任务本身变化了。

&emsp;&emsp;这也是大模型对Agent相关的价值所在，其打破了用户和代码直接的界限，可以让用户直接和底层能力交互，让生产环境某些步骤更加简单高效。

### 1.2 工作流程
&emsp;&emsp;Function Calling最基本的工作流程是：用户提出请求，LLM 理解意图并选择合适的函数，生成参数并调用该函数，接收函数执行结果后整合到最终回复中呈现给用户，从而实现 LLM 与外部工具的交互，扩展其能力并完成更复杂的任务。

&emsp;&emsp;其中LLMs主要参与用户的意图理解，输入是用户期望的promot，输出是相关函数的调用信息。而具体的函数调用是由外部软件完成的。

&emsp;&emsp;最基本的工作流程就是：输入——推理——调用——返回结果。但是通常情况下为了处理更加复杂的任务流程会更加复杂，推理流程也会更加复杂。比如需要区分用户的诉求是咨询诉求还是执行诉求，根据不同的诉求执行不同的单元等等。


### 1.3 示例
&emsp;&emsp;简单的描述比较干燥，下面用一个大模型Function Call的例子来说明LLMs Function Call是如何工作的。首先，我们需要实现期望运行的函数，这里实现了两个含简单的函数，一个是显示图片，一个是图片重设大小。
>需要注意的是这里展示的只是最基本的原理，而实际上很多开元的function call会对工具进行二次封装，比如openai的示例。但是大体流程是一致的。
```python
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    tools=tools
)
```

```python
from PIL import Image
import ollama
import json
import re  # Import the regular expression module

def display_image(image_path):
    """Displays an image from the given path."""
    try:
        img = Image.open(image_path)
        img.show()  # This will open the image using the default image viewer
        return f"Image displayed successfully from {image_path}"
    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    except Exception as e:
        return f"Error displaying image: {e}"


def resize_image(image_path, scale=0.5):
    """Resizes an image to half its original size and saves it with '_resized' suffix."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = img.resize((new_width, new_height))
        new_image_path = image_path.replace(".", "_resized.")  # Add '_resized' before the extension
        resized_img.save(new_image_path)
        return f"Image resized successfully and saved to {new_image_path}"
    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    except Exception as e:
        return f"Error resizing image: {e}"
```

&emsp;&emsp;然后我们需要告诉大模型，这些函数能干什么，参数是什么，返回值是什么。
```python
# Function descriptions (adapt to the model's expected format)
function_descriptions = [
    {
        "name": "display_image",
        "description": "Displays an image from a given file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the image file.  Must be a valid path to an image file (e.g., PNG, JPG, JPEG).",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "resize_image",
        "description": "Resizes an image to half its original size and saves the resized image with a suffix.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the image file to resize.",
                },
                "scale": {
                    "type": "number",
                    "description": "The scale factor to resize the image. Default is 0.5 for half the size.",
                    "default": 0.5,
                },
            },
            "required": ["image_path"],
        },
    },
]
```

&emsp;&emsp;有了上面的函数描述，我们就可以将其告诉大模型我们的意图和相关的函数描述，大模型会根据输入来判断我们期望调用的参数。需要注意的是，在prompt中尽量标准化输出方便后续解析输出。
```python
def run_conversation_ollama(user_prompt, model="deepseek-r1:1.5b-qwen-distill-fp16"):
    """
    Interacts with the local LLM via the ollama API, attempts to identify function calls, and executes them.
    """
    messages = [{"role": "user", "content": user_prompt}]

    # Include function descriptions in the prompt (you might need to adapt this based on the model)
    system_prompt = f"""你是一个图像处理专家，根据给定的函数列表分析出用户期望调用的函数名和对应的参数，返回的格式为:
    ```json
    {{
        "function": "function_name",
        "args": {{
            "image_path": "1.jpg",
            "scale": "0.1"
        }}
    }}
    ```
    可用的函数列表如下:\n{json.dumps(function_descriptions)}
    """
    messages.insert(0, {"role": "system", "content": system_prompt}) # Add system prompt

    try:
        response = ollama.chat(model=model, messages=messages)
        llm_response = response['message']['content']

        print(f"LLM Response: {llm_response}")

        # Attempt to parse the response to identify a function call
        function_name, arguments = extract_function_call(llm_response)  # Implement this function!

        if function_name:
            print(f"Identified function call: {function_name} with arguments: {arguments}")

            # Execute the function
            function_response = execute_function(function_name, arguments)  # Implement this function!

            print(f"Function response: {function_response}")

            # Integrate the function response into a final response
            messages.append({"role": "assistant", "content": llm_response}) # Add assistant response
            messages.append({"role": "function", "name": function_name, "content": function_response}) # Add function response

            # Get a new response from the model with the function result
            second_response = ollama.chat(model=model, messages=messages)
            final_response = second_response['message']['content']
            return final_response
        else:
            return llm_response

    except Exception as e:
        return f"Error during ollama interaction: {e}"
```

&emsp;&emsp;拿到输出后就是从输出中解析出我们需要的参数，比如我本地返回的输出为如下，后续解析参数和调用比较简单就不说了。
```python
<think>
好的，我现在要分析用户提供的信息，并根据给定的函数列表来确定应该调用哪个函数以及对应的参数。
首先，用户提供了一个示例：显示图片，图片路径是1.jpeg。看起来这是一个简单的请求，要求将指定的图像显示出来。根据可用的函数列表，有两个可能相关的函数：display_image 和 resize_image。  
我需要判断用户的需求属于哪种情况。用户明确提到“显示图片”，而不是“ resized image”。因此，应该使用 display_image 函数来完成操作。
接下来，我查看函数的参数。对于 display_image 函数，主要参数是 image_path，这是一个字符串，表示要显示的图像文件路径。用户提供的例子中，image_path 是1.jpeg，所以参数应该是 { "image_path": "1.jpg" }。
不需要使用 resize_image 的函数，因为用户没有提到需要调整大小或缩放。因此，在 args 中不包括 scale 参数。
总结一下，应该调用 display_image 函数，参数是 image_path，“1.jpg”。
</think>

```json
{
    "function": "display_image",
    "args": {
        "image_path": "1.jpg"
    }
}
```json
```

## 2 MCP
### 2.1 简介
&emsp;&emsp;模型上下文协议 (MCP) 是一种旨在标准化和管理大语言模型 (LLM) 上下文信息的协议。它定义了 LLM 如何接收、处理和利用上下文信息，以提高生成结果的相关性、准确性和一致性。MCP 的目标是解决 LLM 在处理长文本、多轮对话和复杂任务时遇到的上下文理解和利用问题。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/v2-1c76c2a5a299ac7f36efaed9fefd75eb_b.webp)

&emsp;&emsp;所以总体上MCP只是一套协议，相比这就意味着可以通过这个协议调用网络上的LLM能力。Function Call是用来和LLM交互的，用来表示如何通过LLMs实现某些功能，更加和native贴近，相比之下MCP则是将这些能力通过协议开放到网络上让用户或者其他C端使用。这里只描述了MCP是什么，具体MCP细节不赘述了。
![](https://github.com/liaokongVFX/MCP-Chinese-Getting-Started-Guide/raw/main/.assets/image-20250223214308430.png)

### 2.2 简单实现一个MCP
&emsp;&emsp;简单实现一个MCP，首先是服务器，服务器是用来处理具体事务的。下面的代码比较简单就是接受客户端的请求然后通过llm解析然后执行具体的调用（这里也用到了function call解析，只不过没有那么标准，没有写function描述）。

```python
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import os
import threading
import time
import ollama
import json

app = Flask(__name__)

# 身份验证 (简单示例)
API_KEY = "your_secret_api_key"

# Ollama 模型名称
OLLAMA_MODEL = "llama2"  # 替换为你实际使用的 Ollama 模型

def authenticate(request):
    api_key = request.headers.get('X-API-Key')
    if api_key == API_KEY:
        return True
    else:
        return False

# 健康检查
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/process_image', methods=['POST'])
def process_image():
    print("processing request...")
    if not authenticate(request):
        return jsonify({"error": "Unauthorized"}), 401

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    instruction = request.form.get('instruction', 'Resize the image to 200x200')  # 默认指令

    try:
        img = Image.open(io.BytesIO(image_file.read()))

        # 调用 Ollama 模型
        prompt = f"""你是一个图像处理专家，能够对图像进行缩放(scale)，灰度化(grayscale)，裁剪处理(crop)。分析输入的指令返回执行的结果格式为(输出结果除了这个json多余的内容不要加)
        ```json
        {{
            "function": "function_name",
            "args": {{
                "width": "200",
                "height": "200"
            }}
        }}
        ```
        "用户的输入为: {instruction}
        """
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False)
        res = response['response']
        i = res.find("{")
        re = res[i:]
        j = re.rfind("}")
        re = re[:j + 1]
        operation_data = json.loads(re)
        # 根据 Ollama 模型的输出进行图像处理
        if operation_data:
            operation = operation_data.get('function')
            if operation == 'scale':
                width = operation_data.get('width', 200)
                height = operation_data.get('height', 200)
                img = img.resize((width, height))
            elif operation == 'grayscale':
                img = img.convert('L')
            elif operation == 'rotate':
                angle = operation_data.get('angle', 90)
                img = img.rotate(angle)
            elif operation == 'crop':
                left = operation_data.get('left', 0)
                top = operation_data.get('top', 0)
                right = operation_data.get('right', 100)
                bottom = operation_data.get('bottom', 100)
                img = img.crop((left, top, right, bottom))
            else:
                return jsonify({"error": "Invalid operation from LLM"}), 400
        else:
            print("No operation needed based on the instruction.")

        # 将处理后的图像转换为 base64 编码
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({"image": img_str}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

&emsp;&emsp;然后是客户端，客户端比较简单就是请求-响应。
```python
import requests
import base64
from PIL import Image
import io

class MCPClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {'X-API-Key': self.api_key}

    def process_image(self, image_path, instruction="Resize the image to 200x200"):
        url = f"{self.base_url}/process_image"
        try:
            with open(image_path, 'rb') as image_file:
                files = {'image': (image_path, image_file, 'image/jpeg')}
                data = {'instruction': instruction}
                print("sending request...")
                response = requests.post(url, files=files, data=data, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error processing image: {e}")
            return None

def display_image_from_base64(base64_image):
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        image.show()  # 使用默认图像查看器显示图像
    except Exception as e:
        print(f"Error displaying image: {e}")

# 示例用法
if __name__ == '__main__':
    # 替换为你的 MCP 服务器地址和 API 密钥
    mcp_client = MCPClient("http://localhost:5001", "your_secret_api_key")

    image_path = "1.jpg"
    instruction = "Enter the image processing instruction: 缩放图片到200x200"

    result = mcp_client.process_image(image_path, instruction=instruction)
    if result:
        processed_image = result['image']
        display_image_from_base64(processed_image)
    else:
        print("Image processing failed.")
```

## 3 总结
&emsp;&emsp;Function Calling 是一种 LLM 的能力，用于生成调用外部函数的指令，而 MCP 是一种自定义的消息传递协议，用于在客户端和服务器之间传递消息。  Function Calling 侧重于利用 LLM 的推理能力来驱动外部系统的执行，而 MCP 侧重于建立可靠的通信通道，以便客户端和服务器能够协同工作。  在你的图像处理应用中，你可以同时使用 Function Calling 和 MCP：  使用 Function Calling 来让 LLM 决定如何处理图像 (例如，选择合适的滤镜或调整参数)，然后使用 MCP 将图像数据和 LLM 生成的指令传递给服务器进行处理。
| 特性          | Function Calling (LLM)                                  | MCP (自定义消息传递协议)                                       |
|---------------|-------------------------------------------------------|--------------------------------------------------------------|
| **核心功能**    | LLM 生成函数调用指令                                   | 客户端和服务器之间的消息传递                                       |
| **执行函数**    | LLM  *不*  执行函数，而是将指令返回给应用程序                       | 服务器执行函数 (例如图像处理)                                         |
| **控制权**      | 应用程序控制函数的执行                                     | 客户端和服务器共同控制协议的实现                                       |
| **通用性**      | 可以与各种外部工具和服务集成                                  | 通常用于特定的应用程序场景                                           |
| **自定义程度**   | 函数的定义和实现由开发者控制，但 LLM 的行为受到预训练模型的限制 | 协议的各个方面 (消息格式、状态码、错误处理等) 都可以自定义                   |
| **应用场景**    | 扩展 LLM 的能力，使其能够完成更复杂的任务                        | 在客户端和服务器之间建立可靠的通信通道，用于特定的应用程序需求 (例如图像处理) |
 
## 4 参考文献
- [MCP](https://modelcontextprotocol.io/introduction)
- [Function calling](https://platform.openai.com/docs/guides/function-calling?api-mode=responses)