# Meta 패밀리 모델로 빌드하기

## 소개 

이 강의에서는 다음 내용을 다룰 예정입니다.
1. 두가지 주요 메타 패밀리 모델 : Llama 3.1과 Llama 3.2 탐색
2. 각 모델의 사용 사례와 시나리오 이해
3. 각 모델의 고유한 기능을 보여주는 예제 코드

# Meta 패밀리 모델

 이 강의에서는 Meta 패밀리와 "Llama Herd"의 두가지 모델인 Llama 3.1과 Llama 3.2에 대해 알아보겠습니다. 
 
 이 모델들은 다양한 변형 버전으로 제공되며, GitHub 모델 마켓플레이스에서 사용할 수 있습니다. GitHub 모델을 통해 [AI 모델을 사용하여 프로토타입](https://docs.github.com/en/github-models/prototyping-with-ai-models?WT.mc_id=academic-105485-koreyst).을 만드는 자세한 내용은 다음과 같습니다.


Model Variants: 
- Llama 3.1 - 70B Instruct 
- Llama 3.1 - 405B Instruct 
- Llama 3.2 - 11B Vision Instruct 
- Llama 3.2 - 90B Vision Instruct 

*Note: Llama 3 is also available on Github Models but won't be covered in this lesson*

## Llama 3.1 

At 405 Billion Parameters, Llama 3.1 fits into the open source LLM category. 

The mode is an upgrade to the earlier release Llama 3 by offering: 

- Larger context window - 128k tokens vs 8k tokens 
- Larger Max Output Tokens - 4096 vs 2048 
- Better Multilingual Support - due to increase in training tokens 

These enables Llama 3.1 to handle more complex use cases  when building GenAI applications including: 
- Native Function Calling - the ability to call external tools and functions outside of the LLM workflow
- Better RAG Performance - due to the higher context window 
- Synthetic Data Generation - the ability to create effective data for tasks such as fine-tuning 

### Native Function Calling 

Llama 3.1 has been fine-tuned to be more effective at making function or tool calls. It also has two built-in tools that the model can identify as needing to be used based on the prompt from the user. These tools are: 

- **Brave Search** - Can be used to get up-to-date information like the weather by performing a web search 
- **Wolfram Alpha** - Can be used for more complex mathematical calculations so writing your own functions is not required. 

You can also create your own custom tools that LLM can call. 

In the code example below: 

- We define the available tools (brave_search, wolfram_alpha) in the system prompt. 
- Send a user prompt that asks about the weather in a certain city. 
- The LLM will respond with a tool call to the Brave Search tool which will look like this `<|python_tag|>brave_search.call(query="Stockholm weather")` 

*Note: This example only make the tool call, if you would like to get the results, you will need to create a free account on the Brave API page and define the function itself` 

```python 
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "meta-llama-3.1-405b-instruct"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)


tool_prompt=f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Tools: brave_search, wolfram_alpha
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant<|eot_id|>
"""

messages = [
    SystemMessage(content=tool_prompt),
    UserMessage(content="What is the weather in Stockholm?"),

]

response = client.complete(messages=messages, model=model_name)

print(response.choices[0].message.content)
```

## Llama 3.2 

Despite being a LLM, one limitation that Llama 3.1 has is multimodality. That is, being able to use different types of input such as images as prompts and providing responses. This ability is one of the main features of Llama 3.2. These features also include: 

- Multimodality -  has the ability to evaluate both text and image prompts 
- Small to Medium size variations (11B and 90B) - this provides flexible deployment options, 
- Text-only variations (1B and 3B) - this allows the model to be deployed on edge / mobile devices and provides low latency 

The multimodal support represents a big step in the world of open source models. The code example below takes both an image and text prompt to get an analysis of the image from Llama 3.2 90B. 


### Multimodal Support with Llama 3.2

```python 
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
)
from azure.core.credentials import AzureKeyCredential

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "Llama-3.2-90B-Vision-Instruct"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(
            content="You are a helpful assistant that describes images in details."
        ),
        UserMessage(
            content=[
                TextContentItem(text="What's in this image?"),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file="sample.jpg",
                        image_format="jpg",
                        detail=ImageDetailLevel.LOW)
                ),
            ],
        ),
    ],
    model=model_name,
)

print(response.choices[0].message.content)
```

## Learning does not stop here, continue the Journey

After completing this lesson, check out our [Generative AI Learning collection](https://aka.ms/genai-collection?WT.mc_id=academic-105485-koreyst) to continue leveling up your Generative AI knowledge!







 
모델 변형:
- Llama 3.1 - 70B Instruct
- Llama 3.1 - 405B Instruct
- Llama 3.2 - 11B Vision Instruct
- Llama 3.2 - 90B Vision Instruct

Llama 3 모델도 GitHub 모델에 제공되지만, 이 강의에서는 다루지 않습니다.

# Llama 3.1

4050억 개의 파라미터를 가진 Llama 3.1은 오픈 소스 대규모 언어 모델(LLM) 카테고리에 속합니다.

이 모델은 이전 버전인 Llama 3보다 다음의  향상된 기능을 제공합니다:

- 더 큰 컨텍스트 윈도우 : 8천 토큰에서 12만 8천 토큰으로 확대
- 더 많은 최대 출력 토큰 수 : 2048에서 4096으로 증가
- 향상된 다국어 지원 : 학습에 사용된 토큰 수 증가

이로 인해 Llama 3.1은 다음과 같은 복잡한 생성 AI 애플리케이션에서 더 효과적으로 활용될 수 있습니다.

- 네이티브 함수 호출: LLM 워크플로우 외부의 외부 도구 및 함수 호출 기능 제공
- 향상된 RAG 성능: 더 큰 컨텍스트 윈도우 덕분에 성능이 향상됨
- 합성 데이터 생성: 파인튜닝과 같은 작업을 위한 효율적인 데이터를 생성할 수 있는 기능

# native 함수 호출

Llama 3.1은 함수나 도구 호출을 더 효과적으로 수행하도록 미세 조정되었습니다. 또한 사용자 프롬프트에 따라 모델이 필요할 때 자동으로 사용할 수 있는 두 가지 내장 도구가 있습니다. 이러한 도구는 다음과 같습니다:

- Brave Search: 날씨와 같은 최신 정보를 얻기 위해 웹 검색 수행
- Wolfram Alpha: 복잡한 수학 계산을 수행하여 사용자가 직접 함수를 작성할 필요가 없음

사용자는 LLM이 호출할 수 있는 사용자 정의 도구를 추가로 만들 수도 있습니다.

### 아래의 코드 예제에서는:

- 시스템 프롬프트에 사용할 수 있는 도구들 (brave_search, wolfram_alpha)을 정의합니다.
- 특정 도시의 날씨를 묻는 사용자 프롬프트를 보냅니다.
- LLM은 Brave Search 도구 호출을 통해 응답하며, 다음과 같이 보입니다: `<|python_tag|>brave_search.call(query="Stockholm weather")`

참고: 이 예제는 도구 호출만 수행하며, 실제 결과를 얻으려면 Brave API 페이지에서 무료 계정을 생성하고 함수를 정의해야 합니다.

# Llama 3.2

대규모 언어 모델(LLM)임에도 불구하고, Llama 3.1에는 **멀티모달리티**라는 한계가 있습니다. 즉, 이미지와 같은 다양한 유형의 입력을 프롬프트로 받아들이고 이에 응답합니다. 이 기능은 Llama 3.2의 주요 기능 중 하나입니다. Llama 3.2의 기능은 다음과 같습니다:

- 멀티모달리티: 텍스트뿐만 아니라 이미지 프롬프트도 평가할 수 있는 기능
- 작은 크기부터 중간 크기까지의 변형(11B 및 90B): 유연한 배포 옵션 제공
- 텍스트 전용 변형(1B 및 3B): 에지/모바일 장치에서 모델을 배포할 수 있으며 저지연성을 제공
