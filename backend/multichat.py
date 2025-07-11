import os
from dotenv import load_dotenv
from enum import Enum


from zhipuai import ZhipuAI # GLM
from openai import OpenAI   # Gemini


import json
from typing import List, Dict, Optional, AsyncGenerator
import asyncio

load_dotenv()

class ClientType(str, Enum):
    DEEPSEEK = "deepseek"
    TONGYI = "tongyi"
    ZHIPUAI = "zhipuai"
    GEMINI = "gemini"


class MultiAIChat:
    def __init__(self, client: str = 'zhipuai'):
        """
        初始化智谱AI聊天客户端

        :param api_key: 智谱AI的API密钥
        :param model: 使用的模型名称，默认为 glm-4-plus
        """
        # 设置默认模型映射
        self.model_mapping = {
            'deepseek': 'deepseek-chat',  # DeepSeek 默认使用 deepseek-chat
            'tongyi': 'qwen-turbo',  # 通义千问默认使用 qwen-turbo
            'zhipuai': 'glm-4-plus',  # 智谱默认使用 glm-4-air
            'gemini': 'gemini-1.5-flash',  # Gemini 默认使用 gemini-1.5-flash
        }

        # 根据客户端类型获取默认模型
        self.client_type = client
        self.model = self.model_mapping.get(client, 'glm-4-plus')

        self.client = self.__init_client(client)
        self.conversation_history: List[Dict[str, str]] = []

    def __init_client(self, client: str):
        self.client_types = {
            "deepseek": OpenAI(api_key=os.environ["DS_API_KEY"], base_url="https://api.deepseek.com/v1"),
            "tongyi": OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "zhipuai": ZhipuAI(api_key=os.environ["GLM_API_KEY"]),
            "gemini": OpenAI(api_key=os.environ["GEMINI_BASE_URL"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
        }
        return self.client_types.get(client)

    def add_message(self, role: str, content: str):
        """
        添加消息到对话历史

        :param role: 消息角色 ('user' 或 'assistant')
        :param content: 消息内容
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取当前对话历史

        :return: 对话历史列表
        """
        return self.conversation_history.copy()

    def clear_history(self):
        """
        清空对话历史
        """
        self.conversation_history = []

    def chat(self, prompt: str, temperature: float = 0, max_tokens: Optional[int] = None) -> str:
        """
        发送消息并获取回复

        :param prompt: 用户输入的提示词
        :param temperature: 控制回复的随机性，0-1之间
        :param max_tokens: 最大token数量
        :return: AI的回复内容
        """

        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": self.conversation_history
                            +[
                                {"role": "user", "content": prompt}
                            ],
                "temperature": temperature
            }

            # 如果指定了max_tokens，添加到请求参数中
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # 发送请求
            response = self.client.chat.completions.create(**request_params)

            # 提取回复内容
            if len(response.choices) > 0:
                ai_reply = response.choices[0].message.content
                # 添加AI回复到历史记录
                self.add_message("user", prompt)
                self.add_message("assistant", ai_reply)
                return ai_reply
            else:
                return "生成错误：没有收到有效回复"

        except Exception as e:
            return f"请求错误: {str(e)}"

    async def chat_stream(self, prompt: str, temperature: float = 0, max_tokens: int | None = None) -> AsyncGenerator[str, None]:
        """
        发送消息并获取流式回复

        :param prompt: 用户输入的提示词
        :param temperature: 控制回复的随机性，0-1之间
        :param max_tokens: 最大token数量
        :yield: AI回复的文本片段
        """
        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": self.conversation_history + [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }

            # 如果指定了max_tokens，添加到请求参数中
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # 发送请求
            response = self.client.chat.completions.create(**request_params)

            # 提取回复内容
            if len(response.choices) > 0:
                ai_reply = response.choices[0].message.content
                # 添加AI回复到历史记录
                self.add_message("user", prompt)
                self.add_message("assistant", ai_reply)
                return ai_reply
            else:
                return "生成错误：没有收到有效回复"

        except Exception as e:
            return f"请求错误: {str(e)}"

    async def chat_stream(self, prompt: str, temperature: float = 0, max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        """
        发送消息并获取流式回复

        :param prompt: 用户输入的提示词
        :param temperature: 控制回复的随机性，0-1之间
        :param max_tokens: 最大token数量
        :yield: AI回复的文本片段
        """
        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": self.conversation_history + [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "stream": True  # 开启流式输出
            }

            # 如果指定了max_tokens，添加到请求参数中
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # 发送流式请求
            stream = self.client.chat.completions.create(**request_params)

            full_response = ""

            # 处理流式响应
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

                    # 添加小延迟以模拟更自然的打字效果
                    await asyncio.sleep(0.01)

            # 将完整的对话添加到历史记录
            self.add_message("user", prompt)
            self.add_message("assistant", full_response)

        except Exception as e:
            yield f"流式请求错误: {str(e)}"



    def get_last_exchange(self) -> Dict[str, str]:
        """
        获取最后一轮对话

        :return: 包含最后一轮用户输入和AI回复的字典
        """
        if len(self.conversation_history) >= 2:
            return {
                "user": self.conversation_history[-2]["content"],
                "assistant": self.conversation_history[-1]["content"]
            }
        return {}

    def save_conversation(self, filename: str):
        """
        保存对话历史到文件

        :param filename: 保存的文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"对话历史已保存到 {filename}")
        except Exception as e:
            print(f"保存失败: {str(e)}")

    def load_conversation(self, filename: str):
        """
        从文件加载对话历史

        :param filename: 要加载的文件名
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"对话历史已从 {filename} 加载")
        except Exception as e:
            print(f"加载失败: {str(e)}")

    def switch_client(self, client: str):
        """
        切换大模型客户端，同时保留当前对话历史

        :param client: 目标客户端类型 (ClientType 枚举值)
        """

        # 更新客户端类型和模型
        self.client_type = client
        self.model = self.model_mapping.get(client, 'glm-4-plus')

        # 重新初始化客户端实例
        self.client = self.client_types.get(client)

        return f"已切换到 {client} 客户端，模型: {self.model}"




# 使用示例
def main():
    # 创建聊天实例
    chat = MultiAIChat(api_key="*******")

    print("智谱AI多轮对话开始！输入 'quit' 退出，'clear' 清空历史，'save' 保存对话")
    print("-" * 50)

    while True:
        user_input = input("\n你: ")

        if user_input.lower() == 'quit':
            print("再见！")
            break
        elif user_input.lower() == 'clear':
            chat.clear_history()
            print("对话历史已清空")
            continue
        elif user_input.lower() == 'save':
            filename = input("请输入保存文件名（如：chat_history.json）: ")
            chat.save_conversation(filename)
            continue
        elif user_input.lower() == 'load':
            filename = input("请输入要加载的文件名: ")
            chat.load_conversation(filename)
            continue
        elif user_input.lower() == 'history':
            print("\n=== 对话历史 ===")
            for i, msg in enumerate(chat.get_conversation_history(), 1):
                role = "你" if msg["role"] == "user" else "AI"
                print(f"{i}. {role}: {msg['content'][:100]}...")
            continue

        # 获取AI回复
        ai_reply = chat.chat(user_input)
        print(f"\nAI: {ai_reply}")


# 简单的单次对话函数（保持向后兼容）
def get_completion(prompt: str, api_key: str, model: str = 'glm-4-plus', temperature: float = 0) -> str:
    """
    单次对话函数，保持与原代码的兼容性

    :param prompt: 提示词
    :param api_key: API密钥
    :param model: 模型名称
    :param temperature: 温度参数
    :return: AI回复
    """
    chat = MultiAIChat(api_key=api_key, model=model)
    return chat.chat(prompt, temperature=temperature)

if __name__ == "__main__":
    main()