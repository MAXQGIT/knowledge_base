from search_knowledge import rrf_fusion
from openai import OpenAI
import time
from collections import deque

'''
大模型多轮问答测试程序
'''

class MultiTurnDialogue:
    def __init__(self, max_turns=5):
        self.client = OpenAI(base_url='http://192.168.0.1:5156/v1',
                             api_key='none')
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)

    def add_history(self, user_msg, assistant_msg):
        self.history.append({"user": user_msg,
                             "assistant": assistant_msg})

    def bulid_messages(self, current_query):
        message = []
        for turn in self.history:
            message.append({'role': "user", 'content': turn["user"]})
            message.append({'role': "assistant", 'content': turn["assistant"]})
        message.append({'role': "user", 'content': current_query})
        return message

    def chat(self, user_input, stream=True, delay=0.02):
        messsages = self.bulid_messages(user_input)
        response = self.client.chat.completions.create(
            model='qwen35_4B',
            messages=messsages,
            temperature=0.7,
            max_tokens=2000,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=stream,
        )
        full_response = ''
        for chunk in response:
            word = chunk.choices[0].delta.content
            if word:
                print(word, end="", flush=True)
                full_response += word
                if delay:
                    time.sleep(delay)
        self.add_history(user_input, full_response)
        return full_response

    def clear_history(self):
        self.history.clear()


if __name__ == "__main__":
    # 创建对话实例，最多保留5轮
    dialogue = MultiTurnDialogue(max_turns=5)
    query = '运行备用'
    knowledge_list= ''.join(rrf_fusion(query))
    question = (
        f"你只能根据以下【参考内容】来回答问题。"
        f"如果【参考内容】中没有明确信息，请直接回答'根据提供的内容，无法回答该问题'。"
        f"不要使用外部知识，不要添加任何解释或额外信息。\n\n"
        f"【参考内容】\n{knowledge_list}\n\n"
        f"【问题】\n{query}\n\n"
        f"【回答】"
    )
    # question=(f'根据内容：“{knowledge_list}”回答问题：“{query}”。\n'
    #           f'不要多余解释，严格按照知识回答')
    dialogue.chat(question)
    #
    # print('\n')
    # print('~~~'*50)
    # dialogue.chat('这些知识描述的是什么？')
