from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai.chat_models import ChatOpenAI

from pixelpilot.tgi_wrapper import LocalTGIChatModel

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="Hello!"),
]

llm_local = LocalTGIChatModel(base_url="http://jelly:8080")
response = llm_local.invoke("hello.")
print(f"TGI Response: \n{response}\n")

llm_cloud = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm_cloud.invoke("hello.")
print(f"OpenAI Response: \n{response}\n")
