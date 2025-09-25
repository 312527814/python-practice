from modelscope.hub.snapshot_download import snapshot_download

# model_dir = snapshot_download('LLM-Research/Llama-3.2-1B',local_dir=r"F:\llm\model")


#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Fengshenbang/Wenzhong-GPT2-110M-chinese-v2',local_dir=r"F:\llm\model")