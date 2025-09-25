from transformers.pipelines import SUPPORTED_TASKS
for k, v in SUPPORTED_TASKS.items():
    print(k, v)