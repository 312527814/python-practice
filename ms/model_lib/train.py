from modelscope.msdatasets import MsDataset
train_dataset =  MsDataset.load('LLM-Research/Llama-3.2-1B-Instruct-evals',cache_dir='F:\llm\dataset')

def cfg_modify_fn(cfg):
  cfg.preprocessor.type='sen-sim-tokenizer'
  cfg.preprocessor.first_sequence = 'sentence1'
  cfg.preprocessor.second_sequence = 'sentence2'
  cfg.preprocessor.label = 'label'
  cfg.preprocessor.label2id = {'0': 0, '1': 1}
  cfg.model.num_labels = 2
  cfg.task = 'text-generation'
  cfg.pipeline = {'type': 'text-generation'}
  cfg.train.max_epochs = 5
  cfg.train.work_dir = '/tmp'
  cfg.train.dataloader.batch_size_per_gpu = 32
  cfg.evaluation.dataloader.batch_size_per_gpu = 32
  cfg.train.dataloader.workers_per_gpu = 0
  cfg.evaluation.dataloader.workers_per_gpu = 0
  cfg.train.optimizer.lr = 2e-5
  cfg.train.lr_scheduler.total_iters = int(len(train_dataset) / cfg.train.dataloader.batch_size_per_gpu) * cfg.train.max_epochs
  cfg.evaluation.metrics = 'seq-cls-metric'
  # 注意这里需要返回修改后的cfg
  return cfg

from modelscope.trainers import build_trainer

# 配置参数
kwargs = dict(
        model="model_id",
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(default_args=kwargs)
trainer.train()