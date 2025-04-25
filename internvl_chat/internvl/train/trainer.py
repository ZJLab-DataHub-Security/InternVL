from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, use_cuda_graph = False,**kwargs):
        super().__init__(**kwargs)
        self.use_cuda_graph = use_cuda_graph

        if self.use_cuda_graph:   
            from internvl.patch import build_graphed_model 
            self.model.train()    
            build_graphed_model(model=self.model,max_seq_len=self.tokenizer.model_max_length,micro_bs=self.args.per_device_train_batch_size,hidden_size=self.model.config.llm_config.hidden_size)  