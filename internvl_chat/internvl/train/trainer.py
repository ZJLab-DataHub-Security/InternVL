import torch
import torch.nn as nn
from typing import Any, Dict, Union
from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, warm_steps = 0,use_cuda_graph = False,cuda_graph_module:str = "self_attn" ,cuda_graph_layer_num=0,**kwargs):
        super().__init__(**kwargs)
        self.warm_steps = warm_steps
        self.steps = 0
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph_module = cuda_graph_module
        self.cuda_graph_layer_num = cuda_graph_layer_num
        
        if self.use_cuda_graph:   
            from internvl.patch import build_graphed_model 
            self.model.train()    
            build_graphed_model(model=self.model,cuda_graph_module=self.cuda_graph_module,cuda_graph_layer_num=self.cuda_graph_layer_num,max_seq_len=self.tokenizer.model_max_length,micro_bs=self.args.per_device_train_batch_size,hidden_size=self.model.config.llm_config.hidden_size)  
            
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        res = super().training_step(model,inputs)
        self.steps += 1
        return res
        