[1mdiff --git a/internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py b/internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py[m
[1mindex 1c5c043..b3cfe05 100644[m
[1m--- a/internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py[m
[1m+++ b/internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py[m
[36m@@ -330,6 +330,7 @@[m [mclass InternVisionEncoder(nn.Module):[m
             return_dict (`bool`, *optional*):[m
                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.[m
         """[m
[32m+[m[41m        [m
         output_hidden_states = ([m
             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states[m
         )[m
[36m@@ -341,10 +342,11 @@[m [mclass InternVisionEncoder(nn.Module):[m
         for idx, encoder_layer in enumerate(self.layers):[m
             if output_hidden_states:[m
                 encoder_states = encoder_states + (hidden_states,)[m
[31m-            if self.gradient_checkpointing and self.training:[m
[32m+[m[32m            if self.gradient_checkpointing and self.training and (not hasattr(self.config,"recompute_num_layers") or idx < self.config.recompute_num_layers):[m
                 layer_outputs = torch.utils.checkpoint.checkpoint([m
                     encoder_layer,[m
[31m-                    hidden_states)[m
[32m+[m[32m                    hidden_states,[m
[32m+[m[32m                    preserve_rng_state=False,)[m
             else:[m
                 layer_outputs = encoder_layer([m
                     hidden_states,[m
[36m@@ -353,7 +355,9 @@[m [mclass InternVisionEncoder(nn.Module):[m
 [m
         if output_hidden_states:[m
             encoder_states = encoder_states + (hidden_states,)[m
[31m-[m
[32m+[m[41m        [m
[32m+[m[41m        [m
[32m+[m[41m        [m
         if not return_dict:[m
             return tuple(v for v in [hidden_states, encoder_states] if v is not None)[m
         return BaseModelOutput([m
[36m@@ -427,4 +431,4 @@[m [mclass InternVisionModel(PreTrainedModel):[m
             pooler_output=pooled_output,[m
             hidden_states=encoder_outputs.hidden_states,[m
             attentions=encoder_outputs.attentions,[m
[31m-        )[m
[32m+[m[32m        )[m
\ No newline at end of file[m
[1mdiff --git a/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py b/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py[m
[1mindex 05f91f4..af56a29 100644[m
[1m--- a/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py[m
[1m+++ b/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py[m
[36m@@ -7,6 +7,7 @@[m
 import warnings[m
 from typing import List, Optional, Tuple, Union[m
 [m
[32m+[m[32mimport torch[m
 import torch.distributed as dist[m
 import torch.utils.checkpoint[m
 import transformers[m
[36m@@ -44,7 +45,7 @@[m [mclass InternVLChatModel(PreTrainedModel):[m
                          'Phi3DecoderLayer', 'Qwen2DecoderLayer'][m
     _supports_flash_attn_2 = True[m
     supports_gradient_checkpointing = True[m
[31m-[m
[32m+[m[41m     [m
     def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):[m
         super().__init__(config)[m
 [m
[36m@@ -85,7 +86,7 @@[m [mclass InternVLChatModel(PreTrainedModel):[m
 [m
         vit_hidden_size = config.vision_config.hidden_size[m
         llm_hidden_size = config.llm_config.hidden_size[m
[31m-[m
[32m+[m[41m        [m
         self.mlp1 = nn.Sequential([m
             nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),[m
             nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),[m
[36m@@ -156,8 +157,8 @@[m [mclass InternVLChatModel(PreTrainedModel):[m
             loss_weight: Optional[List] = None,[m
             loss_reduction_all_gather: Optional[bool] = False,[m
     ) -> Union[Tuple, CausalLMOutputWithPast]:[m
[31m-        return_dict = return_dict if return_dict is not None else self.config.use_return_dict[m
[31m-[m
[32m+[m[32m        return_dict = return_dict if return_dict is not None else self.config.use_return_dict[m[41m [m
[32m+[m[41m        [m
         image_flags = image_flags.squeeze(-1)[m
         input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()[m
 [m
[36m@@ -240,7 +241,7 @@[m [mclass InternVLChatModel(PreTrainedModel):[m
             loss = loss_fct(shift_logits, shift_labels)[m
             if ignore_flag:[m
                 loss = loss * 0.0[m
[31m-[m
[32m+[m[41m        [m
         if not return_dict:[m
             output = (logits,) + outputs[1:][m
             return (loss,) + output if loss is not None else output[m
[36m@@ -281,7 +282,7 @@[m [mclass InternVLChatModel(PreTrainedModel):[m
                 output_hidden_states=True,[m
                 return_dict=True).hidden_states[self.select_layer][m
         vit_embeds = vit_embeds[:, 1:, :][m
[31m-[m
[32m+[m[41m        [m
         h = w = int(vit_embeds.shape[1] ** 0.5)[m
         vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)[m
         vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)[m
[36m@@ -446,4 +447,4 @@[m [mclass InternVLChatModel(PreTrainedModel):[m
         return self.language_model.get_input_embeddings()[m
 [m
     def get_output_embeddings(self):[m
[31m-        return self.language_model.get_output_embeddings()[m
[32m+[m[32m        return self.language_model.get_output_embeddings()[m
\ No newline at end of file[m
[1mdiff --git a/internvl_chat/internvl/patch/__init__.py b/internvl_chat/internvl/patch/__init__.py[m
[1mindex 63f84fe..b0b145c 100644[m
[1m--- a/internvl_chat/internvl/patch/__init__.py[m
[1m+++ b/internvl_chat/internvl/patch/__init__.py[m
[36m@@ -15,7 +15,14 @@[m [mfrom .pad_data_collator import (concat_pad_data_collator,[m
                                 dpo_concat_pad_data_collator,[m
                                 pad_data_collator)[m
 from .phi3_packed_training_patch import replace_phi3_attention_class[m
[31m-from .qwen2_packed_training_patch import replace_qwen2_attention_class[m
[32m+[m[32mfrom .qwen2_packed_training_patch import (replace_qwen2_attention_class,[m
[32m+[m[32m                                          )[m
[32m+[m[32mfrom .qwen2_cuda_graphed_patch import ([m
[32m+[m[32m                                       replace_llm_model_forward,[m
[32m+[m[32m                                       build_graphed_model,[m
[32m+[m[32m                                       replace_decoder_layer_forward,[m
[32m+[m[32m                                       replace_qwen2_self_attn_forward,[m
[32m+[m[32m                                       )[m
 from .train_dataloader_patch import replace_train_dataloader[m
 from .train_sampler_patch import replace_train_sampler[m
 [m
[36m@@ -26,9 +33,14 @@[m [m__all__ = ['replace_llama_attn_with_flash_attn',[m
            'replace_train_dataloader',[m
            'replace_internlm2_attention_class',[m
            'replace_qwen2_attention_class',[m
[32m+[m[32m           'replace_llm_model_forward',[m
[32m+[m[32m           'replace_qwen2_rotary_embedding_forward',[m
[32m+[m[32m           'replace_decoder_layer_forward',[m
[32m+[m[32m           'build_graphed_model',[m
[32m+[m[32m           'replace_qwen2_self_attn_forward',[m
            'replace_phi3_attention_class',[m
            'replace_llama_attention_class',[m
            'pad_data_collator',[m
            'dpo_concat_pad_data_collator',[m
            'concat_pad_data_collator',[m
[31m-           'apply_liger_kernel_to_internvit'][m
[32m+[m[32m           'apply_liger_kernel_to_internvit'][m
\ No newline at end of file[m
[1mdiff --git a/internvl_chat/internvl/patch/qwen2_cuda_graphed_patch.py b/internvl_chat/internvl/patch/qwen2_cuda_graphed_patch.py[m
[1mindex 36aff2c..fb0fdb7 100644[m
[1m--- a/internvl_chat/internvl/patch/qwen2_cuda_graphed_patch.py[m
[1m+++ b/internvl_chat/internvl/patch/qwen2_cuda_graphed_patch.py[m
[36m@@ -2,7 +2,7 @@[m [mimport time[m
 import torch[m
 import torch.nn.functional as F[m
 from typing import Optional, Tuple, List, Union[m
[31m-from types import MethodType[m
[32m+[m
 from ..model.internvl_chat.modeling_internvl_chat import InternVLChatModel[m
 from transformers.models.qwen2.modeling_qwen2 import ([m
                                                       Qwen2FlashAttention2,[m
[36m@@ -12,20 +12,16 @@[m [mfrom transformers.models.qwen2.modeling_qwen2 import ([m
                                                       repeat_kv,[m
                                                       logger,[m
                                                       Qwen2Model,[m
[31m-                                                      Qwen2MLP,[m
[31m-                                                      Qwen2RMSNorm,[m
[31m-                                                      Qwen2Attention,[m
[31m-                                                      QWEN2_ATTENTION_CLASSES[m
                                                       )[m
 from transformers.modeling_outputs import BaseModelOutputWithPast[m
[31m-from transformers.models.qwen2.configuration_qwen2 import Qwen2Config[m
[32m+[m
 from transformers.cache_utils import DynamicCache[m
 from transformers.cache_utils import Cache[m
 from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa[m
 from transformers.utils import (is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10)[m
 if is_flash_attn_2_available():[m
     from flash_attn import flash_attn_func, flash_attn_varlen_func[m
[31m-    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input[m
[32m+[m[32m    from flash_attn.bert_padding import pad_input[m
 [m
 def _get_unpad_data(attention_mask):[m
     seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)[m
[36m@@ -37,7 +33,7 @@[m [mdef _get_unpad_data(attention_mask):[m
         cu_seqlens,[m
         max_seqlen_in_batch,[m
     )[m
[31m-[m
[32m+[m[41m        [m
 def _flash_attention_forward([m
     self,[m
     query_states,[m
[36m@@ -237,7 +233,6 @@[m [mdef replace_qwen2_self_attn_forward():[m
     Qwen2FlashAttention2.forward = flash_attn_forward[m
     Qwen2FlashAttention2._flash_attention_forward = _flash_attention_forward[m
     [m
[31m- [m
 def decoder_layer_forward([m
     self,[m
     hidden_states: torch.Tensor,[m
[36m@@ -478,5 +473,4 @@[m [mdef build_graphed_model(model: InternVLChatModel,cuda_graph_module:str,cuda_grap[m
         [m
         for idx in range(cuda_graph_layer_num):[m
             target_model.layers[idx] = graphed_layers[idx][m
[31m-        del callables[m
[31m-        [m
\ No newline at end of file[m
[32m+[m[32m        del callables[m
\ No newline at end of file[m
[1mdiff --git a/internvl_chat/internvl/train/internvl_chat_finetune.py b/internvl_chat/internvl/train/internvl_chat_finetune.py[m
[1mindex 42f6694..ef71248 100644[m
[1m--- a/internvl_chat/internvl/train/internvl_chat_finetune.py[m
[1m+++ b/internvl_chat/internvl/train/internvl_chat_finetune.py[m
[36m@@ -14,6 +14,7 @@[m [mimport warnings[m
 from copy import deepcopy[m
 from dataclasses import dataclass, field[m
 from functools import partial[m
[32m+[m
 from typing import Dict, Literal, Optional[m
 [m
 import numpy as np[m
[36m@@ -38,6 +39,10 @@[m [mfrom internvl.patch import (concat_pad_data_collator,[m
                             replace_llama_rmsnorm_with_fused_rmsnorm,[m
                             replace_phi3_attention_class,[m
                             replace_qwen2_attention_class,[m
[32m+[m
[32m+[m[32m                            replace_decoder_layer_forward,[m
[32m+[m[32m                            replace_qwen2_self_attn_forward,[m
[32m+[m[32m                            replace_llm_model_forward,[m
                             replace_train_dataloader, replace_train_sampler)[m
 from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,[m
                                       IMG_CONTEXT_TOKEN, IMG_END_TOKEN,[m
[36m@@ -61,6 +66,8 @@[m [mfrom transformers.trainer_utils import get_last_checkpoint[m
 from transformers.utils.logging import (enable_default_handler,[m
                                         enable_explicit_format, set_verbosity)[m
 [m
[32m+[m[32mfrom trainer import CustomTrainer[m
[32m+[m
 # Try to import petrel_client for image loading, fallback to PIL if unavailable[m
 try:[m
     from petrel_client.client import Client[m
[36m@@ -82,7 +89,7 @@[m [mwarnings.filterwarnings('ignore')[m
 logger = logging.getLogger(__name__)[m
 [m
 os.environ['TOKENIZERS_PARALLELISM'] = 'true'[m
[31m-[m
[32m+[m[32mos.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"[m
 [m
 @dataclass[m
 class ModelArguments:[m
[36m@@ -141,6 +148,14 @@[m [mclass ModelArguments:[m
         default=True,[m
         metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},[m
     )[m
[32m+[m[32m    recompute_num_llm_layers: int = field([m
[32m+[m[32m        default=None,[m
[32m+[m[32m        metadata={'help': "Set the num of recomputing language model's layers in training when use gradient checkpointing. Default is 0."}[m
[32m+[m[32m    )[m
[32m+[m[32m    recompute_num_vision_layers: int = field([m
[32m+[m[32m        default=None,[m
[32m+[m[32m        metadata={'help': "Set the num of recomputing vision model's layers in training when use gradient checkpointing. Default is 0."}[m
[32m+[m[32m    )[m
     drop_path_rate: float = field([m
         default=0.0,[m
         metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},[m
[36m@@ -157,7 +172,31 @@[m [mclass ModelArguments:[m
         default=False,[m
         metadata={'help': 'Set to True to use the liger kernel.'}[m
     )[m
[31m-[m
[32m+[m[32m    use_cuda_graph: Optional[bool] = field([m
[32m+[m[32m        default=False,[m
[32m+[m[32m        metadata={[m
[32m+[m[32m            "help": "If set to `True`, will use cuda graph to optimize training. "[m
[32m+[m[32m        },[m
[32m+[m[32m    )[m
[32m+[m[32m    cuda_graph_module: Literal['self_attn', 'decoder_layer'] = field([m
[32m+[m[32m        default='self_attn',[m
[32m+[m[32m        metadata={'help': 'Specify the module of cuda graph opt apply to, Only support apply cuda graph to llm. Default is self_attn.'}[m
[32m+[m[32m    )[m
[32m+[m[32m    cuda_graph_layer_num: int = field([m
[32m+[m[32m        default=0,[m
[32m+[m[32m        metadata={'help': "Set the num of layers which cuda graph applied to. Default is 0."}[m
[32m+[m[32m    )[m
[32m+[m[32m    use_llm_compile: Optional[bool] = field([m
[32m+[m[32m        default=False,[m
[32m+[m[32m        metadata={[m
[32m+[m[32m            "help": "If set to `True`, will use torch.compile to optimize llm model. "[m
[32m+[m[32m        },[m
[32m+[m[32m    )[m
[32m+[m[32m    llm_compile_mode: Literal['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'] = field([m
[32m+[m[32m        default='default',[m
[32m+[m[32m        metadata={'help': 'Specify the mode of torch.compile apply to, when use_llm_compile is True. Default is default.'}[m
[32m+[m[32m    )[m
[32m+[m[41m    [m
 [m
 @dataclass[m
 class DataTrainingArguments:[m
[36m@@ -264,6 +303,10 @@[m [mclass DataTrainingArguments:[m
         default=False,[m
         metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},[m
     )[m
[32m+[m[32m    use_seq_padding: bool = field([m
[32m+[m[32m        default=False,[m
[32m+[m[32m        metadata={'help': 'Whether to padding to max_seq_length, but it must be True when use_cuda_graph is True. Default is False.'},[m
[32m+[m[32m    )[m
 [m
 [m
 class LazySupervisedDataset(Dataset):[m
[36m@@ -906,6 +949,8 @@[m [mdef main():[m
         config.ps_version = model_args.ps_version[m
         config.min_dynamic_patch = data_args.min_dynamic_patch[m
         config.max_dynamic_patch = data_args.max_dynamic_patch[m
[32m+[m[32m        config.llm_config.recompute_num_layers = model_args.recompute_num_llm_layers[m
[32m+[m[32m        config.vision_config.recompute_num_layers = model_args.recompute_num_vision_layers[m
         model = InternVLChatModel.from_pretrained([m
             model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)[m
     else:[m
[36m@@ -938,7 +983,7 @@[m [mdef main():[m
         logger.info('Building InternVLChatModel...')[m
         model = InternVLChatModel(internvl_chat_config, vision_model, llm)[m
     model.img_context_token_id = img_context_token_id[m
[31m-[m
[32m+[m[41m    [m
     assert model.config.downsample_ratio == data_args.down_sample_ratio[m
 [m
     if model_args.mlp_path is not None:[m
[36m@@ -948,6 +993,11 @@[m [mdef main():[m
         logger.info(message)[m
     logger.info('Finished')[m
 [m
[32m+[m[32m    if model_args.use_cuda_graph:[m
[32m+[m[32m        replace_llm_model_forward()[m
[32m+[m[32m        replace_decoder_layer_forward()[m
[32m+[m[32m        replace_qwen2_self_attn_forward()[m
[32m+[m
     patch_size = model.config.vision_config.patch_size[m
     logger.info(f'model.config.force_image_size: {model.config.force_image_size}')[m
     logger.info(f'data_args.force_image_size: {data_args.force_image_size}')[m
[36m@@ -976,8 +1026,10 @@[m [mdef main():[m
     model.vision_model.gradient_checkpointing = True[m
     model.vision_model.encoder.gradient_checkpointing = True[m
     if model_args.grad_checkpoint:[m
[31m-        model.language_model._set_gradient_checkpointing()[m
[31m-[m
[32m+[m[32m        gradient_checkpointing_kwargs = {"use_reentrant": False,"preserve_rng_state": False}[m
[32m+[m[32m        gradient_checkpointing_func = partial(torch.utils.checkpoint.checkpoint, **gradient_checkpointing_kwargs)[m
[32m+[m[32m        model.language_model._set_gradient_checkpointing(gradient_checkpointing_func=gradient_checkpointing_func)[m
[32m+[m[41m        [m
     train_dataset = build_datasets([m
         data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,[m
         dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,[m
[36m@@ -1036,15 +1088,24 @@[m [mdef main():[m
             loss_reduction_all_gather=data_args.loss_reduction_all_gather,[m
         )[m
     else:[m
[31m-        collator = concat_pad_data_collator[m
[31m-[m
[31m-    trainer = Trainer([m
[32m+[m[32m        if data_args.use_seq_padding:[m
[32m+[m[32m            collator = partial(concat_pad_data_collator, max_item_length=tokenizer.model_max_length)[m
[32m+[m[32m        else:[m
[32m+[m[32m            collator = concat_pad_data_collator[m
[32m+[m[32m    if model_args.use_llm_compile:[m
[32m+[m[32m        # torch._dynamo.config.compiled_autograd = True[m
[32m+[m[32m        model.language_model = torch.compile(model=model.language_model,mode=model_args.llm_compile_mode)[m
[32m+[m[32m    trainer = CustomTrainer([m
         model=model,[m
         args=training_args,[m
         train_dataset=train_dataset if training_args.do_train else None,[m
         eval_dataset=None,[m
         tokenizer=tokenizer,[m
         data_collator=collator,[m
[32m+[m[32m        warm_steps=8,[m
[32m+[m[32m        use_cuda_graph=model_args.use_cuda_graph,[m
[32m+[m[32m        cuda_graph_module=model_args.cuda_graph_module,[m
[32m+[m[32m        cuda_graph_layer_num=model_args.cuda_graph_layer_num,[m
     )[m
 [m
     # Training[m
[1mdiff --git a/internvl_chat/internvl/train/trainer.py b/internvl_chat/internvl/train/trainer.py[m
[1mindex 66d6906..3f34db7 100755[m
[1m--- a/internvl_chat/internvl/train/trainer.py[m
[1m+++ b/internvl_chat/internvl/train/trainer.py[m
[36m@@ -28,8 +28,7 @@[m [mclass CustomTrainer(Trainer):[m
     def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:[m
         [m
         res = super().training_step(model,inputs)[m
[31m-        [m
[32m+[m[32m        # print(f"----fwj---in step{self.steps},memory allocate:{torch.cuda.memory_allocated() / 1024**3} GB, peak is {torch.cuda.max_memory_allocated() / 1024**3} GB,reserved is :{torch.cuda.memory_reserved() / 1024**3} GB")[m
         self.steps += 1[m
         return res[m
[31m-        [m
[31m- [m
\ No newline at end of file[m
[32m+[m[41m        [m
\ No newline at end of file[m
[1mdiff --git a/internvl_chat/run_sft.sh b/internvl_chat/run_sft.sh[m
[1mindex 4cc918b..b2978a6 100755[m
[1m--- a/internvl_chat/run_sft.sh[m
[1m+++ b/internvl_chat/run_sft.sh[m
[36m@@ -9,4 +9,4 @@[m [msource base_env.sh[m
 bash shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full_tq.sh[m
 [m
 [m
[31m-#--gpu-metrics-device=all --gpu-metrics-set=ga10x --gpu-metrics-frequency=1000 \[m
\ No newline at end of file[m
[32m+[m[32m#--gpu-metrics-device=all --gpu-metrics-set=ga10x --gpu-metrics-frequency=1000 \[m
[1mdiff --git a/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh b/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh[m
[1mindex b67be72..9c0b15a 100644[m
[1m--- a/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh[m
[1m+++ b/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh[m
[36m@@ -59,10 +59,20 @@[m [mtorchrun \[m
   --max_seq_length 4096 \[m
   --do_train True \[m
   --grad_checkpoint True \[m
[32m+[m[32m  --recompute_num_llm_layers 24 \[m
[32m+[m[32m  --recompute_num_vision_layers 24 \[m
   --group_by_length True \[m
   --dynamic_image_size True \[m
   --use_thumbnail True \[m
   --ps_version 'v2' \[m
[31m-  --deepspeed "zero_stage1_config.json" \[m
[32m+[m[32m  --use_seq_padding True \[m
[32m+[m[32m  --max_steps -1 \[m
[32m+[m[32m  --use_llm_compile False \[m
[32m+[m[32m  --llm_compile_mode "reduce-overhead" \[m
[32m+[m[32m  --use_cuda_graph False \[m
[32m+[m[32m  --cuda_graph_module "self_attn" \[m
[32m+[m[32m  --cuda_graph_layer_num 24 \[m
   --report_to "tensorboard" \[m
   2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"[m
[32m+[m
[32m+[m[32m# --deepspeed "zero_stage1_config.json" \[m[41m [m
\ No newline at end of file[m
[1mdiff --git a/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh b/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh[m
[1mindex 38994e2..3ca52f1 100644[m
[1m--- a/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh[m
[1m+++ b/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh[m
[36m@@ -64,6 +64,14 @@[m [mtorchrun \[m
   --dynamic_image_size True \[m
   --use_thumbnail True \[m
   --ps_version 'v2' \[m
[31m-  --deepspeed "zero_stage1_config.json" \[m
[32m+[m[32m  --use_cuda_graph True \[m
[32m+[m[32m  --use_seq_padding True \[m
[32m+[m[32m  --cuda_graph_module "decoder_layer" \[m
[32m+[m[32m  --cuda_graph_layer_num 24 \[m
[32m+[m[32m  --use_llm_compile True \[m
[32m+[m[32m  --llm_compile_mode "max-autotune-no-cudagraphs" \[m
   --report_to "tensorboard" \[m
   2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"[m
[32m+[m
[32m+[m
[32m+[m[32m#  --deepspeed "zero_stage1_config.json" \[m
\ No newline at end of file[m
