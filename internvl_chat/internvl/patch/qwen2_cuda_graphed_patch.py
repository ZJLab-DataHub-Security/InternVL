import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from ..model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from transformers.models.qwen2.modeling_qwen2 import (
                                                      Qwen2FlashAttention2,
                                                      Qwen2DecoderLayer,
                                                      apply_rotary_pos_emb,
                                                      _flash_supports_window_size,
                                                      repeat_kv,
                                                      logger,
                                                      Qwen2Model,
                                                      )
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
        
def _flash_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    cu_seqlens_k, 
    dropout=0.0,
    softmax_scale=None,
    use_sliding_windows=False,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_sliding_windows (`bool`, *optional*):
            Whether to activate sliding window attention.
    """
    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        causal = self.is_causal and query_length != 1

    # Decide whether to use SWA or not by layer index.
    if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
        use_sliding_windows = False

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        if not use_sliding_windows:
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        else:
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        if not use_sliding_windows:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )
        
        len = cu_seqlens_k.shape[0]
        mask = []
        for i in range(1, len):
            cur_seq_len = cu_seqlens_k[i] - cu_seqlens_k[i - 1]
            mask.append(torch.arange(attn_output.shape[1], device=torch.cuda.current_device()) >= cur_seq_len)
        mask = torch.stack(mask, dim=0)
        attn_output = attn_output.masked_fill(mask.unsqueeze(2).unsqueeze(3), 0.0)
    return attn_output
                
def flash_attn_forward(
    self,
    hidden_states: torch.Tensor, 
    position_ids: Optional[torch.LongTensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    **kwargs,
):    
    bsz, q_len, _ = hidden_states.size()
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len + 1)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, "sliding_window", None) is not None
        and kv_seq_len > self.config.sliding_window
        and self.config.use_sliding_window
    )

    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
            " make sure to upgrade flash-attn library."
        )
    
    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        None,
        q_len,
        cu_seqlens_k,
        dropout=dropout_rate,
        use_sliding_windows=use_sliding_windows,
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    
    return attn_output

def replace_qwen2_self_attn_forward():
    Qwen2FlashAttention2.forward = flash_attn_forward
    Qwen2FlashAttention2._flash_attention_forward = _flash_attention_forward
    
def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None, 
    cu_seqlens_k: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        hidden_states,
        position_ids,
        cu_seqlens_k,
    )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    outputs = (hidden_states,)
    
    return outputs

def replace_decoder_layer_forward():
    Qwen2DecoderLayer.forward = decoder_layer_forward

def qwen2_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    #compute inindices_k, cu_seqlens_k, max_seqlen_in_batch_k ahead.
    if attention_mask is not None:
        _, cu_seqlens_k, _ = _get_unpad_data(attention_mask)
    else:
        seq_len = 0
        cu_seqlens_k = torch.tensor([(seq_len + i * seq_length) for i in range(batch_size + 1)])
    
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__, 
                hidden_states,
                position_ids,
                cu_seqlens_k,

            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids,
                cu_seqlens_k,
            )
        
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
        
def replace_llm_model_forward():
    Qwen2Model.forward = qwen2_model_forward

def build_graphed_model(model: InternVLChatModel,max_seq_len: int, micro_bs: int, hidden_size: int):  
    mempool = torch.cuda.graph_pool_handle()
    
    input_hidden_states = torch.randn(micro_bs,max_seq_len, hidden_size,dtype=torch.bfloat16,device=torch.cuda.current_device(),requires_grad=True)
    input_position_ids = torch.tensor([list(range(max_seq_len)) for _ in range(micro_bs)], device=torch.cuda.current_device(), dtype=torch.long)
    input_cu_seqlens_k = torch.randint(low = 0, high = micro_bs * max_seq_len, size = (micro_bs + 1,), device=torch.cuda.current_device(), dtype=torch.int32,requires_grad=False)     

    target_model = model.language_model.base_model.model.model if model.config.use_llm_lora else model.language_model.model

    static_inputs = []
    layers_module = []
    for idx, decoder_layer in enumerate(target_model.layers):
        layers_module.append(decoder_layer)
        static_inputs.append((input_hidden_states, input_position_ids, input_cu_seqlens_k,))   
    
    callables = tuple(layers_module)
    static_inputs = tuple(static_inputs)
    
    graphed_layers = torch.cuda.make_graphed_callables(callables,static_inputs,pool = mempool)
    
    for idx in range(target_model.config.num_hidden_layers):
        target_model.layers[idx] = graphed_layers[idx]
    del callables
    del static_inputs