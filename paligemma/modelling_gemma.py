import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math
from modelling_siglip import SiglipVisionConfig, SiglipVisionModel


class GemmaConfig:
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            ignore_index=-100,
            image_token_index=256000, # Index for the "<image>" token
            vocab_size=257152,
            projection_dim=2048, #final dimensions that the image features should be reiszed to before feeding to the language model ie what is the ouput size of the linear layer
            hidden_size=2048, # Embedding size
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim



class KVCache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:  #  Basically num_items is used to determine whether we are in prefilling stage or generation stage
        # Each element of key cache is of shape [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        if len(self.key_cache) == 0:        
            return 0
        else:
            return self.key_cache[0].shape[-2] # Return the seq_length
        
    
    def update(
            self,
            key_states:torch.Tensor,
            value_states:torch.Tensor,
            layer_idx:int
    )->Tuple[torch.Tensor, torch.Tensor]:
        # Update the kv cache and return new key,values states
        if len(self.key_cache) <= layer_idx:
            # We have not added anything in this layer
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate the new states with the existing ones

            self.key_cache[layer_idx]=torch.cat((self.key_cache[layer_idx],key_states),dim=-2) # Concatenate along seq_length dimension
            self.value_cache[layer_idx]=torch.cat((self.value_cache[layer_idx],value_states),dim=-2) 

        return self.key_cache[layer_idx],self.value_cache[layer_idx] # Return the updated key,value states
    



class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight=torch.nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # rsqrt means 1/sqrt


    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # Ig this is basically to add some learnable paramters before sending the input to the activation fn
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # Expand the embedding to intermediate size
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False) # Compress back tto the hidden_size

    def forward(self, x):
        z=self.gate_proj(x)
        o=F.gelu(z,approximate="tanh")
        o=o*self.up_proj(x)
        return self.down_proj(o)


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    )-> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states



class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens=torch.nn.Embedding(config.vocab_size,config.hidden_size,self.padding_idx)

        self.layers=torch.nn.ModuleList([GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.norm=GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    )->torch.FloatTensor:
        
        hidden_states = inputs_embeds # [Batch_Size, Seq_Len, Hidden_Size]

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)

        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)   

        return hidden_states # [Batch_Size, Seq_Len, Hidden_Size]


class GemmaForCausalLM(nn.Module):
    def __init__(self, config:GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # LM Head is the linear layer before the softmax

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight # Sharing weights bw embedding layer and lm head

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    )->Tuple:
        
        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        ) # [Batch_Size, Seq_Len, Hidden_Size]

        hidden_states=outputs

        logits=self.lm_head(hidden_states)
        logits=logits.float()

        return_data={
            "logits":logits
        }


        if kv_cache is not None:
            # updated kv cache
            return_data["kv_cache"] = kv_cache

        
        return return_data
    



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape

    if n_rep==1:
        return  hidden_states


    # hidden_states.unsqueeze(2) adds a dimension after num_key_value_heads
    # So the shape becomes batch_size, num_key_value_heads, 1 , seq_len, head_dim
    # expand is used to duplicate the data across that dimension which we added using unsqueeze

    hidden_states=hidden_states.unsqueeze(2).expand(batch_size,num_key_value_heads,n_rep,seq_len,head_dim)

    return hidden_states.reshape(batch_size,num_key_value_heads*n_rep,seq_len,head_dim)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim=dim # Head dim , basically rotatory positional encoding is applied during attention mechanism, attention mech is performed independently for eachc  attention   head , so each head will have its own positional encoding applied to the tokens
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq=1.0/(self.base**(torch.arange(0,self.dim,2,dtype=torch.int64).float()/self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [batch_size, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)

        # Shape of inv_freq is (head_dim // 2,)
        # We have to compute θ = position * inv_freq
        # But position_ids: shape [batch_size, seq_len]
        # inv_freq: shape [head_dim // 2]

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], self.dim // 2, 1)  # [Batch_Size, Head_Dim // 2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [Batch_Size, 1, Seq_Len]

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            #[Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] 
            # =>  [Batch_Size, Head_Dim // 2 , Seq_Len]
            # Transpose => [Batch_Size,Seq_Len ,Head_Dim // 2 ]

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)

            emb = torch.cat((freqs, freqs), dim=-1) # [Batch_Size,Seq_Len ,Head_Dim ]

            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype),sin.to(dtype=x.dtype)


def rotate_half(x):
    # This is different than the original rope implementation
    # bcoz here some of the weights have been permuted or smth

    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the num_heads dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the num_heads dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed








class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):  
        super().__init__()
        self.layer_idx = layer_idx # So basically we pass the layer_idx so that we know which layer it is coz each layer has its own kv cache , so with the help of this we know which kv cache to use

        self.attention_dropout = config.attention_dropout # We wont be using
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # Number of heads for the query
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0    
        
        # shape of q_proj is different from k_proj and v_proj bcoz
        # we are using grouped query attention
        # So the number of heads for query is more than the number of heads for keys and values

        self.q_proj=torch.nn.Linear(self.hidden_size,self.num_heads*self.head_dim,bias=config.attention_bias)
        
        self.k_proj=torch.nn.Linear(self.hidden_size,self.num_key_value_heads*self.head_dim,bias=config.attention_bias)

        self.v_proj=torch.nn.Linear(self.hidden_size,self.num_key_value_heads*self.head_dim,bias=config.attention_bias)
    
        self.o_proj=nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)


        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
            **kwargs,
    )->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, q_len, _ = hidden_states.size() # [Batch_Size, Seq_Len, Hidden_Size]

        query_states = self.q_proj(hidden_states) # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]

        key_states = self.k_proj(hidden_states) # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]

        value_states = self.v_proj(hidden_states) # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]

        
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1,2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
            

        # So essentially what is supposed to happen is that:-
        # 2 query heads are supposed to share 1 key head
        # But we dont have a custom cuda kernel to make use of this
        # So what we do is we create copies of the key heads 
        # so that each query head has its own key head
        # Tho this makes the grouped MQA into simple MHA but yeah this is a naive implementation 

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] x  [batch_size,Num_Heads_Q,head_dim,seq_len_KV] 
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q , Seq_Len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        assert attention_mask is not None

        attn_weights=attn_weights+attention_mask

        attn_weights=F.softmax(attn_weights,dim=-1)

        attn_weights=F.dropout(attn_weights,p=self.attention_dropout,training=self.training)

         # Multiply by the values. 
         # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV] x [Batch_Size, Num_Heads_Q, Seq_Len_KV, Head_Dim] 
         # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size()!=(batch_size,self.num_heads,q_len,self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Make sure the sequence length is the second dimension.
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output=attn_output.view(batch_size,q_len,-1) #  [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        # Currently each embedding is just a concatenation of the independent calculation of each head
        # We need a mixing mechanism which is given by Wo

        attn_output = self.o_proj(attn_output) ## Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]


        return attn_output, attn_weights


class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        # Converts the size of the image features extracted by the vision encoder into the embedding size used by the language model
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        #[Batch_Size, Num_Patches, Hidden_Size]->#[Batch_Size, Num_Patches, projection_dim]
        
        return self.linear(image_features)


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config=self.config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    

    def _merge_input_ids_with_image_features(
            self,
            image_features: torch.Tensor,
            inputs_embeds: torch.Tensor, # Text Embeddings
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape #[Batch_Size, Num_Patches, Hidden_Size]
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size**0.5)

         #Combining embeddings of image tokens ,text tokens

        final_embedding=torch.zeros(batch_size,sequence_length,embed_dim,dtype=inputs_embeds.dtype,device=inputs_embeds.device)
        

        # (batch_size, sequence_length)
        text_mask=(input_ids!=self.config.image_token_index) & (input_ids!=self.pad_token_id)  # text token is something which is not an image placeholder token and is not a padding token
        # (batch_size, sequence_length)
        image_mask=input_ids==self.config.image_token_index
        # (batch_size, sequence_length)
        pad_mask=input_ids==self.config.pad_token_id

        # Adding a dimension at the end to match the shape of the embeddings

        # (batch_size, sequence_length) -> (batch_size, sequence_length,1) 
        text_mask = text_mask.unsqueeze(-1)
        pad_mask= pad_mask.unsqueeze(-1)    
        image_mask=image_mask.unsqueeze(-1)

        # Now we will be making the last dimension to match embed_dim

        text_mask_expanded=text_mask.expand(-1,-1,embed_dim)
        pad_mask_expanded =pad_mask.expand(-1,-1,embed_dim)
        image_mask_expanded=image_mask.expand(-1,-1,embed_dim)

        final_embedding=torch.where(text_mask_expanded,inputs_embeds,final_embedding)

        final_embedding=final_embedding.masked_scatter(image_mask_expanded,scaled_image_features)

        ## Create attention mask

        min_dtype = torch.finfo(dtype).min #gets the smallest possible number (most negative) that can be represented in that data type.
        # Used to simulate negative infinity when creating attention masks.

        q_len = inputs_embeds.shape[1]


        if kv_cache is None or kv_cache.num_items() == 0: # Prefilling 
            # no need to mask any token, because we're in the prefill phase
            causal_mask = torch.full((batch_size,q_len,q_len),fill_value=0,dtype=dtype,device=device)
        else:
            assert q_len == 1 # Since it is generating phase ,query must be single token

            kv_len = kv_cache.num_items() + q_len

            # no need to mask anything becasue each query should be able to attend all previous tokens
            
            causal_mask=torch.full((batch_size,q_len,kv_len),fill_value=0,dtype=dtype,device=device)

        # As you can see above we are not masking anything
        # So the reason is that during inference we dont have to mask anythng bcoz of the explained reasons 
        # But during training we will have to mask the future tokens(tho  this implementation is only concerned about inference)


        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        # When we do the attention computation there will be one attention computation per head sos we will have one attention  matrix for each head

        causal_mask = causal_mask.unsqueeze(1)

        # Causal mask is used to prevent the model from cheating by looking at the future tokens
        # Whereas attention mask is used to mask out padding tokens in the input.

        if kv_cache is not None and kv_cache.num_items() > 0:

            position_ids=attention_mask.cumsum(-1)[:,-1] 
            
            if position_ids.dim()==1:
                position_ids=position_ids.unsqueeze(0) # Adding a batch dimension

            
        else:
            position_ids=attention_mask.cumsum(-1).masked_fill_((attention_mask==0),1).to(device)

        return final_embedding, causal_mask, position_ids
            
    def forward(
            self,
            input_ids:torch.LongTensor=None, #  a tensor with data type int64(long in Pytorch) coz token IDs are integers, not floats.
            pixel_values:torch.FloatTensor=None, # tensor is of data type float32.
            attention_mask: Optional[torch.Tensor]=None,
            kv_cache:Optional[KVCache]=None
    )->Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extract the input embeddings which are of shape (Batch_Size, Seq_Len, Hidden_Size)
        embedding_layer=self.language_model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

