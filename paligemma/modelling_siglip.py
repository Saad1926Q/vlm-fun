from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size=768,  # Size of the embedding vector of the vision transformer that we are going to use
            intermediate_size=3072, # Size of the linear layer that we use in the feedforward NN
            num_hidden_layers=12, # Number of transformer encoder layers(blocks)
            num_attention_heads=12, #Number of attention heads per transformer block
            num_channels=3, # RGB
            image_size=224,
            patch_size=16, # each patch is 16x16
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None, # How many output embeddings will we get from the vision transformer , so basically it takes in a number of ptaches as inputs and returns a sequence of contextualized embeddings for each of the patches
            **kwargs
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens



class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.embed_dim=config.hidden_size
        self.image_size=config.image_size
        self.patch_size=config.patch_size

        self.patch_embedding=torch.nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid" # No Padding
        )

        self.num_patches=(self.image_size//self.patch_size)**2
        self.num_positions=self.num_patches

        self.position_embedding=torch.nn.Embedding(self.num_positions,self.embed_dim)

        self.register_buffer( # buffer is a non-trainable tensor saved inside the model. Used for constants you need at inference but don’t want to train
            "position_ids",
            torch.arange(self.num_positions).expand(1,-1),  # (1,196)
            persistent=False # Normall buffers are stored in state_dict() but persistent=False tells pytorch that we dont need to save it.
        )

    def forward(self,pixel_values:torch.FloatTensor)->torch.Tensor:
        _,_,height,width=pixel_values.shape # batch_size,channels,height,width

        patch_embeds=self.patch_embedding(pixel_values) # (batch_size,embed_dimension,num_patches_H,num_patches_W)

        # num_patches_H=height//patch_size
        # num_patches_W=width//patch_size

        embeddings=patch_embeds.flatten(2) # flatten starting at dimension 2
        # (batch_size,embed_dimension,num_patches_H,num_patches_W) -> (batch_size,embed_dimension,num_patches)
        # where num_patches = num_patches_H x num_patches_W

        embeddings=embeddings.transpose(1,2) # (batch_size,embed_dimension,num_patches) -> (batch_size,num_patches,embed_dimension)

        embeddings=embeddings+self.position_embedding(self.position_ids)

        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale=self.head_dim**(-0.5)  # 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self,hidden_states:torch.Tensor)->Tuple[torch.Tensor,Optional[torch.Tensor]]:
        batch_size,seq_len,_= hidden_states.shape  # (batch_size,num_patches,embed_dimension)

        query_states = self.q_proj(hidden_states) # (batch_size,num_patches,embed_dimension)
        key_states=self.k_proj(hidden_states) # (batch_size,num_patches,embed_dimension)
        value_states=self.v_proj(hidden_states) # (batch_size,num_patches,embed_dimension)

        # Reshaping from (batch_size,num_patches,embed_dimension) -> (batch_size,num_patches,num_heads,head_dim)
        # embedding dimension is split across heads

        query_states=query_states.view(batch_size,seq_len,self.num_heads,self.head_dim)
        key_states=key_states.view(batch_size,seq_len,self.num_heads,self.head_dim)
        value_states=value_states.view(batch_size,seq_len,self.num_heads,self.head_dim)

        # We will be taking transpose along dimensions 1,2 so that we can parallelize the computation
        # (batch_size,num_patches,num_heads,head_dim) -> (batch_size,num_heads,num_patches,head_dim)
        query_states=query_states.transpose(1,2)    
        key_states=key_states.transpose(1,2)
        value_states=value_states.transpose(1,2)

        # now when we will multiply the query and the transpose of key , then batch_size and num_heads dimensions will be preserved
        # (batch_size,num_heads,num_patches,head_dim) * (batch_size,num_heads,head_dim,num_patches)=(batch_size,num_heads,num_patches,num_patches)
        attn_weights=torch.matmul(query_states,key_states.transpose(2,3))*self.scale 

        if attn_weights.shape != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Now calc the weighted sum
        # (batch_size,num_heads,num_patches,num_patches) * (batch_size,num_heads,num_patches,head_dim) =>  (batch_size,num_heads,num_patches,head_dim)
        # Each head will output a list of contextualized embeddings

        attn_output=torch.matmul(attn_weights,value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Now we will merge the contents of the attention heads to get the complete contextualized embeddings

        attn_output=attn_output.transpose(1,2).contiguous() #   (batch_size,num_heads,num_patches,head_dim) ->   (batch_size,num_patches,num_heads,head_dim) 
        attn_output=attn_output.reshape(batch_size,seq_len,self.embed_dim)

        # We dot want each token to be a contextualized version of multiple subtokens each calculated independently from each other by the multihead attention
        # We also want to mix the result of this multi head attention
        # so we multiply by Wo

        attn_output=self.out_proj(attn_output) # Shape remains the same

        return attn_output,attn_weights




class SiglipMLP(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.fc1=nn.Linear(config.hidden_size,config.intermediate_size)
        self.fc2=nn.Linear(config.intermediate_size,config.hidden_size)

    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        hidden_states=self.fc1(hidden_states) # (batch_size,num_patches,embed_dimension) -> (batch_size,num_patches,intermediate_size)

        hidden_states=F.gelu(hidden_states,approximate="tanh")

        hidden_states=self.fc2(hidden_states) # (batch_size,num_patches,intermediate_size) -> (batch_size,num_patches,embed_dimension) 

        return hidden_states
    
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim=config.hidden_size
        self.self_attn=SiglipAttention(config)
        self.layer_norm1=nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp=SiglipMLP(config)
        self.layer_norm2=nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(
            self,
            hidden_states:torch.Tensor
    )->torch.Tensor:
        residual=hidden_states # (batch_size,num_patches,embed_dimension)

        hidden_states = self.layer_norm1(hidden_states) # (batch_size,num_patches,embed_dimension)

        hidden_states,_=self.self_attn(hidden_states=hidden_states) # (batch_size,num_patches,embed_dimension)

        hidden_states=hidden_states+residual # (batch_size,num_patches,embed_dimension)

        residual=hidden_states  # (batch_size,num_patches,embed_dimension)

        hidden_states=self.layer_norm2(hidden_states)  # (batch_size,num_patches,embed_dimension)

        hidden_states = self.mlp(hidden_states)  # (batch_size,num_patches,embed_dimension)

        hidden_states=hidden_states+residual  # (batch_size,num_patches,embed_dimension)

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.layers=torch.nn.ModuleList([SiglipEncoderLayer(config=config) for _ in range(config.num_hidden_layers)])

    def forward(self,input_embeds:torch.Tensor)->torch.Tensor:
        hidden_states=input_embeds  # (batch_size,num_patches,embed_dimension)

        for encoder_layer in self.layers:
            hidden_states=encoder_layer(hidden_states)  # (batch_size,num_patches,embed_dimension)

        return hidden_states
 


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config) # Responsible for extracting the patches
        self.encoder = SiglipEncoder(config)  # List of layers of the transformer
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # takes a batch of images and returns a batch of list of embeddings
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 