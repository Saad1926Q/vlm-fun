from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from torch.linalg import vector_norm
import pickle
import torch
import os


class CaptionsDataset(Dataset):
    def __init__(self,data_path:str,prefix_length:int,normalize_prefix:bool=False):
        super().__init__()
        self.tokenizer:GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data=pickle.load(file=f)
        
        print(f"Data size is {len(all_data["clip_embedding"])}")
        
        self.prefixes = all_data["clip_embedding"]
        captions_dicts = all_data["captions"]  # List of dictionaries with keys => "caption", "image_id","clip_embedding"

        self.image_ids = [caption["image_id"] for caption in captions_dicts]
        self.captions = [caption['caption'] for caption in captions_dicts]
        
        # Check if the tokenized version of the captions have already been saved 
        # We extract the part before .pkl from the data_path by doing data_path[:-4]

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.prefix_indices, self.max_seq_len = pickle.load(f)

        else:
            self.captions_tokens = []
            self.prefix_indices = []
            max_seq_len = 0

            for caption in captions_dicts:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']),dtype=torch.int64))  # encode method converts a string to a sequence of ids
                self.prefix_indices.append(caption["clip_embedding_id"])

                max_seq_len=max(max_seq_len,len(self.captions_tokens[-1]))
            
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.prefix_indices, max_seq_len], f)

            self.max_seq_len=min(64,max_seq_len) # Just to ensure that the max seq len doesnt become too much


    def __len__(self)->int:
        return len(self.captions_tokens)
    
    def pad_or_truncate_tokens(self,idx:int):
        tokens=self.captions_tokens[idx]
        padding=self.max_seq_len-len(tokens)

        if padding>0:
            tokens = torch.cat((tokens,torch.full((padding,),-1,dtype=torch.int64)))

        elif padding<0:
            # If sequence is longer than max_seq_len then we will have to truncate
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[idx] = tokens

        mask=torch.ge(tokens,0)
        tokens[~mask] = 0
        mask=mask.float()
        prefix_mask=torch.ones(self.prefix_length,dtype=torch.float32,device=tokens.device)
        full_mask = torch.cat((prefix_mask,mask))
        
        return tokens,full_mask

    
    def __getitem__(self,idx:int):
        tokens, mask = self.pad_or_truncate_tokens(idx)

        prefix=self.prefixes[self.prefix_indices[idx]]

        if self.normalize_prefix:
            prefix=prefix.float()
            prefix=prefix/vector_norm(prefix,ord=2)

        return tokens,mask,prefix

        