import torch
from transformers import LlamaModel

class LLamaClass(torch.nn.Module):
    def __init__(self):
        super(LLamaClass, self).__init__()
        self.llama= LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf", return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 6)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.llama(
            input_ids, 
            attention_mask=attn_mask, 
           # token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output
