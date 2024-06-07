import torch
from transformers import RobertaForSequenceClassification

class RobertaClassification(torch.nn.Module):
    def __init__(self, num_labels):
        super(RobertaClassification, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits
