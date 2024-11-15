from transformers import T5EncoderModel
from torch import nn

class MaskBasedClassification(nn.Module):
    def __init__(self, numClass):
        super(MaskBasedClassification, self).__init__()
        self.baseModel = T5EncoderModel.from_pretrained('google-t5/t5-small')
        self.linear = nn.Linear(512, numClass)
        self.lossFunc = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            ret = self.baseModel(input_ids, attention_mask)
            logits = self.linear(ret.last_hidden_state[:,0,:])
            loss = self.lossFunc(logits, labels)
            return logits, loss
        else:
            ret = self.baseModel(input_ids, attention_mask)
            logits = self.linear(ret.last_hidden_state[:,0,:])
        return logits, None
