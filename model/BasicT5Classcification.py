from transformers import T5ForSequenceClassification

from torch import nn

class BasicT5Classification(nn.Module):
    def __init__(self, numClass):
        super(BasicT5Classification, self).__init__()
        self.baseModel = T5ForSequenceClassification.from_pretrained('google-t5/t5-small', num_labels=numClass)

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            ret = self.baseModel(input_ids, attention_mask, labels=labels)
            return ret.logits, ret.loss
        else:
            ret = self.baseModel(input_ids, attention_mask)
        return ret.logits, None
