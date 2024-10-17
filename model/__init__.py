from transformers import T5Tokenizer

class TKNZ:
    def __init__(self, device):
        self.device = device
        self.tknz = T5Tokenizer.from_pretrained('google-t5/t5-small')

    def __call__(self, texts):
        ret = self.tknz(texts, return_tensors="pt", padding=True)
        return ret.input_ids.to(self.device), ret.attention_mask.to(self.device)
