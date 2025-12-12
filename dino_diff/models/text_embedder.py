from transformers import T5EncoderModel, T5Tokenizer
import torch.nn as nn


class T5TextEmbedder(nn.Module):
    def __init__(self, pretrained_path="google/flan-t5-small"):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(pretrained_path)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)

    def get_hidden_size(self):
        return self.model.config.d_model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        return embeddings
