from config import TOKENIZER, MLM_MODEL
import torch

class MLMModel:
    """Masked Language Modeling - predict [MASK] tokens in a sentence"""

    def __init__(self):
        self.tokenizer = TOKENIZER
        self.model = MLM_MODEL

    def predict_mask(self, text: str) -> str:
        """predicts word for [MASK]"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        # Get top prediction
        predicted_token_id = logits[0, mask_token_index, :].argmax(dim=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)

        return predicted_token
