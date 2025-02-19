from typing import List, Optional
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines.token_classification import TokenClassificationPipeline

class TokenClassificationChunkPipeline(TokenClassificationPipeline):
    """Extended TokenClassificationPipeline that handles long text by chunking."""
    
    def preprocess(self, sentence: str, offset_mapping=None, **preprocess_params):
        """Process text in chunks to handle long sequences.
        
        Args:
            sentence: Text to process
            offset_mapping: Optional mapping of token offsets
            preprocess_params: Additional preprocessing parameters
        """
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        truncation = bool(self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0)
        
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            max_length=self.tokenizer.model_max_length,
            padding=True
        )
        
        num_chunks = len(inputs["input_ids"])
        for i in range(num_chunks):
            model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            if offset_mapping is not None:
                model_inputs["offset_mapping"] = offset_mapping
            model_inputs["sentence"] = sentence if i == 0 else None
            model_inputs["is_last"] = i == num_chunks - 1
            yield model_inputs

    def _forward(self, model_inputs):
        """Forward pass through the model.
        
        Args:
            model_inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of processed outputs
        """
        # Extract special inputs
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")
        overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        # Model forward pass
        output = self.model(**model_inputs)
        logits = output["logits"] if isinstance(output, dict) else output[0]

        # Prepare outputs
        model_outputs = {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "is_last": is_last,
            **model_inputs,
        }

        # Reshape outputs for postprocessing
        for key in ["input_ids", "token_type_ids", "attention_mask", "special_tokens_mask"]:
            model_outputs[key] = torch.reshape(model_outputs[key], (1, -1))
        model_outputs["offset_mapping"] = torch.reshape(model_outputs["offset_mapping"], (1, -1, 2))

        return model_outputs

class TextAnonymizer:
    """Class for anonymizing text using NER."""
    
    def __init__(self, model_name: str = "Davlan/bert-base-multilingual-cased-ner-hrl"):
        """Initialize the anonymizer.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipe = TokenClassificationChunkPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
        self.ignore_list = set()

    def add_to_ignore_list(self, words: List[str]) -> None:
        """Add words to the ignore list.
        
        Args:
            words: List of words to ignore during anonymization
        """
        self.ignore_list.update(word.lower() for word in words)

    def load_ignore_list_from_file(self, file_path: str, column_name: str) -> None:
        """Load words to ignore from a CSV file.
        
        Args:
            file_path: Path to CSV file
            column_name: Name of column containing words
        """
        words_df = pd.read_csv(file_path)
        words = words_df[column_name].tolist()
        # Handle words with slashes
        words = [subword for word in words for subword in word.split('/')]
        self.add_to_ignore_list(words)

    def anonymize_text(self, text: str) -> str:
        """Anonymize named entities in text.
        
        Args:
            text: Text to anonymize
            
        Returns:
            Anonymized text with entities replaced by their types
        """
        if not text:
            return text
            
        entities = self.pipe(text)
        anonymized_text = text
        
        if entities:
            for entity in entities:
                if entity["word"].lower() in self.ignore_list:
                    continue
                anonymized_text = anonymized_text.replace(
                    entity["word"],
                    f"[{entity['entity_group']}]",
                    1
                )
                
        return anonymized_text

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Anonymize text in a dataframe column.
        
        Args:
            df: Input dataframe
            text_column: Name of column containing text to anonymize
            output_column: Name of column to store anonymized text (defaults to text_column + '_anonymized')
            
        Returns:
            Dataframe with anonymized text
        """
        if output_column is None:
            output_column = f"{text_column}_anonymized"
            
        df[output_column] = df[text_column].apply(self.anonymize_text)
        return df

def main():
    """Example usage of the TextAnonymizer class."""
    # Initialize anonymizer
    anonymizer = TextAnonymizer()
    
    # Add words to ignore
    anonymizer.add_to_ignore_list(['god', 'ok', 'bot', 'jesus'])
    
    # Optional: Load additional words from file
    # anonymizer.load_ignore_list_from_file('mysticality_dict.csv', 'Final Lexicon')
    
    # Example text anonymization
    text = "John and Mary went to Paris last summer."
    anonymized = anonymizer.anonymize_text(text)
    print(f"Original: {text}")
    print(f"Anonymized: {anonymized}")

if __name__ == "__main__":
    main()