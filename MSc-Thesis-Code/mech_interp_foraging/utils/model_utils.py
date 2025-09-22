import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer."""
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def get_hidden_states(model, tokens, layer_idx):
    """Extract hidden states from specified layer."""
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    return outputs.hidden_states[layer_idx]

def find_token_position(tokens, target_word, tokenizer):
    """Find position of target word in tokenized sequence."""
    token_strings = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    for i in range(len(token_strings) - 1, -1, -1):
        if target_word in token_strings[i]:
            return i
    return -1


def get_hidden_states_with_offsets(model, tokenizer, text, layer_idx):
    """
    Gets hidden states and token-to-character offset mappings for a given text and layer.
    """
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer_idx].squeeze(0).cpu() # (Seq_Len, Hidden_Dim)
    offsets = inputs["offset_mapping"].squeeze(0).cpu().tolist() # (Seq_Len, 2)
    
    return hidden_states, offsets

def substring_positions(haystack, needle, start_search=0, end_search=None):
    """
    Finds all occurrences of a needle (as a whole word) in a haystack string.
    Returns a list of [start, end] character indices.
    """
    if end_search is None:
        end_search = len(haystack)
    
    results, start = [], start_search
    while True:
        idx = haystack.find(needle, start, end_search)
        if idx == -1:
            break
        
        # Check for whole word match
        is_word_start = (idx == 0) or (not haystack[idx-1].isalnum())
        is_word_end = (idx + len(needle) == len(haystack)) or (not haystack[idx + len(needle)].isalnum())
        
        if is_word_start and is_word_end:
            results.append([idx, idx + len(needle)])
            
        start = idx + len(needle)
    return results
    
def gather_embeddings_for_span(hidden_states, offsets, span):
    """
    Averages the hidden states of tokens that overlap with a given character span.
    """
    start_needed, end_needed = span
    
    overlapping_vectors = [
        hidden_states[i] for i, (start_offset, end_offset) in enumerate(offsets) 
        if not (end_offset <= start_needed or start_offset >= end_needed) and start_offset != end_offset
    ]
    
    if overlapping_vectors:
        return torch.mean(torch.stack(overlapping_vectors), axis=0).numpy()
    return None

class ComponentPatcher:
    """Context manager for patching component outputs."""
    def __init__(self, model, layer_idx, component_type, patch_tensor, target_position):
        self.model = model
        self.layer_idx = layer_idx
        self.component_type = component_type.lower()
        self.patch_tensor = patch_tensor
        self.target_position = target_position
        self.hook = None

    def _get_target_module(self):
        block = self.model.transformer.h[self.layer_idx]
        if self.component_type == 'mlp':
            return block.mlp
        elif self.component_type == 'attention':
            return block.attn
        else:
            raise ValueError(f"Unknown component_type: {self.component_type}")

    def _patch_hook(self, module, input, output):
        is_attn = isinstance(output, tuple)
        hidden_states = output[0] if is_attn else output
        patched_hidden_states = hidden_states.clone()
        patched_hidden_states[0, self.target_position, :] = self.patch_tensor
        
        if is_attn:
            return (patched_hidden_states,) + output[1:]
        else:
            return patched_hidden_states

    def __enter__(self):
        target_module = self._get_target_module()
        self.hook = target_module.register_forward_hook(self._patch_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()

class AblationPatcher:
    """Context manager to ablate a transformer block by making it an identity function."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hook = None

    def ablation_hook(self, module, module_input, module_output):
        # Return input hidden states unchanged, making the layer an identity function
        return (module_input[0],) + module_output[1:]

    def __enter__(self):
        target_block = self.model.transformer.h[self.target_layer]
        self.hook = target_block.register_forward_hook(self.ablation_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()
