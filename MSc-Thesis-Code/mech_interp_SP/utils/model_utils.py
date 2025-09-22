# utils/model_utils.py

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer."""
    # Using Auto* classes is generally more robust
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Add special tokens required by the SP models if they are missing
    special_tokens_dict = {
        'pad_token': '[PAD]', 
        'additional_special_tokens': ['[SOS]','[SEP]','[PLAN]','[EOS]','[SHORTEST]', '[START_NODE]', '[GOAL]']
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    if num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))

    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def get_hidden_states(model, tokens, layer_idx):
    """Extract hidden states from specified layer."""
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    return outputs.hidden_states[layer_idx]

def find_token_position(tokens, target_word, tokenizer):
    """Find position of target word's last token in a sequence."""
    token_strings = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    # Search backwards to find the last occurrence, which is usually correct for patching
    for i in range(len(token_strings) - 1, -1, -1):
        # Using strip() makes it robust to leading spaces (like 'Ġ') in GPT-2 tokens
        if target_word in token_strings[i] or target_word in token_strings[i].strip():
            return i
    return None # Return None if not found, easier to check than -1

def get_hidden_states_with_offsets(model, tokenizer, text, layer_idx):
    """Gets hidden states and token-to-character offset mappings."""
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[layer_idx].squeeze(0).cpu()
    offsets = inputs["offset_mapping"].squeeze(0).cpu().tolist()
    return hidden_states, offsets

def substring_positions(haystack, needle, start_search=0, end_search=None):
    """Finds all whole-word occurrences of a needle in a haystack."""
    if end_search is None: end_search = len(haystack)
    results, start = [], start_search
    while True:
        idx = haystack.find(needle, start, end_search)
        if idx == -1: break
        is_word_start = (idx == 0) or (not haystack[idx-1].isalnum())
        is_word_end = (idx + len(needle) == len(haystack)) or (not haystack[idx + len(needle)].isalnum())
        if is_word_start and is_word_end:
            results.append([idx, idx + len(needle)])
        start = idx + len(needle)
    return results
    
def gather_embeddings_for_span(hidden_states, offsets, span):
    """Averages the hidden states of tokens that overlap with a character span."""
    start_needed, end_needed = span
    overlapping_vectors = [
        hidden_states[i] for i, (start_offset, end_offset) in enumerate(offsets) 
        if not (end_offset <= start_needed or start_offset >= end_needed) and start_offset != end_offset
    ]
    if overlapping_vectors:
        return torch.mean(torch.stack(overlapping_vectors), axis=0).numpy()
    return None

# --- NEW FUNCTION THAT WAS MISSING ---
def get_component_output(model, tokens, position, layer_idx, component_type):
    """
    Runs a forward pass and captures the output of a specific component.

    Args:
        model: The transformer model.
        tokens: The input tokens.
        position (int): The token position from which to get the activation.
        layer_idx (int): The layer index of the component.
        component_type (str): 'mlp' or 'attention'.

    Returns:
        A detached tensor of the component's output at the specified position.
    """
    cache = {}
    def capture_hook(_, __, output):
        is_attn = isinstance(output, tuple)
        hidden_states = output[0] if is_attn else output
        # Get the activation at the specified token position
        cache['output'] = hidden_states[0, position, :].detach().clone()

    # Determine the target module
    block = model.transformer.h[layer_idx]
    if component_type.lower() == 'mlp':
        target_module = block.mlp
    elif component_type.lower() == 'attention':
        target_module = block.attn
    else:
        raise ValueError(f"Unknown component_type: {component_type}")

    # Register the hook, run the forward pass, and remove the hook
    hook = target_module.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**tokens)
    hook.remove()
    
    return cache.get('output')


class ComponentPatcher:
    """Context manager for patching component outputs."""
    def __init__(self, model, layer_idx, component_type, patch_tensor, target_position):
        self.model = model; self.layer_idx = layer_idx
        self.component_type = component_type.lower()
        self.patch_tensor = patch_tensor; self.target_position = target_position
        self.hook = None

    def _get_target_module(self):
        block = self.model.transformer.h[self.layer_idx]
        if self.component_type == 'mlp': return block.mlp
        elif self.component_type == 'attention': return block.attn
        else: raise ValueError(f"Unknown component_type: {self.component_type}")

    def _patch_hook(self, module, input, output):
        is_attn = isinstance(output, tuple)
        hidden_states = output[0] if is_attn else output
        patched_hidden_states = hidden_states.clone()
        patched_hidden_states[0, self.target_position, :] = self.patch_tensor
        return (patched_hidden_states,) + output[1:] if is_attn else patched_hidden_states

    def __enter__(self):
        self.hook = self._get_target_module().register_forward_hook(self._patch_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook: self.hook.remove()

class AblationPatcher:
    """Context manager to ablate a transformer block."""
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer
        self.hook = None

    def ablation_hook(self, module, module_input, module_output):
        return (module_input[0],) + module_output[1:]

    def __enter__(self):
        self.hook = self.model.transformer.h[self.target_layer].register_forward_hook(self.ablation_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook: self.hook.remove()

# Add this class to utils/model_utils.py

class MultiComponentPatcher:
    """Context manager for patching multiple component outputs at different positions."""
    def __init__(self, model, layer_idx, component_type, patch_dict):
        # patch_dict should be a dictionary of {target_position: patch_tensor}
        self.model = model
        self.layer_idx = layer_idx
        self.component_type = component_type.lower()
        self.patch_dict = patch_dict
        self.hook = None

    def _get_target_module(self):
        block = self.model.transformer.h[self.layer_idx]
        if self.component_type == 'mlp': return block.mlp
        elif self.component_type == 'attention': return block.attn
        else: raise ValueError(f"Unknown component_type: {self.component_type}")

    def _patch_hook(self, module, input, output):
        is_attn = isinstance(output, tuple)
        hidden_states = output[0] if is_attn else output
        patched_hidden_states = hidden_states.clone()
        
        # Iterate through all positions we need to patch
        for position, patch_tensor in self.patch_dict.items():
            patched_hidden_states[0, position, :] = patch_tensor
        
        return (patched_hidden_states,) + output[1:] if is_attn else patched_hidden_states

    def __enter__(self):
        self.hook = self._get_target_module().register_forward_hook(self._patch_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook: self.hook.remove()