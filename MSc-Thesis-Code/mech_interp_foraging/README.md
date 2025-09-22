# MATS Application Code

Clean, minimalistic code to reproduce the mechanistic interpretability experiments for the foraging model.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a trained model checkpoint at `fv1_model/checkpoint-33000/`

## Execution Order

1. **01_training.py** - Train the foraging model (if needed)
2. **02_activation_patching.py** - Minimal pair patching experiments
3. **03_pca_linear_probing.py** - PCA and linear probing analysis
4. **04_direction_swapping.py** - Direction swapping experiments
5. **05_direction_ablation.py** - Direction ablation experiments

## Expected Results

- **Activation Patching**: Layer 0 attention shows causal control over directional updates
- **Cross-context Patching**: ~88% universal move success rate
- **PCA**: Three-stage evolution (noisy → coordinate → functional clustering)
- **Linear Probing**: Peak R² = 0.93 at Layer 8
- **Direction Ablation**: Sharp transition at Layer 7 (0% → 100% accuracy)
- **Direction Swapping**: Cosine similarity ≈1.0 for unconstrained, ≈0.3 for constrained nodes

## Key Findings

- Layer 1 attention handles directional updates
- Layer 7: coordinate system becomes self-sufficient
- Late layers: functional clustering by node type
- Information absorption: direction tokens redundant after Layer 7
