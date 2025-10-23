# Cognitive Maps in Language Models

This repository contains the complete codebase for an MSc thesis investigating whether transformer models develop cognitive maps or task-specific heuristics when trained on spatial navigation tasks.

## Research Question

This work addresses whether generative pre-trained transformers learn generalizable cognitive maps or develop task-specific spatial heuristics when trained on spatial navigation tasks.

Three models were trained on different spatial learning paradigms:
1. **Foraging Model** - Trained on random walks to simulate passive exploratory learning
2. **SP-Hamiltonian Model** - Trained on optimal shortest path demonstrations using structured Hamiltonian contexts
3. **SP-Random Walk Model** - Fine-tuned from SP-Hamiltonian on unstructured random walks

## Repository Structure

### `mech_interp_foraging/` - Foraging Model Experiments
Mechanistic interpretability experiments for the foraging model.

**Files:**
- `exp1_activation_patching.ipynb` - Activation patching experiments to identify causal mechanisms
- `exp2_pca_linear_probing.ipynb` - Principal component analysis and linear probing of hidden representations
- `exp3_direction_swapping.ipynb` - Cross-context transfer experiments
- `exp4_direction_ablation.ipynb` - Direction ablation experiments to test information redundancy
- `reverse_bias.ipynb` - Analysis of reverse bias in minimal context scenarios
- `utils/model_utils.py` - Model loading and hidden state extraction utilities
- `utils/grid_utils.py` - Grid generation and random walk utilities

### `mech_interp_SP/` - Shortest Path Model Experiments
Mechanistic interpretability experiments for shortest path models.

**Files:**
- `sp_direction_ablation.ipynb` - Direction ablation experiments for SP models
- `sp_pc.ipynb` - Principal component analysis of SP model representations
- `utils/model_utils.py` - Model loading and hidden state extraction utilities
- `utils/grid_utils.py` - Grid generation and random walk utilities

### `training_scripts/` - Model Training Code
Training implementations for all three model variants.

**Files:**
- `foraging_model_training.ipynb` - Foraging model training on random walk sequences
- `sp_hamiltonian_training.ipynb` - SP-Hamiltonian model training on structured shortest path data
- `sp_rw_training.ipynb` - SP-Random Walk model fine-tuning on unstructured contexts
- `train_shortest.py` - Python script implementation of shortest path training
- `run_clm.py` - General causal language model training script

### `performance_tests/` - Model Performance Evaluation
Behavioral analysis and generalization testing.

**Files:**
- `foraging_pt.ipynb` - Foraging model performance across grid sizes
- `sp_generalisation.ipynb` - SP model generalization to larger grids
- `sp_models_pt.ipynb` - SP model performance evaluation
- `testSPRW_latecontexts.ipynb` - SP-Random Walk model context length robustness


## Installation and Usage

### Dependencies
```bash
pip install torch transformers numpy matplotlib seaborn scikit-learn networkx tqdm pandas tokenizers
```

### Prerequisites
Trained model checkpoints are required to run experiments. Models should be saved in the expected directory structure.

### Execution
1. **Foraging experiments**: Execute notebooks in `mech_interp_foraging/` sequentially
2. **Shortest path experiments**: Execute notebooks in `mech_interp_SP/`
3. **Model training**: Use notebooks in `training_scripts/`

## Citation

Caroline Baumgartner, MSc Artificial Intelligence for Biomedicine and Healthcare, University College London. Supervised by Dr. Neil Burgess, Dr. Eleanor Spens, and Dr. Petru Manescu.
