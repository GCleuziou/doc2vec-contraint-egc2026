# Doc2Vec with Constraints for Program Embeddings

This repository contains the code accompanying the paper "Enrichissement d'embeddings de code par contraintes expertes pour l'enseignement de la programmation" published by Thibaut Martinet et al. to EGC 2026 [1].

## Overview

This implementation extends the Doc2Vec model by integrating expert constraints on documents (programs). The approach enables the transfer of expert knowledge from document embeddings (D) to word embeddings (W, W'), making the learned representations effective at inference time on new documents.

## Requirements
```
python>=3.8
torch>=2.0
numpy
matplotlib
gensim
transformers
scikit-learn
```

Install dependencies:
```bash
pip install torch numpy matplotlib gensim transformers scikit-learn
```

## Dataset

The experiments use the NC1014 dataset [2], a subset of student Python programs. The dataset is included in this repository.

## Code Structure
```
.
└── doc2vec_constraints_loss.py  # Main implementation
    ├── Doc2VecWithDynamicConstraints  # Model class
    ├── ConstraintManager               # Constraint management
    └── EnhancedDoc2VecTrainerWithDynamicConstraints  # Training and inference
```

## Usage

### Basic Training with Constraints
```python
from doc2vec_constraints_loss import EnhancedDoc2VecTrainerWithDynamicConstraints
import numpy as np

# Initialize trainer
trainer = EnhancedDoc2VecTrainerWithDynamicConstraints(
    embed_dim=50,
    window_size=5,
    negative_samples=5
)

# Define constraints (must-link pairs)
# Example: constraint_matrix[i,j] = 1 means documents i and j should be close
num_docs = len(documents)
constraint_matrix = np.zeros((num_docs, num_docs))
# Add your constraints here...

# Train with constraints
doc_embeddings, context_embeddings, output_embeddings = trainer.train_doc2vec_with_constraints(
    documents=documents,
    epochs=500,
    lr=0.01,
    batch_size=32,
    constraint_matrix=constraint_matrix,
    constraint_lambda=0.1,  # Beta parameter in the paper
    save_directory="checkpoints",
    save_at_epochs=[100, 200, 300, 400, 500]
)
```

### Inference on New Documents
```python
# Load trained model and infer on new documents
inference_results = trainer.infer_new_documents(
    model_or_checkpoint="checkpoints/checkpoint_epoch_500.pth",
    new_documents=new_documents,
    inference_strategy='adaptive',  # or 'fixed' or 'proportional'
    inference_lr=0.001
)

inferred_embeddings = inference_results['embeddings']
```
This code is released under the MIT License.

## References

[1] Th. Martinet, G.  Cleuziou, M. Exbrayat et F. Flouvat. Enrichissement d'embeddings de code par contraintes expertes pour l'enseignement de la programmation. 26èmes Journées Extraction et Gestion des Connaissances (EGC 2026), Jan 2026, Anglet, France (À paraître).

[2] G. Cleuziou and F. Flouvat. Learning student program embeddings using abstract execution traces. In 14th Int. Conf. on Educational Data Mining, pages 252–262, Paris, France, June 2021. 

## Contact

For questions about the code during the review period, please open an issue in this repository.
