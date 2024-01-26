# METRIK: A Meta-knowledge Graph Augmented Transformer for Biomedical Knowledge Graphs Reasoning

## Abstract
ABSTRACT
Multi-hop reasoning on biomedical knowledge graphs is increasingly recognized for its potential in fields such as drug discovery. Despite various efforts, previous models have overlooked the guidance of meta-knowledge graphs in the reasoning process. To address this gap, we introduce METRIK 1: a Meta-knowledge graph Augmented Transformer for biomedical knowledge graphs reasoning. This approach integrates both entity-level and concept-level context to assist models in accurately generating valid tail entities from a given entity. METRIK leverages the Transformer architecture to
translate the multi-hop reasoning into a reasoning path generation task, utilizing entity context to determine subsequent nodes. By defining entity concepts and their relations, biomedical meta-knowledge graphs offer a holistic perspective of entity concept context. Our model leverages this meta-knowledge graph to navigate the generation of diverse paths by selecting relations and entities from various concepts, thereby deducing multiple tail entities. To effectively capture the complex structure of meta-knowledge graphs, we have developed a node-edge symmetric structure encoder. 
This encoder skillfully captures the intricate interactions between nodes and edges, incorporating the impact of connecting edges into the computation of attention weights between nodes
and their first-order neighbor nodes, and vice versa. Moreover, we
have integrated a dual-perspective attention mechanism into the
Transformer. This mechanism concurrently concentrates on both
entity-level and concept-level contexts to generate reasoning paths.
Experimental results validate that our approach not only provides
interpretability but also significantly outperforms baseline mod-
els in key metrics. Besides, the results also demonstrate that our
framework can adeptly handle scenarios involving multiple correct
answers by simultaneously predicting multiple valid tail entities
## Getting Started
Clone this repository and navigate to the `code` directory. `train.py` is the main script for training the model.

## Prerequisites
- PyTorch
- Transformer

## Usage
Run the following command in the `code` directory to train the model:
```bash
python train.py
