# Bio-Inspired VisNet Models (Extended Variants)

This repository provides a Python implementation of VisNet and several biologically-inspired extensions designed to improve feature selectivity, robustness, and cortical plausibility.

## Motivation

VisNet was originally designed as a hierarchical, biologically inspired model of the ventral visual stream that acquires invariances through unsupervised temporal continuity. While effective, the classic formulation makes simplifying assumptions that limit its robustness, selectivity, and generalization in complex visual environments. This repository introduces four VisNet variants that progressively reintroduce biological constraints and computational mechanisms observed in cortex, enhancing both plausibility and performance.

## Model Variants

* **visnet-simplified** — baseline VisNet with fully dense receptive fields (no sparse connectivity)
* **visnet-rbf** — incorporates Radial Basis Function neurons (Gaussian tuning)
* **visnet-md** — integrates Mahalanobis-distance-based learning for covariance-aware adaptation
* **visnet-li** — visnet-simplified enhanced with local inhibition

## visnet-simplified: Baseline Without Sparse Connectivity

The **visnet-simplified** architecture serves as the baseline reference model in this repository. It preserves the original hierarchical structure of VisNet but removes sparse connectivity inside receptive fields, resulting in **fully dense pooling** from one layer to the next. While this version is computationally easier to implement and train, its lack of sparsity significantly weakens the biological plausibility of the model.

From a cortical perspective, the visual pathway exhibits highly structured and localized receptive fields. Neurons in early visual areas (e.g., V1) respond only to a restricted spatial region of the input, and this locality is preserved hierarchically as transformations are integrated – a fact supported by neurophysiological findings. By contrast, visnet-simplified adopts a fully connected receptive field mapping, where every neuron receives equal access to upstream activations. This treatment ignores locality and columnar segregation, effectively ablates spatial inductive bias, and disrupts the emergence of structured feature selectivity.

This simplified model is crucial, however, for ablation analysis: improvements attributed to biologically-inspired sparsity, competition, and lateral structure become measurable only when contrasted with this dense-control variant. It functions as the computational “null hypothesis” against which subsequent extensions can be scientifically benchmarked.

## visnet-rbf: Localized Gaussian Coding via Radial Basis Neurons

The **visnet-rbf** variant replaces traditional McCulloch–Pitts-style neurons with Gaussian radial basis (RBF) units. Instead of producing a broad linear response influenced by the entire high-dimensional input, RBF neurons yield a *localized peak response* around a learned prototype (center). Their activation falls off smoothly as the input deviates from this prototype, following a Gaussian kernel.

This modification is biologically motivated by the selective tuning properties of neurons found in extrastriate visual cortex (V4 and IT), where responses are not globally linear but instead respond strongly to a narrow class of visual features or patterns. Mathematically, the neuron implements:

[
\phi(\mathbf{x}) = \exp\left(-\frac{||\mathbf{x} - \mathbf{c}||^2}{2\sigma^2}\right)
]

where the center **c** functions as a prototype encoded by the weight vector. This ensures that learning is *prototype-driven rather than amplitude-driven*, making the network more robust to irrelevant variations.

In the context of symmetry perception and structured pattern recognition, Gaussian tuning offers a crucial advantage: it enforces **feature locality**, enabling the model to detect subtle structured correspondences (e.g., reflective or bilateral relations) that would be blurred out in a purely dense linear neuron model. This makes visnet-rbf well-suited for biologically plausible intermediate representations of symmetric stimuli.

## visnet-md: Mahalanobis Distance for Covariance-Aware Learning

The **visnet-md** variant replaces the Euclidean-distance-based similarity computation at the core of classical VisNet-style learning with a Mahalanobis-metric formulation. Unlike Euclidean distance, which assumes that all dimensions are independent and identically scaled, the Mahalanobis metric incorporates the covariance structure of the neural representation. This enables neurons to learn tuning surfaces that reflect not only average feature prototypes (centers) but also the *statistical geometry* of the stimulus manifold.

Whereas RBF neurons assume isotropic Gaussian tuning, Mahalanobis units can learn *anisotropic tuning ellipsoids* that shrink or stretch along directions of high or low variance. This allows the model to develop **adaptive feature selectivity** that reflects stimulus statistics. Concretely, the activation of a neuron becomes:

[
\phi(\mathbf{x}) = \exp\left(-\frac{1}{2} (\mathbf{x}-\mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\mathbf{\mu}) \right)
]

where (\mathbf{\Sigma}) is the covariance matrix estimated from input feature activations. Biologically, this corresponds to the emergence of *statistically aligned receptive fields*, which has been hypothesized in IT cortex, where neuronal tuning adapts to the intrinsic structure of the visual feature space.

By incorporating the local covariance structure of layer activations, visnet-md improves selectivity under structured transformations, particularly when feature classes share overlapping raw representations but differ in higher-order dependencies. This makes it particularly valuable in symmetry perception, where covariance relationships between opposing visual fields encode midline structure. In short, visnet-md bridges the gap between biologically motivated receptive field formation and modern statistical manifold learning.

## visnet-li: Local Inhibition Applied to visnet-simplified

The **visnet-li** model builds directly on visnet-simplified by introducing **local inhibitory interactions** within each layer. While retaining fully dense receptive fields, neurons compete with neighboring units via a local inhibitory rule, which suppresses redundant or overlapping activations.

This simple modification emulates cortical lateral inhibition: neurons with similar tuning profiles within a spatial neighborhood mutually suppress each other, promoting **sparsity, decorrelation, and feature competition**. As a result, visnet-li maintains the computational simplicity of the simplified model while reintroducing a biologically relevant mechanism that encourages more selective and stable representations.

## Repository Structure

```
visnet/
├── visnet-simplified/       # Baseline dense model
├── visnet-rbf/              # RBF neuron variant
├── visnet-md/               # Mahalanobis distance learning variant
├── visnet-li/               # Local inhibition variant
├── models/                  # Core layer implementations
├── training/                # Training loops and learning rules
├── experiments/            # Example scripts for datasets
└── utils/                   # Visualization and preprocessing helpers
```

## Installation

```bash
git clone https://github.com/<username>/visnet-bioinspired.git
cd visnet-bioinspired
pip install -r requirements.txt
```

## Quickstart Example

```python
from visnet.models import VisNet

# Initialize any variant
model = VisNet(variant='visnet-rbf', num_layers=4, gabor_frontend=True)

# Train unsupervised
model.train_unsupervised(dataset, epochs=50)

# Visualize receptive fields
model.visualize_layer(layer_index=0)
```

## References

* T. Solls, 2021. *Original VisNet architecture paper*.
* Bishop, C. M., 1995. *Neural Networks for Pattern Recognition*.
* Hebb, D. O., 1949. *The Organization of Behavior*.
* Mahalanobis, P. C., 1936. *Generalized distance in statistics*.
* Hubel, D. H., Wiesel, T. N., 1962. *Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex*.

## Roadmap

* Add spiking variants with eligibility propagation
* Benchmark symmetry perception tasks on synthetic datasets
* Curriculum-based invariance learning
* Compare visnet-md and visnet-li with modern SSL methods (SimCLR, BYOL)
