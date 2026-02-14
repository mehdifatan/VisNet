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

The visnet-rbf variant replaces traditional McCulloch–Pitts-style neurons with Gaussian radial basis (RBF) units. Instead of producing a broad linear response influenced by the entire high-dimensional input, RBF neurons yield a localized peak response around a learned prototype represented by the weight vector w. Their activation falls off smoothly as the input deviates from this prototype, following a Gaussian kernel.

This modification is biologically motivated by the selective tuning properties of neurons found in extrastriate visual cortex (V4 and IT), where responses are not globally linear but instead respond strongly to a narrow class of visual features or patterns.

Mathematically, the neuron implements:

φ(x) = exp( - ||x - w||² / (2σ²) )


where:

x = input feature vector

w = learned prototype (weight vector)

σ = feature-space tuning bandwidth controlling selectivity

The weight vector w functions as a stored prototype. This shifts learning from amplitude-driven encoding (global weighted summation) to prototype-driven similarity coding, making the network more robust to irrelevant variations.

In the context of symmetry perception and structured pattern recognition, Gaussian tuning offers a crucial advantage: it enforces feature locality, enabling the model to detect subtle structured correspondences (e.g., reflective or bilateral relations) that would be blurred in a purely dense linear neuron model. This makes visnet-rbf well-suited for biologically plausible intermediate representations of symmetric stimuli.

## visnet-md: Mahalanobis Distance for Covariance-Aware Learning

The visnet-md variant replaces the Euclidean similarity metric used in classical VisNet-style learning with a Mahalanobis distance formulation. Unlike Euclidean distance, which assumes independent and equally scaled feature dimensions, the Mahalanobis metric incorporates the covariance structure of the neural representation. This allows the model to account for the statistical geometry of the stimulus manifold during learning.

The squared Mahalanobis distance between an input vector x and a weight vector w is defined as:

d_M²(x, w) = (x − w)ᵀ Σ⁻¹ (x − w)


where:

w = learned weight vector (prototype)

Σ = covariance matrix estimated from feature activations

In visnet-md, this distance term defines the geometry of synaptic adaptation. The gradient of the Mahalanobis distance with respect to the weight vector drives learning, yielding covariance-aware updates of the form:

Δw ∝ Σ⁻¹ (x − w)


Importantly, neuronal activation itself follows the standard VisNet competitive structure (e.g., visnet-li). The Mahalanobis formulation modifies the learning dynamics, not the response mechanism.

## visnet-li: Local Inhibition Applied to visnet-simplified

The **visnet-li** model builds directly on visnet-simplified by introducing **local inhibitory interactions** within each layer. While retaining fully dense receptive fields, neurons compete with neighboring units via a local inhibitory rule, which suppresses redundant or overlapping activations.

This simple modification emulates cortical lateral inhibition: neurons with similar tuning profiles within a spatial neighborhood mutually suppress each other, promoting **sparsity, decorrelation, and feature competition**. As a result, visnet-li maintains the computational simplicity of the simplified model while reintroducing a biologically relevant mechanism that encourages more selective and stable representations.

## Installation

```bash
git clone https://github.com/mehdifatan/VisNet.git
cd VisNet
pip install -r requirements.txt
```

## References

1. Rolls, E. T. (2021). *Learning invariant object and spatial view representations in the brain using slow unsupervised learning.* Frontiers in Computational Neuroscience, 15, 686239. [https://doi.org/10.3389/fncom.2021.686239](https://doi.org/10.3389/fncom.2021.686239)

2. Bishop, C. M. (1995). *Neural Networks for Pattern Recognition.* Oxford University Press.

3. Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory.* Wiley.

4. Mahalanobis, P. C. (1936). *On the generalized distance in statistics.* Proceedings of the National Institute of Sciences of India, 2, 49–55.

5. Hubel, D. H., & Wiesel, T. N. (1962). *Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex.* The Journal of Physiology, 160, 106–154. [https://doi.org/10.1113/jphysiol.1962.sp006837](https://doi.org/10.1113/jphysiol.1962.sp006837)


## Roadmap

* Add spiking variants with eligibility propagation
* Benchmark symmetry perception tasks on synthetic datasets
* Curriculum-based invariance learning
* Compare visnet-md and visnet-li with modern SSL methods (SimCLR, BYOL)


## Citation

If you use this repository or the extended VisNet variants, please cite the paper:

@article{fatan2025improvingvisnet,
  title   = {Improving VisNet for Object Recognition},
  author  = {Fatan Serj, Mehdi and Parraga, C. Alejandro and Otazu, Xavier},
  journal = {arXiv preprint arXiv:2511.08897},
  year    = {2025}
}

And optionally the GitHub repository:

@misc{fatan2025visnet,
  author       = {Mehdi Fatan Serj},
  title        = {VisNet: Biologically-Inspired Hierarchical Visual Model (GitHub repository)},
  year         = {2025},
  howpublished = {\url{https://github.com/mehdifatan/VisNet}},
  note         = {Accessed: 2025-10-31}
}











