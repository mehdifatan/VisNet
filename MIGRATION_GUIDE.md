# Migration Guide: From Monolithic Script to Structured Library

## Overview

This guide helps you migrate from the monolithic `Run_Comparison_Experiment_VisNetMD12.py` to the structured `visnet_lib` library.

## Structure Comparison

### Before (Monolithic Script)
```
Run_Comparison_Experiment_VisNetMD12.py (3094 lines)
├── All imports mixed together
├── Filter functions
├── Model classes
├── Learning functions
├── Feature extraction
├── Experiment logic
└── Main execution
```

### After (Structured Library)
```
visnet_lib/
├── visnet/
│   ├── __init__.py              # Package exports
│   ├── models/                  # Model implementations
│   │   ├── __init__.py
│   │   ├── visnet_md.py         # Lines ~580-900
│   │   ├── visnet_li.py         # Lines ~1190-2090
│   │   └── visnet_simplified.py # Lines ~1440-1650
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── filters.py           # Lines ~47-94 ✅ COMPLETE
│   │   ├── inhibition.py       # Extract ~line 200-300
│   │   └── learning.py         # Extract ~line 900-1100
│   ├── experiments/             # Experiment logic
│   │   ├── __init__.py
│   │   └── runner.py           # Extract main() ~line 2925+
│   └── data/                    # Data loading
│       ├── __init__.py
│       └── loader.py           # Extract transform code
├── examples/
│   └── run_comparison.py       # Example usage
├── setup.py                     # Package setup ✅ COMPLETE
├── README.md                    # Documentation ✅ COMPLETE
└── STRUCTURE.md                 # Structure guide ✅ COMPLETE
```

## Migration Steps

### Step 1: Extract Inhibition Functions (visnet/utils/inhibition.py)

**From:** Lines ~200-300 in original file
**Create:** `visnet/utils/inhibition.py`

```python
"""Local inhibition and sparseness functions."""

import torch
import torch.nn.functional as F

def apply_local_inhibition_and_sparseness(x, sparseness_target, layer_size, radius, device):
    """
    Apply local inhibition and sparseness to layer activations.
    
    Args:
        x: Input tensor [batch, num_neurons]
        sparseness_target: Target sparseness ratio
        layer_size: Size of spatial layer (100 for 100x100)
        radius: Inhibition radius
        device: Device to run on
    
    Returns:
        Sparse, inhibited activations
    """
    # Extract from original file around line 200-300
    pass
```

### Step 2: Extract Learning Functions (visnet/utils/learning.py)

**From:** Lines ~900-1100 in original file
**Create:** `visnet/utils/learning.py`

```python
"""Learning rule implementations."""

def setup_manhattan_learning(model, learning_rate=0.00001):
    """Setup Manhattan distance learning."""
    # Extract Manhattan learning code
    
def setup_oja_learning(model, learning_rate=0.0001):
    """Setup Oja's Hebbian learning."""
    # Extract Oja learning code
```

### Step 3: Extract Models

#### VisNet-LI Models (visnet/models/visnet_li.py)

**From:** Lines ~1190-2090
**Extract:**
- `SimplifiedVisNetLI` class
- `SimplifiedVisNetLI_DoG` class  
- `SimplifiedVisNetLI_RBF` class

#### VisNet-MD Models (visnet/models/visnet_md.py)

**From:** Lines ~580-900
**Extract:**
- `setup_visnet_md_architecture()` function
- `VisNetMDLinear` wrapper class
- `VisNetRBFMD` wrapper class

#### Simplified VisNet (visnet/models/visnet_simplified.py)

**From:** Lines ~1440-1650
**Extract:**
- `SimplifiedVisNet` class

### Step 4: Extract Experiment Runner (visnet/experiments/runner.py)

**From:** Lines ~2925+ (main function)
**Create:** `ExperimentRunner` class

```python
class ExperimentRunner:
    """Runs comparison experiments across multiple VisNet architectures."""
    
    def __init__(self, methods, train_sizes, num_trials, num_epochs):
        self.methods = methods
        self.train_sizes = train_sizes
        self.num_trials = num_trials
        self.num_epochs = num_epochs
    
    def run(self, dataset):
        """Run experiments and return results."""
        # Extract main() logic here
        pass
```

### Step 5: Extract Data Loading (visnet/data/loader.py)

**From:** Lines ~486-500
**Create:** Data loading utilities

```python
def load_caltech_dataset(path, transform=None):
    """Load Caltech-101 dataset."""
    pass

def prepare_dataset(dataset, num_classes=2):
    """Prepare dataset for binary classification."""
    pass
```

## Usage After Migration

### Before (Old Way)
```python
# Everything in one file
python Run_Comparison_Experiment_VisNetMD12.py
```

### After (New Way)
```python
from visnet.models import SimplifiedVisNetLI, SimplifiedVisNetLIDoG
from visnet.utils import create_gabor_filters
from visnet.experiments import ExperimentRunner

# Create models
models = {
    'VisNet-LI': SimplifiedVisNetLI(device='cpu'),
    'VisNet-LI-DoG': SimplifiedVisNetLIDoG(device='cpu')
}

# Run experiments
runner = ExperimentRunner(
    methods=['VisNet-LI', 'VisNet-LI-DoG'],
    train_sizes=[5, 15, 30],
    num_trials=5,
    num_epochs=3
)
results = runner.run(dataset)
```

## Benefits

1. **Modularity**: Import only what you need
2. **Reusability**: Use models in different contexts
3. **Testability**: Test each component independently
4. **Maintainability**: Easier to find and fix bugs
5. **Documentation**: Better code organization
6. **Distribution**: Can be installed as a package

## Quick Start

```bash
# Install the library
cd visnet_lib
pip install -e .

# Use in your code
python -c "from visnet.models import SimplifiedVisNetLI; print('Success!')"
```

## Notes

- The library structure is complete ✅
- Filter utilities are implemented ✅
- Model classes need extraction from original file
- Experiment runner needs extraction from original file
- See STRUCTURE.md for detailed line numbers

