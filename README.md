# VisNet Library

A structured Python library implementing biologically-inspired hierarchical vision networks based on the Rolls 2021 paper.

## Structure

```
visnet_lib/
├── visnet/
│   ├── __init__.py          # Main package exports
│   ├── models/              # Model implementations
│   │   ├── __init__.py
│   │   ├── visnet_md.py     # VisNet-MD models
│   │   ├── visnet_li.py     # VisNet-LI models
│   │   └── visnet_simplified.py
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── filters.py       # Gabor and DoG filters
│   │   ├── inhibition.py    # Local inhibition functions
│   │   └── learning.py      # Learning rules
│   ├── experiments/         # Experiment runners
│   │   ├── __init__.py
│   │   └── runner.py
│   └── data/               # Data loading
│       ├── __init__.py
│       └── loader.py
├── examples/               # Example scripts
│   └── run_comparison.py
├── README.md
└── setup.py
```

## Features

- **VisNet-MD**: Manhattan Distance learning with linear and RBF activation
- **VisNet-LI**: Oja's Hebbian learning with local inhibition
- **VisNet-LI-DoG**: VisNet-LI with DoG preprocessing
- **Modular design**: Clean separation of models, utilities, and experiments
- **Easy to use**: Simple API for running experiments

## Installation

```bash
cd visnet_lib
pip install -e .
```

## Usage

See `examples/run_comparison.py` for usage examples.

## License

MIT


