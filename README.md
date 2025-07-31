<!-- Archipelago 2025 hack repo: automatic recognition of objects (people) in images obtained from UAVs -->

# Archipelago 2025 Hack

This repository contains code for the Archipelago 2025 hackathon, focusing on the automatic recognition of objects (people) in images obtained from UAVs (Unmanned Aerial Vehicles).

## Files structure (tree)

```
.
├── .gitignore
├── metadata.json
├── notebooks # Jupyter notebooks for exploration and model training
│   ├── eda
│   └── ml
│       └── example_model
├── README.md
├── resources # Additional resources such as datasets, models, etc.
│   └── weights
│       └── example_model
├── scripts # Python scripts for data processing, model training, and evaluation
│   └── ml
│       └── example_model
└── TODO.md
```

## Development guidelines

### Branching policy

- The `main` branch is used for final best performing solution.
- If you want to work on a new feature or fix a bug, create a new branch from `main` with a descriptive name (e.g., `feature/add-new-model` or `bugfix/fix-detection-issue`).
- If you want to run your separate experiments, create a new branch from `main` with a descriptive name (e.g., `experiment/experiment-1`).
- When your work is done, create a pull request to merge your changes into `main`. Ensure that your code is well-documented and tested before merging.

### Code style

- Follow PEP 8 guidelines for Python code.
- Use meaningful variable and function names.
- Write docstrings for all functions and classes.
- Keep your code modular and organized.
