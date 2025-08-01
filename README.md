# Archipelago 2025 Hack

This repository contains code for the Archipelago 2025 hackathon, focusing on the automatic recognition of objects (people) in images obtained from UAVs (Unmanned Aerial Vehicles).

## Files structure (tree)

```
.
├── .gitignore
├── README.md
├── data # Folder with dataset (excluded from git, ask teammates)
│   ├── archived
│   ├── merged
│   └── raw
├── scripts
│   └── data # Data processing scripts
└── solutions
    ├── examples # Example solutions from orgs
    └── grisha # Grisha's solutions
```

## Development guidelines

### Branching policy

- The `main` branch stores all solutions that are ready for submission.
- If you want to work on a new feature or fix a bug, create a new branch from `main` with a descriptive name (e.g., `feature/add-new-model` or `bugfix/fix-detection-issue`).
- If you want to run your separate experiments, create a new branch from `main` with a descriptive name (e.g., `experiment/your-name/experiment-1`).
- When your work is done, create a pull request to merge your changes into `main`. Ensure that your code is well-documented and tested before merging.

### Code style

- Follow PEP 8 guidelines for Python code.
- Use meaningful variable and function names.
- Write docstrings for all functions and classes.
- Keep your code modular and organized.

## Submission process

1. Create a new branch from `main` for your solution and check out to it.

2. Create a separate directory for your solution in the `solutions` directory (e.g., `solutions/your_name/solution_name`).

3. Develop your solution in this directory and ensure that it follows the development guidelines.

3. If your solution requires additional files (e.g., model weights, configuration files), place them in your solution directory. Ensure that large files are not tracked by Git (use `.gitignore` to exclude them).

4. If your solution requires some specific environment (e.g., specific Python packages or operating system): create a Dockerfile in your solution directory. The Dockerfile should specify the base image, install necessary dependencies. Afrer that, build the Docker image and push it to a Docker Hub repository.

5. Create a `metadata.json` file in your solution directory. This file should contain the name of the Docker image in the format:

```json
{
  "image": "your_dockerhub_username/your_image_name:tag"
}
```

6. Create a `solution.py` file in your solution directory. This file should contain a synchronous function `predict`, which takes an input image and returns the predicted output (e.g., bounding boxes, labels). See `solutions/examples/random_solution` or `solutions/examples/simple_solution` for examples of the `solution.py` file.

7. Zip your solution and upload it to the hackathon [platform](https://xn--e1aaagg3atn2a.xn--2035-43davo0a5a6bk9d.xn--p1ai/ds). If your solution requires additional files (e.g., model weights, configuration files), include them in the zip file. The zip file should at least contain the following files:

```plaintext
.
├── metadata.json # This file contains the name of the Docker image
└── solution.py # This file contains the predict function
```

### What happens after submission?

After you submit your solution zip file, it will be automatically processed by the hackathon platform. The platform will:

1. Download the zip file and extract its contents.

2. Check for the presence of the `metadata.json` and `solution.py` files.

3. Pull the Docker image specified in `metadata.json` from Docker Hub.

4. Build a Docker container using the pulled image.

5. Mount the extracted files from the zip file into the Docker container.

6. Mount the hackathon platform's data into the Docker container as well as mount the `metric_counter.py` and `metric.py` files (see the example in `solutions/examples/`).

7. Iterate over the images in the mounted data directory, calling the `predict` function from `solution.py` for each image.

8. Collect the predictions and save them to a file named `solution_submission.csv` in the mounted directory.

9. Calculate the metrics and save them to a leaderboard.

**You dont need to include `metric_counter.py` and `metric.py` files in your submission. Just ensure that your `metadata.json` and `solution.py` files are correctly formatted and that your Docker image is accessible on Docker Hub.**
