# data-pipeline

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency management.

### Virtual environment

A venv is created with:

```bash
uv venv
```

### Locking dependencies

Generate the lock file with:

```bash
uv lock
```

### Running commands

```bash
uv sync            # create venv and install from the lock file
uv run python ...  # run a script using the project environment
```
