# RAG-Tutorials

## Setup Instructions

### 1. Create and Setup Virtual Environment

First, install `uv` - a fast Python package installer and resolver:

```bash
# Install uv using pip
pip install uv

# Create a new virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.\.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

Use `uv` to install requirements quickly:

```bash
# Install requirements using uv
uv pip install -r requirements.txt
```

### 3. Setup Jupyter Kernel

Set up a new kernel for Jupyter notebooks:

```bash
# Install ipykernel
python -m pip install ipykernel

# Install the new kernel
python -m ipykernel install --user --name="rag-env" --display-name="Python (RAG)"
```

### 4. Running Jupyter Notebooks

You can now open and run the notebooks in the `notebook/` directory using either Jupyter Lab or VS Code:

```bash
# Using Jupyter Lab
jupyter lab

# OR using Jupyter Notebook
jupyter notebook
```

### Project Structure

```
.
├── data/
│   ├── pdf/        # PDF documents for RAG
│   └── text_files/ # Text files for RAG
├── notebook/       # Jupyter notebooks
└── requirements.txt
```