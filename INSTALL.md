# Installation Guide

## Quick Install (Recommended)

```bash
# 1. Update pip and setuptools first
pip install --upgrade pip setuptools wheel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp env_example.txt .env
# Edit .env with your API keys

# 4. Test installation
python setup_and_test.py
```

## Python 3.13 Compatibility Issues

If you encounter installation errors with Python 3.13, try these solutions:

### Solution 1: Update Build Tools
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Solution 2: Individual Package Installation
If the requirements.txt fails, install packages individually:

```bash
# Core build tools
pip install --upgrade setuptools wheel

# API clients
pip install anthropic openai

# Data processing
pip install numpy scikit-learn

# Utilities  
pip install python-dotenv tiktoken
```

### Solution 3: Use conda (Alternative)
```bash
# Create conda environment
conda create -n claude-rag python=3.11
conda activate claude-rag

# Install packages
conda install numpy scikit-learn
pip install anthropic openai python-dotenv tiktoken
```

### Solution 4: Python 3.11 (Most Stable)
If you continue having issues, Python 3.11 is more stable:

```bash
# Using pyenv (macOS/Linux)
pyenv install 3.11.9
pyenv local 3.11.9

# Or download from python.org
```

## Common Error Solutions

### Error: "Cannot import 'setuptools.build_meta'"
```bash
pip install --upgrade setuptools wheel
pip install --no-cache-dir -r requirements.txt
```

### Error: "Microsoft Visual C++ 14.0 is required" (Windows)
- Install Visual Studio Build Tools
- Or use conda: `conda install numpy scikit-learn`

### Error: "Failed building wheel for numpy"
```bash
# Pre-install numpy with conda
conda install numpy
pip install -r requirements.txt
```

### Error: SSL Certificate Issues
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Minimal Installation

If you want to install only essential packages:

```bash
pip install anthropic openai numpy python-dotenv
```

Note: This skips scikit-learn and tiktoken, which means:
- No cosine similarity (will use simple dot product)  
- No token counting (will use character-based estimates)

## Virtual Environment Setup

### Using venv (Recommended)
```bash
python -m venv claude-rag-env
source claude-rag-env/bin/activate  # On Windows: claude-rag-env\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Using conda
```bash
conda create -n claude-rag python=3.11
conda activate claude-rag
pip install -r requirements.txt
```

## API Keys Setup

1. **Get API Keys:**
   - Anthropic: https://console.anthropic.com/
   - OpenAI: https://platform.openai.com/

2. **Create .env file:**
   ```bash
   cp env_example.txt .env
   ```

3. **Edit .env:**
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
   ```

## Verification

Run the setup test to verify everything works:

```bash
python setup_and_test.py
```

## Alternative: Docker Setup

If you prefer Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "example_usage.py"]
```

```bash
docker build -t claude-rag .
docker run -it --env-file .env claude-rag
```

## Troubleshooting

If you're still having issues:

1. **Check Python version:**
   ```bash
   python --version
   ```

2. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

3. **Use verbose installation:**
   ```bash
   pip install -v -r requirements.txt
   ```

4. **Try without binary packages:**
   ```bash
   pip install --no-binary=all -r requirements.txt
   ```

5. **Check for conflicting packages:**
   ```bash
   pip check
   ```

Need help? Create an issue with your error message and Python version. 