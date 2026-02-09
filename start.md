# MSFT GRU Notebook Startup (Windows)

Use these exact commands in PowerShell or Anaconda Prompt:

```powershell
# 1) Go to your repo
cd C:\Users\user\Documents\GitHub\SDP-Technical

# 2) Create/activate env
conda create -n msft-gru python=3.11 -y
conda activate msft-gru

# 3) Upgrade pip
python -m pip install --upgrade pip

# 4) Install PyTorch CUDA build (RTX 3070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5) Install notebook + project deps
pip install -r requirements.txt

# 6) Register kernel for Jupyter
python -m ipykernel install --user --name msft-gru --display-name "Python (msft-gru)"

# 7) Verify GPU is visible
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# 8) Launch Jupyter
jupyter lab
```

Then open:

`C:\Users\user\Documents\GitHub\SDP-Technical\output\jupyter-notebook\msft-gru-1m-recursive-forecast.ipynb`

and select kernel `Python (msft-gru)`.
