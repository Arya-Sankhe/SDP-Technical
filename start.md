# MSFT GRU Notebook Startup (Windows)

Use these exact commands in PowerShell or Anaconda Prompt:

```powershell
# 1) Go to your repo
D:
cd \APPS\Github\SDP-Technical

# 2) Create/activate env
conda create -n msft-gru python=3.11 -y
conda activate msft-gru

# 3) Upgrade pip
python -m pip install --upgrade pip

# 4) Install PyTorch CUDA build (RTX 3070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5) Install notebook + project deps
pip install -r requirements.txt

# 6) Set Alpaca credentials for this terminal session
$env:ALPACA_API_KEY="YOUR_ALPACA_API_KEY"
$env:ALPACA_SECRET_KEY="YOUR_ALPACA_SECRET_KEY"
$env:ALPACA_FEED="delayed_sip"

# Optional: persist credentials for future terminals (run once, then reopen terminal)
# setx ALPACA_API_KEY "YOUR_ALPACA_API_KEY"
# setx ALPACA_SECRET_KEY "YOUR_ALPACA_SECRET_KEY"
# setx ALPACA_FEED "delayed_sip"

# 7) Register kernel for Jupyter
python -m ipykernel install --user --name msft-gru --display-name "Python (msft-gru)"

# 8) Verify GPU is visible
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# 9) Launch Jupyter
jupyter lab
```

Then open:

`C:\Users\user\Documents\GitHub\SDP-Technical\output\jupyter-notebook\msft-gru-1m-recursive-forecast.ipynb`

and select kernel `Python (msft-gru)`.

Stop Jupyter:

```powershell
# Option A (same terminal where Jupyter is running): press Ctrl+C, then confirm with y

# Option B (from another terminal): list servers and stop by port
jupyter server list
jupyter server stop 8888
```

Start Jupyter again later (second launch after closing):

```powershell
D:
cd \APPS\Github\SDP-Technical
conda activate msft-gru

# If you used setx previously, credentials are already persisted.
# If not persisted, set them again each new terminal:
$env:ALPACA_API_KEY="YOUR_ALPACA_API_KEY"
$env:ALPACA_SECRET_KEY="YOUR_ALPACA_SECRET_KEY"
$env:ALPACA_FEED="delayed_sip"

jupyter lab
```
