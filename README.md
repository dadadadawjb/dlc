# dlc
Deep Learning Configuration

## Linux
* Choose `Ubuntu 22.04 LTS`.
* Know common commands.
  ```bash
  ls <path> # list directory
  ll <path> # list long directory
  tree <path> -a -L <layers> # list directory tree, install by `sudo apt install tree`
  cd <path> # change directory
  pwd # print working directory
  clear # clear command line
  touch <path> # make file
  mkdir <path> # make directory
  cp -r <src_path> <dst_path> # copy
  mv <src_path> <dst_path> # move
  rm -rf <src_path> # remove
  ln -s <src_path> <dst_path> # soft link
  du -h <path> --max-depth <depthscp> # show disk usage
  df -h <path> # show disk free
  sudo apt install <pkg_name> # install package
  wget <url> -O <dst_path> # web get
  aria2c -x 16 -s 16 <url> -o <dst_path> # multi-thread download, install by `sudo apt-get install aria2`
  scp -r -P <port> <src_path> <username>@<ip>:<dst_path> -i <your_local_priv_key_path> # secure copy protocol
  zip -r <dst_path>.zip <src_path> # zip
  unzip <src_path>.zip -d <dst_path> # unzip
  tar -czf <dst_path>.tar.gz <src_path> # tar
  tar -xzf <src_path>.tar.gz -C <dst_path> # untar
  ... | grep <str> # find in output
  which ... # find bin path
  [up]/[down] # use command in history
  history # command history
  ps # process status
  htop # show table of processes
  kill -9 <pid> # kill process
  [ctrl]+[c] # cancel process
  [ctrl]+[z] # pause process
  ```
* `~/.bashrc`: Set environment variables, renew by `source ~/.bashrc`.
  ```bash
  export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  ```
* `run.sh`: Run batch commands by `bash run.sh`.
  ```bash
  seqs=(seq1 seq2)
  for seq in "${seqs[@]}"; do
      python train.py --name "$seq"
  done
  ```
* Store `data`, `model`, `software` in big disk, store `code` in small disk, use soft link to connect.

## SSH
* Local: Generate keys by `ssh-keygen -t ed25519 -C "your_email"`, saved at `~/.ssh/id_ed25519`. You can generate many, just save differently.
* Server: Generate keys by `ssh-keygen -t ed25519 -C "your_email"`, saved at `~/.ssh/id_ed25519`. You can generate many, just save differently.
* Server: Authorize local keys by `touch ~/.ssh/authorized_keys` and `echo "your_local_pub_key" >> ~/.ssh/authorized_keys`. You can authorize many, just append new lines.
* Local: Edit config in `~/.ssh/config` as
  ```txt
  Host <abbr>
    HostName <ip>
    Port <port>
    User <username>
    PreferredAuthentications publickey
    IdentityFile "<your_local_priv_key_path>"
  ```
* Tunneling: Achieved by MobaXterm.
* X11 forwarding: Achieved by MobaXterm.

## MobaXterm
1. Download [MobaXterm](https://mobaxterm.mobatek.net).
2. Set `hostname`, `port`, `username`, `password`.
3. Set private key.

## VSCode
1. Download [VSCode](https://code.visualstudio.com).
2. Install `Remote` extension.
3. Remote config by `~/.ssh/config`.

## Git
* Know common commands.
  ```bash
  git init -b main # initialize
  git clone <url> # clone
  git status # status
  git config user.name "your_name" # config name
  git config user.email "your_email" # config email
  git add <path> # add
  git commit -m "your_message" # commit
  git log --all --graph --decorate # history
  git branch # list branches
  git checkout -b <br_name> # create branch
  git remote -vv # list remotes
  git remote add <name> <url> # add remote
  git pull # pull
  git push <name> <local_br_name>:<remote_br_name> # push
  ```
* `.gitignore` and `.gitkeep` for ignoring files.
* GitHub authentication: Add public key on GitHub, check by `ssh -T git@github.com`.
* Multiple users: Edit ssh config in `~/.ssh/config` as
  ```txt
  Host github-<your_name>
    HostName ssh.github.com
    Port 443
    User git
    PreferredAuthentications publickey
    IdentityFile "<your_priv_key_path>"
  ```

## Conda
* Choose [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).
* Know common commands.
  ```bash
  conda create -n <env_name> python=3.10 # create
  conda env create -f environment.yml # create from file
  conda activate <env_name> # activate
  conda deactivate # deactivate
  conda remove -n <env_name> --all # remove environment
  conda env list # list environments
  conda install <pkg_name>=2.0.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ # install package with tsinghua source
  conda uninstall <pkg_name> # uninstall package
  conda list # list packages
  ```
* `environment.yml`
  ```yml
  name: your_env_name
  channels:
    - pytorch
    - nvidia
    - conda-forge
    - defaults
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/
  dependencies:
    - python=3.10
    - pip
    - pytorch::pytorch=2.0.0
    - pytorch::torchvision=0.15.0
    - pip:
      # no real-time info
      - numpy>=2.0.0
      - scipy==1.15.0
      - tqdm
      - -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

## Pip
* Know common commands.
  ```bash
  pip install <pkg_name>==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple # install with tsinghua source
  pip install -r requirements.txt # install from file
  pip install -e <pkg_path>[<extra_name>] # install editable locally
  pip uninstall <pkg_name> # uninstall
  pip list # list packages
  ```
* `requirements.txt`
  ```pip
  numpy>=2.0.0
  scipy==1.15.0
  tqdm
  ```
* `setup.py`
  ```python
  from setuptools import setup, find_packages
  
  basics = [
      'numpy>=2.0.0', 
      'tqdm', 
  ]
  extras = {
      'dev': [
          'scipy==1.15.0', 
      ], 
      'other': [
          'pynput', 
      ], 
  }
  extras['all'] = list(set({pkg for pkgs in extras.values() for pkg in pkgs}))
  
  setup(
      name = 'your_pkg_name', 
      version = '0.0.1', 
      license = 'MIT', 
      description = 'your_pkg_description', 
      author = "your_name", 
      author_email = "your_email", 
      maintainer = "your_name", 
      maintainer_email = "your_email", 
      url = "your_pkg_url", 
      packages = find_packages(), 
      include_package_data = True, 
      install_requires = basics, 
      extras_require = extras, 
      zip_safe = False
  )
  ```
* `pyproject.toml`
  ```toml
  [build-system]
  requires = ["setuptools>=61.0", "wheel"]
  build-backend = "setuptools.build_meta"
  
  [project]
  name = "your_pkg_name"
  version = "0.0.1"
  description = "your_pkg_description"
  license = { text = "MIT" }
  authors = [
    { name = "your_name", email = "your_email" }
  ]
  maintainers = [
    { name = "your_name", email = "your_email" }
  ]
  dependencies = [
    "numpy>=2.0.0", 
    "tqdm", 
  ]
  requires-python = ">=3.7"
  urls = {
    "Homepage" = "your_pkg_url"
  }
  
  [project.optional-dependencies]
  dev = [
    "scipy==1.15.0",
  ]
  other = [
    "pynput",
  ]
  all = [
    "scipy==1.15.0",
    "pynput",
  ]
  
  [tool.setuptools]
  include-package-data = true
  
  [tool.setuptools.packages.find]
  where = ["."]
  ```
* Set cache path in `~/.bashrc` by `export PIP_CACHE_DIR=/data/.cache/pip`, default as `~/.cache/pip`.

## Vim
* Know common commands.
  ```bash
  [i] # insert
  [a] # append
  [o] # open a new line below
  [esc] # return
  [dd] # delete line
  [j] # down
  [k] # up
  [h] # left
  [l] # right
  [w] # next word
  [e] # word end
  [b] # back word
  [0] # beginning line
  [$] # end line
  [I] # insert beginning line
  [A] # append end line
  [ctrl]+[u] # scroll up
  [ctrl]+[d] # scroll down
  [gg] # beginning file
  [G] # end file
  [:wq] # save and quit
  [:q!] # quit without save
  ```

## Tmux
* Install by `sudo apt install tmux`.
* Know common commands.
  ```bash
  tmux new -s <name> # create session
  tmux ls # list sessions
  tmux a -t <name> # attach session
  [ctrl]+[b] [d] # detach session
  [ctrl]+[d] # destory pane, window, session
  [ctrl]+[b] [[] # scrollback
  [ctrl]+[b] [c] # create window
  [ctrl]+[b] [p] # change to previous window
  [ctrl]+[b] [n] # change to next window
  [ctrl]+[b] [,] # rename window
  [ctrl]+[b] ["] # horizonally split pane
  [ctrl]+[b] [%] # vertically split pane
  [ctrl]+[b] [up]/[down]/[left]/[right] # change to other pane
  ```

## Python
* Choose `Python 3.10`.
* Debug by `import pdb; pdb.set_trace()`.
  ```pdb
  n: next
  s: step
  c: continue
  l: list
  p: print
  ```

## PyTorch
* Check CUDA driver version and GPU status by `nvidia-smi`.
* Install [PyTorch](https://pytorch.org/get-started/locally/) according to the CUDA driver version.
* `nvcc` CUDA toolkit is needed only to compile cuda programs, check CUDA toolkit version by `nvcc --version`.
* Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), under `/usr/local/cuda` by default. You can also install by `conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit` for user only.
* Set environment variables in `~/.bashrc` by `export CUDA_HOME=/usr/local/cuda`, `export PATH=$CUDA_HOME/bin:$PATH` and `export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`.
* Check GPU available by `torch.cuda.is_available()`.
* Choose GPU by `CUDA_VISIBLE_DEVICES=0,1 python train.py`.
* Set cache path in `~/.bashrc` by `export TORCH_HOME=/data/.cache/torch`, default as `~/.cache/torch`.

## Seed
```python
import os
import random
import numpy as np
import torch
def setup_seed(seed:int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

## Args
* `argparse`.
  ```python
  import argparse
  def arg_parse() -> argparse.Namespace:
      parser = argparse.ArgumentParser()
      parser.add_argument("--verbose", action="store_true", default=False, help="whether verbose")
      parser.add_argument("--batch_size", type=int, required=True, help="batch size")
      parser.add_argument("--devices", type=str, nargs="+", default=[], help="pytorch device")
      args = parser.parse_args()
      return args
  ```
* `tyro`.
  ```bash
  pip install tyro
  ```
  ```python
  from typing import Tuple
  import tyro
  
  def train(verbose: bool, batch_size: int = 16, devices: Tuple[str, ...] = ()) -> None:
      pass
  
  if __name__ == "__main__":
      tyro.cli(train)
  ```
  ```python
  from typing import Tuple
  from dataclasses import dataclass
  import tyro
  
  @dataclass
  class Config:
      verbose: bool
      batch_size: int = 16
      devices: Tuple[str, ...] = ()
  
  if __name__ == "__main__":
      config = tyro.cli(Config)
  ```

## Configs
* `configargparse`.
  ```bash
  pip install ConfigArgParse
  ```
  ```python
  import configargparse
  def config_parse() -> configargparse.Namespace:
      parser = configargparse.ArgumentParser()
      parser.add_argument('--config', is_config_file=True, help='config file path')
      parser.add_argument("--verbose", action="store_true", default=False, help="whether verbose")
      parser.add_argument("--batch_size", type=int, required=True, help="batch size")
      parser.add_argument("--devices", type=str, nargs="+", default=[], help="pytorch device")
      args = parser.parse_args()
      return args
  ```
  ```python
  # config.txt
  verbose = False
  batch_size = 8
  devices = [cuda:0, cuda:1]
  ```
* `json`.
  ```python
  import json
  def load_config(path:str) -> dict:
      with open(path, 'r', encoding='utf-8') as f:
          config = json.load(f)
      return config
  def save_config(path:str, config:dict) -> None:
      with open(path, 'w', encoding='utf-8') as f:
          json.dump(config, f, ensure_ascii=False, indent=4)
  ```
  ```json
  {
      "verbose": false,
      "batch_size": 8,
      "devices": ["cuda:0", "cuda:1"]
  }
  ```
* `yaml`.
  ```bash
  pip install PyYAML
  ```
  ```python
  import yaml
  def load_config(path:str) -> dict:
      with open(path, 'r', encoding='utf-8') as f:
          config = yaml.load(f, Loader=yaml.FullLoader)
      return config
  def save_config(path:str, config:dict) -> None:
      with open(path, 'w', encoding='utf-8') as f:
          yaml.dump(config, f, allow_unicode=True, sort_keys=False)
  ```
  ```yaml
  # config.yaml
  verbose: false
  batch_size: 8
  devices:
    - cuda:0
    - cuda:1
  ```
* `hydra`.
  ```bash
  pip install hydra-core
  ```
  ```python
  import hydra
  from omegaconf import DictConfig

  @hydra.main(config_path='./configs', config_name='config', version_base='1.2')
  def train(cfg:DictConfig) -> None:
      hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
      output_dir = hydra_cfg['runtime']['output_dir']
      pass

  if __name__ == '__main__':
      train()
  ```
  ```yaml
  # configs/config.yaml
  defaults:
    - training: config # configs/training/config.yaml
    - _self_
  
  abbr: 'exp0'
  verbose: False
  
  hydra:
    run:
      dir: "./outputs/train/${abbr}_${now:%Y_%m_%d_%H_%M_%S}"
  ```

## WandB
```bash
pip install wandb
wandb login
```
```python
import wandb
wandb.init(project="your_proj_name", name="your_exp_name")
for epoch in range(num_epochs):
    wandb.log({
        'Loss/train': train_loss,
        'Loss/val': val_loss,
        'Accuracy/val': val_acc,
        'Input/Image': wandb.Image(image)
    }, step=epoch)
wandb.finish()
```

## TensorBoard
```bash
pip install tensorboard
```
```python
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter(log_dir="your_log_dir")
for epoch in range(num_epochs):
    tb_writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
    tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
    tb_writer.add_image('Input/Image', image, epoch)
tb_writer.close()
```
```bash
tensorboard --logdir=<your_log_dir> --port=6006 # http://localhost:6006
```

## HuggingFace
* Set cache hub path in `~/.bashrc` by `export HF_HOME=/data/.cache/huggingface`, default as `~/.cache/huggingface`.

## Gradio
```bash
pip install gradio
```
```python
import gradio as gr
with gr.Blocks(title="your_title") as demo:
    gr.Markdown("# Your_Title")
    with gr.Column():
        input_images = gr.File(label="Images", file_count="multiple")
        with gr.Row():
            schedule = gr.Dropdown(["linear", "cosine"], value="linear", label="Schedule", info="For aligment")
            niter = gr.Number(value=50, precision=0, minimum=0, maximum=100, label="Iterations", info="For denoising")
            name = gr.Textbox(label="Name", placeholder="NULL", info="Experiment name")
            thr = gr.Slider(label="Threshold", value=5, minimum=1, maximum=10, step=1)
            flag = gr.Checkbox(value=True, label="Mask")

        run_btn = gr.Button("Run")

        output_model = gr.Model3D(label="3D Result")
        output_gallery = gr.Gallery(label="2D Results", columns=4)

        flag.change(set_flag_fn, inputs=[input_images, flag], outputs=niter)
        run_btn.click(run_fn, inputs=[input_images, schedule, niter, name, thr, flag], outputs=[output_model, output_gallery])
demo.launch()
```
