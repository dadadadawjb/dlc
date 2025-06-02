# dlc
Deep Learning Configuration

## Linux
* Choose `Ubuntu 22.04 LTS`.
* Know common commands.
  ```bash
  ls <path> # list directory
  ll <path> # list long directory
  cd <path> # change directory
  pwd # print working directory
  clear # clear command line
  touch <path> # make file
  mkdir <path> # make directory
  cp <src_path> <dst_path> # copy
  mv <src_path> <dst_path> # move
  ln -s <src_path> <dst_path> # soft link
  du -h <path> # show disk usage
  df -h <path> # show disk free
  htop # show table of processes
  sudo apt install <pkg_name> # install package
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

## MobaXterm
1. Download [MobaXterm](https://mobaxterm.mobatek.net).
2. Set `hostname`, `port`, `username`, `password`.
3. Set private key.

## VSCode
1. Download [VSCode](https://code.visualstudio.com).
2. Install extensions.
   * Remote
   * Python
   * Markdown
   * Todo Tree
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
  ```
* GitHub authentication: Add public key.
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
  conda activate <env_name> # activate
  conda deactivate # deactivate
  conda remove -n <env_name> --all # remove
  ```

## Pip
* Know common commands.
  ```bash
  pip install <pkg_name> -i https://pypi.tuna.tsinghua.edu.cn/simple # install with tsinghua source
  pip install -r requirements.txt # install from file
  ```

## Vim

## Tmux

## PyTorch
* Check CUDA version and GPU status by `nvidia-smi`.
* Install [PyTorch](https://pytorch.org/get-started/locally/) according to the CUDA version.
