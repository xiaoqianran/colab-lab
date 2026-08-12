# colab-lab

用 [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)（`colab` / `google-colab-cli`）做实验与开发的工作区。

本地写脚本和 notebook，远程用 Colab 的 CPU / GPU / TPU 运行、装依赖、传文件、拉回产物。本仓库目前是空骨架，后续实验代码会放在这里。

官方 CLI 文档：[googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)

## 开发方式

日常开发走 **colab-cli**，而不是只在浏览器里点 notebook：

1. 本仓库维护脚本（`.py`）和 notebook（`.ipynb`）。
2. 用 `colab new` 申请运行时（需要 GPU 时加 `--gpu`）。
3. 用 `colab install` 在远程装依赖。
4. 用 `colab exec` / `colab run` 把本地代码丢到远程执行。
5. 用 `colab download` 拉回模型、日志等产物，用完 `colab stop` 释放机器。

Colab CLI 目前官方支持 **Linux 和 macOS**。Windows 请在 **WSL** 里安装和使用 `colab`。

## 安装 CLI

推荐 `uv`：

```bash
uv tool install google-colab-cli
```

或：

```bash
pip install google-colab-cli
```

确认：

```bash
colab version
colab --help
```

首次使用需要能访问 Google 账号（CLI 默认 `--auth adc`，也可 `oauth2`）。细节见官方 README。

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `colab new [-s NAME] [--gpu GPU] [--tpu TPU]` | 申请 CPU / GPU / TPU 运行时 |
| `colab sessions` / `colab status` | 查看会话 |
| `colab exec [-s NAME] -f FILE` | 在远程执行本地 `.py` 或 `.ipynb` |
| `colab run [--gpu GPU] SCRIPT` | 临时开机器跑脚本，结束后默认释放 |
| `colab install [-s NAME] PKG...` | 在远程装包（优先 `uv`） |
| `colab upload` / `colab download` | 和远程互传文件 |
| `colab drivemount` | 挂载 Google Drive |
| `colab log [-s NAME] -o FILE` | 导出执行记录 |
| `colab stop [-s NAME]` | 关掉会话、释放资源 |

只有一个活动会话时，多数命令可以省略 `-s`。

GPU 示例：`T4`、`L4`、`G4`、`H100`、`A100`。TPU 示例：`v5e1`、`v6e1`。是否可用取决于 Colab 账号额度。

## 快速开始

CPU 冒烟：

```bash
colab new -s lab
echo "print('hello from colab-lab')" | colab exec -s lab
colab stop -s lab
```

GPU 跑本地脚本（一次性，跑完释放）：

```bash
colab run --gpu T4 path/to/train.py
```

长会话（装依赖 → 跑训练 → 拉 checkpoint）：

```bash
colab new -s trainer --gpu T4
colab install -s trainer torch transformers
colab exec -s trainer -f train.py
colab download -s trainer checkpoints/model.bin ./model.bin
colab stop -s trainer
```

## 仓库约定（后续会补）

计划目录（尚未落地，随实验再加）：

```text
colab-lab/
  README.md
  notebooks/     # 探索用 notebook
  scripts/       # 可被 colab exec / colab run 的脚本
  requirements.txt
```

约定：

- 能复现的实验写成脚本，优先 `colab run` / `colab exec -f`。
- 远程产物不要默认提交；大文件用 Drive / 本地下载，不要塞进 git。
- 用完会话记得 `colab stop`，避免空占计算资源。

## 参考

- [Colab CLI 仓库](https://github.com/googlecolab/google-colab-cli)
- [Introducing the Google Colab CLI](https://developers.googleblog.com/en/introducing-the-google-colab-cli/)
- 浏览器里交互式 agent 工作流另见 [Colab MCP Server](https://github.com/googlecolab/colab-mcp)（本仓库主路径仍是 CLI）
