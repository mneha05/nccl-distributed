NCCL Distributed Training — DDP & Multi-Node Examples

![status](https://img.shields.io/badge/status-scaffold-ready-yellow)

This repository provides reproducible examples and runbooks for launching multi-GPU distributed training using NCCL and PyTorch DDP. It contains launch scripts, environment notes, and a small DDP example (`train_ddp.py`).

Highlights
- `train_ddp.py`: minimal DDP example using `LOCAL_RANK` pattern
- `run_example.ps1`: launcher notes for single-node multi-GPU and multi-node setups
- Guidance for NCCL environment tuning and networking

Architecture diagram

```mermaid
flowchart LR
	A[Launcher (torchrun)] --> B[Processes per GPU]
	B --> C[NCCL Backend]
	C --> D[Gradient Allreduce]
```

Why this impresses
- Shows distributed-training infrastructure knowledge and NCCL tuning awareness — key for NVIDIA infra roles

Demo GIF placeholder:

![ddp-demo](./assets/ddp_launch.gif)

License: MIT

