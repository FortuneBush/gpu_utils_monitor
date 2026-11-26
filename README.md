# GPU Monitor Guard

本项目监控 NVIDIA GPU 的空闲状态，并在 GPU 连续空闲达到阈值后，自动启动一个 Qwen3-30B-A3B 推理进程。每张 GPU 最多启动一个子进程；子进程退出后，监控器会在重试间隔后再次尝试启动。

## 固定路径和文件

- 项目目录：/tmp/wyb/code/vllm
- 模型目录：/tmp/wyb/models/Qwen3-30B-A3B
- 监控入口：vllm_inference_eval.py
- 单 GPU 推理入口：qwen_inference.py
- systemd 监控单元：gpu-monitor-guard.service
- 可选的 systemd 推理模板：qwen-inference@.service
- watchdog 锁文件：/tmp/vllm_inference_eval.lock

如果项目目录或模型目录改变，必须同步修改 systemd 单元、文档和启动命令中的路径。

## 运行要求

- Linux 和可用的 NVIDIA GPU。
- nvidia-smi 可以正常查询 GPU。
- Python 3、CUDA 版 PyTorch 和 transformers。
- 模型目录存在，并包含 Transformers 可加载的 Qwen3-30B-A3B 文件。
- 推理脚本只使用本地模型文件，不会联网下载模型。
- 如果安装 flash-attn，脚本会自动使用 FlashAttention 2。

## 默认启动流程

默认使用直接进程模式，不依赖 systemd。监控程序分为两层：

1. watchdog 持有单实例锁，并在 worker 异常退出后重启 worker。
2. worker 每 30 秒调用 nvidia-smi；当某张 GPU 连续 1800 秒没有 GPU 利用率且没有 compute application 时，启动该 GPU 的 qwen_inference.py 子进程。

前台启动：

~~~bash
cd /tmp/wyb/code/vllm
/usr/bin/python3 vllm_inference_eval.py \
  --idle-threshold 1800 \
  --scan-interval 30 \
  --launch-mode process
~~~

推荐使用 detached watchdog：

~~~bash
cd /tmp/wyb/code/vllm
setsid /usr/bin/python3 vllm_inference_eval.py \
  --idle-threshold 1800 \
  --scan-interval 30 \
  --launch-mode process \
  </dev/null >>gpu-monitor-guard.log 2>&1 &
echo $!
~~~

如果已有 watchdog，第二次启动会因锁文件被拒绝，从而避免重复占用 GPU。一个 watchdog 可以管理多张 GPU。

## 启动验证

~~~bash
ps -eo pid,ppid,stat,etime,cmd | grep -E 'vllm_inference_eval|qwen_inference' | grep -v grep
cat /tmp/vllm_inference_eval.lock
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
tail -f /tmp/wyb/code/vllm/gpu-monitor-guard.log
~~~

正常日志应包含 Started persistent watchdog、Started monitor worker、Started inference process on GPU N、Started continuous inference load 和 GPU N load progress。

### 快速验证自动启动

默认阈值是 30 分钟。需要验证自动启动链路时，可以使用短阈值；该命令会真实加载模型并占用 GPU：

~~~bash
cd /tmp/wyb/code/vllm
setsid /usr/bin/python3 vllm_inference_eval.py \
  --idle-threshold 10 \
  --scan-interval 2 \
  --start-retry-interval 10 \
  --launch-mode process \
  </dev/null >>gpu-monitor-guard.log 2>&1 &
~~~

看到 GPU 推理进程、模型加载完成和 load progress 后，停止测试 watchdog，并按默认参数重新启动。

## 停止流程

优先向 watchdog 发送 SIGTERM。watchdog 会停止 worker，worker 会继续停止它管理的推理子进程：

~~~bash
WATCHDOG_PID="$(cat /tmp/vllm_inference_eval.lock)"
kill -TERM "$WATCHDOG_PID"
~~~

确认没有残留进程：

~~~bash
ps -eo pid,ppid,stat,cmd | grep -E 'vllm_inference_eval|qwen_inference' | grep -v grep
~~~

## systemd 启动

正式的 systemd 单元名称是 gpu-monitor-guard.service。仓库中的单元文件已经使用当前项目路径：

- gpu-monitor-guard.service：默认直接进程模式的 watchdog。
- qwen-inference@.service：可选的每 GPU 推理模板。

在真正由 systemd 作为 PID 1 的主机上安装：

~~~bash
cd /tmp/wyb/code/vllm
sudo install -m 0644 gpu-monitor-guard.service /etc/systemd/system/gpu-monitor-guard.service
sudo install -m 0644 qwen-inference@.service /etc/systemd/system/qwen-inference@.service
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-monitor-guard.service
~~~

当前监控单元使用 --launch-mode process，因此不需要启动推理模板。如果要让监控器通过 systemd 启动每张 GPU 的推理子进程，需要把 ExecStart 改为包含 --launch-mode systemd：

~~~bash
sudo systemctl edit gpu-monitor-guard.service
~~~

覆盖内容：

~~~ini
[Service]
ExecStart=
ExecStart=/usr/bin/python3 /tmp/wyb/code/vllm/vllm_inference_eval.py --idle-threshold 1800 --scan-interval 30 --launch-mode systemd
~~~

然后重新加载并重启：

~~~bash
sudo systemctl daemon-reload
sudo systemctl restart gpu-monitor-guard.service
~~~

如果运行环境不是 systemd PID 1，例如某些容器，systemctl 会报系统未由 systemd 启动；此时使用 detached watchdog 流程。

## 手动推理测试

~~~bash
cd /tmp/wyb/code/vllm
/usr/bin/python3 qwen_inference.py \
  --model-path /tmp/wyb/models/Qwen3-30B-A3B \
  --device 0 \
  --duration 300 \
  --batch-size 256 \
  --max-new-tokens 64 \
  --workload-mode prefill \
  --input-tokens 1024
~~~

--duration 0 表示持续运行直到收到 SIGINT 或 SIGTERM。prefill 模式使用固定输入长度，decode 模式使用自回归生成。

## 主要参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| --idle-threshold | 1800 秒 | 连续空闲多久后启动推理 |
| --scan-interval | 30 秒 | GPU 查询间隔 |
| --start-retry-interval | 300 秒 | 启动失败或子进程退出后的重试间隔 |
| --batch-size | 256 | 每轮推理的 batch size |
| --max-new-tokens | 64 | decode 模式每个 prompt 的生成长度 |
| --workload-mode | prefill | prefill 或 decode |
| --input-tokens | 1024 | prefill 输入长度 |
| --watchdog-restart | 5 秒 | worker 异常退出后的重启延迟 |

完整参数：

~~~bash
/usr/bin/python3 vllm_inference_eval.py --help
~~~

## 常见问题

- nvidia-smi 查询失败：先单独运行 nvidia-smi，确认驱动和 GPU 可用。
- 没有自动启动：默认需要连续空闲 30 分钟；确认 GPU 利用率为 0、没有 compute application，并检查日志。
- 模型加载失败：确认模型目录存在，且当前 Python 环境能导入 torch 和 transformers。
- 进程重复启动：检查锁文件中的 watchdog PID，不要绕过锁文件再启动第二个监控器。
- systemd 找不到服务：确认安装的是 gpu-monitor-guard.service，执行 systemctl daemon-reload，并检查 systemctl status gpu-monitor-guard.service。

## 文件说明

- vllm_inference_eval.py：GPU 空闲监控、watchdog 和推理子进程管理。
- qwen_inference.py：单 GPU Qwen 推理负载。
- gpu-monitor-guard.service：监控 watchdog 的 systemd 单元。
- qwen-inference@.service：可选的每 GPU systemd 推理模板。
- gpu_burn.py：独立的 GPU burn 工具，不是默认监控启动链路的一部分。
