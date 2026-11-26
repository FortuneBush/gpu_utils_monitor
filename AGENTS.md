# AGENTS.md

## 项目范围

这是一个用于 GPU 空闲监控和 Qwen3-30B-A3B 推理负载的 Python 项目。默认模型位于 /tmp/wyb/models/Qwen3-30B-A3B，项目根目录为 /tmp/wyb/code/vllm。

## 启动约定

- 默认入口是 vllm_inference_eval.py。
- 默认使用 --launch-mode process，由 watchdog 直接创建 qwen_inference.py 子进程。
- 默认空闲阈值为 1800 秒，扫描间隔为 30 秒，启动重试间隔为 300 秒。
- 单实例锁文件是 /tmp/vllm_inference_eval.lock。
- 正式的 systemd 单元名称是 gpu-monitor-guard.service。
- qwen-inference@.service 只在选择 --launch-mode systemd 时使用。
- 运行环境没有 systemd PID 1 时，使用 detached watchdog。

标准 detached 启动命令：

~~~bash
cd /tmp/wyb/code/vllm
setsid /usr/bin/python3 vllm_inference_eval.py \
  --idle-threshold 1800 \
  --scan-interval 30 \
  --launch-mode process \
  </dev/null >>gpu-monitor-guard.log 2>&1 &
~~~

停止时向锁文件中的 watchdog PID 发送 SIGTERM。不要只杀 worker 后再启动第二个 watchdog；锁文件用于防止重复监控和重复占用 GPU。

## 修改启动流程时

1. 同步更新 README.md、本文件、gpu-monitor-guard.service 和必要的 qwen-inference@.service 路径。
2. 如果项目目录或模型目录改变，检查所有 WorkingDirectory、ExecStart 和默认参数，不要保留旧目录的断链。
3. 保持直接进程模式无需 systemd 的能力，因为容器或开发环境可能没有 systemd PID 1。
4. 不要把模型、运行日志、__pycache__、虚拟环境或大文件加入 Git；这些路径由 .gitignore 过滤。
5. 监控单元名称和文档统一使用 gpu-monitor-guard.service。

## 验证清单

修改 Python 代码后至少执行：

~~~bash
/usr/bin/python3 -m py_compile vllm_inference_eval.py qwen_inference.py gpu_burn.py
/usr/bin/python3 vllm_inference_eval.py --help
~~~

修改 systemd 单元后执行：

~~~bash
systemd-analyze verify gpu-monitor-guard.service qwen-inference@.service
~~~

如果有 NVIDIA GPU，再检查：

~~~bash
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
~~~

要验证自动启动，可使用 --idle-threshold 10 --scan-interval 2 做短测试；这会真实加载模型并占用 GPU。验证后应停止测试 watchdog，并用默认 1800 秒配置重新启动。

## systemd 约定

主机上安装单元时使用：

~~~bash
sudo install -m 0644 gpu-monitor-guard.service /etc/systemd/system/gpu-monitor-guard.service
sudo install -m 0644 qwen-inference@.service /etc/systemd/system/qwen-inference@.service
sudo systemctl daemon-reload
~~~

当前仓库使用 /tmp/wyb/code/vllm 的绝对路径。路径变更后必须同步更新 unit 文件和文档，并检查 multi-user.target.wants 中是否还存在指向旧路径的符号链接。

## Git 约定

规范远程仓库：https://github.com/FortuneBush/gpu_utils_monitor.git。

提交前检查：

~~~bash
git status --short
git diff --check
git ls-files
~~~

推送前确认远程和分支：

~~~bash
git remote -v
git branch -vv
~~~
