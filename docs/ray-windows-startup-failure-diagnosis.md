# Ray Windows 启动失败诊断

> **日期**: 2026-06-29  
> **主题**: Ray 2.55.1 在 Windows 10 Pro 上 `ray.init()` 超时  

---

## 环境

| 项目 | 值 |
|------|-----|
| Ray | 2.55.1 |
| Python | 3.10.20 |
| OS | Windows 10 Pro 22H2 (19045) |
| Conda env | `jsbsim_rl` |

---

## 故障症状

```
$ python -c "import ray; ray.init(num_cpus=2)"

2026-06-29 20:48:52,084  INFO node.py:421 -- Failed to get node info
  b'GCS cannot find the node with node ID ... The node registration
  may not be complete yet before the timeout. Try increase the
  RAY_raylet_start_wait_time_s config.'

Exception: The current node timed out during startup. This could
happen because some of the raylet failed to startup or the GCS
has become overloaded.
```

即使将 `RAY_raylet_start_wait_time_s` 增加到 60 秒，问题依旧。

---

## 故障链路

```
ray.init()
  │
  ├─ 启动 GCS (gRPC 服务)         ✅ 正常
  │    gcs_server.out 显示正常 gRPC 指标
  │
  ├─ 启动 raylet (C++ 调度器)     ❌ 崩溃
  │    raylet.out / raylet.err 全部输出 "unknown"
  │    raylet 进程静默失败, 无法向 GCS 注册
  │
  ├─ Dashboard / Monitor           ✅ 正常
  │
  └─ GCS 等待 raylet 注册 → 60s 超时
       → Exception: node timed out during startup
```

### 关键日志

**raylet.out** (全部内容):
```
unknown
unknown
unknown
... (重复)
```

**monitor.log** (节选):
```
Monitor started with command: [...]
Autoscaler has not yet received load metrics. Waiting.
Autoscaler has not yet received load metrics. Waiting.
... (持续等待, 因 raylet 从未上线)
```

**gcs_server.out** (正常):
```
NodeInfoGcsService.grpc_server.GetAllNodeInfo - 59 total
NodeInfoGcsService.grpc_server.CheckAlive - 12 total
... (gRPC 指标正常)
```

---

## 根因

**Ray 2.x 已不再支持原生 Windows。**

`raylet` 是 Ray 的核心调度组件, 用 C++ 编译为二进制文件。在 Ray 2.x 中, 该二进制文件依赖 Linux 系统调用, 无法在原生 Windows 上运行。

参考:
- [Ray GitHub Issue: Windows support dropped in 2.x](https://github.com/ray-project/ray/issues/33146)
- Ray 1.13 是最后一个官方支持 Windows 的版本

---

## 解决方案对比

| 方案 | 工作量 | 说明 |
|------|--------|------|
| **WSL2** | 中 | 在 WSL2 Ubuntu 中安装 Ray, 代码通过 `/mnt/c/` 访问, 与 Windows 开发环境共存 |
| **SB3 IPPO** | 低 | 用 SB3 实现独立策略 (每人一个 Actor + 局部观测), 无需 Ray, 在当前 Windows 环境直接运行 |
| **Ray 1.13 降级** | 高 | 最后一个支持 Windows 的版本, API 不兼容当前代码, 需要大量改写 |
| **Docker** | 中 | 在 Docker Desktop 中运行 Ray 容器, 但 Windows Docker 也是基于 WSL2 |

---

## 推荐路径

**短期**: SB3 IPPO (去中心化执行, 每人独立 Actor + 局部观测)

**长期**: WSL2 + Ray (完整 CTDE / MAPPO 支持)

当前 MAPPO 代码 (`formation_mappo_env.py` / `formation_mappo_model.py` / `train_formation_mappo.py`) 已完备, 迁移到 WSL2 后可直接使用。
