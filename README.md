# MOSS-Image Flask 本地工作台

本项目提供 MOSS-Image-SFT 图像生成、固定版本权重下载与 HTTP 断点续传、可选 MLX 中文构思扩写、进度查询与 PNG 下载。服务默认只监听本机。

模型、公开权重与官方推理代码来自复旦大学计算与智能创新学院邱锡鹏教授带领的 OpenMOSS 团队。本仓库是本地工作台，不是官方发布。致谢页在本地的 `/about/`，源文件为 `docs/index.html`。

页内示例图由本工作台生成，提示词是：清晨的森林里，一只橘猫坐在长满青苔的石头上，柔和的阳光穿过树叶，水彩插画风格。

## 启动

```sh
git clone https://github.com/bennix/OpenMOSSImageFlask.git
cd OpenMOSSImageFlask
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./run.sh
```

打开 http://127.0.0.1:7860 。先下载 MOSS 权重，若需要扩写，再下载提示词模型。默认 512×512、30 步；官方推荐 50 步，低步数仅适合链路检查。首次在 Mac 上推理可能较慢，实际性能以本机测试为准。端口可通过 `PORT=7861 ./run.sh` 修改。

## 模型与内存

- 图像模型：`OpenMOSS-Team/MOSS-Image-SFT`，约 27.2 GB，包含 Qwen3.5-9B 文本编码器、DiT 和 VAE。
- 提示词模型：`mlx-community/Qwen3.5-4B-4bit`，把简短构思扩写成图像描述。
- 如果系统已有 `aria2c`，大权重使用 16 连接下载，校验 SHA-256；没有时自动使用 Python HTTP 下载。`.part.aria2` 控制文件用于恢复分段进度，请保留。
- MOSS 大权重优先通过官方 ModelScope 镜像下载，每个文件固定到镜像提交并与 Hugging Face 清单的 SHA-256 比对；中断后通过 `.part` 文件续传。小配置文件使用 Hugging Face。
- 模型版本固定在 `models.json`，图像与提示词权重位于 `models/moss-sft/` 和 `models/prompt/`，不会自动更新。
- 高清放大使用 Real-ESRGAN 官方权重：照片 `RealESRGAN_x4plus`、插画 `RealESRGAN_x4plus_anime_6B`、快速 `RealESRGAN_x2plus`。放大在单独进程里分块运行，不把模型留在 Flask 进程中，也不安装 basicsr。
- 在 36 GB Mac 上，MLX 扩写使用独立子进程，退出后再加载 MOSS。MOSS 编码完毕释放文本编码器，再加载 DiT 与 VAE，避免全部组件同时常驻。仍需为系统和其他应用留出内存；建议先使用 512×512。
- Apple Silicon 使用 PyTorch MPS；CUDA 或 CPU 可运行图像部分。MLX 扩写仅适用于 Apple Silicon。
- MOSS 提示词编码最多 512 token；输出文件保存在 `outputs/<任务ID>/`，包含请求、实际提示词、日志、进度、结果与图片。

## HTTP API

```sh
# 模型下载状态 / 当前任务
curl http://127.0.0.1:7860/api/status

# 下载，可断点续传（model 也可为 prompt）
curl -X POST http://127.0.0.1:7860/api/download \
  -H 'Content-Type: application/json' -d '{"model":"moss"}'

# 生成任务，返回 202 与 id
curl -X POST http://127.0.0.1:7860/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A watercolor orange cat in a sunlit forest", "negative":"", "width":512,"height":512,"steps":30,"seed":42,"enhance":false}'

# 查询任务，用实际 id 替换 TASK_ID
curl http://127.0.0.1:7860/api/jobs/TASK_ID
```

`enhance` 可为 `off`、`caption` 或 `layout`。`layout` 先由本地提示词模型写出官方草稿，再编译成画布中心坐标。`region.kind` 为 `text`（添加文字）、`add`（添加内容）或 `edit`（修改这块）。选区会按该位置重新生成整张图；页面在已有生成结果上沿用原来的种子、宽高和步数，并在上一张的构图上追加，已添加的文字和内容会保留。宽高仍须是 16 的倍数，页面提供常见比例。

单次只运行一个下载或生成任务；其他请求返回 409。任务状态保存在进程内存，Flask 重启后历史文件仍在 `outputs/` 中。不要启用 Flask 自动重载或多 worker；这是供个人本地使用的开发服务。

## 上游与 Mac 适配

`vendor/pipeline_moss_image.py` 来自 MOSS-SFT 仓库版本 `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`，保留 Apache-2.0 许可证于 `vendor/LICENSE`。本地仅修改采样时间嵌入：float64 的小张量在 CPU 计算后搬回推理设备，以兼容不支持 float64 的 MPS；未修改模型权重。

使用固定版本的本地 pipeline，加载组件时不执行从远端动态下载的代码。权重下载流程仍会保留官方仓库里的 Python 文件，但推理不导入它们。

官方资源：[代码和说明](https://github.com/OpenMOSS/MOSS-Image)、[SFT 权重](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)。

## 检查

```sh
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest -q
```

接口测试不下载模型、不执行完整模型推理。实际出图验证应另行检查 `outputs/` 内图片与日志。
