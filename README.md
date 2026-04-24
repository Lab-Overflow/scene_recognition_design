# 摄像头画面 LAN 共享 + 场景识别方案设计

> **目标**：USB 摄像头接在 Host A（Mac），同网段所有设备都能拉到画面；Host C（无相机）读取画面后用模型判断场景类别（舞台 / 广场 / 餐厅 / …）。

---

## 0. 快速启动（可直接实测）

你现在可以直接用本项目做一轮实机测试。已支持 `npm run start` 一键启动。

### 0.1 首次安装

```bash
cd /Users/user_name/Downloads/scene_recognition_design
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-demo.txt

# 一次性配置 key（弹窗填写，长期保存到 .env.local）
npm run config:keys

# 启动前自检（会做模拟帧 smoke test + 单测 + 路由检查）
npm run doctor

# 若你要走 OpenAI 兜底，建议再跑一遍带 key 检查
npm run doctor:openai
```

### 0.2 一键启动（本机摄像头）

```bash
# 纯规则层（YOLO + 模板）
npm run start

# 规则层 + OpenAI 兜底（key 已在 config:keys 保存后可直接跑）
npm run start:openai

# 规则层 + Anthropic 兜底（key 已在 config:keys 保存后可直接跑）
npm run start:anthropic

# 识别 + 可视化看板（推荐）
npm run start:openai:ui
```

### 0.3 用本地端口帧流测试（你当前“网页实时画面”场景）

当你已有单帧接口（如 `http://127.0.0.1:8080/frame.jpg`）时：

```bash
FRAME_URL=http://127.0.0.1:8080/frame.jpg npm run start:url

# 若同时要看可视化看板：
FRAME_URL=http://127.0.0.1:8080/frame.jpg npm run start:openai:ui
```

可选环境变量：

```bash
PROVIDER=openai MODEL=gpt-5.1 SAMPLE_INTERVAL=0.8 DETECTOR=none FRAME_URL=http://127.0.0.1:8080/frame.jpg npm run start:url
```

### 0.4 注意事项

- 如果你走 `--camera-index`（`npm run start`），请先关闭占用摄像头的浏览器页面。
- 如果你走 `FRAME_URL=... npm run start:url`，则不会和浏览器抢摄像头。
- 运行日志会持续打印 JSON（含 `scene`、`route_decision`、`top3`），可直接观察识别稳定性。
- 如果你运行 `npm run start:openai:ui`，会自动起本地看板：`http://127.0.0.1:8787`。

---

## 1. 整体架构

```
┌─────────────────────────────────────────────┐
│ Host A（Mac，接摄像头）                       │
│                                             │
│   [UVC 摄像头]                              │
│        ↓                                    │
│   capture.py (Python + OpenCV)              │
│        ↓                                    │
│   [内存帧队列 / 最新帧缓存]                    │
│        ├───► MJPEG HTTP endpoint  :8080     │
│        └───► （可选）WebSocket     :8081     │
└──────────┬──────────────────────────────────┘
           │ LAN (Wi-Fi / 有线)
           │
     ┌─────┴──────────────────────────┐
     │                                │
     ▼                                ▼
┌──────────────────┐       ┌─────────────────────────┐
│ Host B (纯观看)   │       │ Host C (场景识别)         │
│                  │       │                         │
│ 浏览器:           │       │ analyzer.py:            │
│ <img src=        │       │   定时抓 JPEG           │
│   /stream.mjpg>  │       │   → 预处理              │
│                  │       │   → 模型推理            │
│ 零依赖, 手机也能看 │       │   → 场景标签 + 置信度     │
└──────────────────┘       └─────────────────────────┘
```

关键点：
- **Host A 只做一件事：采集 + 分发**，不跑模型。
- **传输用 MJPEG over HTTP**，通吃浏览器、Python、手机、ffmpeg；消费方有几个都行。
- **Host C 拉帧而不是被推**，采样节奏自己控（对场景识别 1~2 fps 就够，不需要 30 fps）。

---

## 2. 传输层选型

| 方案 | 延迟 | Host B 接入成本 | Host C 接入成本 | 采纳 |
|---|---|---|---|---|
| **MJPEG over HTTP** | 100–300ms | `<img src>` 一行 | `cv2.VideoCapture(url)` 一行 | **是** |
| WebRTC | 50–100ms | 页面+signaling | aiortc，较重 | 否（收益不够） |
| WebSocket JPEG | 100–200ms | JS 自己解码 | Python 自己收包 | 否（MJPEG 更标准） |
| RTSP/RTMP | 200–500ms | 要 video.js 之类 | ffmpeg 拉 | 否（链路重） |

**选 MJPEG 的理由：**
- HTTP 协议自带 `multipart/x-mixed-replace` 支持，浏览器直接渲染；
- Python 侧 OpenCV / ffmpeg / requests + PIL 都支持；
- 没有 signaling、没有握手，每个消费者独立；
- 场景识别对延迟不敏感（分类在秒级），MJPEG 够用。

---

## 3. 采集 + 分发（Host A）

### 3.1 `capture.py` 职责
1. 打开 UVC 相机（`cv2.VideoCapture(0)`）；
2. 后台线程以原生 FPS 写入"最新帧缓存"（单槽，新帧覆盖旧帧）；
3. HTTP handler 对每个 `/stream.mjpg` 请求独立循环：读最新帧 → JPEG 编码 → 写 multipart chunk；
4. 额外提供 `/frame.jpg` 拿单张（给识别端用更省）；
5. 可选：`/latest.json` 返回元数据（FPS、分辨率、上线时间）。

### 3.2 为什么用"最新帧缓存"而不是队列
- 消费者之间速率可能差很多（浏览器 30fps，识别端 1fps）；
- 队列会让慢消费者积压内存、拖慢整体；
- "单槽覆盖" = 每个消费者永远拿到当前最新，丢掉中间帧是对场景识别有利的（少抖动）。

### 3.3 macOS 上的注意事项
- 浏览器和 `cv2.VideoCapture` 不能同时占用同一个 UVC 设备 → 启动 `capture.py` 前必须关闭占用摄像头的浏览器页面；
- 首次运行 `python` 调用摄像头会触发 macOS 权限弹窗，允许后需重启该终端进程。

---

## 4. 模型选型（Host C）

### 4.1 候选对比

| 模型 | 类别 | 优势 | 劣势 | 适合场景 |
|---|---|---|---|---|
| **Places365 (ResNet-50)** | 365 固定场景类 | 本身就是"室内外场景分类"专用，标签匹配度高；CPU 就能跑（~20ms/帧） | 标签固定，加新类要微调 | **主分类器** |
| **OpenCLIP (ViT-B/32)** | 零样本 | 用自然语言定义类别，随时加减；鲁棒性强 | ~400MB 模型，CPU 上 300–500ms/帧，最好有 GPU/MPS | **灵活补充** |
| **SigLIP-2** | 零样本 | 比 CLIP 精度高、更小 | 需要 `transformers`，同样吃算力 | CLIP 的替代 |
| **云端 VLM（Claude/GPT-4V/Gemini）** | 自由问答 | 复杂场景描述能力最强；能回答"这是什么场合" | 调用成本、网络依赖、延迟不稳定 | 高价值帧的精细分析 |

### 4.2 推荐组合：**Places365 主分类 + CLIP 兜底 + VLM 精析（可选）**

```
frame
  │
  ▼
Places365 推理
  │
  ├─ top-1 置信度 ≥ 0.5
  │     → 输出该类别，结束
  │
  └─ 置信度 < 0.5（可能是 365 类没覆盖的场景）
        ▼
       CLIP 零样本，用你自定义的类别列表打分
        │
        ├─ top-1 ≥ 0.25
        │     → 输出 CLIP 类别
        │
        └─ 仍然低
              ▼
             （可选）发 VLM 精析
              或者报 "unknown"
```

### 4.3 自定义类别设计（给 CLIP）

用自然语言描述你真正关心的场景，越具体越稳：

```python
CATEGORIES = [
    "a photo of a stage during a performance",
    "a photo of an empty stage",
    "a photo of a public plaza with people",
    "a photo of the interior of a restaurant",
    "a photo of a bar counter",
    "a photo of a conference hall",
    "a photo of a city street at night",
    "a photo of a retail store",
    # ...
]
```

**经验法则**：
- 每条以 `a photo of a ...` 开头（CLIP 训练分布匹配）；
- 把同一概念的**多种形态**拆成多条（空台 / 演出中分别写），推理时取最大 score 再聚类回父类；
- 避免过于抽象的词（"艺术"、"氛围"）。

---

## 5. 识别策略

### 5.1 采样节奏
- **默认 1 fps**：场景变化是秒级；比这更高只是浪费算力；
- **动态采样**：用简单的帧差（`cv2.absdiff` + 阈值）检测画面变化；
  - 变化小（静止场景）→ 降到 0.2 fps（每 5 秒一次）
  - 变化大（镜头移动 / 人员进出）→ 升到 2 fps
  - 抽样节奏节省 70%+ 算力，且不丢显著切换

### 5.2 时序平滑
**问题**：单帧误判很常见（遮挡、光照瞬变、人物特写）。
**做法**：保留最近 `N=5` 次预测，按置信度加权投票：
```python
from collections import deque
history = deque(maxlen=5)
history.append((label, score))

# 聚合
from collections import Counter
votes = Counter()
for label, score in history:
    votes[label] += score
final_label, final_score = votes.most_common(1)[0]
```

### 5.3 切换检测（触发事件而不只是输出）
应用层更关心"场景变了"这件事，而不是每秒一条标签：
- 维持当前场景 `current`；
- 只有连续 `K=3` 帧 majority 不等于 `current`，才发 `scene_changed` 事件；
- 这样能把"镜头扫过" 这种噪声过滤掉。

---

## 6. "更好的方案"—— 不一定要逐帧送像素

### 6.1 在 Host A 侧先算 embedding
场景识别本质只需要**图像的高维特征向量**，不需要原始像素。可以把 Places365 的 `avgpool` 层输出（2048 维）或 CLIP 的 image embedding（512 维）在 Host A 算好再传给 Host C：

- 单张 JPEG ≈ 100–300 KB
- 一个 CLIP embedding ≈ 2 KB

**带宽降一到两个数量级，延迟也同步降**。

代价：Host A 需要跑模型（但 Places365 在 Mac CPU 上开销很小）。适用场景：多个 Host C 同时在消费；或 LAN 带宽紧张；或想对外网开放拉流。

### 6.2 事件驱动
Host A 每秒本地算一次场景；**只在切换时推送事件**给订阅方（经 WebSocket）：
```json
{ "t": 1714123456.7, "scene": "restaurant_interior", "score": 0.83 }
```
消费者可以是 Host C（落库）、告警系统、可视化 dashboard——**不再关心像素**。

这是如果这套系统要规模化时的最终形态。

---

## 7. 代码骨架（后续可按此实现）

### 7.1 `host_a/capture.py`（约 80 行）
```python
import cv2, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAM_INDEX = 0
PORT = 8080

lock = threading.Lock()
latest_jpeg = [None]

def capture_loop():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.01); continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with lock:
                latest_jpeg[0] = buf.tobytes()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with lock: data = latest_jpeg[0]
                    if data:
                        self.wfile.write(b'--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: ')
                        self.wfile.write(str(len(data)).encode())
                        self.wfile.write(b'\r\n\r\n')
                        self.wfile.write(data)
                        self.wfile.write(b'\r\n')
                    time.sleep(1/30)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif self.path == '/frame.jpg':
            with lock: data = latest_jpeg[0]
            if not data:
                self.send_response(503); self.end_headers(); return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a, **kw): pass  # 静音访问日志

if __name__ == '__main__':
    threading.Thread(target=capture_loop, daemon=True).start()
    print(f'listening on :{PORT}  (/stream.mjpg, /frame.jpg)')
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
```

### 7.2 `host_c/analyzer.py`（Places365 最小版）
```python
import requests, time, io
import torch, torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50

HOST_A = 'http://192.168.x.x:8080/frame.jpg'  # 改成 Host A 的 IP
INTERVAL = 1.0  # 秒/次

# Places365 ResNet-50 权重 & 类别
# weights: https://github.com/CSAILVision/places365
# 官方 weights 不在 torchvision，实际部署时下载 .pth 和 categories_places365.txt
# 这里仅展示推理流程
model = resnet50(num_classes=365)
model.load_state_dict(torch.load('resnet50_places365.pth', map_location='cpu')['state_dict'], strict=False)
model.eval()

with open('categories_places365.txt') as f:
    classes = [l.strip().split(' ')[0][3:] for l in f]

tfm = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

while True:
    try:
        r = requests.get(HOST_A, timeout=2); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert('RGB')
        with torch.no_grad():
            x = tfm(img).unsqueeze(0)
            logits = model(x)
            probs = logits.softmax(-1)[0]
            top = probs.topk(5)
        pairs = [(classes[i], float(s)) for i, s in zip(top.indices, top.values)]
        print(time.strftime('%H:%M:%S'), pairs)
    except Exception as e:
        print('err:', e)
    time.sleep(INTERVAL)
```

### 7.3 `host_c/analyzer_clip.py`（CLIP 零样本兜底）
```python
import open_clip, torch, requests, io, time
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

CATEGORIES = [
    "a photo of a stage during a performance",
    "a photo of an empty stage",
    "a photo of a public plaza",
    "a photo of a restaurant interior",
    "a photo of a bar counter",
    "a photo of a conference hall",
    # ...
]

with torch.no_grad():
    text_feats = model.encode_text(tokenizer(CATEGORIES))
    text_feats /= text_feats.norm(dim=-1, keepdim=True)

HOST_A = 'http://192.168.x.x:8080/frame.jpg'
while True:
    try:
        r = requests.get(HOST_A, timeout=2); r.raise_for_status()
        img = preprocess(Image.open(io.BytesIO(r.content)).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            f = model.encode_image(img); f /= f.norm(dim=-1, keepdim=True)
            sims = (f @ text_feats.T)[0]
            idx = int(sims.argmax())
            print(CATEGORIES[idx], f"{sims[idx]:.3f}")
    except Exception as e:
        print('err:', e)
    time.sleep(1)
```

### 7.4 Host B 观看（浏览器一行 HTML）
```html
<img src="http://192.168.x.x:8080/stream.mjpg" style="width:100%">
```

---

## 8. 部署 / 运行流程

### 8.1 Host A
```bash
# 关闭当前浏览器里占用摄像头的 tab（osmo_camera 那个）
# 环境
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python

# 运行
python capture.py
# 首次会弹摄像头权限；允许后若仍失败，退出终端重开再跑
```
查 IP：`ipconfig getifaddr en0`（Wi-Fi）或 `en1`（有线）。假设是 `192.168.1.50`。

### 8.2 Host B（任意设备）
浏览器打开 `http://192.168.1.50:8080/stream.mjpg` 即可看画面。

### 8.3 Host C（识别机）
```bash
# Places365 版本
pip install torch torchvision pillow requests
# 下载权重 & 类别文件（详见 8.4）
python analyzer.py

# CLIP 版本（算力更强时）
pip install open_clip_torch
python analyzer_clip.py
```

### 8.4 资源准备

**Places365 权重**（约 100MB）：
- GitHub: https://github.com/CSAILVision/places365
- 权重文件: `resnet50_places365.pth.tar`
- 类别列表: `categories_places365.txt`

**OpenCLIP** 首次加载会自动从 HuggingFace 下载权重（约 400MB），缓存到 `~/.cache/huggingface/`。

---

## 9. 扩展建议

| 需求 | 加什么 |
|---|---|
| 识别结果落库 | Host C 把 label+timestamp 写入 SQLite/ClickHouse |
| 多摄像头 | Host A 跑多份 capture.py 监听不同端口 |
| 外网可访问 | 在路由器做 8080 端口转发，或 Cloudflare Tunnel |
| 识别结果实时推给多人 | Host C 产出事件到 Redis Pub/Sub 或 MQTT，再让前端订阅 |
| 更强模型 | 把 analyzer 换成 VLM API（Claude / GPT-4V），在 Places365 置信度不足时才调用，节省成本 |
| 隐私/安全 | capture.py 加 token 检查（Header 或 URL `?k=xxx`），Host A 只绑 LAN IP 不绑 `0.0.0.0`（但 Mac 上 LAN 本身就是内网） |

---

## 10. 里程碑建议

1. **M1：把 capture.py + 浏览器观看跑通**（完成本方案 §3 + §7.1 + §7.4）
2. **M2：接入 Places365，`analyzer.py` 稳定输出标签**（§7.2）
3. **M3：加 CLIP 兜底和时序平滑**（§5.2 + §7.3）
4. **M4：动态采样 + 切换事件**（§5.1 + §5.3）
5. **M5：（可选）转 embedding 传输 / VLM 精析**（§6）
