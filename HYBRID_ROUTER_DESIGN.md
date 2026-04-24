# 混合式场景识别：规则检测 + 模板路由 + VLM 兜底

> **目标**：对摄像头画面做场景判断（舞台/广场/餐厅/…）。用快而便宜的规则层处理大多数帧，只在"拿不准"的时候升级到前沿 VLM（Opus 4.7、GPT-5.4、Gemini 等）。

---

## 0. 术语澄清

| 名字 | 含义 | 是否本方案要做 |
|---|---|---|
| **Router / Specialist** | 运行时：先跑便宜模型；不确定就调贵模型 | **是（本文主线）** |
| **Template / Pattern 匹配** | 手写一套"证据 → 标签"规则 | **是（作为 router 的打分函数）** |
| **SFT / 蒸馏** | 训练期：拿 VLM 回答做数据，训一个小模型 | **可选第二阶段（§8）** |
| **RAG** | 检索期：从知识库召回证据拼进 prompt | 相关但不是核心 |

你说的"类似 SFT"其实更像 **"VLM-as-oracle + template router"**：模板是路由函数，不是训练数据；VLM 回答可以**之后**拿去做 SFT 闭环。本文按两阶段分开设计。

---

## 1. 总架构

```
frame
  │
  ▼
┌──────────────────────────┐
│ Stage 1: 规则检测层        │
│ (YOLO + Open-Vocab Det.)  │
│                           │
│ 输出: Evidence JSON       │
│   {                       │
│     objects: [            │
│       {label, score, bbox}│
│     ],                    │
│     counts: {...},        │
│     global_cues: {        │
│       brightness,         │
│       indoor_score, ...   │
│     }                     │
│   }                       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Stage 2: 模板打分         │
│ 对每个 SceneTemplate:     │
│   score = f(evidence)     │
│ 选 top-1，算 margin        │
└────────┬─────────────────┘
         │
         ▼
     Router 判断
     ┌──────────┼────────────────┐
     ▼          ▼                ▼
  confident   borderline    no-match
  (score 高 & (margin 小 /   (全部低)
   margin 大)  score 中等)
     │          │                │
     │          ▼                ▼
     │   ┌──────────────────────────┐
     │   │ Stage 3: VLM 兜底         │
     │   │  (Opus 4.7 / GPT-5.4 ...) │
     │   │                           │
     │   │ 输入: frame + evidence +  │
     │   │       候选模板 list       │
     │   │ 输出: label + reason      │
     │   └────────┬─────────────────┘
     │            │
     ▼            ▼
  final label (+ trace)
         │
         ▼
  写入事件流 / log（成为将来 SFT 的数据源）
```

---

## 2. Stage 1：规则检测层

### 2.1 选型：YOLO 单一模型不够

YOLO (COCO 80 类) 对"餐厅/舞台/广场"这种场景语义**覆盖不均**：
- 餐厅 → 有 `dining_table`、`wine_glass`（OK）
- 舞台 → 几乎没有直接标签（`microphone` 还行）
- 广场 → 只有 `person` 一堆（信息量低）

**建议组合**：

| 组件 | 作用 | 成本 |
|---|---|---|
| **YOLOv11n/s** | 通用对象（人/桌/椅/车/瓶/餐具/…） | ~20ms/帧 CPU |
| **Grounding DINO** 或 **OWL-ViTv2** | 开放词表检测："stage lights", "podium", "market stall", "DJ booth" | ~200ms/帧 CPU，~50ms GPU |
| **简单全局特征** | 室内外（CLIP 单个 prompt）、色温、运动量 | ~30ms/帧 |

运行时**并行跑**，汇总成一个 `Evidence` 对象。Open-vocab 检测可以**按模板动态装配 query 词表**（见 §3），不用一次跑全量。

### 2.2 Evidence JSON 格式（稳定契约）

```json
{
  "ts": 1714123456.78,
  "image_hash": "p:abc123...",        // 感知哈希，用于缓存
  "resolution": [1280, 720],
  "objects": [
    {"label": "dining_table", "score": 0.89, "bbox": [120,340,560,720], "source": "yolo"},
    {"label": "wine_glass",   "score": 0.76, "bbox": [220,310,260,400], "source": "yolo"},
    {"label": "stage lights", "score": 0.42, "bbox": [0,0,1280,80],     "source": "owlv2"}
  ],
  "counts": {
    "person": 4,
    "dining_table": 1,
    "wine_glass": 3
  },
  "global": {
    "indoor_prob": 0.91,
    "brightness": 0.34,
    "motion_score": 0.05
  }
}
```

这个结构既给模板打分用，也给 VLM 做 prompt 用（§5）。

---

## 3. Stage 2：模板（Scene Template）定义

### 3.1 结构

每个模板 = **(scene 标签, 证据要求, 打分规则)** 三元组。用 YAML/JSON 配置，不入代码：

```yaml
- id: restaurant_interior
  label: restaurant
  description: "室内餐厅，桌椅餐具可见"

  evidence:
    positive:                        # 正向加分
      dining_table: 4.0
      wine_glass:   2.0
      plate:        2.0
      fork:         1.5
      chair:        0.3              # 椅子太泛，弱权
      person:       0.2
    negative:                        # 反向减分（显著不匹配）
      stage_lights: -3.0
      microphone:   -1.5
      car:          -2.0

  required:                          # 硬性要求（不满足整体打 0）
    any_of:
      - [dining_table]
      - [plate, fork]                # 这组里都满足才算

  global_cues:                       # 全局特征的软约束
    indoor_prob: ">=0.5"

  owl_queries:                       # 动态喂给 open-vocab 检测器的 query
    - "plated food"
    - "restaurant table setting"

  threshold:
    confident: 8.0                   # ≥ 这个 → 直接用此模板
    borderline: 4.0                  # 4~8 → 升级 VLM
                                     # < 4 → 完全不匹配
```

### 3.2 打分函数

```python
def score_template(evidence, template):
    # 1. 硬性要求
    counts = evidence["counts"]
    if template.get("required"):
        satisfied = False
        for group in template["required"]["any_of"]:
            if all(counts.get(obj, 0) > 0 for obj in group):
                satisfied = True; break
        if not satisfied:
            return 0.0

    # 2. 全局特征软约束
    for k, expr in template.get("global_cues", {}).items():
        v = evidence["global"].get(k, 0)
        if not eval_expr(v, expr):
            return 0.0

    # 3. 证据加权求和
    s = 0.0
    for obj, w in template["evidence"]["positive"].items():
        s += w * min(counts.get(obj, 0), 3)   # 同一类最多贡献 3 次，避免刷分
    for obj, w in template["evidence"]["negative"].items():
        s += w * counts.get(obj, 0)

    # 4. 置信度加权（可选）
    s *= mean_confidence(evidence, template["evidence"]["positive"])

    return s
```

### 3.3 模板怎么来
1. **冷启动**：人工写 10–30 条覆盖主要场景；
2. **迭代**：看 VLM 兜底回答的分布，把高频未覆盖场景补成模板（半自动工作流见 §8）；
3. **审校**：模板是可读 YAML，产品/运营能直接参与。

---

## 4. Stage 2.5：Router 决策逻辑

路由不是简单看 top-1 score，还要看**与第二名的差距（margin）**和**绝对置信度**。

```python
def route(scores: dict[str, float], cfg):
    # scores: {template_id: score}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_id, top_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_s - second_s

    if top_s >= cfg.confident_threshold and margin >= cfg.confident_margin:
        return ("ACCEPT", top_id, top_s)            # 规则层拍板
    if top_s <= cfg.unknown_threshold:
        return ("VLM", None, top_s)                 # 完全不匹配，交给 VLM 兜底
    return ("VLM", top_id, top_s)                   # 有候选但不确定，带着候选给 VLM
```

三种输出含义：
- **ACCEPT**：直接输出 `top_id`，不调 VLM（目标：70–90% 帧走这条路）
- **VLM with candidate**：模板有倾向但不自信，让 VLM 在 top-K 候选里裁决或补充
- **VLM cold**：规则层完全 miss，VLM 自由回答（最贵，但应该是少数）

建议初始阈值：`confident=8.0, confident_margin=3.0, unknown=2.0`，根据实际打分分布调。

---

## 5. Stage 3：VLM 兜底层

### 5.1 调用契约

**输入**：
- 下采样帧（512px 长边，JPEG 85q，通常 30–80KB）
- Evidence JSON（§2.2 的结构）
- 候选模板描述（top-3）+ 完整可选列表

**Prompt 骨架**：
```
你是场景识别助手。根据图像和下列自动检测证据，输出最匹配的场景类别。

[图像附件]

自动检测证据:
- 检测到的对象（按置信度排序）: dining_table(0.89), wine_glass(0.76) x3, ...
- 人数: 4
- 室内概率: 0.91
- 画面亮度: 0.34

候选场景（按规则打分排序）:
1. restaurant_interior (score=6.2)  —— 室内餐厅
2. conference_hall    (score=4.1)  —— 会议/会谈厅
3. bar_interior       (score=3.5)  —— 酒吧

允许的其他完整标签: [stage, plaza, retail, street, ...]

要求:
1. 先从候选里选；明显不对才用完整列表
2. 输出 JSON:
   {"label": "...", "confidence": 0.0~1.0, "reason": "简短解释"}
```

**要点**：
- 把证据**文本化**一起给 VLM，相当于"已经替它做了一半 CV 工作"，精度和延迟都更好；
- 要求 JSON 输出，便于解析；
- 用**有 structured output / JSON mode** 的 API（Claude、OpenAI、Gemini 都支持）。

### 5.2 缓存

VLM 调用昂贵。加两层缓存：
1. **感知哈希缓存**：`phash(frame) → vlm_result`，24h TTL。同一镜头静止时不重复问；
2. **证据签名缓存**：`hash(sorted(counts)) → vlm_result`，跨帧抵消（空餐厅画面多帧共享一个答案）。

经验值：缓存命中后整体 VLM 调用量能降 70%+。

### 5.3 超时 / 降级
- VLM 超时（>3s）→ 回退到 `top_id` 或 `unknown`，记录但不阻塞流水线；
- API 限流 → 按指数退避重试；再失败就走规则层最佳猜测。

---

## 6. 事件输出 & 持久化

每帧的处理结果都存成一条结构化事件，便于后续分析：

```json
{
  "ts": 1714123456.78,
  "final_label": "restaurant",
  "source": "template" | "vlm_candidate" | "vlm_cold",
  "confidence": 0.82,
  "template_top": {"id":"restaurant_interior","score":8.3,"margin":4.2},
  "vlm_call": null | {"model":"claude-opus-4-7","latency_ms":850,"cost_usd":0.007,"cache_hit":false},
  "evidence": {...}
}
```

写入 SQLite / ClickHouse / JSONL 文件都行。这既是可观测性数据源，也是将来训小模型的训练集。

---

## 7. 可观测性：几个关键指标

| 指标 | 健康目标 | 说明 |
|---|---|---|
| VLM escalation rate | <20%（成熟后 <10%） | 规则层覆盖率 |
| 模板命中分布 | 无单点压倒 | 提示模板粒度失衡 |
| VLM 与规则层 top-1 一致率 | 在 borderline 上 >60% | 低 → 阈值设太紧 |
| 平均 frame→label 延迟 | ACCEPT 路径 <100ms；VLM 路径 <1.5s | |
| VLM 月成本 | 线性可预测 | 缓存异常会飙升 |

建议用 Prometheus / 简单 CSV 定期汇总，画到 Grafana / 小 dashboard。

---

## 8. 第二阶段：真正的 SFT 闭环（可选）

当 VLM 回答积累够多（比如 >5k 条），可以做一次**蒸馏**，把 VLM 的判断能力塞进一个本地小模型：

### 8.1 数据构造
每条 VLM 兜底产生的样本：
```json
{
  "input": {
    "image": "<frame>",
    "evidence": {...}
  },
  "output": {
    "label": "restaurant",
    "reason": "..."
  }
}
```

### 8.2 训练方式
两条路子：

**A. 分类头微调（便宜）**：
- 在 CLIP/SigLIP image encoder 上加线性分类头；
- 用 VLM 标签训一个 N-way 分类器；
- 推理只需一次 image embedding；
- 精度一般能追到 VLM 的 85–90%。

**B. LoRA 微调小 VLM（更贵但更强）**：
- 底座选 Qwen2-VL-2B / MiniCPM-V 2.6 / LLaVA-1.6；
- 训练目标：输入(image+evidence) → 输出 VLM 的 JSON；
- 推理本地跑，消灭 VLM 调用成本；
- 需要一张 16GB+ 显卡。

### 8.3 回接到 router
把训好的小模型作为**第二层 specialist**，插在规则层和 VLM 之间：

```
Evidence → 模板 (ACCEPT?) → 小模型 (ACCEPT?) → VLM
```

随时间推进，VLM 调用占比会持续下降。

---

## 9. 推荐技术栈（Python）

| 层 | 技术 |
|---|---|
| 采集 + MJPEG 分发 | `opencv-python`（已有设计） |
| YOLO | `ultralytics` (YOLOv11n，Mac 上 CoreML / MPS) |
| Open-vocab 检测 | `transformers` + `google/owlv2-base-patch16-ensemble` |
| 全局特征 | `open_clip_torch`（单 prompt："a photo taken indoors"） |
| 模板存储 | YAML 文件 + `pydantic` schema 校验 |
| VLM 调用 | `anthropic` SDK（Claude Opus 4.7）/ `openai` SDK / `google-generativeai` |
| 事件持久化 | SQLite（起步）/ ClickHouse（规模化）/ JSONL（调试） |
| Dashboard | `streamlit` 一页看指标最快 |

---

## 10. 实施里程碑

1. **M1: 最小闭环**
   - 规则层只跑 YOLO → 模板打分 → 直接 ACCEPT 或输出 `unknown`；
   - 无 VLM，无缓存，纯跑通管线。

2. **M2: 接入 VLM 兜底 + 缓存**
   - borderline 和 cold 两条路径都接 Claude Opus 4.7；
   - 感知哈希缓存；
   - 事件写 SQLite。

3. **M3: 扩容规则层**
   - 加 OWL-ViTv2（按模板的 `owl_queries` 动态装配）；
   - 加全局特征；
   - 调阈值，让 escalation rate 降到目标。

4. **M4: 模板迭代工具**
   - 一个 CLI 从事件库里看 VLM 高频回答；
   - 一键生成"可能需要新增的模板"建议；
   - 人工 review 后加入 YAML 库。

5. **M5: SFT 蒸馏（可选）**
   - 攒足数据后训分类头；
   - A/B 对比小模型 vs 纯规则 vs VLM 的一致率 + 成本。

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| 模板爆炸（50+ 条后维护困难） | 引入层级：一级是粗类（室内/室外/人群），二级再细分；打分分层 |
| VLM 在模糊场景也抖 | 同一帧连续问 3 次取多数；或 ensemble 两家 VLM |
| 缓存把"场景已切换"藏住 | 哈希要敏感一点（pHash + 亮度直方图），短 TTL（5 min） |
| 规则层误判高置信度 | 每 N 条 ACCEPT 抽样 1 条走 VLM 做金标校验；偏差高就降阈值 |
| Open-vocab 检测慢 | 只在 margin < M 时才调用；或异步跑，不卡主路径 |

---

## 附录 A：最小版骨架代码

```python
# router.py
import yaml, json, hashlib, time
from pathlib import Path

class SceneRouter:
    def __init__(self, templates_path: str, cfg: dict, vlm_client):
        self.templates = yaml.safe_load(Path(templates_path).read_text())
        self.cfg = cfg
        self.vlm = vlm_client
        self.cache = {}   # 生产环境换成 Redis

    def score_all(self, evidence):
        return {t["id"]: score_template(evidence, t) for t in self.templates}

    def route(self, frame_bytes, evidence):
        scores = self.score_all(evidence)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_id, top_s = ranked[0]
        margin = top_s - (ranked[1][1] if len(ranked) > 1 else 0)

        if top_s >= self.cfg["confident_threshold"] and margin >= self.cfg["confident_margin"]:
            return {"source": "template", "label": self.label_of(top_id),
                    "confidence": min(1.0, top_s/10), "scores": dict(ranked[:3])}

        key = self._hash(frame_bytes, evidence)
        if key in self.cache:
            res = self.cache[key]; res["cache_hit"] = True; return res

        candidates = ranked[:3] if top_s > self.cfg["unknown_threshold"] else []
        res = self.vlm.classify(frame_bytes, evidence, candidates, self.templates)
        res["source"] = "vlm_candidate" if candidates else "vlm_cold"
        res["scores"] = dict(ranked[:3])
        self.cache[key] = res
        return res

    def label_of(self, tid):
        return next(t["label"] for t in self.templates if t["id"] == tid)

    def _hash(self, frame_bytes, evidence):
        h = hashlib.md5(frame_bytes[::100]).hexdigest()[:8]
        sig = tuple(sorted((k, v) for k, v in evidence["counts"].items() if v))
        return f"{h}:{sig}"
```

---

## 小结

- **把"规则 + 模板 + VLM 兜底"看作三段 router，不是 SFT**；
- **Stage 1** 的 Evidence JSON 是全系统的稳定契约；
- **模板用 YAML 写，打分带正负加权 + 硬性要求**，Router 决策不只看 top-1 还看 margin；
- **VLM 只在 borderline/cold 两种情况调**，配两层缓存，目标 escalation rate <20%；
- **每一次 VLM 调用都留档**，是日后 SFT 蒸馏的训练数据来源（§8 是第二阶段）；
- **分 5 个里程碑推进**，M1 纯跑通，M5 做蒸馏闭环。

---

## 12. 本仓库落地实现（2026-04-24）

已在当前项目内补齐可运行骨架，目录如下：

```text
scene_recognition_design/
├── config/
│   └── scene_templates.json        # 24 个预置场景模板（含 hackathon）
├── scene_router/
│   ├── __init__.py
│   ├── types.py                    # Evidence / Decision / Trace 数据结构
│   ├── evidence.py                 # detector 输出归一化 + counts 聚合
│   ├── scorer.py                   # 规则检测层打分（支持复合条件）
│   ├── router.py                   # ACCEPT / VLM_CANDIDATE / VLM_COLD 路由
│   └── vlm.py                      # VLM Stub（便于先跑通）
├── scripts/
│   ├── demo_router.py              # 三组样例：hackathon / restaurant / ambiguous
│   └── route_evidence.py           # evidence JSON -> 路由结果
└── tests/
    └── test_router.py              # 单测：规则层可识别 hackathon
```

### 12.1 预置场景数量

`config/scene_templates.json` 目前内置 **24** 个场景：

- hackathon
- conference_hall
- classroom
- office_open_space
- coworking_space
- stage_performance
- empty_stage
- public_plaza
- restaurant_interior
- bar_interior
- cafe
- retail_store
- supermarket
- street_day
- street_night
- parking_lot
- train_station
- airport_terminal
- gym
- home_kitchen
- bedroom
- hospital_ward
- laboratory
- server_room

### 12.2 Hackathon 规则层识别策略（关键）

`hackathon` 模板使用了“硬条件 + 加权证据 + 复合加分”：

1. 硬条件：`person>=3` 且 `laptop>=2`，并且 `whiteboard/projector_screen/banner` 至少有一个。  
2. 正向证据：`laptop/monitor/keyboard/person/whiteboard/projector_screen/badge/banner`。  
3. 负向证据：`bed/oven/sofa/dining_table`。  
4. 复合加分：
   - `dense_team_work`: `person>=6 && laptop>=4`
   - `event_context`: `badge>=2 && banner>=1`

这保证在你描述的黑客松现场，规则检测层就能直接命中，而不依赖 VLM 兜底。

### 12.3 本地验证命令

```bash
# 运行演示
python scripts/demo_router.py

# 运行测试
python -m unittest discover -s tests -q

# 单条 evidence 路由（你接入 YOLO/OWL 后可直接调用）
python scripts/route_evidence.py /path/to/evidence.json
```

当前 demo 输出中，`hackathon_like` 的路由结果为 `RULE_ACCEPT`，标签为 `hackathon`。

### 12.4 实机 Demo（本机摄像头）

新增脚本：`scripts/live_scene_demo.py`，可以直接跑你当前连接外接摄像头的设备。
另外提供了快捷脚本：`scripts/run_live_demo.sh`。

#### 安装依赖

```bash
cd /Users/russellfool/Downloads/scene_recognition_design
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt
```

#### 仅规则层（YOLO + 模板）

```bash
./scripts/run_live_demo.sh none
```

可选环境变量：`CAMERA_INDEX=1 SAMPLE_INTERVAL=0.8 DETECTOR=none ./scripts/run_live_demo.sh openai gpt-5.1`

如果想先不装 YOLO，只验证整体链路（将主要依赖 VLM）：

```bash
python scripts/live_scene_demo.py --camera-index 0 --detector none --provider openai --model gpt-5.1 --preview
```

#### 规则层 + OpenAI 兜底

```bash
export OPENAI_API_KEY="..."
./scripts/run_live_demo.sh openai gpt-5.1
```

#### 规则层 + Anthropic 兜底

```bash
export ANTHROPIC_API_KEY="..."
./scripts/run_live_demo.sh anthropic claude-opus-4-1
```

#### 若视频流来自另一台主机

如果你在 Host A 暴露了 `http://<host-a-ip>:8080/frame.jpg`：

```bash
python scripts/live_scene_demo.py \
  --frame-url http://192.168.1.50:8080/frame.jpg \
  --provider openai \
  --model gpt-5.1 \
  --sample-interval 1.0
```

运行后会持续打印每帧 JSON，包含：
- `scene`（稳定后输出场景）
- `raw_label`（当前帧原始判定）
- `source`（`rules` 或 `vlm`）
- `route_decision`（`RULE_ACCEPT` / `VLM_CANDIDATE` / `VLM_COLD`）
- `top3`（模板分数前 3）
