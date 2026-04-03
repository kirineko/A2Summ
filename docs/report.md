# A2Summ 端到端视频摘要处理技术报告

## 1. 背景与目标

原始 A2Summ 项目并不是一个“原始视频直接输入”的系统。它默认依赖已经预处理好的多模态特征：

- 视频特征：`GoogleNet/Pool5` 风格的帧级特征
- 文本特征：`RoBERTa` 风格的句子级特征
- 跨模态对齐：视频帧与文本句子的时间对应掩码
- 数据容器：`SumMe/TVSum` 风格的 `h5`

本次改造的目标是把它补成一条可运行的端到端推理链路：

`原始视频 -> 预处理 -> h5/文本特征/ASR/对齐 -> 复用现有模型 -> 摘要 JSON/截图`

这条链路的设计原则是：

- 尽量复用原有模型与 checkpoint
- 尽量贴近 `VASNet/DSNet` 社区常用的 `SumMe/TVSum` 数据组织方式
- 在不重训模型的前提下，引入更真实的文本侧输入
- 让结果能落成结构化文件，便于后续分析、评估和系统集成

---

## 2. 当前系统架构

当前系统可以分成 4 层。

### 2.1 原始视频输入层

输入是用户提供的原始视频文件，例如：

- `video_input/0001.mp4`
- `video_input/0004.mp4`

系统通过 `ffprobe` 获取：

- 总帧数
- 帧率

通过 `ffmpeg` 完成：

- 抽帧
- 音频抽取
- 摘要片段截图导出

### 2.2 多模态预处理层

该层现在由 [video_pipeline.py](/home/linux/A2Summ/video_pipeline.py) 实现，主要职责：

1. 视频抽帧
2. 视觉特征提取
3. Whisper ASR
4. 文本特征构造
5. 视频-文本时间对齐掩码构造
6. `h5` 与 sidecar 文件写出

### 2.3 模型推理层

该层由 [infer_video_summary.py](/home/linux/A2Summ/infer_video_summary.py) 驱动，核心仍然复用原始项目的：

- [models.py](/home/linux/A2Summ/models.py) 中的 `Model_VideoSumm`
- `saved_model/SumMe` 或 `saved_model/TVSum` 中的 checkpoint

这里没有改动原始模型结构，只是在输入准备上把“原本依赖外部预处理数据”的流程补齐了。

### 2.4 结果输出层

输出包括：

- `*.h5`
- `*_text_roberta.npy`
- `*_alignment.npz`
- `*_asr.json`
- `*_metadata.json`
- `result.json`
- `screenshots/*.jpg`

这使系统既能保留中间过程，也能直接提供最终摘要结果。

---

## 3. 输入与输出

## 3.1 输入

当前主要输入参数有：

- `--video`：原始视频路径
- `--dataset`：使用哪套 checkpoint 配置，当前支持 `SumMe` 或 `TVSum`
- `--checkpoint-dir`：模型权重目录
- `--work-dir`：输出目录
- `--output-json`：最终摘要 JSON 路径
- `--asr-model`：当前默认 `openai/whisper-base`

典型输入示例：

```bash
uv run python infer_video_summary.py \
  --video video_input/0001.mp4 \
  --dataset SumMe \
  --checkpoint-dir saved_model/SumMe \
  --work-dir video_output/0001 \
  --output-json video_output/0001/result.json \
  --asr-model openai/whisper-base
```

## 3.2 输出

以 `video_output/0001` 为例，输出包括：

- [video_output/0001/0001.h5](/home/linux/A2Summ/video_output/0001/0001.h5)
- [video_output/0001/0001_text_roberta.npy](/home/linux/A2Summ/video_output/0001/0001_text_roberta.npy)
- [video_output/0001/0001_alignment.npz](/home/linux/A2Summ/video_output/0001/0001_alignment.npz)
- [video_output/0001/0001_asr.json](/home/linux/A2Summ/video_output/0001/0001_asr.json)
- [video_output/0001/0001_metadata.json](/home/linux/A2Summ/video_output/0001/0001_metadata.json)
- [video_output/0001/result.json](/home/linux/A2Summ/video_output/0001/result.json)
- [video_output/0001/screenshots/segment_00.jpg](/home/linux/A2Summ/video_output/0001/screenshots/segment_00.jpg)

其中最终最重要的是 `result.json`，当前包含：

- 基础信息：源视频、帧率、采样帧数
- 中间产物路径
- `asr_segments`
- `summary_ratio`
- `summary_description`
- `selected_segments`
- 每个摘要片段对应截图路径

---

## 4. 运行原理

## 4.1 视频侧

视频侧处理流程如下：

1. 使用 `ffprobe` 获取视频总帧数和帧率
2. 使用 `ffmpeg` 按固定步长采样抽帧
3. 将抽出的帧送入 `GoogLeNet`
4. 取 `avgpool` 输出作为 1024 维视觉特征

为什么选 `GoogLeNet avgpool`：

- 这与 `SumMe/TVSum` 社区常见的 `GoogleNet pool5` 特征分布更接近
- 能最大程度贴近 `VASNet/DSNet` 风格输入
- 对现有 A2Summ 视频摘要分支的输入维度是兼容的

## 4.2 音频/文本侧

文本侧不再使用原来的代理特征，而是接入了 Whisper：

1. 用 `ffmpeg` 从视频中抽取单声道 `16kHz` 音频
2. 用 `transformers` 加载 `openai/whisper-base`
3. 做带时间戳的 ASR
4. 得到一组语音片段：
   - 文本内容
   - `start_sec`
   - `end_sec`

例如当前识别结果：

- `0001.mp4`：`我不想再跟你查`
- `0004.mp4`：`有那么明显吗`

## 4.3 文本特征构造

原模型训练时吃的是 `768` 维 `RoBERTa` 风格句子特征，但当前端到端链路并没有复现完整的原始文本编码流程。

因此当前方案采用折中做法：

1. 对 Whisper 输出的每个 ASR 片段文本做 tokenizer 编码
2. 从 Whisper decoder 的 token embedding 中做平均池化
3. 若维度不为 `768`，通过固定随机投影映射到 `768` 维

这一步的意义是：

- 保持当前模型接口兼容
- 用真实 ASR 文本替换之前完全基于视觉特征伪造的文本特征

需要强调：

这不是与原始训练分布完全一致的 `RoBERTa` 文本特征，而是一种“工程可运行、接口兼容”的近似方案。

## 4.4 视频-文本对齐

当前对齐方式是基于 ASR 时间戳构造：

1. 根据视频采样率和 `sample_every` 确定每个采样帧对应的时间
2. 根据 ASR 片段的 `start_sec/end_sec`
3. 构造：
   - `frame_to_text`
   - `text_to_frame`

这样，跨模态交互不再是平均切分得到的伪对齐，而是：

“哪些采样帧落在这句对白对应的时间段内，哪些就与该文本 token 建立连接”

这比旧版本的等长划分更符合 A2Summ 的跨模态注意力设计初衷。

---

## 5. 数据预处理流程

## 5.1 预处理产物

当前预处理脚本是 [preprocess_video_to_h5.py](/home/linux/A2Summ/preprocess_video_to_h5.py)。

它会产出 5 类核心文件：

### 1. `h5`

与 `SumMe/TVSum` 风格兼容的容器，包含：

- `features`
- `change_points`
- `n_frames`
- `n_frame_per_seg`
- `picks`
- `gtscore`
- `gtsummary`
- `user_summary`

其中推理真正依赖的是：

- `features`
- `change_points`
- `n_frames`
- `n_frame_per_seg`
- `picks`

### 2. `*_text_roberta.npy`

以 `{video_key: ndarray}` 的方式保存文本侧输入，当前为 `768` 维 ASR 片段特征。

### 3. `*_alignment.npz`

保存：

- `frame_to_text`
- `text_to_frame`

用于推理阶段直接复用真实时间对齐结果。

### 4. `*_asr.json`

保存 Whisper 输出的片段化语音识别结果：

```json
[
  {
    "text": "我不想再跟你查",
    "start_sec": 0.0,
    "end_sec": 1.08
  }
]
```

### 5. `*_metadata.json`

保存辅助分析信息：

- 帧率
- 原始总帧数
- 采样间隔
- 采样后帧数
- ASR 片段
- `change_points`

## 5.2 多模态融合是如何做的

当前系统不是在预处理阶段直接把视频和文本“融合成一个向量”，而是采用更符合原框架的方式：

1. 分别构建视频特征和文本特征
2. 构建跨模态时间对齐掩码
3. 在模型内部由 `Model_VideoSumm` 完成跨模态融合

换句话说，预处理阶段负责的是：

- 特征准备
- 对齐准备

模型阶段负责的是：

- 真正的注意力交互
- 跨模态融合
- 摘要分数预测

这种设计与原始 A2Summ 框架是兼容的，因为原模型本来就假设输入是：

- `video`
- `text`
- `video_to_text_mask_list`
- `text_to_video_mask_list`

## 5.3 如何与原有框架融合

当前改造尽量不入侵原有训练框架：

- 不修改 [models.py](/home/linux/A2Summ/models.py) 的模型结构
- 不修改 [train_videosumm.py](/home/linux/A2Summ/train_videosumm.py) 的训练逻辑
- 不修改已有 checkpoint

只在推理前面补上“把原始视频加工成现有模型能吃的张量和文件”的流程。

这意味着：

- 原有研究代码的可重复性尽量不受影响
- 新增端到端能力主要集中在新增脚本中

---

## 6. 当前项目相对原项目的改进与创新

相对原始 A2Summ 仓库，当前项目已经新增了这些能力。

### 6.1 支持原始视频直接输入

原始项目：

- 需要用户先准备数据集特征

当前项目：

- 用户可以直接输入 `.mp4`

### 6.2 支持自动 ASR

原始项目：

- 文本侧依赖预先准备好的 `text_roberta.npy`

当前项目：

- 通过 Whisper 自动生成带时间戳的 ASR 片段

### 6.3 支持自动构造对齐掩码

原始项目：

- 默认依赖已有对齐逻辑或数据集约定

当前项目：

- 根据 ASR 时间戳自动生成视频-文本对齐

### 6.4 支持中间产物导出与可解释分析

当前新增：

- `asr.json`
- `alignment.npz`
- `metadata.json`
- 摘要截图
- 结构化 JSON 摘要结果

这让系统更适合：

- 工程调试
- 误差分析
- 论文复现实验记录

### 6.5 支持更自然的中文摘要描述

当前 `result.json` 已支持结合 ASR 内容生成更自然的中文描述，而不是只有冷冰冰的时间区间。

---

## 7. 当前主要问题

虽然系统已经能完整跑通，但还有明显问题。

### 7.1 文本特征分布与原训练分布不一致

当前文本特征来自：

- Whisper token embedding 平均池化
- 再映射到 768 维

而原始训练时更接近：

- RoBERTa 句子级文本特征

所以当前系统只是“接口对齐”，不等于“分布完全对齐”。

### 7.2 视频分段方式仍然是近似的

当前 `change_points` 是按固定长度切分出来的近似段，不是 `DSNet/VASNet` 风格更严格的镜头分割/KTS 结果。

这会影响：

- 片段边界质量
- 摘要压缩效果
- 短视频场景下的细粒度选段能力

### 7.3 模型未针对新输入分布微调

当前复用的是原始 `SumMe` checkpoint。

但新的输入分布已经改变：

- 原始视频来源不同
- 文本由 ASR 生成
- 对齐方式不同

所以现在系统能跑，但效果不一定稳定。

### 7.4 短视频容易整段保留

在当前测试中：

- `0001.mp4`
- `0004.mp4`

都出现了整段保留，说明模型在这种很短的剧情片段上更倾向于把它视为一个完整事件，而不是进一步做压缩。

### 7.5 当前中文摘要描述还主要依赖对白

现在的自然语言描述主要依赖 ASR 内容，还不能真正理解：

- 人物是谁
- 场景是什么
- 动作和关系是什么

因此它更像“对白驱动的摘要说明”，还不是“视觉语义充分理解后的剧情摘要”。

---

## 8. 改进方案

## 8.1 短期可做

### 1. 引入更接近原始文本分布的编码器

方案：

- 接入中文/多语言文本编码器
- 或直接使用 RoBERTa / BERT 类句向量模型

目标：

- 替代当前 Whisper embedding 投影方案

### 2. 替换固定分段为镜头切分

方案：

- 引入 KTS
- 或基于相邻帧特征差异做 shot boundary detection

目标：

- 让 `change_points` 更接近社区标准预处理

### 3. 更丰富的摘要描述

方案：

- 接入视觉语言模型
- 将截图与对白联合生成中文描述

目标：

- 让 `summary_description` 不只引用对白，而能描述画面事件

## 8.2 中期可做

### 4. 重新构造伪训练集并微调

方案：

- 用端到端管线重做一版特征
- 对现有模型进行微调

目标：

- 缩小训练/推理分布偏差

### 5. 支持更完整的输出

方案：

- 输出真正的摘要视频片段 `summary.mp4`
- 输出多截图摘要卡片

目标：

- 让系统更适合演示和部署

## 8.3 长期可做

### 6. 彻底摆脱旧特征格式

方案：

- 改造为真正端到端的原始视频 + 原始音频多模态模型

目标：

- 不再受限于 `h5 + 1024/768` 的历史接口

---

## 9. 当前如何运行

## 9.1 预处理单独运行

```bash
uv run python preprocess_video_to_h5.py \
  --video video_input/0001.mp4 \
  --output-dir /tmp/a2summ_preprocessed \
  --asr-model openai/whisper-base
```

会生成：

- `0001.h5`
- `0001_text_roberta.npy`
- `0001_alignment.npz`
- `0001_asr.json`
- `0001_metadata.json`

## 9.2 端到端推理运行

```bash
uv run python infer_video_summary.py \
  --video video_input/0001.mp4 \
  --dataset SumMe \
  --checkpoint-dir saved_model/SumMe \
  --work-dir video_output/0001 \
  --output-json video_output/0001/result.json \
  --asr-model openai/whisper-base
```

## 9.3 批量跑多个视频

当前没有单独写批处理脚本，但可以在 shell 里直接循环：

```bash
for f in video_input/*.mp4; do
  stem=$(basename "$f" .mp4)
  uv run python infer_video_summary.py \
    --video "$f" \
    --dataset SumMe \
    --checkpoint-dir saved_model/SumMe \
    --work-dir "video_output/$stem" \
    --output-json "video_output/$stem/result.json" \
    --asr-model openai/whisper-base
done
```

---

## 10. 当前如何查看输入和输出

## 10.1 查看输入视频

输入目录在：

- [video_input](/home/linux/A2Summ/video_input)

例如：

- [video_input/0001.mp4](/home/linux/A2Summ/video_input/0001.mp4)
- [video_input/0004.mp4](/home/linux/A2Summ/video_input/0004.mp4)

## 10.2 查看最终摘要结果

直接看：

- [video_output/0001/result.json](/home/linux/A2Summ/video_output/0001/result.json)
- [video_output/0004/result.json](/home/linux/A2Summ/video_output/0004/result.json)

重点字段：

- `summary_description`
- `selected_segments`
- `asr_segments`
- `summary_ratio`

## 10.3 查看语音识别结果

直接看：

- [video_output/0001/0001_asr.json](/home/linux/A2Summ/video_output/0001/0001_asr.json)
- [video_output/0004/0004_asr.json](/home/linux/A2Summ/video_output/0004/0004_asr.json)

## 10.4 查看摘要截图

直接看：

- [video_output/0001/screenshots/segment_00.jpg](/home/linux/A2Summ/video_output/0001/screenshots/segment_00.jpg)
- [video_output/0004/screenshots/segment_00.jpg](/home/linux/A2Summ/video_output/0004/screenshots/segment_00.jpg)

## 10.5 查看中间对齐信息

可以读取：

- `*_alignment.npz`
- `*_metadata.json`

其中 `metadata` 更适合人工查看，`alignment.npz` 更适合程序分析。

---

## 11. 当前阶段结论

当前系统已经从“只能吃预处理数据的研究代码”演化成了“可以直接接受原始视频并输出结构化摘要结果的原型系统”。

它已经具备：

- 原始视频输入
- 视频特征提取
- Whisper ASR
- 文本特征构造
- 跨模态对齐
- 复用原模型推理
- 结果 JSON 输出
- 摘要截图
- 中文摘要说明

但它仍然是一个“面向研究和验证的工程原型”，不是完全成熟的产品级系统。

目前最核心的价值在于：

- 把原始 A2Summ 的离线研究流程补成了端到端可运行系统
- 为后续微调、替换文本编码器、改进镜头分割、增强中文摘要质量打下了工程基础

如果后续继续推进，最值得优先做的两件事是：

- 用更匹配原训练分布的文本编码器替换当前 Whisper embedding 投影
- 用更标准的 shot segmentation / KTS 替换当前固定分段策略
