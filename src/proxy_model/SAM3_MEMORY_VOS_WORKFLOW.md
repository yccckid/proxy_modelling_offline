# SAM3 锚定的长时记忆双向视频分割

## 1. 目的与适用范围

本方法从 ROS bag 的图像话题中生成目标预制构件的逐帧二值掩膜，用于后续图像遮罩、点云筛选和 GS-SDF 重建。

它面向混凝土等低纹理构件、相机运动、视角由长边转为短边、局部遮挡和尺度变化等情况。掩膜估计阶段只使用 RGB 图像和 SAM3 的视频记忆；LiDAR 点云、相机位姿和雷达位姿不参与掩膜生成、关键帧选择或恢复。它们只在掩膜最终确定后用于下游点云提取。

该实现不使用稠密光流，也不会生成 `flows/`。

```text
Qwen（一次）
  → 图像 SAM3 初始锚点
  → SAM3 Video Memory 初步传播
  → 自动新视角锚点
  → 独立前向/后向传播
  → 概率融合与质量检查
  → SAM3 异常恢复
  → 最终 mask 与点云提取
```

## 2. 图像序列和语义初始化

程序读取 bag 中配置的图像话题，并按“时间戳、原始图像消息序号”稳定排序。每一帧保存以下索引信息：

```text
frame_id, timestamp_ns, source_index, image_path
```

在 `seed_frame` 上，Qwen3-VL-Plus 仅调用一次，将中文提示转换为：

- SAM3 可用的英文目标描述；
- 实例选择模式：单实例 `one` 或同类全部实例 `all`；
- 单实例场景下的初始中心点。

随后图像版 SAM3 分割种子帧。经过质量检查的种子 mask 成为第一个永久锚点。

## 3. 两级记忆

### 3.1 永久锚点

每个永久锚点保存：

- 帧编号；
- 图像 SAM3 验证的二值 mask；
- 掩膜质量分数；
- 目标区域 HSV 外观直方图；
- 最小外接框的长宽比；
- 来源：`sam3_seed`、`sam3_view_anchor`、`sam3_terminal_anchor` 或 `sam3_recovery`。

永久锚点的数量由 `max_anchor_memory` 限制，当前默认值为 8。候选锚点不会因时间早晚直接覆盖已有锚点；只有视角新颖且通过质量、身份检查的帧才会加入。

### 3.2 原生工作记忆

工作记忆由 SAM3 Video Predictor 内部的 memory bank 管理。本项目不实现自定义 SDF memory adapter，也不声称向 SAM3 记忆编码器写入额外的距离场特征。

当前本地 SAM3 视频模型的原生近期记忆有 7 个槽，因此配置为：

```yaml
max_working_memory: 7
```

该值只能降低原生槽数，不能安全地增大超过模型构建时的槽数。视频帧可保留在 CPU 内存以减小显存压力；模型推理、特征编码和记忆传播仍在 CUDA GPU 上执行。

## 4. 初步传播与新视角锚点

初始锚点建立后，SAM3 视频模型先完成一次单锚点初步传播。该结果只用于发现候选视角，不立即作为最终输出。

对暂定掩膜，定义质量分数：

\[
q_t=\frac{
\bar p_t+c_t^{cc}+(1-h_t)+(1-b_t)+q_t^{img}+c_t^{fb}
}{6}.
\]

其中：

- \(\bar p_t\)：前景区域平均置信度；
- \(c_t^{cc}\)：最大连通域面积占总前景面积的比例；
- \(h_t\)：封闭孔洞面积占前景面积的比例；
- \(b_t\)：目标触及图像边缘的比例；
- \(q_t^{img}\)：图像质量，由拉普拉斯清晰度和曝光有效比例组成；
- \(c_t^{fb}\)：前后向一致性。初步单向阶段取 1，仅用于候选排序。

视角新颖性定义为：

\[
d_t=\min_{a\in A}
\left[
0.45(1-\operatorname{IoU}(\bar M_t,\bar M_a))
+0.15\left|\log\frac{r_t}{r_a}\right|
+0.40(1-\cos(\mathbf z_t,\mathbf z_a))
\right].
\]

\(\bar M_t\) 是裁剪、缩放到统一大小后的目标轮廓，\(r_t\) 是外接框长宽比，\(\mathbf z_t\) 是目标区域 HSV 外观直方图。

候选帧必须同时满足：

```yaml
quality_score >= quality_threshold          # 默认 0.60
view_novelty >= view_novelty_threshold      # 默认 0.32
与已有锚点的时间距离 >= min_anchor_time_gap_s  # 默认 2.0 s
```

候选帧会重新运行图像 SAM3。其掩膜质量通过，并且相对已有锚点的 HSV 身份相似度通过时，才写入永久锚点。程序还会尝试在序列终端建立可靠锚点，使中间帧具备双向约束。

## 5. SAM3 视频记忆传播

对于一个永久锚点，程序将：

1. 使用锚点的 SAM3 文本描述；
2. 从锚点 SAM3 mask 提取归一化外接框；
3. 以文本和目标框初始化 SAM3 Video Predictor；
4. 使用其原生 memory bank 传播该对象。

当前公开的 SAM3 Video Predictor 接口接受文本、点和框提示；本实现使用经图像 SAM3 验证的外接框作为视频提示。锚点的完整二值 mask 用于最终锚点输出、质量判定和候选评估，但不会通过非公开接口强行写入模型内部 memory bank。

这种方法追踪的是对象级外观和历史记忆，而非上一帧和当前帧的逐像素对应，因此不依赖混凝土墙内部必须存在可匹配纹理。

## 6. 双向传播与概率融合

相邻永久锚点记为 \(a_L\) 与 \(a_R\)。程序分别创建两个独立的 SAM3 视频会话：

- 从 \(a_L\) 向右传播，得到 \(P_t^\rightarrow\)；
- 从 \(a_R\) 向左传播，得到 \(P_t^\leftarrow\)。

两个方向不共享工作记忆，避免一个方向的漂移污染另一个方向。

前后向一致性为：

\[
c_t^{fb}=
\operatorname{IoU}
\left(
\mathbb{I}[P_t^\rightarrow\geq\tau_m],
\mathbb{I}[P_t^\leftarrow\geq\tau_m]
\right),
\]

其中 \(\tau_m\) 为 `mask_threshold`，默认 0.5。

最终概率采用软融合，不能使用二值 mask 的交集：

\[
P_t=\frac{
w_t^\rightarrow P_t^\rightarrow+w_t^\leftarrow P_t^\leftarrow
}{w_t^\rightarrow+w_t^\leftarrow}.
\]

权重同时考虑该方向的实例置信度和当前帧到锚点的时间距离：

\[
w_t^\rightarrow=\frac{p_t^\rightarrow}{t-a_L},
\qquad
w_t^\leftarrow=\frac{p_t^\leftarrow}{a_R-t}.
\]

序列两端若缺少另一侧锚点，只使用单向结果，并记录 `one_direction_only` 标志。

## 7. 概率图与差异图

SAM3 Video Predictor 当前公开输出为二值 `out_binary_masks` 和实例级 `out_probs`，不直接提供原始逐像素 logits。因此当前实现保存的概率图为：

\[
P_t(u,v)=\mathbb{I}[M_t(u,v)]\times p_t^{instance}.
\]

它是“mask 内的实例置信度图”，不是未经处理的像素级网络 logit。

前后向差异图为：

\[
D_t^{fb}=\left|P_t^\rightarrow-P_t^\leftarrow\right|.
\]

亮区表示两个方向不一致，常对应遮挡、轮廓漂移、孔洞、视角转换或目标身份不稳定。

## 8. 失效检测与 SAM3 恢复

以下任一情况触发当前帧的图像 SAM3 恢复：

- \(c_t^{fb}<\texttt{fb\_iou\_threshold}\)，默认 0.65；
- 综合质量低于 `quality_threshold`；
- mask 为空；
- 面积小于最近可靠 mask 面积中位数的 30%；
- 面积大于最近可靠 mask 面积中位数的 3 倍；
- 连通域、孔洞、边界截断或图像质量异常导致低质量。

恢复提示点选择：

1. 若融合概率最大值不低于 `recovery_prompt_min_confidence`，默认 0.45，取最大概率像素作为正点；
2. 否则回退到时间上最近永久锚点的 mask 质心。

恢复 mask 仅在质量通过时替换当前输出。若其外观身份检查也通过，且永久锚点未满，该恢复帧会加入永久锚点集合。为了避免退化成逐帧 SAM3，单个 bag 的恢复预算由 `max_recovery_frames` 限制，默认 80。

## 9. 最终 mask 后处理

后处理只清除小噪声，不负责修复大面积模型漏分：

- 删除面积小于 `min_component_area_px` 的连通域，默认 100 像素；
- 可选填充面积不超过 `max_hole_area_px` 的封闭小孔洞，默认 400 像素；
- 不填充与图像边缘相连的背景；
- 不进行固定膨胀或大尺度闭运算。

## 10. 输出文件

在 `output.cache_dir` 中生成：

```text
frames/                    # 时间有序 RGB 帧
masks/                     # 最终二值掩膜
overlays/                  # mask 可视化
probabilities/             # mask × 实例置信度图
disagreements/             # 双向概率差异图；单向端部帧无文件
labeled_pcd/               # 下游输出：全局坐标系的 object/non-object PCD
point_image/               # 下游输出：点云投影叠加图
segmentation_metrics.csv   # 每帧质量、来源和效率指标
summary.json               # bag 级统计
```

`segmentation_metrics.csv` 的每帧字段包括：

- `frame_id`、`timestamp_ns`、`source_index`；
- mask、概率图、差异图路径；
- `source`、`nearest_anchor_ids`、`is_permanent_anchor`；
- 前景置信度、连通域比例、孔洞比例、边界比例、清晰度、曝光、综合质量；
- 前后向 IoU、视角新颖性、失败标志；
- VOS 时间、SAM3 调用标志、SAM3 恢复标志。

CSV 还重复记录 bag 级效率指标：SAM3 总调用数、VOS 平均/峰值单帧时间、GPU/主机峰值内存、永久锚点数量、工作记忆容量、每分钟视频处理时间、恢复比例与恢复成功率。

`mask_iou_j` 与 `boundary_f` 字段已预留。没有人工真值 mask 时，这两个字段保持为空，不能用模型自身结果代替真值评价。

## 11. 主要配置

```yaml
segmentation:
  video_model_version: sam3
  video_checkpoint: /home/yc/SAM-AGENT/weights/sam3.pt
  video_compile: false
  mask_threshold: 0.5
  fb_iou_threshold: 0.65
  quality_threshold: 0.60
  view_novelty_threshold: 0.32
  min_anchor_time_gap_s: 2.0
  max_anchor_memory: 8
  max_working_memory: 7
  max_recovery_frames: 80
  recovery_prompt_min_confidence: 0.45
  min_component_area_px: 100
  max_hole_area_px: 400
  allow_hole_filling: true

output:
  save_probabilities: true
  save_disagreements: true
```

这些阈值是当前工程默认值，应在独立人工标注验证帧上调整；不应使用最终测试序列调参后再报告其精度。

