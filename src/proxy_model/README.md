# proxy_model

把 FAST-LIVO2 ROS bag 转换成“仅保留指定物体”的新 bag。输出 bag 保持原话题名和消息类型不变，可直接继续使用 GS-SDF 自带的 `rosbag_to_colmap.py`。

处理流程：

1. 从 bag 顺序提取图像和 camera/lidar 位姿。
2. Qwen-VL 在种子帧上仅调用一次，把中文目标描述解析为 SAM 可用的英文类别和实例中心。
3. SAM 3 在种子帧建立初始锚点；SAM3 原生视频记忆模型在全部帧上进行对象级传播。系统自动选择经 SAM3 验证的新视角锚点，并在前后向预测不一致、质量低或面积突变时才调用 SAM3 恢复，不使用固定间隔重分割。
4. 图像 mask 外像素置黑。
5. 使用逐帧 `inv(T_world_camera) @ T_world_lidar` 把点云投影到图像；仅保留 mask 内且接近该像素栅格最近深度的点。
6. 对点云做轻量多视角一致性检查：同一个点在邻近多个视角中投影到前景 mask 的比例超过阈值才保留。
7. 第二遍按原始 bag 时间顺序写出。所有未处理话题逐消息原样复制，并额外发布每帧 `mono8` mask topic。

## 环境

```bash
cd /home/yc/proxy_modelling
source /opt/ros/noetic/setup.bash
source .venv/bin/activate
pip install -r src/proxy_model/requirements.txt
pip install -e /home/yc/SAM-AGENT/sam3_src
export DASHSCOPE_API_KEY="你的 key"
```

项目要求使用 `.venv`；不要用 `sudo pip`。如果已有 SAM-Agent 环境，可以参照它的已安装版本补齐 `sam3`、`openai` 和相关依赖。

## 配置

编辑 `config/my_data_0621.yaml`：

- `segmentation.prompt`：明确描述场景中的那一个物体。
- `segmentation.seed_frame`：目标清楚可见的帧号；不一定要设为 0。
- `camera`：必须使用输入 bag 原始图像的内参和畸变参数，即此前传给 `rosbag_to_colmap.py` 的值。
- `segmentation.video_model_version`：SAM3 视频后端版本；当前本地 `sam3.pt` 配合 `sam3`。只有安装匹配权重时才改为 `sam3.1`。
- `segmentation.mask_threshold`、`fb_iou_threshold`、`quality_threshold`：二值化、双向一致性和自动恢复阈值。
- `segmentation.max_anchor_memory`、`max_working_memory`：永久锚点与 SAM3 原生近期记忆的容量上限。
- `segmentation.view_novelty_threshold`、`min_anchor_time_gap_s`：自动新视角锚点的外观/形状新颖性和时间 NMS 条件。
- `segmentation.max_recovery_frames`：单个 bag 最多允许的 SAM3 自动恢复次数，避免低质量序列退化成逐帧 SAM3。
- `topics.mask`：输出 bag 中的 mask 话题，默认 `/proxy_model/object_mask`。
- `point_filter.multiview_ratio`：点云多视角 mask 命中比例阈值，默认 `0.9`。
- `output.save_probabilities`：是否在 `cache_dir/probabilities/` 保存视频模型的前景置信度图，默认 `true`。
- `output.save_disagreements`：是否保存 `cache_dir/disagreements/` 中的前后向概率差异图，默认 `true`。
- `output.save_labeled_pcd`：是否输出世界坐标系下的完整标签点云；PCD 的 `label=1`
  表示投影落在 mask 内，`label=0` 表示 non-object，默认 `true`。
- `output.save_point_images`：是否把 mask 内点云投影以紫红色叠加到 overlay，默认 `true`。
- `output.overwrite`：确认允许覆盖旧输出后才设为 `true`。

可先关闭 Qwen 做离线调试：

```yaml
qwen:
  enabled: false
  label: excavator
  mode: one
  center_point: [0.5, 0.5]  # 归一化坐标，也可填写像素坐标
```

## 运行

```bash
cd /home/yc/proxy_modelling
source /opt/ros/noetic/setup.bash
source .venv/bin/activate
python src/proxy_model/scripts/build_object_bag.py \
  --config src/proxy_model/config/my_data_0621.yaml
```

mask、叠加预览、前景置信度图和统计信息保存在配置的 `cache_dir`。不再生成或使用
`flows/`。同级的 `segmentation_metrics.csv` 保存每帧质量项、前后向 IoU、来源、锚点、
恢复标记和 VOS 时间；每行也含该 bag 的 SAM3 调用数、峰值显存/内存、记忆容量和处理
效率汇总。务必先查看 `overlays/` 和 CSV 中的 `failure_flags`，确认目标实例无误，再运行
耗时较长的 GS-SDF。

此外，每个匹配点云帧会新增：

```text
cache_dir/
├── labeled_pcd/  # 世界坐标完整点云，字段 x y z label；CloudCompare 可直接打开
└── point_image/  # mask overlay + mask 内投影点（紫红色）
```

两个目录使用同一个六位点云帧编号。标签仅由当前点投影是否落入对应图像 mask 决定，
不会被后续多视角一致性、最大聚类或输出 bag 点云过滤改变。

## 转 COLMAP

输出 bag 的接口与输入一致，原转换命令只需替换 `--bag_path`：

```bash
python src/GS-SDF/scripts/rosbag_convert/rosbag_to_colmap.py \
  --bag_path src/GS-SDF/data/my_bag/my_data_0621_object.bag \
  --image_topic /origin_img/compressed \
  --image_pose_topic /aft_mapped_to_init_cam \
  --mask_topic /proxy_model/object_mask \
  --point_topic /cloud_registered_body \
  --point_pose_topic /aft_mapped_to_init_lidar \
  --output_dir src/GS-SDF/data/my_data_0621_object_colmap \
  --fx 863.4241 --fy 863.4171 --cx 640.6808 --cy 518.3392 \
  --width 1280 --height 1024 \
  --k1=-0.1080 --k2=0.1050 --p1=-0.00012872 --p2=0.000057923 \
  --skip_point
```

`--skip_point` 可确保转换脚本使用相同帧索引导出图像和点云。

转换后会得到与 `images/`、`depths/` 平行的 `masks/` 文件夹。`src/GS-SDF/config/colmap/colmap_example.yaml` 已加入：

```yaml
mask_path: "masks"
```

GS-SDF 训练时会自动读取逐帧 mask，并把 RGB、DSSIM、normal 正则和背景 alpha 抑制限制到 mask 约束下。
