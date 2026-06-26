# Mask-enhanced object reconstruction pipeline

本文档说明 `mask_enhanced` 分支相对原始流程做了什么改进，以及如何从原始 ROS bag 生成只约束目标物体的 GS-SDF 输入数据并启动训练。

整体流程从：

```text
原始 bag -> rosbag_to_colmap -> GS-SDF
```

变为：

```text
原始 bag -> proxy_model 目标分割/点云过滤/object bag
        -> rosbag_to_colmap 导出 images/depths/masks
        -> GS-SDF 使用 mask 约束训练和剪枝
```

## 1. mask 分割与 object bag 生成改进

相关代码：

- `src/proxy_model/scripts/build_object_bag.py`
- `src/proxy_model/src/proxy_model/pipeline.py`
- `src/proxy_model/src/proxy_model/segmenter.py`
- `src/proxy_model/src/proxy_model/geometry.py`

### 改进内容

`proxy_model` 会读取原始 bag，生成一个新的 object bag。新的 bag 中：

1. 原图像话题仍然使用原话题名和原消息类型。
2. 图像中非目标 mask 区域会被置黑，只保留目标物体 RGB。
3. 点云话题仍然使用原话题名和原消息类型。
4. 点云会被投影到图像，只有落在目标 mask 内的点被保留。
5. 新增逐帧 mask 话题，默认：

   ```text
   /proxy_model/object_mask
   ```

   类型为 `sensor_msgs/Image`，编码为 `mono8`，前景为 255，背景为 0。

6. 其他不涉及的话题会原样复制到输出 bag。
7. 输出缓存中会保存：

   ```text
   .proxy_model_cache/<scene>/
   ├── frames/      # 原始抽帧
   ├── masks/       # 每帧二值 mask，000000.png ...
   ├── overlays/    # 可视化叠加图，便于检查 mask 质量
   └── summary.json # 统计信息
   ```

### 分割策略

分割不是每帧都调用大模型。当前设计是：

1. Qwen-VL 只在种子帧调用一次。

   它把中文目标描述解析成 SAM 可用的目标文本、实例模式和可选中心点。

2. SAM 在种子帧和周期关键帧调用。

   `sam_interval: 3` 表示每 3 帧调用一次 SAM；`sam_interval: 1` 表示每帧都调用 SAM。

3. 非关键帧使用光流传播 mask。

   使用 OpenCV Farneback 光流把上一帧 mask warp 到当前帧，降低 SAM 调用频率。

4. 后续 SAM 分割会受光流先验约束。

   参数：

   ```yaml
   sam_refine_margin_px: 12
   ```

   表示将光流传播得到的 mask 膨胀 12 像素，形成一个允许 SAM 结果出现的局部带状区域。SAM 输出后会与这个区域取交集，防止后续帧突然分到过大的背景区域。如果交集太小，则回退到光流传播结果。

### 点云过滤策略

第一层过滤：当前帧 mask 投影过滤。

对每个点云点执行：

```text
point_lidar -> camera frame -> image pixel
```

投影公式本质上是：

```text
T_camera_lidar = inv(T_world_camera) @ T_world_lidar
p_camera = T_camera_lidar * p_lidar
u = fx * x / z + cx
v = fy * y / z + cy
```

然后要求：

- 深度在 `[min_depth_m, max_depth_m]` 内；
- 像素在图像范围内；
- 像素落在当前帧 mask 前景内；
- 同一个局部像素栅格内，只保留接近最近深度的点，避免目标轮廓后面的背景点混入。

第二层过滤：多视角一致性。

参数：

```yaml
point_filter:
  multiview_window: 2
  multiview_min_views: 2
  multiview_ratio: 0.9
```

含义是：对当前帧前后各 `multiview_window` 帧做检查。一个点如果能被多个视角看到，则它投影到前景 mask 的比例必须大于等于 `multiview_ratio`，例如 0.9。这样可以减少某一帧 mask 偶然过大导致的背景点保留。

### 与之前使用方式的区别

之前：

```text
原始 bag 直接进入 rosbag_to_colmap
```

现在：

```text
先用 proxy_model 生成 object bag，再把 object bag 转为 COLMAP 格式
```

也就是说，`rosbag_to_colmap.py` 的输入 bag 不再是原始 bag，而是 `proxy_model` 输出的 object bag。

### 使用方式

准备环境：

```bash
cd /home/yc/proxy_modelling
source /opt/ros/noetic/setup.bash
source .venv/bin/activate
pip install -r src/proxy_model/requirements.txt
pip install -e /home/yc/SAM-AGENT/sam3_src
export DASHSCOPE_API_KEY="你的 DashScope API key"
```

编辑配置，例如：

```yaml
input_bag: ../../GS-SDF/data/my_bag/my_data_0625.bag
output_bag: ../../GS-SDF/data/my_bag/my_data_0625_object.bag

topics:
  image: /origin_img/compressed
  image_pose: /aft_mapped_to_init_cam
  pointcloud: /cloud_registered_body
  pointcloud_pose: /aft_mapped_to_init_lidar
  mask: /proxy_model/object_mask

segmentation:
  prompt: "分割出黑色行李箱"
  seed_frame: 0
  sam_interval: 3
  sam_refine_margin_px: 12

point_filter:
  multiview_window: 2
  multiview_min_views: 2
  multiview_ratio: 0.9
```

运行：

```bash
python src/proxy_model/scripts/build_object_bag.py \
  --config src/proxy_model/config/my_data.yaml
```

运行完成后，先检查：

```text
.proxy_model_cache/<scene>/overlays/
```

确认目标 mask 稳定后，再继续后续转换和训练。

如果只想把缓存中的 mask 复制到某个 COLMAP 数据目录，也可以使用：

```bash
python src/proxy_model/scripts/build_object_bag.py \
  --config src/proxy_model/config/my_data.yaml \
  --masks-dir src/GS-SDF/data/<scene_colmap>/masks
```

## 2. rosbag_to_colmap 转换改进

相关代码：

- `src/GS-SDF/scripts/rosbag_convert/rosbag_to_colmap.py`

### 改进内容

新增参数：

```bash
--mask_topic /proxy_model/object_mask
```

如果提供该参数，转换脚本会：

1. 从 object bag 中读取逐帧 mask 话题；
2. 按时间戳把 mask 与图像帧对齐；
3. 对 mask 执行与图像一致的去畸变/重映射；
4. 保存到与 `images/`、`depths/` 平行的目录：

   ```text
   output_dir/
   ├── images/
   ├── depths/
   ├── masks/
   └── sparse/0/
   ```

5. mask 文件名与 image 文件名保持一致，例如：

   ```text
   images/000000.png
   masks/000000.png
   ```

如果某一帧找不到 mask，脚本会使用全前景 mask，并打印 warning。正常使用时不应长期依赖这个 fallback。

### 与之前使用方式的区别

之前命令不需要 mask：

```bash
python src/GS-SDF/scripts/rosbag_convert/rosbag_to_colmap.py \
  --bag_path <raw.bag> \
  --image_topic /origin_img/compressed \
  --image_pose_topic /aft_mapped_to_init_cam \
  --point_topic /cloud_registered_body \
  --point_pose_topic /aft_mapped_to_init_lidar \
  --output_dir src/GS-SDF/data/<scene_colmap> \
  --fx ... --fy ... --cx ... --cy ... \
  --width ... --height ...
```

现在推荐：

```bash
python src/GS-SDF/scripts/rosbag_convert/rosbag_to_colmap.py \
  --bag_path src/GS-SDF/data/my_bag/data_0621_object_mask_enhanced/my_data_0621_object_mask.bag \
  --image_topic /origin_img/compressed \
  --image_pose_topic /aft_mapped_to_init_cam \
  --mask_topic /proxy_model/object_mask \
  --point_topic /cloud_registered_body \
  --point_pose_topic /aft_mapped_to_init_lidar \
  --output_dir src/GS-SDF/data/my_bag/data_0621_object_mask_enhanced \
  --fx 1294.2997611696601 \
  --fy 1293.8035067346466 \
  --cx 625.69717868846817 \
  --cy 499.69240629695406 \
  --width 1280 \
  --height 1024 \
  --k1 -0.08469119792190298 \
  --k2 0.13301102580164631 \
  --p1 0.00012196606307049965 \
  --p2 0.0031281992500534028 \
  --k3 0.034877906107274127 \
  --skip_point
```

主要区别：

- `--bag_path` 使用 object bag；
- 新增 `--mask_topic`；
- 输出目录中会多出 `masks/`；
- 推荐保留 `--skip_point`，使图像、点云、mask 的帧选择更一致。

## 3. GS-SDF mask 约束改进

相关代码：

- `src/GS-SDF/config/colmap/colmap_example.yaml`
- `src/GS-SDF/config/base.yaml`
- `src/GS-SDF/include/data_loader/data_parsers/base_parser.*`
- `src/GS-SDF/include/data_loader/data_parsers/colmap_parser.hpp`
- `src/GS-SDF/include/neural_mapping/neural_mapping.cpp`
- `src/GS-SDF/include/neural_gaussian/neural_gaussian.h`

### 数据读取改进

COLMAP 配置中新增：

```yaml
mask_path: "masks"
```

GS-SDF 的 dataparser 会读取：

```text
dataset_path/masks/*.png
```

并按照训练帧索引与 `images/` 对齐。mask 会被 resize 到训练尺度，例如：

```yaml
camera:
  scale: 0.5
```

如果 `masks/` 数量与 `images/` 数量不一致，会打印 warning 并禁用逐帧 mask。

### loss 改进

RGB loss 和 DSSIM loss 只在 mask 前景区域计算：

```cpp
color_loss = loss::rgb_loss(render_color, gt_color, mask);
dssim_loss = loss::dssim_loss(render_color, gt_color, mask);
```

normal regularization 也只在 mask 内计算：

```cpp
render_alpha = render_alpha * mask;
normal_error = (normal_residual * mask).sum() / mask.sum()
```

这样可以避免背景区域的黑色像素参与光度监督，降低 GS-SDF 在非目标区域拟合背景/空洞/错误 normal 的倾向。

### 背景 alpha 抑制

`base.yaml` 中新增：

```yaml
mask_alpha_weight: 0.05
```

它会惩罚非 mask 区域的渲染不透明度：

```text
background_alpha_loss = mean(render_alpha outside mask)
```

直观理解：如果某个非目标区域被高斯渲染出明显 alpha，就会被 loss 惩罚。

如果想关闭该项：

```yaml
mask_alpha_weight: 0.0
```

### mask-based Gaussian pruning

`base.yaml` 中新增：

```yaml
mask_prune_ratio: 0.9
```

训练过程中，GS-SDF 会周期性检查当前高斯中心在多个训练视角中的投影：

1. 把每个 Gaussian 的 3D 中心投影到若干训练图像；
2. 统计它在多少个可见视角中落入 mask；
3. 计算：

   ```text
   foreground_ratio = in_mask_views / visible_views
   ```

4. 如果某个 Gaussian 至少在 2 个视角可见，但 `foreground_ratio < mask_prune_ratio`，则剪枝。

这个策略比单帧硬删更稳，因为它使用多视角统计，能容忍少数帧 mask 抖动。

如果想关闭该项：

```yaml
mask_prune_ratio: 0.0
```

### 与之前使用方式的区别

之前的 COLMAP 配置没有 mask：

```yaml
color_path: "images"
depth_path: "depths"
```

现在需要增加：

```yaml
mask_path: "masks"
```

训练命令基本不变：

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash

rosrun neural_mapping neural_mapping_node train \
  src/GS-SDF/config/colmap/colmap_example.yaml \
  src/GS-SDF/data/my_data_0625_object_colmap
```

如果你的数据目录没有 `masks/`，GS-SDF 会退化为原来的无 mask 流程；但要得到目标物体约束效果，必须提供 `masks/` 并在配置中设置 `mask_path`。

## 4. 推荐完整命令顺序

### Step 1: 生成 object bag

```bash
cd /home/yc/proxy_modelling
source /opt/ros/noetic/setup.bash
source .venv/bin/activate

python src/proxy_model/scripts/build_object_bag.py \
  --config src/proxy_model/config/my_data.yaml
```

### Step 2: 检查 mask

查看：

```text
.proxy_model_cache/<scene>/overlays/
.proxy_model_cache/<scene>/summary.json
```

如果 mask 偏大，优先调整：

```yaml
sam_interval: 1       # 更频繁调用 SAM
sam_refine_margin_px: 8
mask_dilate_px: 0
flow_mask_dilate_px: 0
```

如果 mask 偏小，优先调整：

```yaml
sam_refine_margin_px: 16
mask_dilate_px: 2
```

### Step 3: object bag 转 COLMAP

```bash
python src/GS-SDF/scripts/rosbag_convert/rosbag_to_colmap.py \
  --bag_path src/GS-SDF/data/my_bag/my_data_0625_object.bag \
  --image_topic /origin_img/compressed \
  --image_pose_topic /aft_mapped_to_init_cam \
  --mask_topic /proxy_model/object_mask \
  --point_topic /cloud_registered_body \
  --point_pose_topic /aft_mapped_to_init_lidar \
  --output_dir src/GS-SDF/data/my_data_0625_object_colmap \
  --fx <fx> --fy <fy> --cx <cx> --cy <cy> \
  --width <width> --height <height> \
  --k1 <k1> --k2 <k2> --p1 <p1> --p2 <p2> --k3 <k3> \
  --skip_point
```

### Step 4: 确认 GS-SDF 配置

`src/GS-SDF/config/colmap/colmap_example.yaml` 中确认：

```yaml
dataset_type: 6
color_path: "images"
mask_path: "masks"
depth_path: "depths"
```

`src/GS-SDF/config/base.yaml` 中确认：

```yaml
mask_alpha_weight: 0.05
mask_prune_ratio: 0.9
```

### Step 5: 训练

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash

rosrun neural_mapping neural_mapping_node train \
  src/GS-SDF/config/colmap/colmap_example.yaml \
  src/GS-SDF/data/my_data_0625_object_colmap
```

## 5. `gpuAtomicAdd` 与 warp-level grouped reduction 的区别

这部分对应这次 CUDA 编译兼容修复，相关文件在：

```text
src/GS-SDF/submodules/gsplat_cpp/submodules/gsplat/gsplat/cuda/csrc/
```

原始 gsplat 代码使用：

```cpp
cg::labeled_partition(...)
warpSum(...)
```

它的思想是：同一个 warp 内，如果多个线程都在给同一个 Gaussian 或同一个 camera 累加梯度，先在 warp 内把这些线程按 label 分组，再组内求和，最后由一个线程执行一次 `atomicAdd`。

优点：

- atomic 操作次数少；
- GPU 上通常更快；
- 对高并发同目标梯度累加更友好。

缺点：

- 依赖 `cooperative_groups::labeled_partition`；
- 在当前 CUDA/nvcc/架构组合下该 API 不可用，所以会直接编译失败。

当前修复改为：

```cpp
gpuAtomicAdd(...)
```

也就是每个线程直接把自己的梯度贡献 atomic 加到全局梯度张量里。

优点：

- CUDA 兼容性更好；
- 代码路径更简单；
- 不依赖 `labeled_partition`。

缺点：

- atomic 操作次数更多；
- backward kernel 可能略慢；
- 如果很多线程同时写同一个 Gaussian 的梯度，atomic contention 会更高。

### 对结果的影响

对数学目标来说，两者都在做同一件事：

```text
最终梯度 = 所有线程贡献的梯度之和
```

所以理论上训练目标不变，渲染前向结果不变，最终重建结果不应该发生系统性变化。

可能存在的细微差异主要来自浮点加法顺序：

- warp-level grouped reduction 是先组内求和，再 atomic；
- `gpuAtomicAdd` 是更多线程以不确定顺序直接 atomic。

浮点加法不是严格结合律，因此最后几位小数可能不同。这种差异通常表现为训练过程中的微小随机性，不是算法语义变化。

实际影响可以概括为：

- 准确性：基本不变；
- 可复现性：可能有极小数值差异；
- 速度：backward 可能略慢；
- 编译稳定性：明显提升。

如果后续换到支持 `labeled_partition` 的 CUDA/编译环境，也可以再恢复原始 warp-level reduction 来追求更高性能。
