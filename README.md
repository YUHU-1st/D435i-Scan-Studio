# D435i Scan Studio

面向 Intel RealSense D435i 的免费、开源物体扫描工作站。软件采用“实时引导 + 离线高质量重建”两阶段流程：扫描时优先保证操作者知道哪里没扫到、哪里数据差；结束后再利用保存的无损深度关键帧优化轨迹并重新融合，而不是把最终精度押在实时预览上。

## 先说结论：现成免费方案能不能直接满足？

没有发现一款现成免费软件能够同时完成以下全部要求：D435i 实时采集、物体级 ROI、已扫/未扫视角、逐像素与模型区域置信度伪彩、轨迹优化、表面去噪、OBJ，以及可靠的 STEP 输出。

可复用的免费组件如下：

| 组件 | 能做什么 | 仍缺少什么 |
|---|---|---|
| [Intel librealsense](https://github.com/realsenseai/librealsense) / RealSense Viewer | 相机驱动、标定检查、录制、深度后处理；仓库中也有 KinFu 示例 | 不是完整的物体扫描工作流，没有覆盖率、模型置信度和 STEP |
| [Open3D](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/index.html) | RGB-D 里程计、TSDF、位姿图、ICP、网格处理和 GUI | 官方实时 Dense SLAM 示例没有重定位，并明确提示准确度和鲁棒性有限 |
| [RTAB-Map](https://github.com/introlab/rtabmap) | 回环检测、图优化、RGB-D 建图、网格导出 | 更偏场景/机器人建图，不是物体扫描界面，也不生成工程 CAD STEP |
| [Handy3DScanner](https://github.com/state-of-the-art/Handy3DScanner) | D415/D435(i) 采集、点云预览、PCD/GLB | 仓库已于 2024 年归档；没有本项目所需置信度、覆盖引导和 STEP |
| [forest_3d_scanner](https://github.com/Forestjylee/forest_3d_scanner) | RealSense 在线/离线重建、位姿图与 PLY | 依赖 2020 年的 Python 3.6 栈；没有覆盖/置信度和 OBJ/STEP 全链路 |
| [CloudCompare](https://www.cloudcompare.org/) / [MeshLab](https://www.meshlab.net/) | 很强的手工点云和网格清理 | 不是 D435i 一体化采集与覆盖引导程序 |
| [FreeCAD](https://www.freecad.org/) | 免费 OpenCascade CAD 内核，可导出 STEP | 网格转出的通常是三角分面 B-Rep，不会自动变成干净的平面、圆柱和 NURBS CAD 特征 |

因此，本仓库把 RealSense、Open3D 和 FreeCAD 串成了一套可运行的软件，并补上覆盖与置信度逻辑。

## 功能

- D435i 的 848×480@30 FPS 深度与 1280×720@30 FPS 彩色同步，并把 RGB 对齐到深度网格。
- High Accuracy 预设、视差域边缘保持空间滤波；默认关闭移动相机容易产生拖影的时间滤波和启发式补洞。
- Open3D frame-to-model 稠密跟踪与稀疏体素 TSDF 实时预览。
- 彩色画面、深度伪彩、逐像素质量伪彩、当前相机与 ROI 三维显示。
- 24×7 方位/俯仰视角覆盖图：红色未扫、绿色已扫、白框当前视角。
- 模型置信度伪彩：结合观测次数、输入深度质量和观察方向多样性。
- 跟踪 fitness、RMSE、采集 FPS、关键帧数、覆盖率实时提示；异常运动帧不参与融合。
- 原始 16-bit PNG 深度与高质量 JPEG 彩色关键帧留存，可重复调整参数重建。
- 离线 ROI 点云去离群、鲁棒 point-to-plane ICP、候选回环、位姿图优化和高分辨率 TSDF 重融合。
- 小连通域去噪、重复/退化/非流形元素清理、二次误差简化与 Taubin 低收缩平滑。
- `model.obj`、`model.ply`、`model_confidence.ply`、`view_coverage.png` 和质量 `report.json`。
- 安装 FreeCAD 后自动生成 `model.step`（分面 B-Rep）。
- 可读取 RealSense `.bag`，也可从已保存会话重新建模。

## 安装（Windows 10/11）

要求：64 位 Windows、USB 3.x、D435i、[uv](https://docs.astral.sh/uv/)。Open3D 0.19 的正式 Windows wheel 支持 Python 3.12，所以项目固定使用 Python 3.12。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

如需 STEP，再安装免费的 [FreeCAD](https://www.freecad.org/downloads.php)。程序会自动搜索常见安装路径；也可以在 `config.example.yml` 的 `export.freecad_cmd` 填入 `FreeCADCmd.exe` 的绝对路径。

检查相机：

```powershell
.\run.ps1 --list-devices
```

启动界面：

```powershell
.\run.ps1
```

读取已录制的 BAG：

```powershell
.\run.ps1 --bag D:\captures\object.bag
```

重新处理以前的会话：

```powershell
.\run.ps1 --reconstruct .\output\scan_20260909_120000
```

运行测试：

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD ".uv-cache"
uv run pytest
```

## 推荐扫描流程

1. 先在 RealSense Viewer 检查深度图。使用 USB 3、848×480@30、High Accuracy，确认深度比例正常；相机经历跌落、温漂或长期使用后，应先做官方 On-Chip/Tare 校准。D435i 的 IMU 也可用 SDK 附带工具另行校准。
2. 让物体完全静止，把相机对准物体中心，保持物体位于白色十字附近。当前版本的主模式是“相机绕静止物体”；转台上物体运动属于实验用法。
3. 按实际物体外形设置 ROI。ROI 应比物体各方向大约多 1–3 cm，但不要把桌面、墙面等大面积背景包进去。
4. 点击“开始扫描”。相机距离通常先从 0.3–0.8 m 尝试；缓慢、连续地绕物体移动，避免单帧大于界面门限的跳动。
5. 查看“未扫/已扫”页，覆盖水平一圈，并增加上方和下方两圈。对凹槽、遮挡处和掠射角区域补扫；红色格并不代表某个确定的物体三角面，而代表尚未采集的观察方向。
6. 跟踪 RMSE 持续升高或状态提示拒绝时，退回刚才的位置，重新看到已扫描的、有几何或纹理特征的区域，再继续前进。
7. 点击“结束并建模”。输出位于 `output/scan_日期_时间/`。先看 `report.json` 的覆盖率、水密性、自交和 STEP 状态，再使用模型。

### 影响精度最大的实物条件

- 透明、镜面、黑亮、重复纹理和细毛发不是 D435i 主动双目深度的理想目标。可在允许时使用可清洗的显影喷剂；不要指望后处理恢复相机从未测到的表面。
- 曝光和投射器功率应稳定。低纹理物体需要投射器产生纹理，但阳光或其他红外源可能干扰。
- 相机过远时深度噪声会明显增加；“体素 2.5 mm”只是重建采样间距，不代表测量误差就是 2.5 mm。
- 平整度来自正确标定、多视角 TSDF 平均、离群剔除与低收缩平滑。任何软件都不能在未知材质、距离、温度和标定状态下无条件保证平面度。做尺寸验收时，必须用已知平板、量块或标定件测量 RMS/峰谷误差。

## 置信度的含义

D435i 不提供经过标定的逐像素概率置信度流，因此界面显示的是“操作质量分数”，而不是毫米级不确定度：

- 像素质量：有效深度、距离、局部深度跳变、相邻关键帧一致性的加权分数。
- 模型置信度：该表面体素的观测次数、像素质量均值、观察方向数量。
- 视角覆盖：相机相对 ROI 中心的方位/俯仰格是否以良好跟踪质量访问过。

软件不会把“模型里不存在的背面”伪装成低置信度三角面；这类缺失由红色视角格和低覆盖率提醒。

## OBJ 与 STEP 的重要区别

`model.obj` 是扫描结果的原生表达：三角网格。自动 STEP 输出的流程是：先把模型降到 `step_max_triangles`，再由 FreeCAD/OpenCascade 把每个三角面变成 B-Rep 面，并在闭合时尝试组成 Solid。因此：

- 它是合法的 STEP 交换文件，但曲面仍由许多平面三角片组成。
- 网格不水密时，STEP 通常只能成为 Shell。
- 它适合归档、查看、装配占位和部分下游交换，不等于参数化、可编辑的机械 CAD。
- 如果目标是机械件逆向设计，需要再做平面、圆柱、圆锥、孔和自由曲面的分割/拟合及公差约束。这需要针对零件类型定制，无法从任意有机物体扫描全自动可靠完成。

## 输出目录

```text
scan_YYYYMMDD_HHMMSS/
├── color/                 # 关键帧 RGB JPEG
├── depth/                 # 关键帧 16-bit 原始深度 PNG
├── settings.yml           # 本次参数
├── session.json           # 内参、深度比例、轨迹、质量
├── model_raw.ply          # 清理前 TSDF 网格
├── model.ply              # 最终彩色网格
├── model.obj              # 最终 OBJ
├── model_confidence.ply   # 置信度顶点色
├── view_coverage.png      # 视角覆盖图
├── model_for_step.stl     # STEP 转换中间网格
├── model.step             # FreeCAD 可用时生成
└── report.json            # 质量与导出报告
```

OBJ/PLY 顶点坐标按米保存（OBJ 格式本身不声明单位）；STEP 在转换前显式放大 1000 倍，因此 FreeCAD/OpenCascade 中的单位为毫米。

## 参数建议

- 小物体、显卡/CPU和内存充足：`voxel_size_m: 0.0015–0.0025`。
- 30–80 cm 普通物体：默认 `0.0025`。
- 大物体或内存不足：`0.004–0.008`。
- 表面细节被抹掉：减少 `smooth_iterations`，不要盲目减小体素。
- 孤立噪点多：缩紧 ROI、改善深度画面和曝光，再调整滤波；不要只增加平滑次数。
- 移动相机时保持 `enable_temporal_filter: false`。仅在相机和物体均静止、做多帧平均测试时考虑开启。

## 已知边界

- 在线 Open3D Dense SLAM 没有完整重定位。软件会拒绝明显坏帧，离线也会优化轨迹，但彻底丢失后仍建议重扫。
- D435i IMU 已有采集与标定工具，但当前重建未做紧耦合视觉惯性优化；对小型物体，几何 frame-to-model 通常比未经严格时延/外参建模的简单 IMU 拼接更可靠。
- 当前 ROI 是轴对齐盒；强烈凹陷和内部不可见表面仍需要改变视角或多次装夹后做额外配准。
- 自动 STEP 是分面 B-Rep。解析曲面/NURBS 逆向工程是下一层、零件类型相关的功能。

## 开发结构

- `camera.py`：RealSense 采集、对齐、滤波。
- `realtime.py`：在线 TSDF 和跟踪门限。
- `coverage.py` / `quality.py`：视角覆盖和置信度。
- `session.py`：可恢复关键帧会话。
- `offline.py`：ICP、位姿图、TSDF、去噪、平滑、导出与报告。
- `step_export.py` / `freecad_mesh_to_step.py`：FreeCADCmd 桥接。
- `app.py`：Open3D 原生 GUI 和三维视窗。

本项目采用 MIT 许可证。
