# 表面同源关系导出：独立入口 v1

## 本次边界

- 仅新增模块、入口、测试、本文档，不修改现有 `pycompod/__init__.py`、
  `complex.py`、`vg.py`、`run_psdr_compod.py` 或 `ply2off.py`。
- 现有 GUI 的 SCP/OBJ 模式仍调用旧入口；本次不切换其进程、配置或运行目录。
- 不改变 BSP、平面检测、图割、树简化、表面几何和后续高度分区规则。
- **已有 GUI 跑批不会自动多出同源 JSON**。等批次结束后再显式接入新入口。

## 引用和使用

本仓库仍直接使用自己的 `pycompod` 源码；新模块不加入包的自动导入。

```powershell
# 示例输出目录必须与正在跑批的目录分开；所有输出文件必须尚不存在。
.\windows_compat\.venv\Scripts\python.exe -B -X utf8 `
  .\windows_compat\run_psdr_compod_provenance.py `
  --npz <psdr_refined.npz> --center-json <center.json> `
  --out <新目录>\surface.ply
```

新增 `surface.provenance.json`；可用 `--provenance-out` 指定其他新文件。
`--keep-classes`、`--reg-area`、`--no-simplify`、`--save-cells` 与旧入口语义相同。
新入口拒绝覆盖输入、表面、sidecar、体积网格、平移临时文件和需要生成的 classes 清理副本。
清理 classes 的 NPZ 副本保存在新输出旁，不写回源 NPZ 所在目录。
`--save-cells` 在新入口生成 `<输出stem>_volume_mesh.ply`，避免复用固定体积文件名；
同源 sidecar 只对应表面 PLY，不对应体积网格。

**继承的已知限制：** 当前旧库在保留非空 `classes` 时，法向标签路径会因
`point_class_weights` 未初始化而失败；新旧入口均已复现。默认入口会清空副本中的
`classes`，不走此失败分支。此次不修复旧标签逻辑，也不把该可选分支算作成功支持；
空 `classes` 搭配 `--keep-classes` 已验证正常。

库接口也可以独立使用，必须针对**同一次重建、同一个仍存活的 cc 对象**：

```python
from pycompod.surface_provenance import write_surface_provenance, load_surface_provenance

cc.save_surface(out_file=surface_path, triangulate=False)
# 如果调用了 shift_ascii_ply(surface_path, center)，必须传同一个原始 center。
write_surface_provenance(cc, surface_path, coordinate_offset=center, source_npz=npz_path)
metadata = load_surface_provenance(surface_path)
```

没有平移时省略 `coordinate_offset`。目前只接受原始 python/python_exact 后端的
ASCII PLY；不对三角化、`save_simplified_surface`、外部重排或旧 OFF 猜测面片谱系。
`save_surface` 之前的现有树简化仍支持，与单独的简化表面导出不同。

## 输出契约

| 字段 | 含义 |
| --- | --- |
| `schema` | `compod-surface-provenance-v1` |
| `mesh_sha256` | 实际平移后 PLY 的完整字节哈希；消费者必须校验 |
| `vertex_count` / `face_count` | 与配套 PLY 完全一致 |
| `face_plane_ids[i]` | PLY 第 i 个面（零起始）的 `supporting_plane_id` |
| `face_graph_edges[i]` | 生成该面的两个体单元编号；仅用于追溯，不是表面面片邻接图 |
| `planes[id].coefficients` | 网格坐标系中的 `[a,b,c,d]`，方程为 `a*x+b*y+c*z+d=0` |
| `planes[id].source_group_ids` | 输入 NPZ 的原始平面组行号列表；可能多对一 |
| `planes[id].source_group_status` | `exact`、`bounding_box` 或明确不可追溯的原因 |
| `coordinate_offset_from_compod` | 实际从中心化坐标加回的平移量 |
| `input_npz` | 实际进入 VertexGroup 的 NPZ 文件名与哈希；清理 classes 时指向副本 |

`supporting_plane_id` 是 COMPOD 预处理后的输入平面编号，不是 NPZ 原始行号、
不是切分后的 `split_id`，更不是建筑构件语义标签。按
`merged_plane_from_input_planes` 和 `plane_order` 追溯原始组；若 VertexGroup
启用了不能完整追溯原始组的平面聚类，保留内部支撑 ID，但原始组明确为不可用。
若关闭重复平面合并，则每个内部平面只追溯自身原始组，不能将库内等参数评估表
误当成实际发生的多对一合并。
负数 ID 保留其数值，标为 `bounding_box`，不混同于检测平面。

编号仅在**单次重建**中有效；不能跨 SmID、跨输入或跨运行直接比较整数编号。

## 保证对应关系的方式

不靠法向相近或平面拟合推断源 ID。在旧 `save_surface` 完成后，以它同样的
体单元界面遍历顺序读取真实 `supporting_plane_id`，并对每个 PLY 面的顶点集合
和对应界面做精确的序列化坐标比对。数量、面序或坐标不符立即失败，不生成 sidecar。
读取/校验前后再次计算 PLY 哈希，拒绝校验过程中变化的文件。

该方法不复制和修改表面提取算法；依赖当前后端的面序契约。未来换提取后端或
修改导出顺序时必须更新契约和回归，不允许直接去掉对应校验。

## 后续 PLY → OFF 与高度分区

**本次没有接入这些阶段。** 旧转换器只返回 V/F，会丢失这份额外信息。

后续必须显式维护 `OFF面 -> PLY父面 -> 支撑平面`：

- 一个 PLY 面拆成多个三角形：所有子面继承同一个父面与支撑 ID。
- 丢弃面：同步去掉映射，记录原因；不能仅按输出面数截断或 zip。
- 重排、合面后：更新映射和 OFF 哈希，不能复用旧 PLY 哈希。
- 同源片按同向、真实共边的连通分量恢复，保留洞和断开部分；不能凭 XY 接近跨墙连接。
- 恢复应早于逐片进深、细缝和小面积处理；同一倾斜平面也不等于处处同高。
- 不同支撑平面仍可依现有坡结构/业务高差规则归并，但应保留来源 ID 集合与归并原因。

## 数值与参数记录

| 位置/项 | 当前值与来源 | 作用及边界 |
| --- | --- | --- |
| `surface_provenance.LEGACY_PLY_DECIMAL_DIGITS` | 6，继承现有 PLY 写出及平移小数位 | 复现“先写中心化点、再加 center 写点”的两次舍入；不是拟合容差 |
| `run_psdr_compod_provenance --reg-area` | 0.8，沿用旧入口 | 图割面积正则权重，未新增标定；本次未改变 |
| `--no-simplify` | 默认不设置，沿用旧入口 | 仍使用现有树简化；不是按源平面合并表面 |
| 平面平移公式 | `d_mesh = d_compod - n·center` | 只变换坐标表达，不修改平面、归并或高度规则 |
| SHA256 分块读取 | 1 MiB | 仅 IO 内存块大小，不影响几何或身份判定 |

上述实际值与运行元数据通过 sidecar、入口命令与文件哈希追溯。

## 验证

```powershell
.\windows_compat\.venv\Scripts\python.exe -B -X utf8 `
  .\windows_compat\test_surface_provenance.py -v
.\windows_compat\.venv\Scripts\python.exe -B -X utf8 `
  .\windows_compat\test_surface_provenance_integration.py -v
```

覆盖 BSP 多片同源、内部界面排除、预处理编号回溯、负数包围盒编号、平移与两次
舍入、旧文件不变、拒绝覆盖、面序错位、旧 sidecar、输入哈希与坏元数据。
测试只在 `%TEMP%` 生成小型合成文件。

集成测试使用 8 个输入平面组的立方体：内部切平面实际触发 BSP 分片，重复屋面组
验证预处理谱系；同时比较新旧完整入口的 PLY 字节是否一致，并确认输入 NPZ 只读。
导出专项测试用解析立方体 occupancy，完整入口对照仍执行真实法向图割；二者不混淆。
固定夹具为每组 5×5 点、共 200 点，无随机采样。解析立方体导出 10 个面片；
其中 4 个源平面各输出 2 片，重复屋面组的原始编号确认为 `[5, 7]`。
新旧完整入口分别验证默认树简化和 `--no-simplify`；默认分支的表面、体积网格
均逐字节一致。入口覆盖拒绝、输入路径冲突和 classes 副本位置另有固定回归。
测试子进程的 BLAS/OpenMP 线程数限制为 1，每个入口最长等待 60 秒，仅为测试资源预算，
不写入产品默认参数，也不改变正在运行的 GUI 环境。

### 调试记录（2026-11-12）

- 初版夹具每组只有四个角点，法向标签使用严格面内点而未收集到有效投票，
  旧入口自身即输出 0 个界面；内部切平面在立方体内部也没有支撑点。
  修正合成数据的面内覆盖，不改产品参数或挑选有利图割权重。
- 附加 `--keep-classes` 非空检查发现上述旧库缺陷；保留为新旧一致失败的回归，
  不是忽略异常后继续生成不完整 sidecar。
- 最终 30 项契约测试、15 项集成/入口测试通过；其中 1 项验证非空 `classes`
  保留分支的新旧一致失败，不代表该分支已修复或能够成功重建。
- 最终执行结果与命令输出保存在本地 `%TEMP%\compod_provenance_*.log`；
  不上传数据、测试日志或源码，不覆盖/删除旧失败日志。

真实业务样本、D 盘转换映射和高度分区效果尚需独立验收；契约测试通过不表示
最终屋面语义或 GUI 成果已验收。
