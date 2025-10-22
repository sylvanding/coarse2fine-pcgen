# MONAI Generative Models Cursor Rules 使用指南

## 📚 已创建的Cursor Rules

为 `GenerativeModels/` 文件夹和 `monai_diffusion/` 目录创建了三个专门的Cursor Rules：

### 1. `monai-generative.mdc` - 完整API参考
**路径**: `.cursor/rules/monai-generative.mdc`

**内容概览**:
- ✅ MONAI Generative库简介和核心功能
- ✅ 网络模块详解 (DiffusionModelUNet, AutoencoderKL, VQVAE等)
- ✅ 调度器详解 (DDPMScheduler, DDIMScheduler, PNDMScheduler)
- ✅ 推理器详解 (DiffusionInferer, LatentDiffusionInferer)
- ✅ 损失函数和评估指标
- ✅ 三个完整使用场景示例:
  - 场景1: 3D体素无条件生成
  - 场景2: 条件生成（类别引导）
  - 场景3: 潜在扩散模型（节省显存）
- ✅ 关键技巧和最佳实践 (显存优化、学习率调度、EMA权重等)
- ✅ 常见问题和解决方案
- ✅ 参考资料链接

**适用场景**: 需要查阅MONAI Generative API文档、了解模型架构、学习使用方法时。

### 2. `monai-integration.mdc` - 项目集成规范
**路径**: `.cursor/rules/monai-integration.mdc`  
**自动应用**: 在 `monai_diffusion/**/*.py` 文件中自动激活

**内容概览**:
- ✅ 标准路径设置和导入模式
- ✅ VoxelDiffusionPipeline完整实现（点云→体素→扩散→采样）
- ✅ 训练脚本示例 (`train_voxel_diffusion()`)
- ✅ 配置文件规范 (YAML配置)
- ✅ 日志和监控规范 (TensorBoard, WandB)
- ✅ 错误处理和日志规范
- ✅ 最佳实践总结（推荐做法 vs 避免做法）
- ✅ 环境依赖检查
- ✅ 参考项目脚本

**适用场景**: 在 `monai_diffusion/` 文件夹下编写代码时，自动提供最佳实践建议。

### 3. `monai-quick-reference.mdc` - 快速参考
**路径**: `.cursor/rules/monai-quick-reference.mdc`

**内容概览**:
- ✅ 最小训练脚本模板（可直接复制）
- ✅ 最小条件生成脚本模板
- ✅ 四种常用配置组合:
  - 快速原型（小模型）
  - 标准训练（中等模型）
  - 高质量生成（大模型）
  - 潜在扩散（节省显存）
- ✅ 常用数据预处理函数
- ✅ 常用工具函数 (参数统计、显存估算、检查点保存等)
- ✅ 可视化工具
- ✅ 体素转点云（三种采样方法）
- ✅ 调试技巧
- ✅ 性能优化技巧
- ✅ 评估指标计算
- ✅ 完整训练模板（包含所有最佳实践）

**适用场景**: 需要快速找到代码示例、复制粘贴常用代码片段时。

## 🚀 如何使用

### 在Cursor中使用Rules

#### 方法1: 手动引用
在对话中输入 `@` 符号，然后选择对应的rule：
```
@monai-generative 如何使用DiffusionModelUNet？
@monai-integration 在monai_diffusion下如何正确导入库？
@monai-quick-reference 给我一个最小训练脚本
```

#### 方法2: 自动应用（monai-integration）
在 `monai_diffusion/` 文件夹下的任何 `.py` 文件中，`monai-integration` rule会自动应用，AI会自动参考这些最佳实践。

#### 方法3: 使用fetch_rules工具
AI可以通过描述自动获取对应的rule：
```
User: 我想在monai_diffusion下训练扩散模型
AI: [自动fetch monai-integration和monai-generative rules]
```

### 典型工作流

#### 场景A: 开始新项目
1. 阅读 `monai_diffusion/readme.md` 了解整体架构
2. 引用 `@monai-quick-reference` 获取最小训练脚本模板
3. 引用 `@monai-integration` 了解如何集成点云数据流

#### 场景B: 实现特定功能
```
User: @monai-generative 我想实现条件生成，如何使用cross-attention?
AI: [提供场景2的详细代码和解释]
```

#### 场景C: 调试问题
```
User: 训练时显存不足，怎么办？
AI: [参考monai-generative的"常见问题"部分提供解决方案]
```

#### 场景D: 优化性能
```
User: @monai-quick-reference 如何加速训练？
AI: [提供性能优化技巧：混合精度、数据加载优化等]
```

## 📖 Rule内容速查表

| Rule | 主要内容 | 何时使用 |
|------|---------|---------|
| **monai-generative** | 完整API文档、使用示例、最佳实践 | 学习MONAI Generative、查阅API、了解最佳实践 |
| **monai-integration** | 项目集成、Pipeline实现、配置管理 | 在monai_diffusion文件夹编写代码 |
| **monai-quick-reference** | 代码模板、工具函数、快速示例 | 需要快速上手、复制代码片段 |

## 🔍 核心主题索引

### 模型架构
- DiffusionModelUNet: `@monai-generative` → "网络模块" → "扩散模型网络"
- AutoencoderKL: `@monai-generative` → "网络模块" → "自编码器网络"
- 配置选择: `@monai-quick-reference` → "常用配置组合"

### 调度器
- DDPM vs DDIM vs PNDM: `@monai-generative` → "调度器模块"
- 可视化噪声调度: `@monai-quick-reference` → "调试技巧"

### 训练
- 基础训练循环: `@monai-generative` → "场景1: 3D体素无条件生成"
- 完整训练脚本: `@monai-quick-reference` → "完整训练模板"
- 训练优化: `@monai-generative` → "关键技巧和最佳实践"

### 条件生成
- 类别条件: `@monai-generative` → "场景2: 条件生成"
- 快速模板: `@monai-quick-reference` → "最小条件生成脚本"

### 潜在扩散
- 完整流程: `@monai-generative` → "场景3: 潜在扩散模型"
- 配置: `@monai-quick-reference` → "配置4: 潜在扩散"

### 数据处理
- 点云→体素: `@monai-integration` → "数据流集成"
- 体素→点云: `@monai-quick-reference` → "体素转点云"
- 数据增强: `@monai-quick-reference` → "体素数据增强"

### 工具和调试
- 参数统计: `@monai-quick-reference` → "计算模型参数量"
- 显存估算: `@monai-quick-reference` → "估算显存占用"
- 梯度检查: `@monai-quick-reference` → "检查梯度"
- 可视化: `@monai-quick-reference` → "可视化体素"

### 问题解决
- 显存不足: `@monai-generative` → "常见问题" → "问题1"
- 训练不稳定: `@monai-generative` → "常见问题" → "问题2"
- 生成质量差: `@monai-generative` → "常见问题" → "问题3"
- 采样速度慢: `@monai-generative` → "常见问题" → "问题4"

## 📦 更新的文件

### 新创建的Rules
- `.cursor/rules/monai-generative.mdc` (完整API参考)
- `.cursor/rules/monai-integration.mdc` (项目集成规范)
- `.cursor/rules/monai-quick-reference.mdc` (快速参考)

### 更新的文档
- `monai_diffusion/readme.md` (大幅扩展，包含完整使用指南)

## 💡 使用建议

### 对于初学者
1. 先阅读 `monai_diffusion/readme.md` 了解整体架构
2. 使用 `@monai-quick-reference` 获取最小训练脚本，快速上手
3. 遇到问题时查阅 `@monai-generative` 的"常见问题"部分

### 对于开发者
1. 在 `monai_diffusion/` 下编写代码时，`monai-integration` 会自动提供最佳实践
2. 需要实现特定功能时，参考 `@monai-generative` 的三个场景示例
3. 使用 `@monai-quick-reference` 快速查找工具函数和代码片段

### 对于高级用户
1. 参考 `@monai-generative` 的"关键技巧和最佳实践"优化模型
2. 使用 `@monai-integration` 的Pipeline模式组织代码
3. 根据 `@monai-quick-reference` 的配置组合选择最适合的模型规模

## 🎯 下一步行动

现在你可以：

1. **立即开始**: 复制 `@monai-quick-reference` 中的"最小训练脚本"开始实验
2. **深入学习**: 阅读 `@monai-generative` 了解各个模块的详细API
3. **集成项目**: 参考 `@monai-integration` 将扩散模型集成到现有点云生成流程
4. **优化模型**: 使用 `@monai-generative` 的最佳实践提升模型性能

## 📞 获取帮助

在Cursor中，随时可以：
```
@monai-generative 我想了解[具体问题]
@monai-integration 如何在项目中[具体操作]
@monai-quick-reference 给我一个[具体功能]的代码示例
```

AI会根据这些rules提供精准的、符合项目规范的解答。

---

**文档版本**: v1.0  
**创建日期**: 2025-10-21  
**适用范围**: `GenerativeModels/` 和 `monai_diffusion/` 目录

