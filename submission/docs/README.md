# DWTS数据预处理系统

将"与星共舞"（Dancing with the Stars）比赛数据从宽表格式转换为适合时序分析的长表格式。

## 功能特性

- **宽表转长表**：将每个选手一行的宽表数据展开为每个选手每周一行的面板数据
- **智能数据清洗**：自动识别并删除选手被淘汰后的无效数据（N/A、0或空值）
- **特征工程**：
  - 计算裁判平均分（Judge_Avg_Score）
  - 行业编码（Industry_Code）
  - Z-score标准化（Score_Scaled）
- **时序数据集划分**：按赛季顺序划分训练集（前80%）和测试集（后20%）
- **数据验证和可视化**：生成分数分布直方图

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python main.py
```

这将使用默认的输入文件（`2026 MCM Problem C Data.csv`）并生成输出文件（`Processed_DWTS_Long_Format.csv`）。

### 自定义输入输出

```bash
python main.py --input your_input.csv --output your_output.csv
```

### 参数说明

- `--input`: 输入CSV文件路径（默认: `2026 MCM Problem C Data.csv`）
- `--output`: 输出CSV文件路径（默认: `Processed_DWTS_Long_Format.csv`）

## 输出格式

输出CSV文件包含以下字段：

| 字段 | 说明 |
|------|------|
| Season | 赛季编号 |
| Week | 周次（1-11） |
| Name | 选手姓名 |
| Age | 年龄 |
| Industry_Code | 行业数字编码 |
| Judge_Avg_Score | 裁判平均分 |
| Score_Scaled | Z-score标准化分数 |
| Placement | 最终排名 |

## 数据处理流程

1. **加载数据**：读取原始CSV文件
2. **宽表转长表**：将宽表格式转换为长表格式
3. **数据清洗**：删除选手被淘汰后的无效数据
4. **特征工程**：计算派生特征
5. **数据集划分**：按时间顺序划分训练集和测试集
6. **保存输出**：生成处理后的CSV文件
7. **验证和可视化**：生成统计报告和分数分布直方图

## 测试

运行所有测试：

```bash
pytest tests/ -v
```

运行特定测试：

```bash
# 单元测试
pytest tests/test_data_loader.py -v

# 属性测试
pytest tests/test_properties.py -v

# 集成测试
pytest tests/test_integration.py -v
```

## 项目结构

```
.
├── main.py                          # 主程序入口
├── requirements.txt                 # 依赖包
├── README.md                        # 项目文档
├── src/
│   └── preprocessing_pipeline.py    # 核心处理模块
├── tests/
│   ├── test_data_loader.py         # DataLoader单元测试
│   ├── test_feature_engineer.py    # FeatureEngineer单元测试
│   ├── test_properties.py          # 基于属性的测试
│   └── test_integration.py         # 端到端集成测试
└── .kiro/
    └── specs/
        └── dwts-data-preprocessing/ # 规格文档
```

## 技术栈

- **Python 3.8+**
- **pandas**: 数据处理
- **matplotlib**: 数据可视化
- **hypothesis**: 基于属性的测试
- **pytest**: 测试框架

## 注意事项

- 输入数据必须包含必需的列：`season`, `celebrity_name`, `celebrity_age_during_season`, `celebrity_industry`, `placement`, 以及 `week1` 到 `week11` 的裁判分数列
- 系统会自动识别并删除无效数据（N/A、0、空值）
- 数据集划分保持时间顺序，不进行随机打乱
- 输出文件使用UTF-8编码

## 预期输出

- 处理后的数据约2700-3000行（取决于有效数据量）
- 训练集约占80%的赛季数据
- 测试集约占20%的赛季数据
- 生成 `judge_score_distribution.png` 直方图文件
