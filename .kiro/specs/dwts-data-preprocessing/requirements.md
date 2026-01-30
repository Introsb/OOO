# Requirements Document

## Introduction

本文档定义了"与星共舞"（Dancing with the Stars）数据预处理系统的需求。该系统将原始比赛数据从宽表格式转换为适合时序分析的长表格式，并执行数据清洗、特征工程和数据集划分。

## Glossary

- **System**: 数据预处理系统
- **Wide_Format**: 宽表格式，每个选手占一行，包含所有周次的分数列
- **Long_Format**: 长表格式，每个选手每周占一行的面板数据结构
- **Valid_Score**: 有效分数，非N/A、非0且非空的分数值
- **Judge_Avg_Score**: 裁判平均分，每周3-4个裁判分数的算术平均值
- **Industry_Code**: 行业编码，将文本行业类别转换为数字分类
- **Score_Scaled**: 标准化分数，使用Z-score方法消除赛季间偏差
- **Elimination_Point**: 淘汰点，选手被淘汰后的时间点
- **Training_Set**: 训练集，按时间顺序的前80%赛季数据
- **Test_Set**: 测试集，按时间顺序的后20%赛季数据

## Requirements

### Requirement 1: 数据加载

**User Story:** 作为数据分析师，我需要加载原始CSV数据文件，以便进行后续的数据转换和处理。

#### Acceptance Criteria

1. WHEN 系统启动时，THE System SHALL 读取名为"2026 MCM Problem C Data.csv"的文件
2. WHEN 文件不存在时，THE System SHALL 返回明确的错误信息并终止执行
3. WHEN 文件格式无效时，THE System SHALL 返回描述性错误信息
4. THE System SHALL 保留所有原始列的数据类型和值

### Requirement 2: 宽表转长表

**User Story:** 作为数据分析师，我需要将宽表格式转换为长表格式，以便进行时序分析和面板数据建模。

#### Acceptance Criteria

1. WHEN 执行转换时，THE System SHALL 将每个选手的所有周次数据展开为独立的行
2. WHEN 转换完成后，THE Long_Format SHALL 包含字段：Season, Week, Name, Age, Industry, Placement
3. WHEN 转换完成后，THE Long_Format SHALL 为每个选手的每个周次创建一行数据
4. THE System SHALL 保留原始数据中的Week列标识（week1到week11）
5. THE System SHALL 将Week标识转换为数字（1到11）

### Requirement 3: 数据清洗

**User Story:** 作为数据分析师，我需要自动剔除选手被淘汰后的无效数据，以确保分析只包含选手存活期间的真实表现。

#### Acceptance Criteria

1. WHEN 处理分数数据时，THE System SHALL 识别N/A、0和空值为无效数据
2. WHEN 选手在某周被淘汰后，THE System SHALL 删除该选手该周之后的所有数据行
3. WHEN 清洗完成后，THE System SHALL 只保留包含Valid_Score的数据行
4. THE System SHALL 保持每个选手数据的时间连续性（不跳过中间周次）

### Requirement 4: 裁判平均分计算

**User Story:** 作为数据分析师，我需要计算每周的裁判平均分，以便统一不同裁判数量的评分标准。

#### Acceptance Criteria

1. WHEN 计算Judge_Avg_Score时，THE System SHALL 对每周的所有裁判分数求算术平均值
2. WHEN 某周有3个裁判时，THE System SHALL 计算3个分数的平均值
3. WHEN 某周有4个裁判时，THE System SHALL 计算4个分数的平均值
4. WHEN 某周的裁判分数包含无效值时，THE System SHALL 只使用有效分数计算平均值
5. THE System SHALL 将Judge_Avg_Score添加为新列到Long_Format数据中

### Requirement 5: 行业编码

**User Story:** 作为数据分析师，我需要将文本行业类别转换为数字编码，以便在机器学习模型中使用。

#### Acceptance Criteria

1. WHEN 执行行业编码时，THE System SHALL 为每个唯一的行业文本分配一个唯一的整数代码
2. WHEN 相同行业出现多次时，THE System SHALL 分配相同的Industry_Code
3. THE System SHALL 将Industry_Code添加为新列到Long_Format数据中
4. THE System SHALL 保留原始Industry列以便追溯

### Requirement 6: 分数标准化

**User Story:** 作为数据分析师，我需要使用Z-score标准化分数，以消除不同赛季间的评分偏差。

#### Acceptance Criteria

1. WHEN 执行标准化时，THE System SHALL 使用Z-score公式：(x - mean) / std
2. WHEN 计算Z-score时，THE System SHALL 使用所有有效分数的全局均值和标准差
3. THE System SHALL 将Score_Scaled添加为新列到Long_Format数据中
4. WHEN 标准差为0时，THE System SHALL 处理该边缘情况并返回合理的值

### Requirement 7: 时序数据集划分

**User Story:** 作为数据分析师，我需要按时间顺序划分训练集和测试集，以保持时序数据的完整性。

#### Acceptance Criteria

1. WHEN 划分数据集时，THE System SHALL 按Season字段排序所有数据
2. WHEN 确定划分点时，THE System SHALL 计算唯一赛季数量的80%作为训练集边界
3. THE Training_Set SHALL 包含按时间顺序的前80%赛季的所有数据
4. THE Test_Set SHALL 包含按时间顺序的后20%赛季的所有数据
5. THE System SHALL NOT 随机打乱数据顺序
6. THE System SHALL NOT 在同一赛季内分割数据（整个赛季要么在训练集要么在测试集）

### Requirement 8: 数据输出

**User Story:** 作为数据分析师，我需要将处理后的数据保存为CSV文件，以便后续分析使用。

#### Acceptance Criteria

1. WHEN 处理完成后，THE System SHALL 将Long_Format数据保存为"Processed_DWTS_Long_Format.csv"
2. THE System SHALL 确保输出文件包含所有必需字段：Season, Week, Name, Age, Industry_Code, Judge_Avg_Score, Score_Scaled, Placement
3. THE System SHALL 使用UTF-8编码保存文件
4. WHEN 输出文件已存在时，THE System SHALL 覆盖原文件

### Requirement 9: 数据验证和可视化

**User Story:** 作为数据分析师，我需要验证处理结果的正确性，以确保数据质量符合预期。

#### Acceptance Criteria

1. WHEN 处理完成后，THE System SHALL 打印最终数据的形状（行数和列数）
2. THE System SHALL 生成Judge_Avg_Score的分布直方图
3. THE System SHALL 将直方图保存为图像文件
4. THE System SHALL 在控制台输出训练集和测试集的行数统计
