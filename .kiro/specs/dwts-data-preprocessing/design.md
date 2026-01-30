# Design Document: DWTS Data Preprocessing System

## Overview

本系统是一个Python数据预处理管道，用于将"与星共舞"比赛的原始宽表数据转换为适合时序分析的长表格式。系统采用pandas库进行数据操作，matplotlib进行可视化，并实现了完整的ETL（提取、转换、加载）流程。

核心设计原则：
- **时序完整性**：保持数据的时间顺序，不进行随机打乱
- **数据质量**：自动识别和剔除无效数据
- **可追溯性**：保留原始字段以便验证
- **模块化**：每个处理步骤独立封装，便于测试和维护

## Architecture

系统采用管道架构（Pipeline Architecture），数据按顺序流经以下阶段：

```
Raw CSV → Load → Wide-to-Long Transform → Clean → Feature Engineering → Split → Output CSV
                                                          ↓
                                                    Validation & Visualization
```

主要组件：
1. **DataLoader**: 负责读取和验证输入文件
2. **WideToLongTransformer**: 执行宽表到长表的转换
3. **DataCleaner**: 识别和删除无效数据
4. **FeatureEngineer**: 计算派生特征（Judge_Avg_Score, Industry_Code, Score_Scaled）
5. **TimeSeriesSplitter**: 按时间顺序划分训练集和测试集
6. **DataValidator**: 验证输出并生成可视化

## Components and Interfaces

### 1. DataLoader

**职责**：加载原始CSV文件并进行初步验证

**接口**：
```python
class DataLoader:
    def load(self, filepath: str) -> pd.DataFrame:
        """
        加载CSV文件
        
        参数:
            filepath: CSV文件路径
            
        返回:
            pd.DataFrame: 原始数据框
            
        异常:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式无效
        """
```

**实现要点**：
- 使用pandas.read_csv()读取文件
- 检查必需列是否存在（Season, Name, Age, Industry, Placement, week1-week11）
- 验证数据类型的合理性

### 2. WideToLongTransformer

**职责**：将宽表格式转换为长表格式

**接口**：
```python
class WideToLongTransformer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将宽表转换为长表
        
        参数:
            df: 宽格式数据框
            
        返回:
            pd.DataFrame: 长格式数据框，包含列：
                Season, Week, Name, Age, Industry, Placement, Judge_Scores
        """
```

**实现要点**：
- 识别week1到week11列（可能包含多个裁判分数的子列）
- 使用pd.melt()或pd.wide_to_long()进行转换
- 将Week标识（如"week1"）转换为数字1
- 保留每周的所有裁判分数以便后续计算平均值

### 3. DataCleaner

**职责**：删除选手被淘汰后的无效数据

**接口**：
```python
class DataCleaner:
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据，删除无效记录
        
        参数:
            df: 长格式数据框
            
        返回:
            pd.DataFrame: 清洗后的数据框，只包含有效数据
        """
    
    def is_valid_score(self, score: Any) -> bool:
        """
        判断分数是否有效
        
        参数:
            score: 分数值
            
        返回:
            bool: True表示有效，False表示无效
        """
```

**实现要点**：
- 识别无效值：pd.isna(), 值为0, 空字符串
- 对每个选手，找到第一个无效周次，删除该周及之后的所有数据
- 使用groupby按选手分组处理
- 保持每个选手数据的连续性

### 4. FeatureEngineer

**职责**：计算派生特征

**接口**：
```python
class FeatureEngineer:
    def add_judge_avg_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算裁判平均分
        
        参数:
            df: 包含裁判分数的数据框
            
        返回:
            pd.DataFrame: 添加了Judge_Avg_Score列的数据框
        """
    
    def add_industry_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加行业编码
        
        参数:
            df: 包含Industry列的数据框
            
        返回:
            pd.DataFrame: 添加了Industry_Code列的数据框
        """
    
    def add_score_scaled(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加标准化分数
        
        参数:
            df: 包含Judge_Avg_Score列的数据框
            
        返回:
            pd.DataFrame: 添加了Score_Scaled列的数据框
        """
```

**实现要点**：
- **Judge_Avg_Score**: 使用mean()计算每行的裁判分数平均值，自动忽略NaN
- **Industry_Code**: 使用pd.Categorical或LabelEncoder进行编码
- **Score_Scaled**: 使用Z-score公式 (x - μ) / σ，其中μ和σ是全局统计量
- 处理边缘情况：标准差为0时返回0

### 5. TimeSeriesSplitter

**职责**：按时间顺序划分训练集和测试集

**接口**：
```python
class TimeSeriesSplitter:
    def split(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分数据集
        
        参数:
            df: 完整数据框
            train_ratio: 训练集比例（默认0.8）
            
        返回:
            Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 测试集)
        """
```

**实现要点**：
- 获取所有唯一赛季并排序
- 计算划分点：int(len(unique_seasons) * train_ratio)
- 将前N个赛季的所有数据分配给训练集
- 将剩余赛季的所有数据分配给测试集
- 不打乱数据顺序

### 6. DataValidator

**职责**：验证输出并生成可视化

**接口**：
```python
class DataValidator:
    def validate_and_report(self, df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        验证数据并生成报告
        
        参数:
            df: 完整处理后的数据框
            train_df: 训练集
            test_df: 测试集
        """
    
    def plot_score_distribution(self, df: pd.DataFrame, output_path: str) -> None:
        """
        绘制分数分布直方图
        
        参数:
            df: 包含Judge_Avg_Score的数据框
            output_path: 输出图像路径
        """
```

**实现要点**：
- 打印df.shape
- 打印训练集和测试集的行数
- 使用matplotlib.pyplot.hist()绘制Judge_Avg_Score分布
- 保存图像为PNG格式

### 7. PreprocessingPipeline

**职责**：协调所有组件，执行完整的预处理流程

**接口**：
```python
class PreprocessingPipeline:
    def __init__(self):
        """初始化所有组件"""
    
    def run(self, input_path: str, output_path: str) -> None:
        """
        执行完整的预处理流程
        
        参数:
            input_path: 输入CSV文件路径
            output_path: 输出CSV文件路径
        """
```

**实现要点**：
- 按顺序调用所有组件
- 处理异常并提供清晰的错误信息
- 确保输出文件包含所有必需字段

## Data Models

### 输入数据模型（宽表格式）

```python
# 每行代表一个选手
{
    "Season": int,           # 赛季编号
    "Name": str,             # 选手姓名
    "Age": int,              # 年龄
    "Industry": str,         # 行业类别
    "Placement": int,        # 最终排名
    "week1_judge1": float,   # 第1周裁判1分数
    "week1_judge2": float,   # 第1周裁判2分数
    "week1_judge3": float,   # 第1周裁判3分数
    "week1_judge4": float,   # 第1周裁判4分数（可选）
    # ... week2 到 week11 的类似结构
}
```

### 中间数据模型（长表格式，清洗前）

```python
# 每行代表一个选手的一周数据
{
    "Season": int,
    "Week": int,             # 周次（1-11）
    "Name": str,
    "Age": int,
    "Industry": str,
    "Placement": int,
    "judge1": float,         # 裁判1分数
    "judge2": float,         # 裁判2分数
    "judge3": float,         # 裁判3分数
    "judge4": float,         # 裁判4分数（可选）
}
```

### 输出数据模型（长表格式，处理后）

```python
# 每行代表一个选手的一周有效数据
{
    "Season": int,
    "Week": int,
    "Name": str,
    "Age": int,
    "Industry": str,         # 保留原始行业文本
    "Industry_Code": int,    # 行业数字编码
    "Judge_Avg_Score": float,  # 裁判平均分
    "Score_Scaled": float,   # Z-score标准化分数
    "Placement": int
}
```

### 数据约束

- Season: 正整数
- Week: 1-11之间的整数
- Age: 正整数
- Judge_Avg_Score: 通常在0-40之间（假设每个裁判最高10分）
- Score_Scaled: Z-score，通常在-3到+3之间
- Industry_Code: 非负整数
- Placement: 正整数


## Correctness Properties

*属性（Property）是系统在所有有效执行中应保持为真的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性是人类可读规范和机器可验证正确性保证之间的桥梁。*

### Property 1: 宽表到长表转换的行数扩展

*对于任意*宽表数据框，如果一个选手有N个周次的数据（week1到weekN），那么转换后的长表应该为该选手创建N行数据。

**Validates: Requirements 2.1, 2.3**

### Property 2: 输出数据结构完整性

*对于任意*处理后的数据框，它必须包含所有必需的列：Season, Week, Name, Age, Industry, Industry_Code, Judge_Avg_Score, Score_Scaled, Placement，并且原始Industry列应被保留。

**Validates: Requirements 2.2, 4.5, 5.3, 5.4, 6.3, 8.2**

### Property 3: Week标识转换正确性

*对于任意*week标识字符串（如"week1", "week5", "week11"），转换后的数字应该等于标识中的数字部分（1, 5, 11）。

**Validates: Requirements 2.5**

### Property 4: 无效分数识别

*对于任意*分数值，如果它是N/A、0、空值或NaN，那么is_valid_score()函数应返回False；否则应返回True。

**Validates: Requirements 3.1**

### Property 5: 淘汰后数据清除

*对于任意*选手的时序数据，如果第K周是第一个包含无效分数的周次，那么清洗后的数据应该只包含该选手第1周到第K-1周的数据，第K周及之后的所有数据应被删除。

**Validates: Requirements 3.2, 3.3**

### Property 6: 时间连续性保持

*对于任意*选手在清洗后的数据中，该选手的Week列应该是连续的整数序列（如1,2,3,4或1,2,3），不应有跳跃（如1,3,5）。

**Validates: Requirements 3.4**

### Property 7: 裁判平均分计算正确性

*对于任意*一行数据的裁判分数集合，Judge_Avg_Score应该等于所有有效裁判分数的算术平均值（sum / count），自动排除无效值。

**Validates: Requirements 4.1, 4.4**

### Property 8: 行业编码的双射性

*对于任意*两个行业文本值，如果它们相同，则它们的Industry_Code必须相同；如果它们不同，则它们的Industry_Code必须不同（一对一映射）。

**Validates: Requirements 5.1, 5.2**

### Property 9: Z-score标准化公式正确性

*对于任意*Judge_Avg_Score值x，其对应的Score_Scaled值应该等于(x - μ) / σ，其中μ是所有Judge_Avg_Score的全局均值，σ是全局标准差。

**Validates: Requirements 6.1, 6.2**

### Property 10: 时序数据集划分的赛季完整性

*对于任意*处理后的数据框，按Season排序后，训练集应包含前80%赛季的所有行，测试集应包含后20%赛季的所有行，且训练集和测试集的赛季集合应该不相交（无重叠）。

**Validates: Requirements 7.2, 7.3, 7.4, 7.6**

### Property 11: 数据加载保真性

*对于任意*有效的CSV文件内容，加载后的数据框应该保留所有原始列的数据类型和值，不应有数据丢失或类型转换错误。

**Validates: Requirements 1.4**

### Property 12: 数据顺序保持

*对于任意*数据框，在按Season排序后，训练集和测试集内部的行顺序应该与原始排序后的顺序一致，不应被随机打乱。

**Validates: Requirements 7.1, 7.5**

## Error Handling

### 文件操作错误

1. **文件不存在**：
   - 捕获FileNotFoundError
   - 返回清晰的错误消息："Error: Input file '{filepath}' not found."
   - 终止执行

2. **文件格式无效**：
   - 捕获pd.errors.ParserError或ValueError
   - 返回描述性消息："Error: Invalid CSV format in '{filepath}'. {details}"
   - 终止执行

3. **缺少必需列**：
   - 检查必需列是否存在
   - 返回消息："Error: Missing required columns: {missing_columns}"
   - 终止执行

### 数据处理错误

1. **标准差为0**：
   - 在计算Z-score时检查std是否为0
   - 如果std == 0，将所有Score_Scaled设为0
   - 记录警告："Warning: Standard deviation is 0, all scores are identical. Setting Score_Scaled to 0."

2. **空数据框**：
   - 在每个处理步骤后检查数据框是否为空
   - 如果为空，返回错误："Error: No valid data remaining after {step_name}"
   - 终止执行

3. **无效的Week标识**：
   - 使用正则表达式验证week列名格式
   - 如果格式不匹配，记录警告并跳过该列
   - 继续处理其他有效列

### 输出错误

1. **无法写入文件**：
   - 捕获PermissionError或IOError
   - 返回消息："Error: Cannot write to '{filepath}'. Check permissions."
   - 终止执行

2. **磁盘空间不足**：
   - 捕获OSError
   - 返回消息："Error: Insufficient disk space to write output file."
   - 终止执行

## Testing Strategy

本系统采用**双重测试方法**，结合单元测试和基于属性的测试（Property-Based Testing, PBT）以确保全面覆盖：

### 单元测试（Unit Tests）

单元测试用于验证特定示例、边缘情况和错误条件：

1. **特定示例测试**：
   - 测试加载特定的示例CSV文件
   - 测试文件不存在时的错误处理
   - 测试输出文件生成
   - 测试可视化文件生成

2. **边缘情况测试**：
   - 测试3个裁判的情况
   - 测试4个裁判的情况
   - 测试标准差为0的情况
   - 测试空数据框的处理

3. **集成测试**：
   - 测试完整的端到端流程
   - 验证输出文件的UTF-8编码
   - 验证文件覆盖行为

### 基于属性的测试（Property-Based Tests）

基于属性的测试用于验证跨所有输入的通用属性。我们将使用**Hypothesis**库（Python的PBT库）来实现。

**配置要求**：
- 每个属性测试最少运行**100次迭代**
- 每个测试必须用注释标记对应的设计属性
- 标记格式：`# Feature: dwts-data-preprocessing, Property {N}: {property_text}`

**属性测试列表**：

1. **Property 1测试**：生成随机宽表数据，验证转换后的行数
2. **Property 2测试**：生成随机数据框，验证所有必需列存在
3. **Property 3测试**：生成随机week标识，验证转换正确性
4. **Property 4测试**：生成随机分数值（包括有效和无效），验证识别正确性
5. **Property 5测试**：生成随机选手时序数据（包含无效点），验证清洗后只保留有效部分
6. **Property 6测试**：生成随机选手数据，验证Week列的连续性
7. **Property 7测试**：生成随机裁判分数集合，验证平均值计算
8. **Property 8测试**：生成随机行业列表，验证编码的一致性和唯一性
9. **Property 9测试**：生成随机分数数据，验证Z-score公式
10. **Property 10测试**：生成随机多赛季数据，验证训练/测试集划分
11. **Property 11测试**：生成随机CSV内容，验证加载保真性
12. **Property 12测试**：生成随机数据，验证排序后顺序保持

**测试框架设置**：

```python
from hypothesis import given, strategies as st, settings
import hypothesis.extra.pandas as pdst

# 配置Hypothesis
settings.register_profile("dwts", max_examples=100, deadline=None)
settings.load_profile("dwts")

# 示例：Property 7的测试
@given(st.lists(st.floats(min_value=0, max_value=10), min_size=3, max_size=4))
def test_judge_avg_score_calculation(judge_scores):
    # Feature: dwts-data-preprocessing, Property 7: 裁判平均分计算正确性
    # 测试逻辑...
```

### 测试覆盖目标

- **单元测试**：覆盖具体示例和边缘情况，目标覆盖率 > 80%
- **属性测试**：覆盖通用正确性属性，确保所有12个属性都有对应测试
- **集成测试**：验证组件间的交互和端到端流程

### 测试数据生成

为了支持属性测试，需要实现以下数据生成器：

1. **宽表数据生成器**：生成符合DWTS格式的随机宽表数据
2. **分数生成器**：生成有效和无效的分数值
3. **行业生成器**：生成随机行业文本
4. **赛季数据生成器**：生成多赛季的随机数据

这些生成器应该能够生成边缘情况，如：
- 只有1周数据的选手
- 所有周次都有效的选手
- 第一周就被淘汰的选手
- 只有1个赛季的数据
- 所有分数相同的情况（标准差为0）
