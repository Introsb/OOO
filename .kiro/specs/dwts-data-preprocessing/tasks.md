# Implementation Plan: DWTS Data Preprocessing System

## Overview

本实施计划将DWTS数据预处理系统分解为离散的编码步骤。每个任务都建立在前面的任务之上，最终形成一个完整的ETL管道。实施将使用Python、pandas进行数据操作、matplotlib进行可视化、Hypothesis进行基于属性的测试。

## Tasks

- [x] 1. 设置项目结构和核心接口
  - 创建项目目录结构：`src/`, `tests/`, `data/`
  - 创建主模块文件：`src/preprocessing_pipeline.py`
  - 设置测试框架：安装pytest和hypothesis
  - 创建requirements.txt文件（pandas, matplotlib, hypothesis, pytest）
  - _Requirements: 所有需求的基础设施_

- [x] 2. 实现DataLoader组件
  - [x] 2.1 实现DataLoader类的load方法
    - 使用pandas.read_csv()读取CSV文件
    - 实现文件存在性检查
    - 实现必需列验证（Season, Name, Age, Industry, Placement, week1-week11）
    - 实现错误处理（FileNotFoundError, ValueError）
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [x] 2.2 为DataLoader编写单元测试
    - 测试成功加载有效文件
    - 测试文件不存在的错误处理
    - 测试无效格式的错误处理
    - _Requirements: 1.2, 1.3_
  
  - [x] 2.3 编写Property 11的属性测试
    - **Property 11: 数据加载保真性**
    - 生成随机CSV内容，验证加载后数据保持一致
    - **Validates: Requirements 1.4**

- [x] 3. 实现WideToLongTransformer组件
  - [x] 3.1 实现宽表到长表的转换逻辑
    - 识别week1到week11列（包括裁判分数子列）
    - 使用pd.melt()或pd.wide_to_long()进行转换
    - 提取Week数字（从"week1"提取1）
    - 保留所有裁判分数列以便后续计算
    - _Requirements: 2.1, 2.2, 2.4, 2.5_
  
  - [x] 3.2 编写Property 1的属性测试
    - **Property 1: 宽表到长表转换的行数扩展**
    - 生成随机宽表数据，验证输出行数 = 选手数 × 周次数
    - **Validates: Requirements 2.1, 2.3**
  
  - [x] 3.3 编写Property 3的属性测试
    - **Property 3: Week标识转换正确性**
    - 生成随机week标识，验证转换为正确的数字
    - **Validates: Requirements 2.5**

- [x] 4. 实现DataCleaner组件
  - [x] 4.1 实现is_valid_score方法
    - 检查N/A、0、空值、NaN
    - 返回布尔值表示分数是否有效
    - _Requirements: 3.1_
  
  - [x] 4.2 实现clean方法
    - 按选手分组（groupby Name和Season）
    - 找到每个选手的第一个无效周次
    - 删除该周次及之后的所有数据
    - 确保保持时间连续性
    - _Requirements: 3.2, 3.3, 3.4_
  
  - [x] 4.3 编写Property 4的属性测试
    - **Property 4: 无效分数识别**
    - 生成随机分数值（包括有效和无效），验证识别正确性
    - **Validates: Requirements 3.1**
  
  - [x] 4.4 编写Property 5的属性测试
    - **Property 5: 淘汰后数据清除**
    - 生成包含无效点的选手数据，验证清洗后只保留有效部分
    - **Validates: Requirements 3.2, 3.3**
  
  - [x] 4.5 编写Property 6的属性测试
    - **Property 6: 时间连续性保持**
    - 生成随机选手数据，验证Week列的连续性
    - **Validates: Requirements 3.4**

- [x] 5. Checkpoint - 确保数据加载和清洗功能正常
  - 运行所有测试确保通过
  - 如有问题请询问用户

- [x] 6. 实现FeatureEngineer组件
  - [x] 6.1 实现add_judge_avg_score方法
    - 计算每行的裁判分数平均值
    - 使用pandas的mean()方法，自动忽略NaN
    - 处理3-4个裁判的不同情况
    - 添加Judge_Avg_Score列
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 6.2 实现add_industry_code方法
    - 使用pd.Categorical或sklearn.LabelEncoder进行编码
    - 确保相同行业获得相同编码
    - 添加Industry_Code列
    - 保留原始Industry列
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 6.3 实现add_score_scaled方法
    - 计算全局均值和标准差
    - 应用Z-score公式：(x - μ) / σ
    - 处理标准差为0的边缘情况（设为0并记录警告）
    - 添加Score_Scaled列
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 6.4 编写Property 7的属性测试
    - **Property 7: 裁判平均分计算正确性**
    - 生成随机裁判分数集合，验证平均值计算
    - **Validates: Requirements 4.1, 4.4**
  
  - [x] 6.5 编写Property 8的属性测试
    - **Property 8: 行业编码的双射性**
    - 生成随机行业列表，验证编码的一致性和唯一性
    - **Validates: Requirements 5.1, 5.2**
  
  - [x] 6.6 编写Property 9的属性测试
    - **Property 9: Z-score标准化公式正确性**
    - 生成随机分数数据，验证Z-score公式应用正确
    - **Validates: Requirements 6.1, 6.2**
  
  - [x] 6.7 编写单元测试处理边缘情况
    - 测试3个裁判的情况
    - 测试4个裁判的情况
    - 测试标准差为0的情况
    - _Requirements: 4.2, 4.3, 6.4_

- [x] 7. 实现TimeSeriesSplitter组件
  - [x] 7.1 实现split方法
    - 按Season排序数据
    - 获取唯一赛季列表并排序
    - 计算80%划分点
    - 将前80%赛季分配给训练集
    - 将后20%赛季分配给测试集
    - 不打乱数据顺序
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  
  - [x] 7.2 编写Property 10的属性测试
    - **Property 10: 时序数据集划分的赛季完整性**
    - 生成随机多赛季数据，验证训练/测试集划分正确
    - **Validates: Requirements 7.2, 7.3, 7.4, 7.6**
  
  - [x] 7.3 编写Property 12的属性测试
    - **Property 12: 数据顺序保持**
    - 生成随机数据，验证排序后顺序在训练/测试集中保持
    - **Validates: Requirements 7.1, 7.5**

- [x] 8. 实现DataValidator组件
  - [x] 8.1 实现validate_and_report方法
    - 打印数据框形状（df.shape）
    - 打印训练集和测试集行数
    - 验证所有必需列存在
    - _Requirements: 9.1, 9.4_
  
  - [x] 8.2 实现plot_score_distribution方法
    - 使用matplotlib.pyplot.hist()绘制Judge_Avg_Score分布
    - 添加标题、轴标签
    - 保存为PNG文件
    - _Requirements: 9.2, 9.3_
  
  - [x] 8.3 编写Property 2的属性测试
    - **Property 2: 输出数据结构完整性**
    - 生成随机数据框，验证所有必需列存在
    - **Validates: Requirements 2.2, 4.5, 5.3, 5.4, 6.3, 8.2**
  
  - [x] 8.4 编写单元测试
    - 测试直方图文件生成
    - 测试控制台输出格式
    - _Requirements: 9.2, 9.3_

- [x] 9. 实现PreprocessingPipeline主协调器
  - [x] 9.1 实现Pipeline类的__init__方法
    - 初始化所有组件（DataLoader, Transformer, Cleaner, Engineer, Splitter, Validator）
    - _Requirements: 所有需求_
  
  - [x] 9.2 实现Pipeline类的run方法
    - 按顺序调用所有组件
    - 实现异常处理和错误消息
    - 保存输出CSV文件（UTF-8编码）
    - 处理文件覆盖情况
    - 调用验证和可视化
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  
  - [x] 9.3 编写端到端集成测试
    - 使用示例CSV文件测试完整流程
    - 验证输出文件生成
    - 验证UTF-8编码
    - 验证文件覆盖行为
    - _Requirements: 8.1, 8.3, 8.4_

- [x] 10. 创建命令行接口和文档
  - [x] 10.1 创建main.py入口文件
    - 使用argparse解析命令行参数
    - 接受输入文件路径和输出文件路径参数
    - 调用PreprocessingPipeline.run()
    - _Requirements: 1.1, 8.1_
  
  - [x] 10.2 创建README.md文档
    - 说明系统功能和使用方法
    - 提供安装说明
    - 提供使用示例
    - 说明输出文件格式
    - _Requirements: 所有需求_

- [x] 11. 实现Hypothesis数据生成器
  - [x] 11.1 创建测试数据生成器模块
    - 实现宽表数据生成器
    - 实现分数生成器（有效和无效）
    - 实现行业文本生成器
    - 实现多赛季数据生成器
    - 确保能生成边缘情况（1周数据、第一周淘汰、标准差为0等）
    - _Requirements: 支持所有属性测试_

- [x] 12. Final Checkpoint - 运行完整测试套件
  - 运行所有单元测试
  - 运行所有属性测试（每个至少100次迭代）
  - 运行集成测试
  - 验证测试覆盖率 > 80%
  - 确保所有测试通过，如有问题请询问用户

## Notes

- 每个任务都引用了具体的需求以便追溯
- Checkpoint确保增量验证
- 属性测试验证通用正确性属性
- 单元测试验证特定示例和边缘情况
- 所有属性测试应配置为运行至少100次迭代
