# 计算机建模 - 课程作业

本仓库包含计算机建模课程的课后作业。

## 目录结构

```
计算机建模/
├── assignment_1/          # 作业1
│   ├── 1.py
│   └── Figure_1.png
│
├── assignment_2/          # 作业2 - 人口增长模型分析
│   ├── code_and_output/
│   │   ├── population_model.py      # 人口增长模型分析程序
│   │   ├── 1949-2023人口数据-实验1.xlsx
│   │   ├── README.md
│   │   └── output/                # 输出结果
│   │       ├── output.txt         # 终端输出
│   │       ├── 信赖域反射法TRF.png
│   │       ├── Levenberg-Marquardt.png
│   │       ├── Dogbox信赖域法.png
│   │       └── BFGS拟牛顿法.png
│   │
│   └── 参考资料/
│       ├── 实验一-实验内容与要求.docx
│       ├── 上机实验一参考资料.pptx
│       └── 1949-2023人口数据-实验1.xlsx
└── 
```

## assignment\_2 - 人口增长模型分析

### 实验内容

使用1949-2023年中国人口数据，建立并对比三种经典人口增长模型，通过多种参数估计方法进行拟合与预测分析。

### 包含模型

- 指数增长模型
- 改进指数增长模型
- Logistic增长模型
- Gompertz增长模型
- Von Bertalanffy增长模型

### 参数估计方法

- 信赖域反射法(TRF)
- Levenberg-Marquardt算法
- Dogbox信赖域法
- BFGS拟牛顿法

### 运行方式

```bash
cd assignment_2/code_and_output
python population_model.py
```

### 输出

- 终端显示参数估计结果、模型检验指标、预测值汇总表
- 4张可视化图表（每种参数估计方法一张）

