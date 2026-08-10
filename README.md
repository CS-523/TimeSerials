## 任务

1. 分析并处理数据，清理异常、缺失等情况；

2. 建立过程预测模型，输入一段时间的数据（比如周期0-20、25-40等，长度自定，需要起始数据点可变（即可以从不同时间点开始输入一段时间的数据）；输入的参数可自选）；输出未来一段时间的数据（x1-x8和y4必选，y1-y3和y可选，）；输出长度可自定义（不少于6个数据点/每次），越长越好，也可以迭代生成；

3. 建模过程中，分析不同参数的特点和它们之间的关联性；

4. 建立优化模型，输入一段时间数据，优化未来一段时间的x3、x4、x6、x8；使y4或者y最大。

## 汇报内容（PPT）

1. 对项目内容的理解

2. 技术路线

3. 处理过程和结果

4. 分析总结，拓展等

## Data Description

* **Timeseries Data**: Each CSV file contains timeseries data for a single experiment.

* **Headers**: 15 columns in total:

  * **`x1` to `x8`**: Experiment variables measured at every **30-minute timestep**.

  * **`y1` to `y4`**: Target variables measured at **specific timesteps** (not all rows will have values for these).

  * **`Y`**: Final target value representing the **overall outcome** of the entire experiment.

  * **`datime`**: Timestamp of the measurement (e.g., `YYYY-MM-DD HH:MM:SS`).

  * **`周期` (zhouqi)**: Timestep index (e.g., 1, 2, 3, ...) for the timeseries.

  * 空行可删除；x空白，y1-y4数据可与前一行的x对齐

## Columns

| Column Name | Description                          | Type         |
| ----------- | ------------------------------------ | ------------ |
| `datime`    | Timestamp of measurement             | Timestep     |
| `x1`-`x8`   | Process variables (30-min intervals) | Inputs       |
| `y1`-`y4`   | Intermediate targets                 | Outputs      |
| `Y`         | Final experiment outcome             | Final Target |
| `周期`        | Timestep index (sequential counter)  | Metadata     |

💡 **Notes**:

* Each CSV corresponds to one individual experiment.

* Missing values in `y1`-`y4` indicate no measurement was taken at that timestep.

* `Y` is a single value for one CSV.

