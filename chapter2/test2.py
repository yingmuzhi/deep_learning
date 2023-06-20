'''pandas的基本使用'''

import pandas as pd
import torch

# 存储数据/读取数据
import pandas as pd
data_file = "./_.csv"
data = pd.read_csv(data_file)
print(data)
data.to_csv(data_file)

# 创建Pandas DataFrame
DF_NAME = [
    "channel",
    "position",
    "dir_name",
]
# 创建空DataFrame
df_rows_num = 10
df = pd.DataFrame(
    data=None, 
    index=range(df_rows_num),
    columns=DF_NAME,
)
print(df)

# 新增新的一行数据
# add one row - method 1: generage a dictionary, then using .loc[]  || recommanded ||
df_one_row_dict = {
    "channel":  "retardance",
    "position": 4,
    "dir_name": "None",
}
df.loc[0, :] = df_one_row_dict
# add one row - method 2: generate one dataframe, then using concat.
df_one_row_dict = {
    "channel":  ["retardance"],
    "position": [4],
    "dir_name": ["None"],
}
df_one_row = pd.DataFrame(df_one_row_dict)
df = pd.concat([df, df_one_row], axis=0)
print(df)

# 筛选数据
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 类型转换 - 转为numpy
numpy_data = df.to_numpy()
print(numpy_data)
# 转为tensor: 用.values只取值（将index剔除）；用.astype("float64")指定ndarray的数据类型。
df_position = df.loc[:, "position"]
df_position = df_position.dropna(axis=0, how="all")
print(df_position)
torch_data = torch.tensor(df_position.values.astype("float64"), dtype=torch.float64)    # ndarray转换类型
print(torch_data)