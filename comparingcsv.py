import pandas as pd

# 讀取第一個 CSV 文件
df1 = pd.read_csv('data.csv')  # 替換為第一個文件的名稱
# 讀取第二個 CSV 文件
df2 = pd.read_csv('brca_data_w_subtypes.csv')  # 替換為第二個文件的名稱

# 檢查兩個 DataFrame 是否具有相同的列名
if list(df1.columns) != list(df2.columns):
    print("兩個 CSV 文件的列名不匹配。")
else:
    # 確保兩個 DataFrame 的索引一致
    df1 = df1.sort_index(axis=1)
    df2 = df2.sort_index(axis=1)

    # 比較兩個 DataFrame，找出不同之處
    diff = df1.compare(df2)

    # 打印出不同之處
    if diff.empty:
        print("兩個 CSV 文件內容相同。")
    else:
        print("兩個 CSV 文件有以下不同：")
        print(diff)

# 打印兩個 DataFrame 的列名 
print("data.csv 的列名：", df1.columns) 
print("brca_data_w_subtypes.csv 的列名：", df2.columns)