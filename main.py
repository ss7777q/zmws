import pandas as pd
import numpy as np
import sqlite3
import os

csvfile='data.csv'
dbfile='data.db'


if not os.path.exists(csvfile):
    print("csv文件不存在")
try:
    df=pd.read_csv(csvfile)
    cnt=sqlite3.connect(dbfile)
    df.to_sql('data',con=cnt,if_exists='replace',index=False)
    print('成功导入数据库')
except:
    print("导入时出错")
finally:
    if 'cnt' in locals():
        cnt.close()
