gray.shape

gray.shape = gray.shape[0]*gray.shape[1]

gray.shape

import pandas as pd

gray_df = pd.Series(gray)
gray_df.describe()

gray_df.plot(kind='kde')
