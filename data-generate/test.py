import pandas

# df = pandas.read_pickle('/small-ase2022/data/mvd_down_40_train.pkl')
# for i in range(len(df)):
#     print(df[i])

def test_read_pkl(pkl_filename):
    df = pandas.read_pickle(pkl_filename)
    for i in range(len(df)):
        e = 1
        # print(df[i]['code'])
    return df
df = test_read_pkl('/small-ase2022/temp.pkl')
#df = test_read_pkl('/small-ase2022/data/mvd_nr_40_test.pkl')
print(df)
print(len(df))
print(df['code'][0])
print(df['label'])
print(df['code1'][0])
