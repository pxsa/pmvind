# Predicting missing values ​​in non-numerical dataset
There are plenty real-world datasets that have missing values, which definitely isn't good. if you have missing value in your dataset you'll face to a big problem for training your data. Because some popular packages such as **scikit-learn**, **keras** and etc don't seem to like missing values at all. So in this article I want to show my thesis to you which introduce useful solutions for this problem.
## Abstract

## Content
- [Abstract](#Abstract)
- [Content](#Content)
- [Algorithm](#Algorithm)

## Algorithm
```go
# Begin
CT  = ExtractCompleteTuples(df)
ICT = ExtractInCompleteTuples(df)

# The number of partitions
m = 5
# The number of attributes
s = df.columns.size    

T   = GenerateTuplePartitions(ICT, CT, m, s)
Tp  = [[0]] * m
Tpp = [[0]] * m

CTS    = [[0]] * (m)
CTS[0] = np.array(CT.copy())

for i in range(1, m):
    KNNImputation(CTS[i-1], T[i])
    CTS[i] =  Merge(CTS[i-1], T[i])

CTS[0] =  Merge(CTS[m-1], T[m-i])
```
Don't worry if you don't understand even a little bit, it will get easier soon.
