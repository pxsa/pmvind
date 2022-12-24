# Predicting missing values ​​in non-numerical dataset
There are plenty real-world datasets that have missing values inside, which definitely isn't good. if you have missing value in your dataset you'll face to a big touble for training your data. Because some popular packages such as **scikit-learn**, **keras** and etc don't seem to like missing values at all. So in this article I want to show my thesis to you which introduce useful solutions for this problem.
## Abstract

## Content
- [Abstract](#Abstract)
- [Content](#Content)
- [Algorithm](#Algorithm)
    - [ExtractCompleteTuples](#ExtractCompleteTuples)
    - [ExtractInCompleteTuples](#ExtractInCompleteTuples)

## Algorithm
```python
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
    CTS[i] = Merge(CTS[i-1], T[i])

CTS[0] = Merge(CTS[m-1], T[m-i])
```
This is the main part of the algorithm that is the beginning of the story. Don't worry if you don't understand even a little bit, it will get easier soon. For imputating missing values in dataset we use `KNNI` with some tricks. As you can see we devide the dataset into two different parts:
- CT  (complete tuples)
- ICT (Incomplete tuples)

## ExtractCompleteTuples
```python
def ExtractCompleteTuples(df):
    # getting the rows without null values
    CT = df.dropna()
    return CT
```
In this function we get rid of all rows that contains Nan values, it means our dataframe `CT` will only have complete rows.
## ExtractInCompleteTuples
```python
def ExtractInCompleteTuples(df):
    # getting only the rows with null values
    ICT = df[df.isnull().any(axis=1)]
    # print(ICT.shape)
    return ICT.values
```
In this function our dataframe `ICT` is full of rows which at least have one missing value.

> The combination of `CT` and `ICT` will be the full version of our dataset.

