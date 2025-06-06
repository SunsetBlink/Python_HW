import numpy as np
import pandas as pd
# # 1. Разобраться как использовать мультииндексные ключи в данном примере
import pandas as pd

index = [
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
]

population = [
    101,
    201,
    102,
    202,
    103,
    203,
]

pop = pd.Series(population, index=index)
pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }
)

#pop_df_1 = pop_df.loc[('city_1',2020), 'something']
#pop_df_1 = pop_df.loc[(['city_1', 'city_3'], slice(None)), ['total', 'something']]
#pop_df_1 = pop_df.loc[(['city_1', 'city_3'], slice(None)), 'something']



# 2. Из получившихся данных выбрать данные по
index = pd.MultiIndex.from_product(
     [
        ['city_1', 'city_2'],
        [2010, 2020]
     ],
     names=['city', 'year']
 )
columns = pd.MultiIndex.from_product(
    [
        ['person_1','person_2','person_3'],
        ['job_1','job_2']
    ],
    names = ['worker','job']
)
data = np.random.randint(1, 100, size=(4, 6))
df = pd.DataFrame(data, index=index, columns=columns)
idx = pd.IndexSlice

#- 2020 году (для всех столбцов)
#result = df.loc[idx[2020,:]]
# - job_1 (для всех строк)
#result = df.loc[idx[:,['job_1']]]
# - для city_1 и job_2
#result = df.loc[idx[:,['job_1']]]



# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3

result = df.loc[:, idx[['person_1', 'person_3'], :]]
# - все данные по первому городу и первым двум person-ам (с использование срезов)
result = df.loc[idx['city_1', :], idx[['person_1', 'person_2'], :]]
# Приведите пример (самостоятельно) с использованием pd.IndexSlice
result = df.loc[idx['city_2', 2020], idx['person_2', :]]


#4. Привести пример использования inner и outer джойнов для Series (данные примера скорее всего нужно изменить)
ser1 = pd.Series(['a', 'b', 'c'], index=[1,2,3])
ser2 = pd.Series(['b', 'c', 'f'], index=[4,5,6])
print (pd.concat([ser1, ser2], join='outer'))
print (pd.concat([ser1, ser2], join='inner'))
