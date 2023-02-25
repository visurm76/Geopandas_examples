from timeit import timeit

import pandas as pd
import numpy as np

print(pd.__version__)

data = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(data)
print(data.values, data.index)
print('_________________________________')

# Объект Series для словаря
print('Объект Series для словаря')
population_dict = {'California': 3546457, 'Texas': 654654}
population = pd.Series(population_dict)  # population = pd.Series({'California':3546457, 'Texas':654654})
print(population)
print(population['California'])
print('_________________________________')
num = pd.Series(5, index=[100, 200, 300])
print(num)
print('_________________________________')

# Выбрать, какие индексы отображать
print('Выбрать, какие индексы отображать')
num2 = pd.Series({2: 'a', 1: 'c', 3: 'b'}, index=[1, 3])
print(num2)
print('_________________________________')

# Двумерные массивы
print('Двумерные массивы')
area_dict = {'California': 875432, 'Texas': 65465654}
states = pd.DataFrame({'population': population, 'area_dict': area_dict})
print(states)
print(states.index)
print(states.values)
print('_________________________________')
# Доступ к конкретному столбцу
print('Доступ к конкретному столбцу')
print(states['area_dict'])
print('_________________________________')

# Создание объектов DataFrame
print('Создание объектов DataFrame')
breeds = {'pinus': 12321, 'picea': 879798}
num3 = pd.Series(breeds)
num_breeds = pd.DataFrame(num3, columns=['index'])
print(num_breeds)
print('_________________________________')

# Index

indA = pd.Index([1, 2, 3, 4, 5])
indB = pd.Index([2, 3])
indAB = indA.intersection(indB)

# Пересечение
print(f"indAB {indAB}")
print('_________________________________')

# Объект, как словарь
print('Объект, как словарь')
data = pd.Series([1, 56, 31, 54, 32], index=['a', 'b', 'c', 'd', 'e'])
print(data['b'])
print(data.keys())
print(list(data.items()))
print(data['a':'c'])
print(data[['a', 'c']])  # Конкретные индексы
print(data > 10)
print('_________________________________')

# Индексаторы loc - явное обращение по индексу
print('Индексаторы loc - явное обращение по индексу')
print(data.loc['a':'b'])

# Индексаторы iloc - неявное обращение по индексу
print('Индексаторы iloc - неявное обращение по индексу')
print(data.iloc[::-1])
print('_________________________________')

# Объект DataFrame как словарь
print('Объект DataFrame как словарь')
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})

pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})

dates = pd.DataFrame({'area': area, 'pop': pop})
print(dates)
print()
print(dates['pop'])
print()
print(dates.area)
print('_________________________________')

# Добавление нового столбца
print('Добавление нового столбца')
dates['density'] = dates['pop'] / dates['area']
print(dates)
print('_________________________________')
print(dates['Florida':'Illinois'])
print('_________________________________')
print()

# Можно ссылаться по номеру строк
print('Можно ссылаться по номеру строк')
print(dates[0:3])
print('_________________________________')
print()
dates[dates.density > 100]
print(dates[dates.density > 100])
print('_________________________________')
print()
# Универсальные функции: сохранение индекса
print('Универсальные функции: сохранение индекса')
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
print(ser)
print('_________________________________')
print()
df = pd.DataFrame(rng.randint(0, 10, (3, 4)), \
                  columns=['A', 'B', 'C', 'D'])
print(df)
print('_________________________________')
print()
# Выравнивание индексов в объектах Series
print('Выравнивание индексов в объектах Series')
areas = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                   'California': 423967}, name='area')
populations = pd.Series({'California': 38332521, 'Texas': 26448193,
                         'New York': 19651127}, name='population')
print(populations / areas)
print(areas.index.union(populations.index))

# Универсальные функции: выполнение операции
# между объектами DataFrame и Series
print('Универсальные функции: выполнение операции между объектами DataFrame и Series')
A = rng.randint(10, size=(3, 4))
print(A)
print('_________________________________')
print(A - A[0])
print('_________________________________')
print()

# Обработка отсутствующих данных
print('Обработка отсутствующих данных')
vals1 = np.array([1, None, 3, 4])
print(vals1)
print('_________________________________')
print()
# Выявление пустых значений
print('Выявление пустых значений')
data2 = pd.Series([1, np.nan, 'hello', None])
print(data2)
print(data2.isnull())
print('_________________________________')
print()
# булевы маски можно использовать непосредственно в качестве индекса объектов
# Series или DataFrame
print('Булевы маски можно использовать непосредственно в качестве индекса объектов Series или DataFrame')
print(data2[data2.notnull()])
print('_________________________________')
print()
# Удаление пустых значений
print('Удаление пустых значений')
print(data2.dropna())
print('_________________________________')
print()
# Заполнение пустых значений
print('Заполнение пустых значений')
data3 = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
print(data3)
print('_________________________________')
print(data3.fillna(0))
print('_________________________________')
# Можно задать параметр заполнения по направлению «вперед», копируя предыду-
# щее значение в следующую ячейку:
print('Можно задать параметр заполнения по направлению «вперед», копируя предыдущее значение в следующую ячейку')
print(data3.fillna(method='ffill'))
print('_________________________________')
# заполнение по направлению «назад»
print('заполнение по направлению «назад»')
print(data3.fillna(method='bfill'))
print('_________________________________')
# Для объектов DataFrame опции аналогичны
print('Для объектов DataFrame опции аналогичны')
df1 = pd.DataFrame([[1, np.nan, 2], [2, 3, 5], [np.nan, 4, 6]])
print(df1)
print('_________________________________')
print(df1.fillna(method='bfill', axis=1))
print('_________________________________')

# Мультииндексирование
# в данном случаи индексирование идет по двум параметрам в котреже
print('Мультииндексирование в данном случаи индексирование идет по двум параметрам в котреже')
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956, 18976457, 19378102, 20851820, 25145561]
pop = pd.Series(populations, index=index)
print(pop)
print('_________________________________')
index2 = pd.MultiIndex.from_tuples(index)
print(index2)
print('_________________________________')
pop = pop.reindex(index2)
print(pop)
print('_________________________________')
print('используем срезы для выборки данных, например год 2010')
print(pop[:,2010]) # используем срезы для выборки данных
print('_________________________________')
print(pop['California',2010]) # используем индексы для выборки данных
print('_________________________________')
# Превращаем мультиндексный объект Series в индексированный обычным способом DataFrame
print('Превращаем мультиндексный объект Series в индексированный обычным способом DataFrame')
pop_df = pop.unstack()
print(pop_df)
print('_________________________________')
# Обратная операция unstack
print('Обратная операция unstack')
print(pop_df.stack())
print('_________________________________')
# Добавим еще один столбец
print('Добавим еще один столбец')
pop_df = pd.DataFrame({'total': pop, 'under18': [9267089, 9284094,4687374, 4318033,5906301, 6879014]})
print(pop_df)
print('_________________________________')
# вычисляем по годам долю населения младше 18 лет
print('Вычисляем по годам долю населения младше 18 лет')
f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()
print(f_u18)
print('_________________________________')