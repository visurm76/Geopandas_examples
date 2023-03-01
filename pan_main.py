from timeit import timeit
from sys import argv
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
print(pop[:, 2010])  # используем срезы для выборки данных
print('_________________________________')
print(pop['California', 2010])  # используем индексы для выборки данных
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
pop_df = pd.DataFrame({'total': pop, 'under18': [9267089, 9284094, 4687374, 4318033, 5906301, 6879014]})
print(pop_df)
print('_________________________________')
# вычисляем по годам долю населения младше 18 лет
print('Вычисляем по годам долю населения младше 18 лет')
f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()
print(f_u18)
print('_________________________________')
print('Методы создания мультииндексовНаиболее простой метод создания мультииндексирова')
print()
dfm = pd.DataFrame(np.random.rand(4, 2),
                   index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                   columns=['data1', 'data2'])
print(dfm)
print('_________________________________')
print('Явные конструкторы для объектов MultiIndex')
multindex = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
print(multindex)
print('Или из списка кортежей, задающих все значения индекса в каждой из точек')
multindex2 = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
print(multindex2)
print('_________________________________')
print('Задать названия для уровней объекта MultiIndex')
pop.index.names = ['state', 'year']
print(pop)
print('_________________________________')
print('# Иерархические индексы и столбцы')
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'],
                                      ['HR', 'Temp']],
                                     names=['subject', 'type'])
# Имитационная модель
datas = np.round(np.random.randn(4, 6), 1)
print(datas)
print('_________________________________')
datas[:, ::2] *= 10
print(datas)
print('_________________________________')
datas += 37
print(datas)
print('_________________________________')
print('# Создаем объект DataFrame')
health_data = pd.DataFrame(datas, index=index, columns=columns)
print(health_data)
print('_________________________________________________________')
print('Сделаем выборки')
print(health_data['Guido'])
print('________________________')
print('Выборка с помощью явного инденсирования')
print(pop.loc['California':'New York'])
print('Частиная индексация объектов Series')
print(pop[:, 2000])
print('_________________________________________')
print(pop[pop > 22000000])
print('_________________________________________')
print(pop[['California', 'Texas']])
print('_________________________________________')
print('Частиная индексация объектов DataFrame')
print(health_data['Guido', 'HR'])
print('_________________________________________')
print(health_data.iloc[:2, :2])
print('_________________________________________')
print(health_data.loc[:, ('Bob', 'HR')])
print('_________________________________________')
print()
print('Лучше в данном случае использовать объект\
IndexSlice, предназначенный библиотекой Pandas как раз для подобной ситуации')
idx = pd.IndexSlice
print(health_data.loc[idx[:, 1], idx[:, 'HR']])
print('______________________________________')
print('Создание и перестройка индексов')
print(pop)
print()
pop_flat = pop.reset_index(name='population')
print(pop_flat)
print('______________________________________')
print('set_index объекта DataFrame, возвращающего мультииндексированный объект DataFrame')
print(pop_flat.set_index(['state', 'year']))
print('______________________________________')
print('Нужно усреднить результаты измерений показателей \
по двум визитамв течение года, используем метод mean')
data_mean = health_data.groupby(level='year').median()
print(data_mean)
print('______________________________________')
print('Далее, воспользовавшись ключевым словом axis, можно получить и среднее зна-\
                                                        чение по уровням по столбцам')
print(data_mean.mean(axis=1, level='type'))
print('______________________________________')
print('Объединение наборов данных: конкатенация\
и добавление в конец')
print('конкатенацию объектов Series и DataFrame с помощью\
функции pd.concat')


def make_df(cols, ind):
    """Быстро создаем объект DataFrame"""
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)


print(make_df('ABC', range(3)))
print('______________________________________')
print('Простая конкатенация с помощью метода pd.concat')
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
print(pd.concat([ser1, ser2]))
print('______________________________________')
print('Конкатенация объектов более высокой размерности, таких как DataFrame')
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(df1);
print(df2);
print(pd.concat([df1, df2]))
print('______________________________________')
print('Объединение наборов данных: конкатенация и добавление в конец')
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(df3);
print(df4);
print(pd.concat([df3, df4], axis=1))
print('______________________________________')
print("Изучим объединение следующих двух объектов DataFrame, у которых столбцы (но не все!) называются одинаков")
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
print(df5);
print(df6);
print(pd.concat([df5, df6]))
print(pd.concat([df5, df6], join='inner'))  # join='inner'
print('______________________________________')
print('Метод append()')
print(df1);
print(df2);
print(df1.append(df2))
print('______________________________________')
print('Соединения «один-к-одному»')
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'], \
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'], 'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.merge(df1, df2)
print(df3)
print('______________________________________')
print('Соединения «многие-к-одному»')
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print(pd.merge(df3, df4))
print('______________________________________')
print('Соединения «многие-ко-многим»')
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting', 'Engineering', 'Engineering',
                              'HR', 'HR'], 'skills': ['math', 'spreadsheets', 'coding',
                                                      'linux', 'spreadsheets', 'organization']})
print(pd.merge(df1, df5))
print('______________________________________')
print('Ключевое слово on для определения ключа')
# Этот параметр работает только в том случае, когда в левом и правом объектах
# DataFrame имеется указанное название столбца
print(pd.merge(df1, df2, on='employee'))
print('______________________________________')
print('Ключевые слова left_on и right_on')
# Иногда приходится выполнять слияние двух наборов данных с различными именами
# столбцов.
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(pd.merge(df1, df3, left_on="employee", right_on="name"))
print('______________________________________')
# Результат этой операции содержит избыточный столбец, который можно при жела-
# нии удалить. Например, с помощью имеющегося в объектах DataFrame метода drop()
print(pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1))
print('______________________________________')
print('Ключевые слова left_index и right_index.Иногда удобнее вместо слияния по столбцу выполнить слияние по индексу.')
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(df1a);
print(df2a)
print('______________________________________')
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
print('______________________________________')
print('Для удобства в объектах DataFrame реализован метод join(), выполняющий по умолчанию слияние по индексам')
print(df1a.join(df2a))
print('______________________________________')
print('Задание операций над множествами для соединений\
Во всех предыдущих примерах мы игнорировали один важный нюанс выполнения\
соединения — вид используемой при соединении операции алгебры множеств. Это\
играет важную роль в случаях, когда какое-либо значение есть в одном ключевом\
столбце, но отсутствует в другом.')
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])
print(df6);
print(df7);
print(pd.merge(df6, df7))
print('______________________________________')
# Можно указать это явным об-
# разом, с помощью ключевого слова how, имеющего по умолчанию значение 'inner'
print(pd.merge(df6, df7, how='inner'))
print('______________________________________')
print("Другие возможные значения ключевого слова how: 'outer', 'left' и 'right")
print(pd.merge(df6, df7, how='outer'))
print('______________________________________')
print(pd.merge(df6, df7, how='left'))
print('______________________________________')
print(pd.merge(df6, df7, how='right'))
print('______________________________________')
print('Пересекающиеся названия столбцов:ключевое слово suffixes')
# Вам может встретиться случай, когда в двух входных объектах присутствуют кон-
# фликтующие названия столбцов.
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(df8);
print(df9);
print(pd.merge(df8, df9, on="name"))
print('______________________________________')
# Если подобное поведение,
# принятое по умолчанию, неуместно, можно задать пользовательские суффиксы
# с помощью ключевого слова suffixes
print(pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]))
print('______________________________________')
print('Примеры: данные по штатам')
pop = pd.read_csv('data-USstates/state-population.csv')
areas = pd.read_csv('data-USstates/state-areas.csv')
abbrevs = pd.read_csv('data-USstates/state-abbrevs.csv')
print(pop.head()); print(); print(areas.head()); print(); print(abbrevs.head())
