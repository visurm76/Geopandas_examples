import pandas as pd

print(pd.__version__)

data = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(data)
print(data.values, data.index)
print('_________________________________')

# Объект Series для словаря

population_dict = {'California': 3546457, 'Texas': 654654}
population = pd.Series(population_dict)  # population = pd.Series({'California':3546457, 'Texas':654654})
print(population)
print(population['California'])
print('_________________________________')
num = pd.Series(5, index=[100, 200, 300])
print(num)
print('_________________________________')

# Выбрать, какие индексы отображать

num2 = pd.Series({2: 'a', 1: 'c', 3: 'b'}, index=[1, 3])
print(num2)
print('_________________________________')

# Двумерные массивы
area_dict = {'California': 875432, 'Texas': 65465654}
states = pd.DataFrame({'population': population, 'area_dict': area_dict})
print(states)
print(states.index)
print(states.values)
print('_________________________________')
# Доступ к конкретному столбцу
print(states['area_dict'])
print('_________________________________')

# Создание объектов DataFrame
breeds = {'pinus': 12321, 'picea': 879798}
num3 = pd.Series(breeds)
num_breeds = pd.DataFrame(num3, columns=['index'])
print(num_breeds)
print('_________________________________')

# Index
indA = pd.Index([1,2,3,4,5])
indB = pd.Index([2,3])
indAB = indA.intersection(indB)

# Пересечение
print(f"indAB {indAB}")
print('_________________________________')

# Объект, как словарь
data = pd.Series([1,54,31,54,32], index=['a', 'b', 'c', 'd', 'e'])
print(data['b'])
print(data.keys())
print(list(data.items()))
print(data['a':'c'])
print('_________________________________')

