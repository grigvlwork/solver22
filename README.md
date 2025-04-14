# solver22
# Решение задачи 22 из ЕГЭ по информатике

Этот репозиторий содержит код для решения некоторых вариантов задачи 22 из ЕГЭ по информатике, связанной с планированием зависимых процессов. Цель — анализировать граф зависимостей между задачами и находить оптимальные параметры выполнения: максимальный параллелизм, его длительность, критический путь и т.д.  

Данный решатель нужен прежде всего для тех случаев, когда вы столкнулись с вариантом, в котором неизвестны ответы, а вы хотите убедиться, что решили верно, или хотите понять, к какому количеству процессов стремиться, или какая длительность пика должна быть.



## Формат входных данных

Для подготовки текстового файла с входными данными:

1. Откройте файл Excel с данными задачи.
2. Скопируйте все строки, исключая строку заголовка.
3. Вставьте содержимое в обычный текстовый файл (например, `input.txt`).

Пример содержимого файла:   
102	4	0  
103	6	0  
104	8	0  
105	10	102  
106	9	103;104  

Каждая строка содержит:
- ID процесса
- Длительность (целое число)
- Список зависимостей через `;` (если зависимостей нет — указано `0`)

## Ограничения

Некоторые задачи не решаются при слишком больших значениях длительности процессов (обычно свыше 100). Это связано с ограничениями используемой модели оптимизации.   
Их можно попробовать решить уменьшив длительность в 100-1000 раз и посмотреть, как будут располагаться процессы, а затем перейти к исходным значениям и подобрать точное значение промежутка.

## Установка зависимостей

Убедитесь, что у вас установлен Python 3.7 или новее.

Установите зависимости из файла `requirements.txt`:

```
pip install -r requirements.txt
```
## Запуск    
Основной код запускается из файла solver.py.
```
python solver.py
```
Далее нужно ввести имя файла, выбрать тип задачи и, если нужно, ввести дополнительные параметры.



