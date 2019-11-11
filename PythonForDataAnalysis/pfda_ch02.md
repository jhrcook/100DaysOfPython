## 2.1 The Python Interpreter

Here I can type some Markdown and code.

I set up this Jupyter notebook by following the [instructions](https://code.visualstudio.com/docs/python/jupyter-support) from VS Code


```python
print('Hello world')
```

    Hello world


Some basic commands for working with a Jupyter notebook in VS Code:

- `ESC` and `Enter` to choose command mode and editing mode, respectively.
- `M` and `Y` to switch between markdown and code for a selected cell.
- `A` and `B` to add a new cell above or below the one selected.
- `Z` to delete the selected cell.
- arrows or `K` and `J` to move up and down the selected cells.
- `Ctrl+Enter` runs the currently selected cell.
- `Shift+Enter` runs the currently selected cell and makes a new one below.


## 2.2 IPython Basics

Below is an example of running some code.


```python
import numpy as np

data = {i : np.random.randn() for i in range(7)}

print(data)
```

    {0: -0.9247333079107192, 1: -0.8790015980782923, 2: 0.0903812076588127, 3: 1.7729559410349256, 4: -0.4912993201666181, 5: 1.2044170139944357, 6: 0.9866700994612237}


### Scalar types

The Python standard library has a few built-in types for single (scalar) values.
Dates and times are handled by the 'datetime' module and are discussed separately.
The rest of the types are `None`, `str`, `bytes`, `float`, `bool`, and `int`.

`int` and `float` are the numeric types.
`int` can handle arbitariliy large integers.
`float` is a double-precision (64-bit) value.
They can be expressed using scientific notation.


```python
a = 1e10
b = 1e-10
```

Integer division that does not result in an integer will yeild a floating-point number.


```python
2/3
```




    0.6666666666666666



There is also the floor operator which drops the remainder.


```python
3//2
```




    1



### Strings

Strings literals can be written using " " or ' ' and multi-line strings use triple-quotes ''' '''.
It appears to be most common to use a single quote.
Strings are immutable - they cannot be modified.


```python
a = 'This is a string.'
b = a.replace('string', 'longer string')
a
b
```




    'This is a longer string.'



Strings are a *sequence* of Unicode characters.


```python
snakiBoi = 'python'
for char in snakiBoi:
    print(char)
```

    p
    y
    t
    h
    o
    n


Strings can be indexed just like lists or tuples.


```python
snakiBoi[:3]
```




    'pyt'



Strings can be catenated using the `+` operator.


```python
"one pl" + "us two"
```




    'one plus two'



### Dates and times

The built-in 'datetime' module provides `datetime`, `date`, and `time` types.


```python
from datetime import datetime, date, time
dt  = datetime(2011, 10, 29, 20, 30, 21)
dt.day
```




    29




```python
dt.minute
```




    30




```python
dt.date()
```




    datetime.date(2011, 10, 29)




```python
dt.strftime('%m/%d/%Y %H:%M')
```




    '10/29/2011 20:30'



Datetime objects can be added and subtracted, too, returning a `datetime.timedelta` type.


```python
dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt2 - dt
delta
```




    datetime.timedelta(days=17, seconds=7179)



The `timedelta` object can be added to a `datetime` object to shift it.


```python
dt + delta
```




    datetime.datetime(2011, 11, 15, 22, 30)



### Control Flow

#### if, elif, and else


```python
x = -5
if x < 0:
    print('Negative.')
elif x == 0:
    print('Equal to zero.')
elif 0< x < 5:
    print('Positive, but smaller than 5.')
else:
    print('Positive and greater than or equal to 5.')
```

    Negative.


It is possible to chain comparisons.


```python
4 > 3 > 2 > 1
```




    True



#### for

`for` loops are for iterating over a collection or an iterator.
the standard syntax is

```python
for value in collection:
    # do something with `value`
```

A step in the loop can be skipped by using the `continue` keyword.


```python
sequence = [1, 2, None, 4, None, 6]
total = 0
for value in sequence:
    if value is None:
        continue
    total += value
total
```




    13



A `for` loop can be exited with the `break` keyword.


```python
sequence = [1, 2, 0, 4, 6, 5, 2, 1]
total_until_5 = 0
for value in sequence:
    if value == 5:
        break
    total_until_5 += value
total_until_5
```




    13



#### while


```python
x = 256
total = 0
while x > 0:
    if total > 500:
        break
    total += x
    x = x // 2
x
```




    4



#### range

The `range` function returns an *iterator* that yields a sequence of evenly spaced integers.
It includes the first value and excludes the last value.


```python
range(10)
```




    range(0, 10)




```python
list(range(10))
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
list(range(0, 20, 2))
```




    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]




```python
list(range(5, 0, -1))
```




    [5, 4, 3, 2, 1]




```python

```




    []



#### Ternary expressions

Single-line if-else statement.
The general syntax is:

```txt
value = <true-expr> if <condition> else <false-expr>
```


```python
x = 5
'Non-negative' if x >= 0 else 'Negative'
```




    'Non-negative'


