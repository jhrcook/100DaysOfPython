# Chpater 3. Built-in Data Structures, Functions, and Files

## 3.1 Data Structures and Sequences

### Tuple

Fixed-length, immutable sequence of Python objects.


```python
tup = 4, 5, 6
tup
```




    (4, 5, 6)




```python
nested_tup = (4, 5, 6), (7, 8)
nested_tup
```




    ((4, 5, 6), (7, 8))



Can convert a sequence or iterator to a tuple with the `tuple` function.


```python
tuple([4, 0, 2])
```




    (4, 0, 2)




```python
tuple('string')
```




    ('s', 't', 'r', 'i', 'n', 'g')



Indexing a tuple is standard.


```python
tup[1]
```




    5



While the tuple is not mutable, mutable objects within the tuple can be modified in place.


```python
tup = 'foo', [1, 2], True
tup[1].append(3)
tup
```




    ('foo', [1, 2, 3], True)



Tuples can be concatenated using the `+` operator or repeated with the `*` operator and an integer.
Note that the objects are not copied, just the references to them.


```python
(4, None, 'foo') + (6, 0) + ('bar', )
```




    (4, None, 'foo', 6, 0, 'bar')




```python
('foo', 'bar') * 4
```




    ('foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar')




```python
tup = ([1, 2], 'foo')
tup = tup * 4
tup[0].append(3)
tup
```




    ([1, 2, 3], 'foo', [1, 2, 3], 'foo', [1, 2, 3], 'foo', [1, 2, 3], 'foo')



Tuples can be unpacked by position.


```python
tup = (4, 5, 6)
a, b, c = tup
b
```




    5




```python
tup = 4, 5, (6, 7)
a, b, (c, d) = tup
d
```




    7




```python
a, b = 1, 2
b, a = a, b
a
```




    2



A common use of variable unpacking is iterating over sequences of tuples or lists.


```python
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c, in seq:
    print(f'a={a}, b={b}, c={c}')
```

    a=1, b=2, c=3
    a=4, b=5, c=6
    a=7, b=8, c=9


Another common use is return multiple values from a function (discuessed later).

There is specific syntax if you only want the first few values and put the rest into another tuple.


```python
values = tuple(range(5))
a, b, *rest = values
a
```




    0




```python
b
```




    1




```python
rest
```




    [2, 3, 4]



If you don't want the other values, the convention is to assign them to a variable called `_`.


```python
a, b, *_ = values
```


```python
a, b, *_ = values
```

### List

Variable-lengthed and the contents can be modified in place.


```python
a_list = [2, 3, 7, None]
a_list
```




    [2, 3, 7, None]




```python
tup = 'foo', 'bar', 'baz'
b_list = list(tup)
b_list
```




    ['foo', 'bar', 'baz']




```python
b_list[1]
```




    'bar'




```python
b_list[1] = 'peekaboo'
b_list
```




    ['foo', 'peekaboo', 'baz']



Elements can be added, inserted, removed, etc.


```python
b_list.append('dwarf')
b_list
```




    ['foo', 'peekaboo', 'baz', 'dwarf']




```python
b_list.insert(1, 'red')
b_list
```




    ['foo', 'red', 'peekaboo', 'baz', 'dwarf']




```python
b_list.pop(2)
```




    'peekaboo'




```python
b_list
```




    ['foo', 'red', 'baz', 'dwarf']




```python
b_list.append('foo')
b_list.remove('foo')
b_list
```




    ['red', 'baz', 'dwarf', 'foo']



Lists can be concatenated using the `+` operator.
Alternatively, an existing list can be extended using the `extend` method and passing another list.


```python
[4, None, 'foo'] + [7, 8, (2, 3)]
```




    [4, None, 'foo', 7, 8, (2, 3)]




```python
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
x
```




    [4, None, 'foo', 7, 8, (2, 3)]



A list can be sorted in place.


```python
a = [7, 2, 5, 1, 3]
a.sort()
a
```




    [1, 2, 3, 5, 7]



Sort has a few options, one being `key` that allows us to define the function used for sorting.


```python
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
b
```




    ['He', 'saw', 'six', 'small', 'foxes']



The 'bisect' module implements binary search and insertion into a sorted list.
This finds the location of where to insert a new element to maintain the sorted list.
`bisect.bisect(list, value)` finds the location for where the element should be added, `bisect.insort` actually inserts the element.


```python
import bisect
c = [1, 2, 2, 2, 2, 3, 4, 7]
bisect.bisect(c, 2)
```




    5




```python
bisect.bisect(c, 5)
```




    7




```python
bisect.insort(c, 6)
c
```




    [1, 2, 2, 2, 2, 3, 4, 6, 7]



Specific elements of a list can be accessed using *slicing*.


```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[1:5]
```




    [2, 3, 7, 5]




```python
seq[3:4] = [6, 7, 8, 9]
```


```python
seq
```




    [7, 2, 3, 6, 7, 8, 9, 5, 6, 0, 1]




```python
seq[:5]
```




    [7, 2, 3, 6, 7]




```python
seq[5:]
```




    [8, 9, 5, 6, 0, 1]




```python
seq[-4:]
```




    [5, 6, 0, 1]




```python
seq[-6:-2]
```




    [8, 9, 5, 6]



A step can also be included after another `:`.


```python
seq[::2]
```




    [7, 3, 7, 9, 6, 1]




```python
seq[::-1]
```




    [1, 0, 6, 5, 9, 8, 7, 6, 3, 2, 7]



### Built-in sequence functions

There are a number of useful built-in functions specifically for sequence types.

`enumerate` builds an iterator of the sequence to return each value and its index.


```python
some_list = ['foo', 'bar', 'baz']
for i, v in enumerate(some_list):
    print(f'{i}. {v}')
```

    0. foo
    1. bar
    2. baz


`sorted` returns a *new* sorted list from the elements of a sequence.


```python
sorted([7, 1, 2, 6, 0, 3, 2])
```




    [0, 1, 2, 2, 3, 6, 7]




```python
sorted('horse race')
```




    [' ', 'a', 'c', 'e', 'e', 'h', 'o', 'r', 'r', 's']



`zip` pairs up elements of a number of sequences to create a list of tuples.


```python
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two','three']
zipped = zip(seq1, seq2)
list(zipped)
```




    [('foo', 'one'), ('bar', 'two'), ('baz', 'three')]




```python
seq2 = ['one', 'two']
list(zip(seq1, seq2))
```




    [('foo', 'one'), ('bar', 'two')]




```python
for a, b in zip(seq1, seq2):
    print(f'{a} - {b}')
```

    foo - one
    bar - two


A list of tuples can also be "un`zip`ped".


```python
pitchers = [
    ('Nolan', 'Ryan'),
    ('Roger', 'Clemens'),
    ('Curt', 'Schilling')
]
first_names, last_names = zip(*pitchers)
first_names
```




    ('Nolan', 'Roger', 'Curt')




```python
last_names
```




    ('Ryan', 'Clemens', 'Schilling')



The `reversed` function iterates over the sequence in reverse order.


```python
list(reversed(range(10)))
```




    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]



### Dictionaries
