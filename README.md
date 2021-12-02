# AI-Mafia---Machine-Learning


![](https://bids.berkeley.edu/sites/default/files/styles/400x225/public/projects/numpy_project_page.jpg?itok=flrdydei)


# What is Numpy?
---

NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays.

It is the fundamental package for scientific computing with Python. It contains among other things:
- a powerful N-dimensional array object
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities

Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data.
Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.
# Installation
---

- **Mac** and **Linux** users can install NumPy via pip command:
    ```
    pip install numpy
    ```

- **Windows** does not have any package manager analogous to that in linux or mac. Please download the pre-built windows installer for NumPy from here (according to your system configuration and Python version). And then install the packages manually.


Once you are done, just type this in python interpreter:
```python
import numpy as np
```

If you are still experiencing some issues, then Stack Overflow is your friend!

If no errors appear,congo! You have successfully installed NumPy. 
Lets move ahead...

## Arrays in NumPy
---
NumPy’s main object is the homogeneous multidimensional array.
- It is a table of elements (usually numbers), all of the same type, indexed by a tuple of positive integers.
- In NumPy dimensions are called *axes*. The number of axes is *rank*.
- NumPy’s array class is called **ndarray**. It is also known by the alias **array**. 

For example:
```python
[[ 1, 2, 3],
 [ 4, 2, 5]]
```  
This array has:
- rank = 2 (as it is 2-dimensional or it has 2 axes)
- first dimension(axis) length = 2, second dimension has length = 3.
- overall shape can be expressed as: (2, 3)

**Consider the example shown below:**
import numpy as np

# creating array object
arr = np.array([[ 1, 2, 3],
                [ 4, 2, 5]])

# printing type of arr object
print("Array is of type: ", type(arr))

# printing array dimensions (axes)
print("No. of dimensions: ", arr.ndim)

# printing shape of array
print("Shape of array: ", arr.shape)

# printing size (total number of elements) of array
print("Size of array: ", arr.size)

# printing type of elements in array
print("Array stores elements of type: ", arr.dtype)
## Array creation
---
There are various ways to create arrays in NumPy.

- For example, you can create an array from a regular Python **list** or **tuple** using the **array** function. The type of the resulting array is deduced from the type of the elements in the sequences.


- Often, the elements of an array are originally unknown, but its size is known. Hence, NumPy offers several functions to create arrays with **initial placeholder content**. These minimize the necessity of growing arrays, an expensive operation. **For example:** np.zeros, np.ones, np.full, np.empty, etc.


- To create sequences of numbers, NumPy provides a function analogous to range that returns arrays instead of lists.
   - **arange:** returns evenly spaced values within a given interval. **step** size is specified.
   - **linspace:** returns evenly spaced values within a given interval. **num** no. of elements are returned.
   
   
- **Reshaping array:** We can use **reshape** method to reshape an array. Consider an array with shape (a1, a2, a3, ..., aN). We can reshape and convert it into another array with shape (b1, b2, b3, ....., bM). The only required condition is:   <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*a1 x a2 x a3 .... x aN = b1 x b2 x b3 .... x bM *. (i.e original size of array remains unchanged.)


- **Flatten array:** We can use **flatten** method to get a copy of array collapsed into **one dimension**. It accepts *order* argument. Default value is 'C' (for row-major order). Use 'F' for column major order.


**Note:** Type of array can be explicitly defined while creating array.

**Consider the example shown below:**
# creating array from list with type float
a = np.array([[1, 2, 4], [5, 8, 7]], dtype = 'float')
print("Array created using passed list:\n", a)

# creating array from tuple
b = np.array((1 , 3, 2))
print("\nArray created using passed tuple:\n", b)

# creating a 3X4 array with all zeros
c = np.zeros((3, 4))
print("\nAn array initialized with all zeros:\n", c)

# create a constant value array of complex type
d = np.full((3, 3), 6, dtype = 'complex')
print("\nAn array initialized with all 6s. Array type is complex:\n", d)

# create an array with random values
e = np.random.random((2,2))
print("\nA random array:\n", e)

# create a sequence of integers from 0 to 30 with steps of 5
f = np.arange(0, 30, 5)
print("\nA sequential array with steps of 5:\n", f)

# create a sequence of 10 values in range 0 to 5
g = np.linspace(0, 5, 10)
print("\nA sequential array with 10 values between 0 and 5:\n", g)

# reshaping 3X4 array to 2X2X3 array
arr = np.array([[1, 2, 3, 4],
                [5, 2, 4, 2],
                [1, 2, 0, 1]])
newarr = arr.reshape(2, 2, 3)
print("\nOriginal array:\n", arr)
print("Reshaped array:\n", newarr)

# flatten array
arr = np.array([[1, 2, 3], [4, 5, 6]])
flarr = arr.flatten()
print("\nOriginal array:\n", arr)
print("Fattened array:\n", flarr)
## Array Indexing
---

Knowing the basics of array indexing is important for analysing and manipulating the array object.
NumPy offers many ways to do array indexing.

- **Slicing:** Just like lists in python, NumPy arrays can be sliced. As arrays can be multidimensional, you need to specify a slice for each dimension of the array.


- **Integer array indexing:** In this method, lists are passed for indexing for each dimension. One to one mapping of corresponding elements is done to construct a new arbitrary array.


- **Boolean array indexing:** This method is used when we want to pick elements from array which satisfy some condition.

**Consider the exemplar program given below:**
# an exemplar array
arr = np.array([[-1, 2, 0, 4],
                [4, -0.5, 6, 0],
                [2.6, 0, 7, 8],
                [3, -7, 4, 2.0]])

# slicing
temp = arr[:2, ::2]
print("Array with first 2 rows and alternate columns(0 and 2):\n", temp)

# integer array indexing example
temp = arr[[0, 1, 2, 3], [3, 2, 1, 0]]
print("\nElements at indices (0, 3), (1, 2), (2, 1), (3, 0):\n", temp)

# boolean array indexing example
cond = arr > 0    # cond is a boolean array
temp = arr[cond]
print("\nElements greater than 0:\n", temp)
## Basic operations
---

Plethora of built-in arithmetic functions are provided in NumPy.

- **Operations on single array:** We can use overloaded arithmetic operators to do element-wise operation on array to create a new array. In case of +=, -=, *= operators, the exsisting array is modified.

**Here are some examples:**
a = np.array([1, 2, 5, 3])

# add 1 to every element
print("Adding 1 to every element:", a+1)

# subtract 3 from each element
print("Subtracting 3 from each element:", a-3)

# multiply each element by 10
print("Multiplying each element by 10:", a*10)

# square each element
print("Squaring each element:", a**2)

# modify existing array
a *= 2
print("Doubled each element of original array:", a)

# transpose of array 
a = np.array([[1, 2, 3], [3, 4, 5], [9, 6, 0]])
print("\nOriginal array:\n", a)
print("Transpose of array:\n", a.T)

- **Unary operators:** Many unary operations are provided as a method of **ndarray** class. This includes sum, min, max, etc. These functions can also be applied row-wise or column-wise by setting an axis parameter.

**Here are some examples:**
arr = np.array([[1, 5, 6], 
                [4, 7, 2], 
                [3, 1, 9]])

# maximum element of array
print("Largest element is:", arr.max())
print("Row-wise maximum elements:", arr.max(axis = 1))

# minimum element of array
print("Column-wise minimum elements:", arr.min(axis = 0))

# sum of array elements
print("Sum of all array elements:", arr.sum())

# cumulative sum along each row
print("Cumulative sum along each row:\n", arr.cumsum(axis = 1))
---
- **Binary operators:** These operations apply on array elementwise and a new array is created. You can use all basic arithmetic operators like +, -, /, *, etc. In case of +=, -=, *= operators, the exsisting array is modified.

**Here are some examples:**
a = np.array([[1, 2], 
              [3, 4]])
b = np.array([[4, 3], 
              [2, 1]])

# add arrays
print("Array sum:\n", a + b)

# multiply arrays (elementwise multiplication)
print("Array multiplication:\n", a*b)

# matrix multiplication
print("Matrix multiplication:\n", a.dot(b))

- **Universal functions (ufunc):** NumPy provides familiar mathematical functions such as sin, cos, exp, etc. These functions also operate elementwise on an array, producing an array as output.

**Note:** All the operations we did above using overloaded operators can be done using ufuncs like np.add, np.subtract, np.multiply, np.divide, np.sum, etc.
# create an array of sine values
a = np.array([0, np.pi/2, np.pi])
print("Sine values of array elements:", np.sin(a))

# exponential values
a = np.array([0, 1, 2, 3])
print("Exponent of array elements:", np.exp(a))

# square root of array values
print("Square root of array elements:", np.sqrt(a))
## Sorting array
There is a simple **np.sort** method for sorting NumPy arrays.
Let's explore it a bit.
a = np.array([[1, 4, 2],
              [3, 4,6],
              [0, -1, 5]])

# sorted array
print("Array elements in sorted order:\n", np.sort(a, axis = None))

# sort array row-wise
print("Row-wise sorted array:\n", np.sort(a, axis = 1))

# specify sort algorithm
print("Column wise sort by applying merge-sort:\n", np.sort(a, axis = 0, kind = 'mergesort'))

# example to show sorting of structured array
## set alias names for dtypes
dtypes = [('name', 'S10'), ('grad_year', int), ('cgpa', float)]
## values to be put in array
values = [('Hrithik', 2009, 8.5), ('Ajay', 2008, 8.7), ('Pankaj', 2008, 7.9), ('Aakash', 2009, 9.0)]
## creating array
arr = np.array(values, dtype = dtypes)
print("\nArray sorted by names:\n", np.sort(arr, order = 'name'))
print("Array sorted by grauation year and then cgpa:\n", np.sort(arr, order = ['grad_year', 'cgpa']))
# Stacking and Splitting

Several arrays can be stacked together along different axes.

- **np.vstack:** To stack arrays along vertical axis.

- **np.hstack:** To stack arrays along horizontal axis.

- **np.column_stack:** To stack 1-D arrays as columns into 2-D arrays.

- **np.concatenate:** To stack arrays along specified axis (axis is passed as argument).
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

# vertical stacking
print("Vertical stacking:\n", np.vstack((a, b)))

# horizontal stacking
print("\nHorizontal stacking:\n", np.hstack((a, b)))

c = [5, 6]

# stacking columns
print("\nColumn stacking:\n", np.column_stack((a, c)))

# concatenation method 
print("\nConcatenating to 2nd axis:\n", np.concatenate((a, b), 1))
For splitting, we have these fuctions:

- **np.hsplit:** Split array along horizontal axis.

- **np.vsplit:** Split array along vertical axis.

- **np.array_split:** Split array along specified axis.
a = np.array([[1, 3, 5, 7, 9, 11],
              [2, 4, 6, 8, 10, 12]])

# horizontal splitting
print("Splitting along horizontal axis into 2 parts:\n", np.hsplit(a, 2))

# vertical splitting
print("\nSplitting along vertical axis into 2 parts:\n", np.vsplit(a, 2))
# Some More Numpy Functions - Statistics
- min, max
- mean 
- median
- average
- variance
- standard devidation
a = np.array([[1,2,3,4],[7,6,2,0]])
print(a)
print(np.min(a))
#Specify axis for the direction in case of multidim array
print(np.min(a,axis=0))
print(np.min(a,axis=1))
b = np.array([1,2,3,4,5])
m = sum(b)/5
print(m)

print(np.mean(b))
print(np.mean(a,axis=0))
print(np.mean(a,axis=1))
c = np.array([1,5,4,2,0])
print(np.median(c))

# Mean vs Average is Weighted
print(np.mean(c))

# Weights 
w = np.array([1,1,1,1,1])
print(np.average(c,weights=w))

# weighted mean => n1*w1 + n2*w2 / n1+n2 

# Standard Deviation
u = np.mean(c)
myStd = np.sqrt(np.mean(abs(c-u)**2))
print(myStd)

#Inbuilt Function
dev= np.std(c)
print(dev)

#Variance
print(myStd**2)
print(np.var(c))
# Numpy Random Module

- rand : Random values in a given shape.
- randn : Return a sample (or samples) from the “standard normal” distribution.
- randint : Return random integers from low (inclusive) to high (exclusive).
- random : Return random floats in the half-open interval [0.0, 1.0) 
- choice : Generates a random sample from a given 1-D array
- Shuffle : Shuffles the contents of a sequence
a = np.arange(10) + 5
print(a)

np.random.seed(1)
np.random.shuffle(a)
print(a)

#Returns values from a Standard Normal Distributions
a = np.random.randn(2,3)
print(a)

a = np.random.randint(5,10,3)
print(a)

#Randoly pick one element from a array
element = np.random.choice([1,4,3,2,11,27])
print(element)
