#                  THEROY BASED QUESTIONS FROM NumPY :-


#🔹 🔰 Basic Level
#1️⃣ What is NumPy?
'''NumPy is a Python library used for working with arrays and performing fast mathematical 
operations.'''

#2️⃣ Difference between Python list and NumPy array?


#3️⃣ How to create a NumPy array?
import numpy as np
a = np.array([1,2,3])
a

#4️⃣ Difference between np.zeros() and np.ones()?

'''np.zeros() → Creates array filled with 0
np.ones() → Creates array filled with 1'''

#5️⃣ What does np.arange() do?
'''Creates array with values in a range.
np.arange(1,6)  # [1 2 3 4 5]'''

#6️⃣ What is np.eye()?
'''Creates an identity matrix (1 on diagonal, 0 elsewhere).'''

#7️⃣ Difference between ndim, shape, and size?

'''ndim → Number of dimensions

shape → Rows and columns

size → Total number of elements'''

#8️⃣ How to check data type?
a.dtype

#🔹 🟡 Lower-Intermediate
#9️⃣ What is broadcasting?

'''Broadcasting allows NumPy to perform operations on arrays of different shapes by automatically adjusting their sizes.'''

#🔟 Difference between copy() and b = a?
'''b = a → Both share same memory
a.copy() → Creates new independent array'''

#1️⃣1️⃣ Difference between reshape() and flatten()?

'''reshape() → Changes shape

flatten() → Converts to 1D array'''

#1️⃣2️⃣ How to generate random integers?
#np.random.randint(1,10)

#1️⃣3️⃣ Difference between rand() and randint()?

'''rand() → Random float (0–1)

randint() → Random integer'''

#1️⃣4️⃣ How to change data type?
#a.astype(float)

#🔹 🟠 Moderate Level
#1️⃣5️⃣ What happens if shapes are incompatible in broadcasting?

#NumPy gives a ValueError.

#1️⃣6️⃣ Difference between view() and copy()?
'''view() → Shares memory
copy() → New memory'''

#1️⃣7️⃣ How to perform element-wise operations?
'''a + b
a * b
Operations are applied to each element.'''

#1️⃣8️⃣ Difference between np.sum(a) and a.sum()?
'''Both give same result.
One is NumPy function, other is array method.'''

#1️⃣9️⃣ How to select elements?
'''a[0]        # First element
a[1:3]      # Slicing
a[0,1]      # 2D indexing'''

#2️⃣0️⃣ Create 3×3 random matrix (0–1):
#np.random.rand(3,3)


#2️⃣1️⃣ What is the difference between ravel() and flatten()?
'''flatten() → Returns a copy
ravel() → Returns a view (if possible)
👉 ravel() is faster
👉 flatten() is safer'''

#2️⃣2️⃣ What is vectorization in NumPy?

'''Vectorization means performing operations on entire arrays without using loops.
example:-
a + b
It is faster than using for loop.'''

#2️⃣3️⃣ What is slicing in NumPy?
'''Selecting a portion of array using index ranges.
a[1:4]'''

#2️⃣4️⃣ What is fancy indexing?
'''Selecting elements using list or array of indices.
a[[0,2,4]]'''

#2️⃣5️⃣ What is boolean indexing?
'''Selecting elements based on condition.
a[a > 5]'''

#2️⃣6️⃣ What is difference between np.vstack() and np.hstack()?
'''vstack() → Stack vertically (row-wise)
hstack() → Stack horizontally (column-wise)'''

#2️⃣7️⃣ What is np.concatenate()?
#Joins two or more arrays along a given axis.

#2️⃣8️⃣ What is axis in NumPy?
'''Axis defines direction of operation.
axis=0 → Column-wise
axis=1 → Row-wise'''

#2️⃣9️⃣ What is np.where()?
'''Returns indices where condition is true.
np.where(a > 5)'''

#3️⃣0️⃣ What is difference between np.dot() and * operator?
'''* → Element-wise multiplication
np.dot() → Matrix multiplication'''

#3️⃣1️⃣ What is np.clip()?
'''Limits values within a range.
np.clip(a, 0, 10)'''

#3️⃣2️⃣ What is np.unique()?
#Returns unique elements from array.

#3️⃣3️⃣ What is np.argsort()?
'''Returns indices that would sort the array.'''

#3️⃣4️⃣ What is difference between np.mean() and np.average()?
'''mean() → Simple average
average() → Can use weights'''

#3️⃣5️⃣ What is broadcasting rule in simple words?
'''Two arrays are compatible if:
Dimensions are equal
OR
One of them is 1
Otherwise → Error'''

#3️⃣6️⃣ How to check if two arrays are equal?
#np.array_equal(a, b)

#3️⃣7️⃣ What is np.linspace()?
'''Generates evenly spaced numbers between two values.
np.linspace(0,10,5)'''

#3️⃣8️⃣ What is difference between np.empty() and np.zeros()?
'''zeros() → Fills with 0
empty() → Creates array without initializing values (random garbage values)'''

#3️⃣9️⃣ What is np.diag()?
'''Extracts or creates diagonal of matrix.'''

#4️⃣0️⃣ What is memory efficiency advantage of NumPy?
'''NumPy arrays store data in continuous memory blocks, making them:
Faster
Less memory consuming
Efficient for mathematical computation'''


# PRACTICAL BASED / CODING BASED QUESTIONS FROM NumPy:-

#🟢 BASIC LEVEL (1–10)
#1️⃣ Create a NumPy array of numbers from 1 to 10.
import numpy as np
#arr1=np.arange([1,11])
a=np.arange(1,11)
print(a)

#2️⃣ Create a 3×3 array filled with zeros.
import numpy as np
a=np.zeros((3,3))
print(a)

#3️⃣ Create a 2×4 array filled with ones (integer type).
import numpy as np
a=np.ones((2,4),dtype=int)
print(a)

#4️⃣ Create an identity matrix of size 4×4.
import numpy as np
a=np.array([4,4])
print(a)

#5️⃣ Generate 5 random integers between 1 and 20.
import numpy as np
a=np.random.randint(1,21,size=5)
print(a)

#6️⃣ Find the shape and size of this array:
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.shape)
print(a.size)


#7️⃣ Extract the first row from a 2D array.
import numpy as np
a=np.array([[1,2,3,4],[4,5,6,7]])
print(a[0])

#8️⃣ Extract the last element from:
a = np.array([10,20,30,40,50])
print(a[-1])


#9️⃣ Reshape this array into 2×3:
a = np.array([1,2,3,4,5,6])
a.resize([2,3])
print(a)

# OR
import numpy as np
a = a.reshape(2,3)
print(a)


#🔟 Convert array data type from int to float.
import numpy as np
a=np.array([1,2,3,4,5,6])
a=a.astype(float)
print(a)

#🔹 🟡 MODERATE LEVEL (11–20)
#1️⃣1️⃣ Find the sum of all elements in an array.
import numpy as np
a=np.array([1,2,3,4,5,6])
print(np.sum(a))


#1️⃣2️⃣ Find the mean of each column in a 2D array.
import numpy as np
a=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(np.mean(a,axis=0))

#1️⃣3️⃣ Multiply two arrays element-wise.
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = a * b

print(result)

#OR in 2D array:-
import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

result = a * b

print(result)


#1️⃣4️⃣ Perform matrix multiplication of two 2×2 matrices.


#1️⃣5️⃣ Find the maximum and minimum value in array.
import numpy as np
a=np.array([1,2,3,4,6,5])
print('Maximum value of this array : ',np.max(a))
print('Minimum value of this array : ',np.min(a))


#1️⃣6️⃣ Find indices where values are greater than 5.

print(np.argmax(a))

#1️⃣7️⃣ Reverse a NumPy array.
import numpy as np
a=np.array([1,2,3,4,5])
b=(a[::-1])
print(b)


#1️⃣8️⃣ Remove duplicate values from array.
import numpy as np
num=[9,10,8,5,6,5,8]
num2=np.unique(num)
print(num2)


#1️⃣9️⃣ Sort an array in descending order.
import numpy as np
num=np.array[9,19,87,12,3,40,100]
b=np.sort(num)          # short in ascending order
print(b)


c=np.sort(num)[::-1]        #short in descending order
print(c)

#2️⃣0️⃣ Replace all values greater than 10 with 10.
a = np.array([2,14,4,30,5,9])
a[a > 10] = 10
print(a)

# OR :--

a = np.array([2,14,4,30,5,9])
a = np.where(a > 10, 10, a)
print(a)


#🔹 🟠 ADVANCED LEVEL (21–30)
#2️⃣1️⃣ Normalize an array (scale values between 0 and 1).

import numpy as np
a = np.array([10, 20, 30, 40, 50])
normalized = (a - np.min(a)) / (np.max(a) - np.min(a))
print(normalized)


#2️⃣2️⃣ Find the second largest number in an array.
num=np.array([1,3,2,5,4,10])
max_num=np.sort(num)[-2]
print(max_num)

#method 2 :- using partation method (most important )
second_largest = np.partition(num, -2)[-2]
print(second_largest)


#2️⃣3️⃣ Count frequency of each unique element.
a=np.array([9,9,8,5,7,4,2,3,4])
unique_values=np.unique(a)
print(unique_values)
count_unique=np.count(unique_values)
print(count_unique)

#2️⃣4️⃣ Flatten a 3D array into 1D.
import numpy as np
a=np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])
print(a)
b=a.flatten()
print(b)

#2️⃣5️⃣ Stack two arrays vertically and horizontally.
import numpy as np
a=np.array([1,2,3,4,5])
b=np.array([6,7,8,9,10])
c=np.vstack((a, b))
print(c)
d=np.hstack((a,b))
print(d)

#2️⃣6️⃣ Create a checkerboard pattern (8×8 matrix of 0 and 1).
import numpy as np                          # Import NumPy
checkerboard = np.zeros((8, 8), dtype=int)  # Create 8x8 matrix filled with 0
checkerboard[1::2, ::2] = 1                 # Set 1s in alternating pattern (odd rows, even columns)
checkerboard[::2, 1::2] = 1                 # Set 1s in alternating pattern (even rows, odd columns)
print(checkerboard)                         # Print final checkerboard

#2️⃣7️⃣ Create a diagonal matrix from given array.
import numpy as np
a=np.array([2,3,4,5,6,7,8,9,11,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
digonal_matrix=np.eye(a)
print(digonal_matrix)

#2️⃣8️⃣ Replace NaN values with 0.
import numpy as np
a = np.array([11,np.nan,np.nan,22,20,np.nan])  # Create array with NaN values
a[np.isnan(a)] = 0                       # Replace NaN values with 0 using boolean masking
print(a)                                 # Print updated array

#2️⃣9️⃣ Find common elements between two arrays.
import numpy as np
a=np.array([6,3,5,7,9])
b=np.array([9,6,7,8,3])
c=np.intersect1d(a,b)
print(c)

# without using built-in function.
import numpy as np
a=np.array([1,2,3,4,5,12])
b=np.array([7,6,5,1,2,3,4])
common = a[np.isin(a, b)]
print(common)


#3️⃣0️⃣ Compute dot product of two vectors.
import numpy as np
a=np.array([1,2,3,4,5])
b=np.array([9,8,7,7,6])
c=np.dot(a,b)
print(c)

#OR
import numpy as np
a=np.array([1,2,3,3,4])
b=np.array([7,6,5,4,3])
c = a @ b
print(c)

# OR
import numpy as np
a=np.array([1,2,3,3,4])
b=np.array([7,6,5,4,3])
c =np.sum(a*b)
print(c)



#🔴 INTERVIEW LEVEL (31–40)
#These are commonly asked in Data Analyst / ML interviews.
#3️⃣1️⃣ Find row-wise sum of 2D array.
import numpy as np
a=np.array([[1,2,3,4,5],[10,2,3,5,1]])
add=np.sum(a,axis=1)
print(add)


#3️⃣2️⃣ Standardize an array (mean = 0, std = 1).
import numpy as np
a=np.array([1,2,3,4,5])
mean_a=np.mean(a)
print(mean_a)
std_a=np.std(a)
print(std_a)
standardized = (a - mean_a) / std_a  # Apply standardization formula
print(standardized)                  # Print standardized array

#3️⃣3️⃣ Find correlation coefficient between two arrays.
'''Correlation measures the strength and direction of linear relationship between two variables.
Correlation tells us how strongly two variables are related.     Value range: -1 to +1
Value	Meaning
+1	    Perfect positive correlation
0	    No correlation
-1	    Perfect negative correlation'''

import numpy as np
a=np.array([1,3,2,4,5])
b=np.array([4,1,2,6,5])
c=np.corrcoef(a,b)
print(c)

#3️⃣4️⃣ Create sliding window of size 3 from array.
[1,2,3,4,5]

[[1,2,3],
 [2,3,4],
 [3,4,5]]

#3️⃣5️⃣ Check if two arrays are equal.


#3️⃣6️⃣ Find top 3 largest values.
a=np.array([10,20,30,40,50])
b=np.sort(a)
top_3=b[-3::]
print(top_3)

#3️⃣7️⃣ Count occurrences of each value.
import numpy as np                          # Import NumPy
a = np.array([1,2,2,3,3,4,4,4,6,6])         # Create array
values, counts = np.unique(a, return_counts=True)   # Get unique values and their counts
print(values)                               # Unique values
print(counts)                               # Occurrence count

#3️⃣8️⃣ Convert 1D array to 2D column vector.
import numpy as np
a=np.array([1,2,3,4,5,6,7,8,8,9,10,11])
b=a.reshape(4,3)
print(b)

#3️⃣9️⃣ Randomly shuffle rows of a 2D array.
# value will change whenever you  run this code , it is suffle
import numpy as np
a = np.array([[1,2],
              [3,4],
              [5,6],
              [7,8]])
np.random.shuffle(a)   #→ shuffles along first axis (rows)
print(a)


#4️⃣0️⃣ Generate 1000 random numbers and calculate their mean and standard deviation.

import numpy as np
# Generate 1000 random numbers between 0 and 1
a = np.random.rand(1000)
# Calculate mean
mean_value = np.mean(a)
# Calculate standard deviation
std_value = np.std(a)
print("Mean:", mean_value)
print("Standard Deviation:", std_value)
