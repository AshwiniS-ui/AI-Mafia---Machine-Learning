{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "221e449a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hello world!\n"
     ]
    }
   ],
   "source": [
    "print(\"hello world!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2476e96b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: numpy in c:\\users\\ashwini s\\anaconda3\\lib\\site-packages (1.19.5)\n"
     ]
    }
   ],
   "source": [
    "!pip install numpy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a721a1bc",
   "metadata": {},
   "source": [
    "# We are going to learn numpy now!\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f24563c",
   "metadata": {},
   "source": [
    "Numpy!!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "3a7ead03",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "8b4f4264",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.19.5\n",
      "blas_mkl_info:\n",
      "  NOT AVAILABLE\n",
      "blis_info:\n",
      "  NOT AVAILABLE\n",
      "openblas_info:\n",
      "    library_dirs = ['D:\\\\a\\\\1\\\\s\\\\numpy\\\\build\\\\openblas_info']\n",
      "    libraries = ['openblas_info']\n",
      "    language = f77\n",
      "    define_macros = [('HAVE_CBLAS', None)]\n",
      "blas_opt_info:\n",
      "    library_dirs = ['D:\\\\a\\\\1\\\\s\\\\numpy\\\\build\\\\openblas_info']\n",
      "    libraries = ['openblas_info']\n",
      "    language = f77\n",
      "    define_macros = [('HAVE_CBLAS', None)]\n",
      "lapack_mkl_info:\n",
      "  NOT AVAILABLE\n",
      "openblas_lapack_info:\n",
      "    library_dirs = ['D:\\\\a\\\\1\\\\s\\\\numpy\\\\build\\\\openblas_lapack_info']\n",
      "    libraries = ['openblas_lapack_info']\n",
      "    language = f77\n",
      "    define_macros = [('HAVE_CBLAS', None)]\n",
      "lapack_opt_info:\n",
      "    library_dirs = ['D:\\\\a\\\\1\\\\s\\\\numpy\\\\build\\\\openblas_lapack_info']\n",
      "    libraries = ['openblas_lapack_info']\n",
      "    language = f77\n",
      "    define_macros = [('HAVE_CBLAS', None)]\n"
     ]
    }
   ],
   "source": [
    "print(np.__version__)\n",
    "np.show_config()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "8bdf1483",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "Z = np.zeros(10)\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "8819e5b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "800 bytes\n"
     ]
    }
   ],
   "source": [
    "Z = np.zeros((10,10))\n",
    "print(\"%d bytes\" % (Z.size * Z.itemsize))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "b89eb53e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "Z = np.zeros(10)\n",
    "Z[4] = 1\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "7a8adec7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33\n",
      " 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]\n"
     ]
    }
   ],
   "source": [
    "Z = np.arange(10,50)\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "e8ba3d93",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26\n",
      " 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2\n",
      "  1  0]\n"
     ]
    }
   ],
   "source": [
    "Z = np.arange(50)\n",
    "Z = Z[::-1]\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "13af369c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n"
     ]
    }
   ],
   "source": [
    "Z = np.arange(9).reshape(3,3)\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "6e08d2f5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(array([0, 1, 4], dtype=int64),)\n"
     ]
    }
   ],
   "source": [
    "nz = np.nonzero([1,2,0,0,4,0])\n",
    "print(nz)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "7ed89a5e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]]\n"
     ]
    }
   ],
   "source": [
    "Z = np.eye(3)\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "58a1cdcd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[0.88475981 0.79283792 0.3336451 ]\n",
      "  [0.04207673 0.61929374 0.57930564]\n",
      "  [0.00707491 0.48841006 0.59986087]]\n",
      "\n",
      " [[0.39370792 0.47031835 0.39972801]\n",
      "  [0.52281437 0.89983616 0.24253961]\n",
      "  [0.47191451 0.64608253 0.42439965]]\n",
      "\n",
      " [[0.45189245 0.55435436 0.67859321]\n",
      "  [0.39988688 0.51941059 0.83141309]\n",
      "  [0.68740838 0.8210891  0.47288468]]]\n"
     ]
    }
   ],
   "source": [
    "Z = np.random.random((3,3,3))\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "983fd981",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.004919124064745595 0.9984874478139694\n"
     ]
    }
   ],
   "source": [
    "Z = np.random.random((10,10))\n",
    "Zmin, Zmax = Z.min(), Z.max()\n",
    "print(Zmin, Zmax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "9cacfb2e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.555932125913559\n"
     ]
    }
   ],
   "source": [
    "Z = np.random.random(30)\n",
    "m = Z.mean()\n",
    "print(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "acaa96af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
      " [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]\n"
     ]
    }
   ],
   "source": [
    "Z = np.ones((10,10))\n",
    "Z[1:-1,1:-1] = 0\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "2e47b8dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 1. 1. 1. 1. 1. 0.]\n",
      " [0. 1. 1. 1. 1. 1. 0.]\n",
      " [0. 1. 1. 1. 1. 1. 0.]\n",
      " [0. 1. 1. 1. 1. 1. 0.]\n",
      " [0. 1. 1. 1. 1. 1. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0.]]\n"
     ]
    }
   ],
   "source": [
    "Z = np.ones((5,5))\n",
    "Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "fca45e86",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "nan\n",
      "False\n",
      "False\n",
      "nan\n",
      "True\n",
      "False\n"
     ]
    }
   ],
   "source": [
    "print(0 * np.nan)\n",
    "print(np.nan == np.nan)\n",
    "print(np.inf > np.nan)\n",
    "print(np.nan - np.nan)\n",
    "print(np.nan in set([np.nan]))\n",
    "print(0.3 == 3 * 0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "4e0f1207",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 0 0 0 0]\n",
      " [1 0 0 0 0]\n",
      " [0 2 0 0 0]\n",
      " [0 0 3 0 0]\n",
      " [0 0 0 4 0]]\n"
     ]
    }
   ],
   "source": [
    "Z = np.diag(1+np.arange(4),k=-1)\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "663d4910",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 0 1 0 1 0 1]\n",
      " [1 0 1 0 1 0 1 0]\n",
      " [0 1 0 1 0 1 0 1]\n",
      " [1 0 1 0 1 0 1 0]\n",
      " [0 1 0 1 0 1 0 1]\n",
      " [1 0 1 0 1 0 1 0]\n",
      " [0 1 0 1 0 1 0 1]\n",
      " [1 0 1 0 1 0 1 0]]\n"
     ]
    }
   ],
   "source": [
    "Z = np.zeros((8,8),dtype=int)\n",
    "Z[1::2,::2] = 1\n",
    "Z[::2,1::2] = 1\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f07fdb2b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1, 5, 3)\n"
     ]
    }
   ],
   "source": [
    "print(np.unravel_index(99,(6,7,8)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f93d3aa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
