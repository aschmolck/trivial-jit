# Some JIT examples for working with numpy arrays.
from trivial_jit import *
import os
import numpy as np
import ctypes


DEBUG = bool(os.getenv("DEBUG", False))


# An ASM loop template (notice the yield) -- something for which
# you might use a macro in a traditional assembler.
@contextmanager
def down_under(cnt):
    While = Loop()
    SUB(cnt, 1)
    JB(While.end)
    with While:
        yield
        SUB(cnt, 1)
        JNB(While.begin)


# Let's manually write an assembler function that sums up a c-style "array".
# Of course C doesn't have real arrays, so the function will take an integer
# specifying a length and a pointer to a buffer of doubles of that length, and
# returns their sum.
def make_array_summer():
    n = Argument(int64_t)
    a = Argument(ptr(const_double_))
    # def Sum(n: int64_t, a: ptr(const_double)) -> double_
    with Function("Sum", (n, a), double_) as asm_function:
        cnt = GeneralPurposeRegister64()
        reg_a = GeneralPurposeRegister64()
        acc = XMMRegister()
        LOAD.ARGUMENT(cnt, n)
        LOAD.ARGUMENT(reg_a, a)
        XORPD(acc, acc)
        with down_under(cnt):
            ADDSD(acc, [reg_a])
            ADD(reg_a, 8)
        RETURN(acc)
    encoded = asm_function.finalize(abi.detect()).encode()
    if DEBUG:
        print("SUM asm:", encoded.format())
    python_function = encoded.load()  # expose as python func
    return python_function


sum_c_array = make_array_summer()


# Let's test it.
def test_manual_array_summation():
    n = 100
    # Convert a range to a buffer of doubles with ctypes.
    doubles = (ctypes.c_double * n)(*range(n))
    assert (sum_c_array(n, doubles)) == sum(range(n))


# Now let's convert the psedu-array summer to something that can work on an
# actual (numpy) array type.


# First we need a function to convert a numpy array to a length and a data
# buffer pointer that  we can pass to a jitted function.
def narray_data_ptr(a):
    ctypes_double_ptr = ctypes.POINTER(ctypes.c_double)
    return a.size, a.ctypes.data_as(ctypes_double_ptr)


def adapt_c_array_fun_for_numpy(f):
    return lambda a: f(*narray_data_ptr(a))


sum_numpy_array = adapt_c_array_fun_for_numpy(sum_c_array)


# Let's test it.
def test_numpy_array_summation():
    n = 100
    ary = np.arange(n + 0.0)
    assert (sum_numpy_array(ary)) == sum(range(n))


# Now, let's automate this and write a helper that JITs an arbitrary expression
# into a numpy array reduction.
def array_reducer(name, vars, expr):
    assert len(vars) == 2
    n = Argument(int64_t)
    a = Argument(ptr(const_double_))
    init = Argument(double_)
    op_args = free_vars(expr)
    assert len(op_args) == 2
    with Function(name, (init, n, a), double_, debug_level=10 * DEBUG) as asm_function:
        cnt = GeneralPurposeRegister64()
        reg_a = GeneralPurposeRegister64()
        reg_acc = XMMRegister()
        LOAD.ARGUMENT(cnt, n)
        LOAD.ARGUMENT(reg_a, a)
        LOAD.ARGUMENT(reg_acc, init)
        with down_under(cnt):
            last_reg = jit_expr(expr, dict(zip(op_args, [reg_acc, [reg_a]])))
            MOVSD(reg_acc, last_reg)
            ADD(reg_a, 8)
        RETURN(reg_acc)
    encoded = asm_function.finalize(abi.detect()).encode()
    if DEBUG:
        print("Array_reducer {name=} {vars=} {expr=} asm", encoded.format())
    return encoded.load()


def make_array_aggregator(name, args, expr, initial):
    raw_aggregator = array_reducer(name, args, to_ast(expr))

    def f(a, initial=initial):
        return raw_aggregator(initial, *narray_data_ptr(a))

    f.__name__ = name
    return f


asum = make_array_aggregator("Asum", ["t", "x"], "t+x", 0.0)
aprod = make_array_aggregator("Aprod", ["t", "x"], "t*x", 1.0)


def test_array_reduction():
    assert asum(np.array([])) == 0.0
    assert asum(np.array([1.0, 2.0])) == 3.0
    assert asum(np.array([1.0, 2.0, 3.0])) == 6.0
    assert asum(np.array([1.0, 2.0, 3.0]), 4.0) == 10.0

    assert aprod(np.array([])) == 1.0
    assert aprod(np.array([1.0, 2.0])) == 2.0
    assert aprod(np.array([1.0, 2.0, 3.0])) == 6.0
    assert aprod(np.array([1.0, 2.0, 3.0]), 4.0) == 24.0


if __name__ == "__main__":
    test_manual_array_summation()
    test_numpy_array_summation()
    test_array_reduction()
