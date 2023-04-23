"""A for JIT arithmetic expressions to x64 machine code.

The generated code is exposed as a python-callable-function.
"""
from contextlib import contextmanager
import ctypes
import operator as op
import math

from peachpy import *
from peachpy.x86_64 import *

from parser import canonicalize_num, to_ast, OPS


# Peachpy's Constant._parse_float64 is broken for denormals, let's fix it.
import struct


def float_bits(f):
    return struct.unpack("<Q", struct.pack("<d", f))[0]


Constant._parse_float64 = float_bits

expr_n33 = to_ast("3 + (8 - 7.5) * 10 / 5  - (2 + 5 * 7)")
expr_qf = to_ast("(-b+(b^2-4*a*c)^0.5)/(2*a)")


def free_vars(expr):
    """Return the free variables in `expr`.

    >>> free_vars(to_ast("(-b+(b^2-4*a*c)^0.5)/(2*a)")) == {'a', 'b', 'c'}
    True
    """
    if type(expr) is str:
        return frozenset([expr] if expr.isidentifier() else [])
    return frozenset(e for x in expr[1:] for e in free_vars(x))


def _NEGSD(reg_x):
    "Negate reg_x (xmm)."
    temp = XMMRegister()
    MOVSD(temp, Constant.float64(-0.0))
    return XORPD(reg_x, temp)


def _POWSD(reg_x, reg_y):
    """reg_x**reg_y, where reg_y must be an integral non-negative double.

    Uses one additional General purpose register and clobbers Flags (OF, CF, SF, ZF, PF).

    TODO: add support for negative powers and return NaN if assumptions are violated.
    """
    lbl_even = Label()
    lbl_loop = Label()
    lbl_loop_check = Label()
    cnt = GeneralPurposeRegister64()
    CVTSD2SI(cnt, reg_y)  # convert double in reg_y to int in cnt register.
    TEST(cnt, cnt)
    MOVSD(reg_y, Constant.float64(1.0))
    JNZ(lbl_loop_check)
    MOVSD(reg_x, Constant.float64(1.0))

    with LABEL(lbl_loop):
        SHR(cnt, 1)
        JNC(lbl_even)
        MULSD(reg_y, reg_x)
    with LABEL(lbl_even):
        MULSD(reg_x, reg_x)
    with LABEL(lbl_loop_check):
        CMP(cnt, 1)
        JG(lbl_loop)

    MULSD(reg_x, reg_y)


ASM = {
    op.add: ADDSD,
    op.truediv: DIVSD,
    op.mul: MULSD,
    op.sub: SUBSD,
    op.neg: _NEGSD,
    op.pow: _POWSD,
    math.sqrt: lambda x: SQRTSD(x, x),
}


def simplify_expr(expr):
    if not isinstance(expr, tuple):
        return expr
    (op, *args) = expr
    # Generic pow with float exponent is a bit of a pain without libm;
    # cheat by rewriting simple common cases, and then do exponentiation
    # by squaring for integers.
    #
    # First simplify (OPS['-'], '123') to '-123'.
    if op is OPS["-"] and isinstance(arg := args[0], str) and not arg.isidentifier():
        return "-" + arg

    if op is OPS["^"]:
        b, e = map(simplify_expr, args)
        if isinstance(e, str):
            if e.startswith("-"):
                return (OPS["/"], "1", simplify_expr((OPS["^"], b, e[1:])))
            if e == "0.5":
                return (OPS["√"], b)
            if e == "2":
                return (OPS["*"], b, b)
            if e == "1":
                return b
            if e == "0":
                return "1"
            if e.isdecimal():  # Can handle integer powers at runtime.
                return expr
        raise ValueError(f"Cannot yet handle ^{e}")
    return (op, *map(simplify_expr, args))


def jit_expr(expr, env):
    if type(expr) is str:
        if expr.isidentifier():
            val = env[expr]
        else:
            val = Constant.float64(float(expr))
        tmp = XMMRegister()
        MOVSD(tmp, val)
        return tmp
    raw_op, *raw_args = expr
    op = ASM[raw_op.fun]
    args = [jit_expr(raw_arg, env) for raw_arg in raw_args[::-1]][::-1]
    op(*args)
    return args[0]


# Create a jitted function callable from python that evaluates
# body_expr.
def jit_fun(name, arg_names, body_expr):
    assert frozenset(arg_names) == free_vars(body_expr)
    args = tuple(Argument(double_, name) for name in arg_names)
    with Function(name, args, double_) as asm_function:
        regs = tuple(XMMRegister() for _ in range(len(arg_names)))
        for reg, arg in zip(regs, args):
            LOAD.ARGUMENT(reg, arg)
        expr_asm = jit_expr(simplify_expr(body_expr), dict(zip(arg_names, regs)))
        RETURN(expr_asm)

    return asm_function.finalize(abi.detect()).encode().load()


def jit_evaluator(expr, name="f_jit"):
    """Return a function (named `name`) that evaluates `expr`.

    The returned function will take the free variables in `expr` in alphabetical order.

    Examples:

    >>> jit_evaluator(to_ast("x^13"))(3) == 3**13
    True
    >>> f = jit_evaluator(to_ast('(2+3)/2'))
    >>> f()
    2.5
    >>> f = jit_evaluator(to_ast("30*x/5  - (2 + 5 * 7)"))
    >>> f(5)
    -7.0
    >>> jit_evaluator(to_ast('(-b+(b^2-4*a*c)^0.5)/(2*a)'))(5, 6, 1)
    -0.2
    """
    return jit_fun(name, sorted(free_vars(expr)), expr)
