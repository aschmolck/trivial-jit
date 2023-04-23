"""Arithmetic JIT Testing Strategy:

 1. A few ad hoc examples, as a basic sanity check.
 2. Generatve tests that make sure that parsing roundtrips, and the jitted code
    "kind of" gives the same answer as the equivalent expression string eval'ed in
    python.

A straightfoward testing strategy for parsing is randomly generating expression
trees, pretty printing them to an infix string and verifying that parsing the
string back roundtrips.

Testing evaluation is a bit trickier. We'd like to make use of python's eval as
a ground truth for what an expression should evaluate to. But there are some
intentional differences between the JIT and python syntax and semantics:

 1. Syntactic differences: Exponention is denoted by `^` not `**` and all
    numbers are floating point, not integers (e.g. 1 really means 1.0).

 3. Semantic differences: 1.0/0.0 results in  (as it does in numpy), not a
    ZeroDivisionError; similarly, 0/0 should be NaN. Also, large exponents in
    python can produce OverflowError, and an exponent of 0.5 can result in a
    complex result (rather than NaN as in the JIT). Furthermore, exponentiation
    can give numerically slightly different results, since we don't use libm
    and instead use repeated squaring for integer exponents etc.

The syntactic differences can be solved by pre-processing the input string.
The semantic differences are trickier to address, as e.g. the ZeroDivisionError
can occur anywhere in the expression and python does not have resumable
Exceptions (e.g. `3 + -1/(1/0)` should be 3, but we can just tell python to
continue evaluation with the value infinity where it encountered the exception).
So we adopt the following strategy: we have a python eval based evaluator
(`py_eval`) that on ZeroDivisionError (or OverflowError) always returns NaN. The
JIT result is only allowed to disagree with `py_eval` if the latter returned a
NaN. In that case we then also compare against `naive_evaluator`, which
tree-based interpreter of arithmetic expressions, that can capture either
python's or the JIT's zero division semantics. Additionally, we need to allow
for some slight numerical differences, but only if the expression contains an
exponentiation.
"""

from math import isnan, isfinite
from hypothesis import strategies as st, example, given, settings, Phase

from parser import *
from trivial_jit import *


def unparse(ast):
    if not isinstance(ast, tuple):
        return str(ast)
    if len(ast) == 2:
        (unary_op, arg) = ast
        return (
            f"{unary_op.op}{unparse(arg)}"
            if type(arg) is not tuple or not unary_op.left_first(arg[0])
            else f"{unary_op.op}({unparse(arg)})"
        )
    (binary_op, arg1, arg2) = ast
    o = binary_op.op
    o = {"~": "-"}.get(o, o)
    x = unparse(arg1)
    y = unparse(arg2)
    if type(arg1) is tuple and not arg1[0].left_first(binary_op):
        x = f"({x})"
    if type(arg2) is tuple and len(arg2) > 2 and binary_op.left_first(arg2[0]):
        y = f"({y})"
    return f"{x} {o} {y}" if o != "^" else f"{x}^{y}"


def py_eval(s):
    """Python's `eval`, souped up to be more similar to JIT semantics."""
    # 123 -> 123.0; tricky because we don't want 1e+23 -> 1e+23.0.
    floatified = re.sub(
        "(?<![a-zA-Z0-9_.])(?<![eE][+-])([0-9]+)(?![a-zA-Z0-9_.])", r"\1.0", s
    )
    try:
        if isinstance(ans := eval(floatified.replace("^", "**")), complex):
            ans = float("NaN")
        return ans
    except (ZeroDivisionError, OverflowError):
        return float("NaN")


def naive_evaluator(expr, name="f_naive", div_by_0_is_inf=True):
    """Return a function (named `name`) that evaluates `expr`.

    The returned function will take the free variables in `expr` in alphabetical order.

    By default, `naive_evaluator` is a drop-in-replacement for `jit_evaluator`;
    the returned function will be orders of magnitude slower but should return
    *identical* results (unless exponentiation is used, in which case there can
    be small numerical differences).  However if `div_by_0_is_inf` is set to
    False, 1/0 will return NaN like `py_eval`, not `inf` like by default.

    Examples:

    >>> naive_evaluator(to_ast('a+2*b^3'))(.5,1.)  # free vars are passed i
    2.5
    >>> naive_evaluator(to_ast('a/0'))(-1.)
    -inf
    >>> naive_evaluator(to_ast('a/0'), div_by_0_is_inf=False)(-1.)
    nan
    """
    zero_div = ZeroDivisionError if div_by_0_is_inf else type("Never", (Exception,), {})

    def f(*args):
        env = dict(zip(sorted(free_vars(expr)), args))

        def eval_expr(expr):
            if type(expr) is str:
                return float(expr) if expr not in env else env[expr]
            try:
                op, [*args] = expr[0], map(eval_expr, expr[1:])
                return float(op(*args))
            except zero_div:
                return handle_zero_div(op, *args)

        try:
            return eval_expr(expr)
        except (ZeroDivisionError, TypeError, OverflowError):
            return float("NaN")

    f.__name__ = name

    return f


def handle_zero_div(op, a, b):
    assert type(a) == type(b) == float
    if op == OPS["^"]:
        assert b < 0 and a == 0
        return math.copysign(float("inf"), a)
    assert op == OPS["/"], "Only ^ and / should cause ZeroDivisionError"
    assert b == 0
    if a == 0 or math.isnan(a):
        # NB, can't just raise, since nan doesn't *always* propagate
        return float("nan")
    return math.copysign(float("inf"), a) * math.copysign(1, b)


assert naive_evaluator(to_ast("--9007199254741195"))() == 9007199254741195.0


assert naive_evaluator(expr_qf)(5, 6, 1) == -0.2
assert naive_evaluator(expr_n33)() == -33.0


def test_parse_roundtrips_handpicked():
    equiv_exprs = [
        ["0.1 + 0.2 + 0.3", "(0.1 + 0.2) + 0.3"],
        ["-1^2", "-(1^2)"],
        ["-1^-2", "-(1^(-2))"],
        ["1 + 2 * 3", "1 + (2 * 3)"],
        ["(1 + 2) * 3"],
    ]
    for canon, *equivs in equiv_exprs:
        canon_val = py_eval(canon)
        canon_ast = to_ast(canon)
        assert canon == unparse(canon_ast)
        assert naive_evaluator(canon_ast, div_by_0_is_inf=False)() == canon_val
        for equiv in equivs:
            assert py_eval(equiv) == canon_val
            assert to_ast(unparse(to_ast(equiv))) == canon_ast


unary_ops = st.sampled_from([OPS["-"]])
binary_ops = st.sampled_from([o for o in OPS.values() if o.assoc != "u"])
numbers = (
    st.floats(min_value=0, allow_nan=False, allow_infinity=False).map(canonicalize_num)
).map(str)


# The jit evaluator can only do square roots and integer exponents.
def simple_powers_only(op_x_y):
    op, x, y = op_x_y
    if op is not OPS["^"]:
        return True
    if y[0] == OPS["-"] and isinstance(y[1], str):
        y = "-" + y[1]
    return isinstance(y, str) and (
        abs(pow := float(y)) < 1e6 and (pow == int(pow) or abs(pow) == 0.5)
    )


# An ast is either a number, or, recursively, a 2 element tuple
# of an unary operator and an ast, or a 3 element tuple of a binary operator and
# an ast.
ast = st.recursive(
    numbers,
    lambda child_ast: st.tuples(unary_ops, child_ast)
    | st.tuples(binary_ops, child_ast, child_ast).filter(simple_powers_only),
)


def same(x, y):
    return x == y or isnan(x) and isnan(y)


def similar(x, y):
    return (
        x == y
        or abs(x - y) / max(abs(x), abs(y)) < 1e-6
        # Make e.g x=inf, and y=nan OK, because python evaluation will
        # trap on e.g 1.0/0.0 (which we map to NaN), but the CPU (or numpy)
        # will just give inf.
        or not (isfinite(x) or isfinite(y))
    )


@given(ast)
# @settings(phases=[p for p in Phase if p != Phase.shrink])
@example(ast=(OPS["^"], (OPS["/"], "0", "0"), "0"))
def test_parse_roundtrips(ast):
    assert to_ast(as_str := unparse(ast)) == ast, "Didn't roundtrip"
    py_ans = py_eval(as_str)
    naive_ans = naive_evaluator(ast, div_by_0_is_inf=False)()
    jit_ans = jit_evaluator(ast)()

    assert same(py_ans, naive_ans)
    # We do exponentiation by repeated squaring, which can return slightly different results.
    naive_ans_with_zero_division = naive_evaluator(ast, div_by_0_is_inf=True)()
    if not same(naive_ans_with_zero_division, jit_ans):
        assert similar(naive_ans_with_zero_division, jit_ans) and (
            "^" in as_str
            or "/" in as_str
            and not isfinite(naive_ans_with_zero_division)
        )


def test_associativity():
    assert to_ast("1 * 2 * 3") == to_ast("(1 * 2) * 3")
    assert to_ast("1 * 2 * 3") != to_ast("1 * (2 * 3)")
    assert to_ast("1 / 2 / 3") == to_ast("(1 / 2) / 3")
    assert to_ast("1^2^3") == to_ast("1^(2^3)")
    assert to_ast("1^2^3") != to_ast("(1^2)^3")

    assert unparse(to_ast("1^"))
