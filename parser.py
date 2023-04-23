import re
import math
import operator
from typing import Callable, Literal, NamedTuple

eof = {"eof"}


def canonicalize_num(num):
    return repr(integer if (integer := int(num)) == num else num)


def lex(s):
    ops_rex = re.compile(r"\s*(>=|=<|<=|=>|[(){}*/^<>=\[\]]|(?<![0-9.][eE])[+-])\s*")
    for tok in ops_rex.split(s):
        if tok:
            try:
                yield canonicalize_num(float(tok))
            except ValueError:
                yield tok
    yield eof


class Op(NamedTuple):
    op: str
    prec: int
    assoc: Literal["l", "r", "u"]  # left-associative, right-associative, unary
    fun: Callable

    def __call__(self, *args):
        return self.fun(*args)

    def __repr__(self):
        return f"op({self.op!r:})"

    def left_first(self, other):
        return self.prec > other.prec or self.prec == other.prec and other.assoc == "l"

    def arity(self):
        return 1 if self.assoc == "u" else 2

    def apply(self, stack):
        n = self.arity()
        stack[-n:] = [(self,) + tuple(stack[-n:])]


OP_GROUPS = """
add+l sub~l
truediv/l mul*l
neg-u
pow^r
sqrt√u
""".strip()
OPS = {
    o: Op(o, prec, assoc, getattr(operator, fun) if fun != "sqrt" else math.sqrt)
    for prec, op_groups in enumerate(OP_GROUPS.split("\n"))
    for [(fun, o, assoc)] in map(
        re.compile(r"^(\w+)(\W+)(\w+)$").findall, op_groups.split()
    )
}


def parse_expr(flat, waitfor=eof):
    exprs = []
    ops = []
    last_was_op = True
    while True:
        x = next(flat)
        if x == waitfor:
            while ops:
                ops.pop().apply(exprs)
            (ans,) = exprs
            return ans
        if o := OPS.get(x):
            if last_was_op:
                assert o.assoc == "u"
            else:
                o = o if x != "-" else OPS["~"]
                while ops and ops[-1].left_first(o):
                    ops.pop().apply(exprs)
            ops.append(o)
            last_was_op = True
        else:
            exprs.append(x if x != "(" else parse_expr(flat, ")"))
            last_was_op = False


def to_ast(x):
    return parse_expr(lex(x))
