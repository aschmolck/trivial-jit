# How to build a simplistc x64 JIT in under 200 lines of Python

This repo [parses](parser.py) arithmetic expressions (+,-,*,/,^,√) into a
simple ast (~64 LOC) and [compiles](trivial_jit.py) the ast to x64 machine code
(~100LOC), which it exposes as a callable python functions. Parsing and
evaluation are checked with property-based [tests](test_jit.py).

The small code size is achieved by cheating as much as possible. For example,
x64 floating point instructions have less legacy baggage than the integer
instructions, so we use floating point throughout.

There's also a [simple demo](array_jit.py) of how to run JIT'ed functions over
arrays. On my mahcine, the jit'ed sum function is about 20x faster for a 100M
element array than python's `sum` and only about 3x slower than `numpy.sum`.


## Installation

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

(If you have [direnv](https://github.com/direnv/direnv) you can skip the first
two steps).

## Running

```
python
>>> from trivial_jit import to_ast, jit_evaluator
>>> def f(x): 2*x**0.5
>>> f_jit = jit_evaluator(to_ast('2*x^0.5'))
>>> f(3)
3.4641016151377544
>>> f_jit(3)
3.4641016151377544
```

## Motivation
I can imagine some cases where it might be handy to have the ability to generate
native code from pure python and without having to have a complex compiler
toolchain installed.

But more importantly, having a smattering of x64 assembler knowledge is handy,
but annoyinng to come by for various reasons (historical baggage, widely
divergent syntax and calling conventions, poor tooling and paucity of learning
resources).

Turns out python makes for a mighty fine assembler DSL and interactive
playground with many perks over traditional toolchains.

- Enjoy something much closer to Intel syntax (`mov rax, 1`) than
  gcc/clang/gas/objdump (`movq $1, %rax`) with default settings). This matters,
  because all the reference material uses Intel syntax.

- Build higher-level abstractions using python, an expressive high-level
  language you already know anway, instead of some weird and limited macro
  language completely particular to one of the half a dozen popular x64
  assemblers (gas, nasm, masm, tasm, yasm, ...).

- These abstractions can be much more sophisticated than what would be possible
  with a
  traditional assembler (it's easy to write code that "templates" over single
  and double precision, control over opcode encoding, interleaving scalar and
  simd streams etc).

- Have different calling conventions (windows vs rest of the
  world) and similar annoyances taken care of automatically.

- No need to juggle registers allocation yourself if you don't want to, PeachPy
  comes with a simple register allocator.

- Much nicer dev experience: auto-completion! Write some ASM int the repl
  and expose it as a python function, and try it out immediately!

# Some Resources to have at hand whilst reading through the code

- https://www.felixcloutier.com/x86/ (pro-tip: the many different jump
  instructions, e.g. JE, JNE, JZ are under
  [Jcc](https://www.felixcloutier.com/x86/Jcc);
  [SETcc](https://www.felixcloutier.com/x86/SETcc) and a few others follow the same pattern)
- https://godbolt.org/ to see what x64 assembler say a C compiler will spit out
  for the same arithmetic expression
- https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf
  (unfortunately AT\&T syntax)
- on that topic see https://staffwww.fullcoll.edu/aclifton/courses/cs241/syntax.html


# For digging deeper

- The Intel reference manuals: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
- Agner Fog's ASM and optimization resources: https://agner.org/optimize/
- Peter Cordes's Stackoverflow posts, the top rated one: https://stackoverflow.com/questions/40354978
  gives an excellent concrete example of optimizing a routine with x64 asm, and
  all the considerations that go into it.

