import dataclasses
import math
import random
import sys

RANDOM_SCALE = 2**16
RANDOM_FLOAT_DECIMALS = 10


def format_value(v: float, int_mode: bool = False):
    return f"{v:d}" if int_mode else f"{v:0.{RANDOM_FLOAT_DECIMALS}g}"


def generate_random_float(mu: float, sigma: float) -> float:
    return float(format_value(random.gauss(mu, sigma=RANDOM_SCALE * sigma)))


def get_value(value: 'MathematicalExpression | float') -> float | int:
    int_mode = value.int_mode if isinstance(value, MathematicalExpression) else False
    return value.eval if isinstance(value, MathematicalExpression) else (int(value) if int_mode else value)


def get_expression(value: 'MathematicalExpression | float') -> str:
    int_mode = value.int_mode if isinstance(value, MathematicalExpression) else False
    return value.expression if isinstance(value, MathematicalExpression) else format_value(value, int_mode)


# mathematical operations

def eval_plus_op(value1, value2): return get_value(value1) + get_value(value2)
def eval_minus_op(value1, value2): return get_value(value1) - get_value(value2)
def eval_mul_op(value1, value2): return get_value(value1) * get_value(value2)
def eval_div_op(value1, value2): return get_value(value1) / get_value(value2)
def eval_and_op(value1, value2): return float(round(get_value(value1)) & round(get_value(value2)))
def eval_or_op(value1, value2): return float(round(get_value(value1)) | round(get_value(value2)))
def eval_xor_op(value1, value2): return float(round(get_value(value1)) ^ round(get_value(value2)))
def eval_shr_op(value1, value2): return float(round(get_value(value1)) >> round(get_value(value2)))
def eval_shl_op(value1, value2): return float(round(get_value(value1)) << round(get_value(value2)))


def exp_plus_op(value1, value2): return f"({get_expression(value1)}+{get_expression(value2)})"
def exp_minus_op(value1, value2): return f"({get_expression(value1)}-{get_expression(value2)})"
def exp_mul_op(value1, value2): return f"({get_expression(value1)}*{get_expression(value2)})"
def exp_div_op(value1, value2): return f"({get_expression(value1)}/{get_expression(value2)})"
def exp_or_op(value1, value2): return f"({get_expression(value1)}|{get_expression(value2)})"
def exp_and_op(value1, value2): return f"({get_expression(value1)}&{get_expression(value2)})"
def exp_xor_op(value1, value2): return f"({get_expression(value1)}⊻{get_expression(value2)})"
def exp_shr_op(value1, value2): return f"({get_expression(value1)}>>{get_expression(value2)})"
def exp_shl_op(value1, value2): return f"({get_expression(value1)}<<{get_expression(value2)})"


MATH_OPERANDS = {
    # name: (eval fn, exp fn, int mode, unwanted values, positive only)
    "plus": (eval_plus_op, exp_plus_op, False, [0.0], False),
    "minus": (eval_minus_op, exp_minus_op, False, [0.0], False),
    "mul": (eval_mul_op, exp_mul_op, False, [0.0], False),
    "div": (eval_div_op, exp_div_op, False, [0.0], False),
    "or": (eval_or_op, exp_or_op, True, [], False),
    "and": (eval_and_op, exp_and_op, True, [], False),
    "xor": (eval_xor_op, exp_xor_op, True, [], False),
    # TODO: Fix overflow error
    # "shr": (eval_shr_op, exp_shr_op, True, [0.0], True),
    # "shl": (eval_shl_op, exp_shl_op, True, [0.0], True),
}


# mathematical functions

def eval_sin_fn(value): return math.sin(get_value(value))
def eval_cos_fn(value): return math.cos(get_value(value))
def eval_tan_fn(value): return math.tan(get_value(value))
def eval_cot_fn(value): return 1/math.tan(get_value(value))
def eval_sec_fn(value): return 1/math.cos(get_value(value))
def eval_csc_fn(value): return 1/math.sin(get_value(value))
def eval_sqrt_fn(value): return math.sqrt(get_value(value))
def eval_log_fn(value): return math.log(get_value(value))
def eval_exp_fn(value): return math.exp(get_value(value))


def exp_sin_fn(value): return f"sin({get_value(value)})"
def exp_cos_fn(value): return f"cos({get_value(value)})"
def exp_tan_fn(value): return f"tan({get_value(value)})"
def exp_cot_fn(value): return f"cot({get_value(value)})"
def exp_sec_fn(value): return f"sec({get_value(value)})"
def exp_csc_fn(value): return f"csc({get_value(value)})"
def exp_sqrt_fn(value): return f"sqrt({get_value(value)})"
def exp_log_fn(value): return f"log({get_value(value)})"
def exp_exp_fn(value): return f"exp({get_value(value)})"


MATH_FUNCTIONS = {
    # name: (eval fn, exp fn, unwanted values, positive only)
    None: (None, None, [], False),
    "sin": (eval_sin_fn, exp_sin_fn, [0.0], False),
    "cos": (eval_cos_fn, exp_cos_fn, [0.0], False),
    "tan": (eval_tan_fn, exp_tan_fn, [0.0], False),
    "cot": (eval_cot_fn, exp_cot_fn, [0.0], False),
    "sec": (eval_sec_fn, exp_sec_fn, [], False),
    "csc": (eval_csc_fn, exp_csc_fn, [], False),
    "sqrt": (eval_sqrt_fn, exp_sqrt_fn, [], True),
    # TODO: Fix overflow error
    # "log": (eval_log_fn, exp_log_fn, [0.0], True),
    # "exp": (eval_exp_fn, exp_exp_fn, [0.0], False),
}


@dataclasses.dataclass(frozen=True)
class MathematicalExpression:
    """
    Represent a mathematical expression

    - if `value2` is not defined, it is a mathematical value, and `function` MAY be defined for it
    - if `value2` is defined it is a mathematical equation, and `operand` MUST be defined for it
    """

    value1: 'MathematicalExpression | float'
    value2: 'MathematicalExpression | float | None' = None
    operand: str | None = None
    function: str | None = None

    @property
    def eval_op(self):
        return MATH_OPERANDS[self.operand][0] if self.operand else False

    @property
    def exp_op(self):
        return MATH_OPERANDS[self.operand][1] if self.operand else False

    @property
    def eval_fn(self):
        return MATH_FUNCTIONS[self.function][0] if self.function else False

    @property
    def exp_fn(self):
        return MATH_FUNCTIONS[self.function][1] if self.function else False

    @property
    def int_mode(self):
        return MATH_OPERANDS[self.operand][2] if self.operand else False

    @property
    def eval(self) -> float:
        """the numerical result of evaluating the mathematical expression as a float"""
        if self.value2 is None:
            return self.eval_fn(self.value1) if self.eval_fn else get_value(self.value1)
        else:
            assert self.eval_op
            return self.eval_op(self.value1, self.value2)

    @property
    def expression(self) -> str:
        """the full mathematical expression as a string"""
        if self.value2 is None:
            return self.exp_fn(self.value1) if self.exp_fn else get_expression(self.value1)
        else:
            assert self.exp_op
            return self.exp_op(self.value1, self.value2)

def generate_random_value(mu: float, sigma: float) -> MathematicalExpression:
    rand_function, (_, _, unwanted_values, positive_only) = random.choice(list(
        MATH_FUNCTIONS.items()
    ))
    unwanted_values = set(unwanted_values)
    rand_num = None
    while rand_num is None or rand_num in unwanted_values:
        rand_num = generate_random_float(mu, sigma)
        if positive_only and rand_num < 0:
            rand_num = -rand_num
    
    return MathematicalExpression(
        value1=rand_num,
        function=rand_function,
    )

def generate_random_equation(length: int) -> MathematicalExpression:
    value = MathematicalExpression(
        value1=generate_random_float(0, 1/length)
    )
    for _ in range(length):
        rand_operand, (_, _, _, unwanted_values, positive_only) = random.choice(list(
            MATH_OPERANDS.items()
        ))
        unwanted_values = set(unwanted_values)
        if positive_only and value.eval < 0 or value.eval in unwanted_values:
            continue
        rand_value = generate_random_value(0, 1/length)
        reverse_order = random.randint(0, 1)
        if reverse_order:
            value, rand_value = rand_value, value
        value = MathematicalExpression(
            value1=value,
            value2=rand_value,
            operand=rand_operand,
        )
    return value


def generate_expresson(target: int, n_iters: int) -> MathematicalExpression:
    sys.setrecursionlimit((n_iters + 100) * 3)  # TODO: how can we make it less recursive so that we don't hit the recursion limit?

    value = MathematicalExpression(
        value1=generate_random_float(0, 1/n_iters)
    )
    for i in range(n_iters):
        if i == (n_iters-1):
            sub_expression_target_value = round(target-value.eval, RANDOM_FLOAT_DECIMALS)
        else:
            sub_expression_target_value = generate_random_float(0, 1/n_iters)
        if sub_expression_target_value == 0:
            continue
        rand_exp = generate_random_equation(length=random.randint(2, 6))
        if round(rand_exp.eval, RANDOM_FLOAT_DECIMALS) == 0:
            continue
        sub_value_value1 = sub_expression_target_value
        sub_value_value2 = rand_exp
        reverse_order = random.randint(0, 1)
        if reverse_order:
            sub_value_value1, sub_value_value2 = sub_value_value2, sub_value_value1
        sub_value = MathematicalExpression(
            value1=sub_expression_target_value,
            value2=rand_exp,
            operand="mul",
        )
        sub_value = MathematicalExpression(
            value1=sub_value,
            value2=rand_exp.eval,
            operand="div",
        )
        reverse_order = random.randint(0, 1)
        if reverse_order:
            value, sub_value = sub_value, value
        value = MathematicalExpression(
            value1=value,
            value2=sub_value,
            operand="plus",
        )
    return value


def main():
    import argparse

    argparser = argparse.ArgumentParser(description="Random Equation Generator")
    argparser.add_argument(
        "--target", "-t",
        type=float,
        required=True
    )
    argparser.add_argument(
        "--num-iterations", "-n",
        type=int,
        default=20,
    )
    argparser.add_argument(
        "--seed", "-s",
        type=int
    )

    args = argparser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    result = generate_expresson(target=args.target, n_iters=args.num_iterations)
    print(result.expression[1:-1])


if __name__ == "__main__":
    main()
