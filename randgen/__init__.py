import dataclasses
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


def eval_plus_fn(value1, value2): return get_value(value1) + get_value(value2)
def eval_minus_fn(value1, value2): return get_value(value1) - get_value(value2)
def eval_mul_fn(value1, value2): return get_value(value1) * get_value(value2)
def eval_div_fn(value1, value2): return get_value(value1) / get_value(value2)
def eval_and_fn(value1, value2): return float(round(get_value(value1)) & round(get_value(value2)))
def eval_or_fn(value1, value2): return float(round(get_value(value1)) | round(get_value(value2)))
def eval_xor_fn(value1, value2): return float(round(get_value(value1)) ^ round(get_value(value2)))
def eval_shr_fn(value1, value2): return float(round(get_value(value1)) >> round(get_value(value2)))
def eval_shl_fn(value1, value2): return float(round(get_value(value1)) << round(get_value(value2)))


def exp_plus_fn(value1, value2): return f"({get_expression(value1)}+{get_expression(value2)})"
def exp_minus_fn(value1, value2): return f"({get_expression(value1)}-{get_expression(value2)})"
def exp_mul_fn(value1, value2): return f"({get_expression(value1)}*{get_expression(value2)})"
def exp_div_fn(value1, value2): return f"({get_expression(value1)}/{get_expression(value2)})"
def exp_or_fn(value1, value2): return f"({get_expression(value1)}|{get_expression(value2)})"
def exp_and_fn(value1, value2): return f"({get_expression(value1)}&{get_expression(value2)})"
def exp_xor_fn(value1, value2): return f"({get_expression(value1)}⊻{get_expression(value2)})"
def exp_shr_fn(value1, value2): return f"({get_expression(value1)}>>{get_expression(value2)})"
def exp_shl_fn(value1, value2): return f"({get_expression(value1)}<<{get_expression(value2)})"


MATH_FUNCTIONS = {
    # name: (eval fn, exp fn, int mode, unwanted values, positive only)
    "plus": (eval_plus_fn, exp_plus_fn, False, [0.0], False),
    "minus": (eval_minus_fn, exp_minus_fn, False, [0.0], False),
    "mul": (eval_mul_fn, exp_mul_fn, False, [0.0], False),
    "div": (eval_div_fn, exp_div_fn, False, [0.0], False),
    "or": (eval_or_fn, exp_or_fn, True, [], False),
    "and": (eval_and_fn, exp_and_fn, True, [], False),
    "xor": (eval_xor_fn, exp_xor_fn, True, [], False),
    # TODO: Fix overflow error
    # "shr": (eval_shr_fn, exp_shr_fn, True, [0.0], True),
    # "shl": (eval_shl_fn, exp_shl_fn, True, [0.0], True),
}


@dataclasses.dataclass(frozen=True)
class MathematicalExpression:
    value1: 'MathematicalExpression | float'
    value2: 'MathematicalExpression | float | None' = None
    operand: str | None = None

    @property
    def eval_fn(self):
        return MATH_FUNCTIONS[self.operand][0] if self.operand else False

    @property
    def exp_fn(self):
        return MATH_FUNCTIONS[self.operand][1] if self.operand else False

    @property
    def int_mode(self):
        return MATH_FUNCTIONS[self.operand][2] if self.operand else False

    @property
    def eval(self) -> float:
        """the numerical result of evaluating the Mathematical Value as a float"""
        if self.value2 is not None:
            assert self.eval_fn
            return self.eval_fn(self.value1, self.value2)
        return self.value1.eval if isinstance(self.value1, MathematicalExpression) else (int(self.value1) if (self.int_mode) else self.value1)

    @property
    def expression(self) -> str:
        """the full mathematical expression as a string"""
        if self.value2 is not None:
            assert self.exp_fn
            return self.exp_fn(self.value1, self.value2)
        int_mode = self.value1.int_mode if isinstance(self.value1, MathematicalExpression) else False
        return self.value1.expression if isinstance(self.value1, MathematicalExpression) else format_value(self.value1, int_mode)


def generate_random_expresson(length: int) -> MathematicalExpression:
    value = MathematicalExpression(
        value1=generate_random_float(0, 1/length)
    )
    for _ in range(length):
        rand_operand, (_, _, _, unwanted_values, positive_only) = random.choice(list(
            MATH_FUNCTIONS.items()
        ))
        unwanted_values = set(unwanted_values)
        if positive_only and value.eval < 0 or value.eval in unwanted_values:
            continue
        rand_num = None
        while rand_num is None or rand_num in unwanted_values:
            rand_num = int(generate_random_float(0, 1/length))
            if positive_only and rand_num < 0:
                rand_num = -rand_num
        reverse_order = random.randint(0, 1)
        if reverse_order:
            value, rand_num = rand_num, value
        value = MathematicalExpression(
            value1=value,
            value2=rand_num,
            operand=rand_operand,
        )
    return value


def generate_expresson(target: int, n_iters: int) -> MathematicalExpression:
    sys.setrecursionlimit((n_iters + 100) * 3)  # TODO: fix this

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
        rand_exp = generate_random_expresson(length=random.randint(2, 6))
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
        default=25,
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
