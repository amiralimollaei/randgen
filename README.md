# **Rand**om Equation **Gen**erator

Generates random mathematical equations that evaluate to a specified target, can be used to generate really long expressions to test parsers.

## Usage

clone this repository and run

```sh
uv sync
```

then you can generate random equations using

```sh
uv run randgen --target <number>
```

below is the result of `randgen --help`

```text
usage: randgen [-h] --target TARGET [--num-iterations NUM_ITERATIONS] [--seed SEED]

Random Equation Generator

options:
  -h, --help            show this help message and exit
  --target TARGET, -t TARGET
  --num-iterations NUM_ITERATIONS, -n NUM_ITERATIONS
  --seed SEED, -s SEED
```

## License

MIT
