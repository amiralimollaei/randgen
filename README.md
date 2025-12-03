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

```
usage: randgen [-h] --target TARGET [--num-iterations NUM_ITERATIONS] [--seed SEED]
```

## License

MIT
