# LLM's: RAGs and more

By using this repository, you can easily build a local RAG (from some documents in a folder) with a (pre-trained or not) LLM.

The repository also includes a UI made using Streamlit which, of course, you can deploy locally, too.

## Pre-requisites

You need the following:

- `python = ">=3.10,<3.13"` (see [pyproject.toml](./pyproject.toml))
- `make`
- `HF_TOKEN` in a `.env` file

## Installation

To install all dependencies for the project, run the following command:

```bash
make install-init # only first time
make install
```

## Run app locally

To run the UI locally, run the following command:

```bash
make ui
```

## Usage

For more examples, see [examples](./examples) and [tutorials](./tutorials).

## Contributing

If you wish to contribute to this repository, see [here](./contributing.md).
