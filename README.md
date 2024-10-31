# LLM's: RAGs and more

This repository shows you how one can locally build a RAG from some documents in a folder for a pre-trained LLM.

## Contributing

If you wish to contribute to this repository, see [here](./contributing.md).

## Pre-requisites

You need the following:

- Python and a virtual environment (even better if you have `pyenv`)
- `make`
- `HUGGINGFACE_TOKEN` declared in a `.env` file.

## Installation

To install all dependencies for the project, run the following command:

```bash
make install-init # only first time
make install
```

## Run app locally

To run the UI, run the following command:

```bash
make ui
```

## Usage

For more examples, see [examples](./examples) and [tutorials](./tutorials).
