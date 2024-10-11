# Italian LLms

**Status:** In progress  
**Priority:** High  
**Description:**

Show we can set up a RAG+LLM to answer questions about legal stuff, loading the database from [here](https://bancadatigiurisprudenza.giustiziatributaria.gov.it/ricerca).

Test on [Colab](https://colab.research.google.com/drive/1QP7avU0pY2Qc0u46tOMsf8Hu5ueeNU4W?usp=sharing), quick and dirty.

Models to test:

- [Fauno](https://huggingface.co/andreabac3/Fauno-Italian-LLM-7B): On Mac, got the error `OSError: andreabac3/Fauno-Italian-LLM-7B does not appear to have a file named config.json. Checkout 'https://huggingface.co/andreabac3/Fauno-Italian-LLM-7B/tree/main' for available files.`... Try on Colab.
- ~~[Umberto](https://github.com/musixmatchresearch/umberto)~~: This seems to be a model for masking...
- [Cerbero](https://github.com/galatolofederico/cerbero-7b)
