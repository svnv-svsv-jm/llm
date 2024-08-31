DEFAULT_TEMPLATE = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

"""

ITALIAN_PROMPT_TEMPLATE = """
<|system|>
Rispondi alla domanda con la tua conoscenza. Usa il seguente contesto per aiutarti:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>
"""
