from typing import Iterable, Sequence

def consume_iterable(chunks: Iterable[str]) -> None:
    # On peut juste itérer
    for c in chunks:
        print("chunk:", c)

def consume_sequence(embeddings: Sequence[Sequence[float]]) -> None:
    # On peut indexer
    print("premier vecteur:", embeddings[0])
    print("dimension:", len(embeddings[0]))

# Iterable: un generator (pas indexable)
def chunk_generator():
    gen = []
    for i in range(3):
        gen.append(i)
        #print(f"chunk-{i}")
    return gen
chunks = chunk_generator()
print(chunks)
consume_iterable(chunks)

# # Sequence: une liste (indexable)
# embeddings = [
#     [0.1, 0.2, 0.3],
#     [0.4, 0.5, 0.6],
# ]
# consume_sequence(embeddings)
