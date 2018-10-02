from data.loader import VQALoader
from data.processor import run

train_loader = VQALoader("train", True, True, 32).get()
val_loader = VQALoader("val", True, True, 32).get()

i = 0
for _ in train_loader:
    print(i)
    i += 1

i = 0
for _ in val_loader:
    print(i)
    i += 1