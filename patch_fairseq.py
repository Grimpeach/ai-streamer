import re
import os

path = r".venv\Lib\site-packages\fairseq\dataclass\configs.py"

if not os.path.exists(path):
    print("Файл не найден. Проверьте, установлен ли fairseq.")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Магическая регулярка, которая лечит баг с dataclasses в Python 3.11
code = re.sub(
    r"(\w+):\s*([A-Za-z0-9_]+Config)\s*=\s*\2\(\)", 
    r"\1: \2 = field(default_factory=\2)", 
    code
)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("Fairseq успешно пропатчен под Python 3.11!")