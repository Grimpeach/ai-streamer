import re
import os

path = r".venv\Lib\site-packages\hydra\conf\__init__.py"

if not os.path.exists(path):
    print(f"Файл {path} не найден.")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Лечим баг со всеми изменяемыми классами (OverrideDirname, RunDir, SweepDir и т.д.)
code = re.sub(
    r"(\w+):\s*([A-Za-z0-9_]+)\s*=\s*\2\(\)", 
    r"\1: \2 = field(default_factory=\2)", 
    code
)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("Библиотека Hydra успешно пропатчена под Python 3.11!")