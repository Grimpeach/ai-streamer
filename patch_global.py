import os, re

folders = [
    r".venv\Lib\site-packages\fairseq",
    r".venv\Lib\site-packages\hydra"
]

count = 0
for folder in folders:
    if not os.path.exists(folder):
        continue
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            
            orig = code
            
            # 1. Откатываем default_factory, если он где-то остался
            code = re.sub(
                r"(\w+):\s*([A-Za-z0-9_]+)\s*=\s*field\(default_factory=\2\)", 
                r"\1: \2 = \2()", 
                code
            )
            
            # 2. Добавляем unsafe_hash=True ко ВСЕМ датаклассам в файле
            code = re.sub(r"@dataclass(?!\()", "@dataclass(unsafe_hash=True)", code)
            code = code.replace("@dataclass()", "@dataclass(unsafe_hash=True)")
            
            if code != orig:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                count += 1

print(f"Глобальный патч применен! Успешно вылечено файлов: {count}")