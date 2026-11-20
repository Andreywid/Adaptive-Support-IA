import json
import os
import tempfile
from pathlib import Path
from typing import Any


# grava JSON de forma atômica: temporário na mesma pasta e troca com os.replace
def atomic_write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix="." + path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        with temp:
            temp.write(text)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp.name, path)
    except BaseException:
        Path(temp.name).unlink(missing_ok=True)
        raise
