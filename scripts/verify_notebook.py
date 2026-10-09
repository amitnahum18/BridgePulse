import sys
import re
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks" / "experiment.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
client = NotebookClient(notebook, timeout=180, kernel_name="bridgepulse", resources={"metadata": {"path": str(path.parent)}})
client.execute()
images = sum("image/png" in output.get("data", {}) for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs)
assert images >= 2, f"Expected two plots, found {images}"
assert not any(re.search(r"\?{2,}|\ufffd", cell.source) for cell in notebook.cells)
nbformat.write(notebook, path)
print(f"Executed notebook successfully with {images} embedded plots.")
print(f"Environment: {sys.executable}")
