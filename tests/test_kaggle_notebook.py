import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "neurochip_twin_ai4s_demo.ipynb"
COMPETITION = "ai-4-s-open-innovation-artificial-intelligence-for-life-scien"


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_kaggle_notebook_embeds_current_core_source():
    notebook = _load_notebook()
    source = (ROOT / "src" / "neurochip_twin.py").read_text(encoding="utf-8")
    marker = "\nif __name__ == \"__main__\":"
    assert marker in source
    expected = source.rsplit(marker, 1)[0].rstrip() + "\n"
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    mirrored_source = "".join(code_cells[0]["source"])
    assert mirrored_source == expected
    compile(mirrored_source, "neurochip_twin.py", "exec")


def test_kaggle_notebook_is_competition_linked_and_offline():
    notebook = _load_notebook()
    metadata_path = NOTEBOOK.with_name("kernel-metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["competition_sources"] == [COMPETITION]
    assert metadata["code_file"] == NOTEBOOK.name
    assert metadata["id"].rsplit("/", 1)[-1] == "neurochip-twin-ai4s-reproducible-synthetic-demo"
    assert metadata["title"] == "Neurochip Twin AI4S Reproducible Synthetic Demo"
    assert metadata["is_private"] is False
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == []
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ).lower()
    assert "submission category: end-to-end system" in markdown
    assert "synthetic" in markdown
    assert "not neural organ-on-chip biological validation" in markdown
    run_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert 'scenario="compound_specific"' in "".join(run_cells[-1]["source"])
