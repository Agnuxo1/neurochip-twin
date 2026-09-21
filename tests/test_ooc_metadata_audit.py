from __future__ import annotations

from zipfile import ZipFile

from src.ooc_metadata_audit import read_xlsx_table, summarize_ooc_rows


def test_ooc_summary_preserves_quality_and_missingness() -> None:
    summary = summarize_ooc_rows(
        [
            {
                "imageID": "a",
                "cell type": "A549",
                "seeding density, cells/ml": "5,000,000.00",
                "time after seeding, h": "2",
                "day": "1",
                "Decision 1/2 (good/bad)": "1",
                "flow rate": "",
            },
            {
                "imageID": "b",
                "cell type": "A549",
                "seeding density, cells/ml": "",
                "time after seeding, h": "4",
                "day": "1",
                "Decision 1/2 (good/bad)": "2",
                "flow rate": "3.5 uL/min",
            },
        ]
    )
    assert summary["rows"] == 2
    assert summary["unique_nonempty_image_ids"] == 2
    assert summary["quality_counts_mapped"] == {"bad": 1, "good": 1}
    assert summary["numeric_fields"]["seeding_density_cells_per_ml"]["missing"] == 1
    assert summary["numeric_fields"]["flow_rate"]["max"] == 3.5
    assert summary["split_column_present"] is False


def test_xlsx_reader_keeps_empty_middle_cells(tmp_path) -> None:
    path = tmp_path / "table.xlsx"
    shared = """<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><si><t>id</t></si><si><t>quality</t></si><si><t>x</t></si><si><t>1</t></si></sst>"""
    sheet = """<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData><row r=\"1\"><c r=\"A1\" t=\"s\"><v>0</v></c><c r=\"B1\" t=\"s\"><v>1</v></c><c r=\"C1\" t=\"s\"><v>2</v></c></row><row r=\"2\"><c r=\"A2\" t=\"s\"><v>3</v></c><c r=\"C2\" t=\"s\"><v>3</v></c></row></sheetData></worksheet>"""
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)

    assert read_xlsx_table(path) == [
        {"id": "1", "quality": "", "x": "1"},
    ]
