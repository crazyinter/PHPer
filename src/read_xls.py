from pathlib import Path
import xlrd

C = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\data\raw\CHERRY-main")

for name in ["Interactiondata/VHM_PAIR_TAX.xls", "Interactiondata/TEST_PAIR_TAX.xls"]:
    print("=" * 70)
    print(name)
    book = xlrd.open_workbook(str(C / name))
    for sh in book.sheets():
        print(f"sheet: {sh.name} rows={sh.nrows} cols={sh.ncols}")
        for r in range(min(6, sh.nrows)):
            vals = [str(sh.cell_value(r, c))[:28] for c in range(min(sh.ncols, 12))]
            print("  ", vals)
