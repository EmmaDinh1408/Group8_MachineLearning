"""Tiện ích dựng notebook: nhúng mã module vào notebook bằng %%writefile để Run all chạy độc lập."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADER = """# {title}

**Môn học:** Học máy (Machine Learning) — Học kỳ I, năm học 2026-2027  
**Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách Khoa, ĐHQG-HCM**  
**GVHD:** *[điền tên giảng viên]*  
**Nhóm:** *[tên nhóm]* — *[Họ tên – MSSV – email]* (2-3 thành viên)  
**Mã nguồn GitHub:** *[điền link repo nếu có]*

> **Hướng dẫn chạy:** chọn *Runtime → Change runtime type → T4 GPU* (khuyến nghị), sau đó *Runtime → Run all*.
> Notebook tự cài thư viện, tự tải dữ liệu công khai về máy chủ Colab, **không** mount Google Drive.
"""


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


def module_cell(fname):
    with open(os.path.join(ROOT, "modules", fname), encoding="utf-8") as f:
        body = f.read()
    return code(f"%%writefile modules/{fname}\n{body}")


def build(path, title, cells, modules):
    nb = nbf.v4.new_notebook()
    cells = [md(HEADER.format(title=title))] + cells
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    }
    out = os.path.join(ROOT, "notebooks", path)
    nbf.write(nb, out)
    return out


def setup_cells(pip_line, modules, extra_imports=""):
    cells = [
        md("## 0. Cài đặt môi trường & nạp module\n\nCác module Python hỗ trợ (`modules/*.py`) được ghi trực tiếp ra máy chủ Colab bằng `%%writefile` "
           "(cùng nội dung với thư mục `modules/` trong bài nộp / GitHub), nên notebook chạy độc lập mà không cần Drive."),
        code(f"{pip_line}\n!mkdir -p modules reports/figures features data"),
    ]
    cells += [module_cell(m) for m in modules]
    cells.append(code(
        "import sys, warnings\n"
        "sys.path.insert(0, 'modules')\n"
        "warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "from common import *\n"
        "set_seed(42); ensure_dirs()\n"
        f"{extra_imports}"
    ))
    return cells
