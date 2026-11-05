"""
标准答案适配：把不同来源的标准答案文件统一成 {题号: 答案}。

支持三种入口，因为标准答案在另一台机器上，格式未确认：
  .docx  读表格，按表头找“题号 / 题目 / 答案”三列
  .csv   按表头匹配列名，兼容 UTF-8(BOM) 与 GBK
  .json  支持 [{"id","question","answer"}, ...]、{"1": "答案", ...}、{"answers": [...]}

用法：
    python -m eval.answers <标准答案文件> --out data/private/answers.json
    python -m eval.answers <文件> --dry-run        # 只识别格式，不写文件

保密约束：这个模块只输出计数与覆盖率，不打印任何答案正文。
标准答案与派生文件都写在 data/private/ 下，该目录已被 .gitignore 排除。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import QUESTIONS_PATH, DATA_DIR

PRIVATE_DIR = DATA_DIR / "private"
DEFAULT_OUT = PRIVATE_DIR / "answers.json"

# 表头同义词。真实文件里的列名不一定叫“答案”，先把常见写法覆盖掉。
ID_KEYS = {"题号", "序号", "编号", "id", "no", "num", "number"}
QUESTION_KEYS = {"题目", "问题", "提问", "question", "q"}
ANSWER_KEYS = {"答案", "标准答案", "参考答案", "answer", "a", "gold"}


def _norm(s: str) -> str:
    """表头归一化：去空白、去括号内容、转小写。"""
    s = str(s).strip().lower().replace(" ", "").replace("\u3000", "")
    for ch in "()（）[]【】:：":
        s = s.replace(ch, "")
    return s


def _pick(header: List[str]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """从表头里找出 (题号列, 题目列, 答案列) 的下标。"""
    idx_id = idx_q = idx_a = None
    for i, h in enumerate(header):
        n = _norm(h)
        if idx_id is None and n in ID_KEYS:
            idx_id = i
        elif idx_a is None and n in ANSWER_KEYS:
            idx_a = i
        elif idx_q is None and n in QUESTION_KEYS:
            idx_q = i
    return idx_id, idx_q, idx_a


def _looks_like_id(s: str) -> bool:
    s = str(s).strip()
    return s.isdigit() and 1 <= int(s) <= 10000


def _from_rows(rows: List[List[str]], source: str) -> Tuple[Dict[int, str], dict]:
    """从二维表格提取答案。返回 (answers, diagnostics)。"""
    if not rows:
        raise ValueError("表格为空")

    header = [c for c in rows[0]]
    idx_id, idx_q, idx_a = _pick(header)
    start = 1
    mapping = "表头匹配"

    # 表头里找不到答案列时，退化为按位置取：如果第一列本身就是数字，
    # 认为格式是 [题号, 题目, 答案]；否则把最后一列当答案。
    if idx_a is None:
        first = rows[0][0] if rows[0] else ""
        if _looks_like_id(first):
            start = 0
            idx_id, idx_q, idx_a = 0, 1, len(rows[0]) - 1
            mapping = "无表头，按位置 [题号, 题目, 答案]"
        else:
            raise ValueError(f"没找到答案列，表头为 {header[:8]}")

    answers: Dict[int, str] = {}
    skipped = 0
    dup = 0
    for r in rows[start:]:
        if not r:
            continue
        if idx_id >= len(r) or idx_a >= len(r):
            skipped += 1
            continue
        raw_id = str(r[idx_id]).strip()
        if not _looks_like_id(raw_id):
            skipped += 1
            continue
        ans = str(r[idx_a]).strip()
        if not ans:
            skipped += 1
            continue
        qid = int(raw_id)
        if qid in answers:
            dup += 1
        answers[qid] = ans

    diag = {
        "format": source,
        "mapping": mapping,
        "header": header[:8],
        "id_col": idx_id,
        "question_col": idx_q,
        "answer_col": idx_a,
        "rows": len(rows),
        "parsed": len(answers),
        "skipped": skipped,
        "duplicates": dup,
    }
    return answers, diag


def load_docx(path: Path) -> Tuple[Dict[int, str], dict]:
    import docx
    doc = docx.Document(str(path))
    if not doc.tables:
        raise ValueError(f"{path.name} 里没有表格；若是纯段落格式，请导出成 CSV")
    table = doc.tables[0]
    rows = []
    for row in table.rows:
        rows.append([c.text.strip() for c in row.cells])
    return _from_rows(rows, "docx-table")


def load_csv(path: Path) -> Tuple[Dict[int, str], dict]:
    text = None
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"{path.name} 编码无法识别")
    rows = [row for row in csv.reader(text.splitlines()) if any(c.strip() for c in row)]
    return _from_rows(rows, "csv")


def load_json(path: Path) -> Tuple[Dict[int, str], dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("answers", data)
    answers: Dict[int, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if _looks_like_id(k) and isinstance(v, str) and v.strip():
                answers[int(k)] = v.strip()
        fmt = "json-dict"
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            qid = item.get("id", item.get("题号"))
            ans = item.get("answer")
            if qid is None or ans is None:
                continue
            if _looks_like_id(qid) and str(ans).strip():
                answers[int(qid)] = str(ans).strip()
        fmt = "json-list"
    else:
        raise ValueError("JSON 顶层不是对象或数组")
    diag = {"format": fmt, "parsed": len(answers), "rows": len(data)}
    return answers, diag


LOADERS = {".docx": load_docx, ".csv": load_csv, ".json": load_json}


def load_answers(path: Path) -> Tuple[Dict[int, str], dict]:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"不支持的格式 {ext}，请用 docx / csv / json")
    return LOADERS[ext](path)


def coverage(answers: Dict[int, str], questions: List[dict]) -> dict:
    """标准答案对题库的覆盖情况。"""
    qids = {int(q["id"]) for q in questions}
    hit = qids & set(answers)
    return {
        "questions": len(qids),
        "answered": len(hit),
        "missing": sorted(qids - set(answers))[:20],
        "extra": sorted(set(answers) - qids)[:20],
    }


def save(answers: Dict[int, str], diag: dict, out: Path, source: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {**diag, "source_file": source.name},
        "answers": {str(k): v for k, v in sorted(answers.items())},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_saved(path: Path = DEFAULT_OUT) -> Dict[int, str]:
    """读回 save() 写出的文件，给 run_eval 用。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in payload.get("answers", {}).items()}


def main() -> None:
    p = argparse.ArgumentParser(description="标准答案适配：docx / csv / json → answers.json")
    p.add_argument("path", help="标准答案文件")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--dry-run", action="store_true", help="只识别格式，不写文件")
    args = p.parse_args()

    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"文件不存在: {src}")

    answers, diag = load_answers(src)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    cov = coverage(answers, questions)

    print(f"来源文件: {src.name}（{src.suffix.lower()}）")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print(f"题库覆盖: {cov['answered']}/{cov['questions']}")
    if cov["missing"]:
        print(f"  题库中有、答案里没有的题号（前 20）: {cov['missing']}")
    if cov["extra"]:
        print(f"  答案里有、题库中没有的题号（前 20）: {cov['extra']}")

    if args.dry_run:
        print("dry-run：未写文件")
        return
    if cov["answered"] == 0:
        raise SystemExit("没有匹配到任何题目，未写文件。请核对题号列，或导出成 CSV 再试。")
    out = Path(args.out)
    save(answers, diag, out, src)
    print(f"已写入 {out}（该目录已被 .gitignore 排除）")


if __name__ == "__main__":
    main()
