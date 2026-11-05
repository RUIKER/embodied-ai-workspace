"""
端到端冒烟：用**虚构语料**跑 ingest → build_index → 标准答案适配 → 四档评估。

作用：在真实语料还没到本机之前，验证链路能通、代码没坏、指标不是恒为 0。
约束：只读 data/smoke/ 下的虚构文本，不碰 data/raw/；产物全部写在 data/smoke/。

用法：
    python scripts/smoke_test.py            # 完整链路（首次会下载 bge 模型）
    python scripts/smoke_test.py --no-index # 跳过建索引（只验证切分与答案适配）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "smoke"

# 必须在 import config 之前设好，config 在导入时读环境变量
os.environ["KB_RAW_DIR"] = str(SMOKE / "raw")
os.environ["KB_PROCESSED_DIR"] = str(SMOKE / "processed")
os.environ["KB_CHROMA_DIR"] = str(SMOKE / "chroma_db")
os.environ["KB_QUESTIONS"] = str(SMOKE / "questions.json")
os.environ["KB_EVAL_DIR"] = str(SMOKE)
os.environ["KB_COLLECTION"] = "smoke_test"

sys.path.insert(0, str(ROOT))

DOCS = {
    "虚构_政策.txt": """【虚构语料，仅用于冒烟测试，与真实知识库无关】
临江市于 2025 年发布《智能装备产业跃升行动计划》，提出到 2027 年智能装备产业规模突破 800 亿元，
年均增速保持在 15% 以上，其中具身智能整机及核心零部件产值占比不低于三成。
行动计划明确三项重点任务：一是建设智能装备中试基地，二是设立每年 3 亿元的产业引导基金，
三是引进培育 50 家专精特新企业。政策同时提出对首台套装备给予最高 20% 的购置补贴，
对首次采购本地整机产品的企业再给予 5% 的额外奖励。

牵头单位是临江市工业和信息化局，配套文件包括《智能装备产业人才专项实施细则》与
《中试基地认定与管理办法》。人才细则规定，对年薪 50 万元以上的核心技术人员，
按其个人所得税地方留成部分给予三年全额补贴；对整建制引进的创新团队，
给予最高 500 万元的项目资助。

资金支持之外，行动计划还提出用地保障：对列入市重点的智能装备项目，
优先安排新增建设用地指标，允许分期缴纳土地出让价款，最长不超过两年。
对租用标准厂房的企业，前三年租金按市场价的 50% 收取。

在空间布局上，临江市规划了"一核两翼"的产业格局：核心区位于临江高新区，
重点布局整机与核心零部件；东翼为临港片区，侧重物流机器人与港口自动化；
西翼为临江经开区，侧重系统集成与检测认证服务。

行动计划同时设定了考核指标：到 2027 年，全市智能装备规上企业研发投入强度不低于 4%，
每万人高价值发明专利拥有量达到 25 件，培育省级以上创新平台 15 个。
考核由市工信局牵头，每年一季度对上一年度完成情况进行评估并向社会公布。
""",
    "虚构_产业链.txt": """【虚构语料，仅用于冒烟测试，与真实知识库无关】
智能装备产业链分为上游核心零部件、中游本体制造、下游系统集成三个环节。
上游核心零部件包括谐波减速器、无框力矩电机、六维力传感器与实时控制器，
其中谐波减速器与六维力传感器的国产化率最低，是产业链最容易被卡住的两环。

临江精工是国内谐波减速器的主要供应商之一，2024 年出货量约 18 万台，在国内市场占有率约 12%，
仅次于两家日本厂商。其产品精度等级覆盖 15 弧秒到 1 弧分，寿命测试可达 1 万小时。
龙辰传动主攻行星滚柱丝杠，产品主要用于人形机器人线性关节，单台价值量约 3000 元，
2024 年小批量出货 2 万套，良品率约 85%。

传感器环节，本地企业以六维力传感器和柔性触觉传感器为主，但高端产品仍依赖进口。
六维力传感器的国产价格约为进口产品的六成，精度指标差 1 到 2 个等级，
在打磨、装配等对力控精度要求高的场景中，客户仍倾向选择进口件。

中游本体环节，临江本地整机企业数量偏少，多数企业仍以系统集成为主，这是产业链的主要短板。
全市规模以上智能装备企业约 320 家，其中整机企业不到 30 家，产值占比不足 20%。
整机企业少带来的直接后果是：核心零部件企业缺少本地验证场景，产品迭代速度慢于沿海地区。

下游系统集成环节相对成熟，本地集成商在汽车零部件、光伏、锂电三个行业积累了较多案例。
集成项目的平均毛利率约 22%，低于整机环节的 35%，但回款周期更短，现金流压力小。
产业链的另一个结构性问题是人才：减速器设计、力控算法两类岗位的本地供给缺口约 40%。
""",
    "虚构_场景.txt": """【虚构语料，仅用于冒烟测试，与真实知识库无关】
临江市规划了三类具身智能应用场景：工业制造、公共服务、特种作业。
工业制造场景以汽车零部件产线的柔性上下料为代表，单条产线可替代 6 名工人，
投资回收期约 2.5 年，目前已落地 18 条产线，主要集中在武进片区的零部件园区。

公共服务场景包括科技馆导览机器人与社区助老陪护，目前处于试点阶段，覆盖 12 个社区。
导览机器人日均交互约 300 次，常见问题集中在展项位置与参观路线，
陪护机器人则以用药提醒和跌倒检测为主，误报率约 5%。

特种作业场景面向电力巡检与化工园区巡检，对防护等级和续航要求最高。
电力巡检机器人需要在高电磁干扰环境下工作，防护等级要求 IP65 以上，
单次续航不低于 6 小时；化工园区巡检还要求防爆认证，整机成本因此高出普通机型约 40%。

三类场景的共性瓶颈是数据采集成本高、场景碎片化，单场景难以摊薄部署成本。
调研显示，一个中等规模场景的数据采集与标注费用约占项目总投入的 35%，
而场景之间的数据复用率不足 20%，导致每进入一个新场景都要重新投入。

在商业模型上，本地企业主要采用三种方式：整机销售、租赁运营、按效果付费。
整机销售的毛利最高但回款慢；租赁运营的毛利约 30%，客户接受度最高；
按效果付费目前只有两家企业在做，主要面向物流分拣场景，按件计费，单价约 0.03 元。
""",
}

QUESTIONS = [
    (1, "临江市智能装备产业跃升行动计划提出到 2027 年产业规模突破多少？",
     "到2027年智能装备产业规模突破800亿元"),
    (2, "该行动计划提出的三项重点任务是什么？",
     "建设智能装备中试基地、设立每年3亿元的产业引导基金、引进培育50家专精特新企业"),
    (3, "首台套装备的购置补贴比例是多少？",
     "对首台套装备给予最高20%的购置补贴"),
    (4, "智能装备产业链的上游包括哪些核心零部件？",
     "谐波减速器、无框力矩电机、六维力传感器与实时控制器"),
    (5, "临江精工的谐波减速器国内市场占有率是多少？",
     "2024年出货量约18万台，国内市场占有率约12%"),
    (6, "行星滚柱丝杠的单台价值量是多少？",
     "单台价值量约3000元"),
    (7, "临江产业链的主要短板是什么？",
     "中游本体环节整机企业数量偏少，多数企业仍以系统集成为主"),
    (8, "临江市规划了哪三类具身智能应用场景？",
     "工业制造、公共服务、特种作业"),
    (9, "公共服务场景目前覆盖多少个社区？",
     "覆盖12个社区"),
    (10, "三类应用场景的共性瓶颈是什么？",
     "数据采集成本高、场景碎片化，单场景难以摊薄部署成本"),
]


def make_corpus():
    if SMOKE.exists():
        shutil.rmtree(SMOKE)
    (SMOKE / "raw").mkdir(parents=True)
    for name, text in DOCS.items():
        (SMOKE / "raw" / name).write_text(text, encoding="utf-8")
    qs = [{"id": i, "question": q, "answer": None} for i, q, _ in QUESTIONS]
    (SMOKE / "questions.json").write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[1/4] 虚构语料 {len(DOCS)} 篇、题目 {len(qs)} 道 → {SMOKE}")


def make_answers():
    from eval.answers import save
    answers = {i: a for i, _, a in QUESTIONS}
    diag = {"format": "synthetic", "parsed": len(answers), "rows": len(QUESTIONS)}
    out = SMOKE / "answers.json"
    save(answers, diag, out, Path("synthetic.json"))
    print(f"[2/4] 标准答案 {len(answers)} 条 → {out}")
    return out


def run_pipeline(no_index: bool):
    from knowledge.ingest import ingest
    stats = ingest()
    if not stats.get("total"):
        raise SystemExit("切分结果为 0，冒烟失败")
    print(f"[3/4] ingest 完成，{stats['total']} 个 chunk")
    if no_index:
        print("[4/4] 跳过建索引")
        return
    from knowledge.build_index import build_index
    coll = build_index()
    print(f"[4/4] 索引完成，{coll.count()} 条")


def run_eval(answers_path: Path):
    from eval.run_eval import run_eval as _run
    return _run(answers_path=answers_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    args = p.parse_args()

    make_corpus()
    answers_path = make_answers()
    run_pipeline(args.no_index)
    if args.no_index or args.no_eval:
        return
    table = run_eval(answers_path)
    ok = table and any(r["hit_rate"] > 0 for r in table)
    print("\n冒烟结果:", "通过" if ok else "失败（四档命中率全为 0）")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
