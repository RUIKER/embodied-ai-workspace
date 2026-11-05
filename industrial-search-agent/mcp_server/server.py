"""
第3周:MCP Server —— 把知识库检索封装为 MCP 工具,供 Agent 调用。

暴露工具:
1. industrial_knowledge_search(query, mode?)   通用知识检索,返回 topK 片段+来源
2. enterprise_query(name)                       企业查询(按企业名过滤检索)
3. component_param(name)                        零部件参数查询(按零部件关键词检索)

协议:Streamable HTTP / JSON,符合 AgentOS MCP 规范。

运行模式:
- 若已安装 `mcp` SDK(Python 3.10+):以标准 MCP stdio/http server 启动,可直接对接 AgentOS。
- 否则(Python 3.9 等):以 FastAPI 暴露等价的 JSON-RPC 风格 HTTP 接口,作为本地 demo 回退。

Agent 调用流程(简历要点):
  用户问题 → Agent(LLM)判断是否调用工具 → 调用 MCP Server
    → 工具内部走 RAG 检索 → 返回片段+来源 → Agent 据此生成最终回答
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import retrieve
from config import FINAL_TOP_K


# ===== 工具实现(与协议无关,纯函数)=====
def tool_industrial_knowledge_search(query: str, mode: str = "rerank", top_k: int = FINAL_TOP_K) -> dict:
    """通用知识检索工具。"""
    results = retrieve(query, top_k=top_k, mode=mode)
    return {
        "query": query,
        "mode": mode,
        "count": len(results),
        "results": [{
            "id": r.get("id"),
            "text": r["text"],
            "category": r.get("category"),
            "layer": r.get("layer"),
            "source": r.get("source"),
            "score": round(r.get("rerank_score", r.get("score", 0.0)), 4),
        } for r in results],
    }


def tool_enterprise_query(name: str, top_k: int = 5) -> dict:
    """企业查询工具:以企业名为查询做检索。"""
    return tool_industrial_knowledge_search(name, mode="rerank", top_k=top_k)


def tool_component_param(name: str, top_k: int = 5) -> dict:
    """零部件参数查询工具:以零部件名为查询做检索。"""
    return tool_industrial_knowledge_search(name, mode="hybrid", top_k=top_k)


# 工具注册表(供两种启动模式共用)
TOOLS = {
    "industrial_knowledge_search": {
        "description": "工业具身智能知识库语义检索。输入自然语言query,返回最相关的知识片段及来源。mode可选:bm25/vector/hybrid/rerank(默认rerank)。",
        "func": tool_industrial_knowledge_search,
        "params": {"query": "str", "mode": "str", "top_k": "int"},
    },
    "enterprise_query": {
        "description": "按企业名查询其在知识库中的相关信息(产品、地位、市占率等)。",
        "func": tool_enterprise_query,
        "params": {"name": "str", "top_k": "int"},
    },
    "component_param": {
        "description": "按零部件名(如减速器、伺服电机、传感器)查询参数与供应商信息。",
        "func": tool_component_param,
        "params": {"name": "str", "top_k": "int"},
    },
}


# ===== MCP SDK 模式(Python 3.10+)=====
def run_mcp_server():
    """用 mcp SDK 启动标准 MCP server(Streamable HTTP)。"""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("industrial-knowledge")

    @mcp.tool()
    def industrial_knowledge_search(query: str, mode: str = "rerank", top_k: int = FINAL_TOP_K) -> str:
        """工业知识库语义检索,返回 topK 片段+来源。"""
        return json.dumps(tool_industrial_knowledge_search(query, mode, top_k), ensure_ascii=False)

    @mcp.tool()
    def enterprise_query(name: str, top_k: int = 5) -> str:
        """按企业名查询知识库相关信息。"""
        return json.dumps(tool_enterprise_query(name, top_k), ensure_ascii=False)

    @mcp.tool()
    def component_param(name: str, top_k: int = 5) -> str:
        """按零部件名查询参数与供应商。"""
        return json.dumps(tool_component_param(name, top_k), ensure_ascii=False)

    # Streamable HTTP
    mcp.run(transport="streamable-http")


# ===== FastAPI 回退模式(Python 3.9 本地 demo)=====
def run_fastapi_server(host: str = "127.0.0.1", port: int = 8000):
    """无 mcp SDK 时,用 FastAPI 暴露等价 JSON 接口,便于本地 demo。"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    app = FastAPI(title="Industrial Knowledge MCP (FastAPI fallback)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/tools")
    def list_tools():
        return {"tools": {k: {"description": v["description"], "params": v["params"]} for k, v in TOOLS.items()}}

    @app.post("/tools/{name}")
    def call_tool(name: str, params: dict):
        if name not in TOOLS:
            return {"error": f"unknown tool: {name}"}
        try:
            return TOOLS[name]["func"](**params)
        except TypeError as e:
            return {"error": f"参数错误: {e}"}

    print(f"MCP(FastAPI 回退)启动:http://{host}:{port}/tools  (Python 3.9 本地 demo)")
    print("调用示例: POST /tools/industrial_knowledge_search  body: {\"query\":\"谐波减速器\"}")
    uvicorn.run(app, host=host, port=port)


def main():
    # 优先标准 MCP SDK,不可用则回退 FastAPI
    try:
        import mcp  # noqa: F401
        run_mcp_server()
    except ImportError:
        print("[提示] 未安装 mcp SDK(需 Python 3.10+),回退到 FastAPI 本地 demo 模式。")
        run_fastapi_server()


if __name__ == "__main__":
    main()
