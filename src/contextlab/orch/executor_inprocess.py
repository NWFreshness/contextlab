"""In-process tool executor.

Slice 7 ships one executor: `InProcessExecutor`. Slice 8 swaps in a sandboxed
implementation behind the same `Executor` protocol — the brief is explicit
that the protocol stays, the implementation changes. The tools registered
here are the ones listed in `config/orchestrator.yaml: tools`.

Tools in this slice:
    retrieve   {query, k?, mode?} -> calls Slice 1 retrieval, returns hits as
                                     a ToolResult whose content is a JSON
                                     payload the assembler can re-ingest.
    read_chunk {chunk_id}          -> loads a single chunk from data/chunks.jsonl
                                     and returns its text as a ToolResult.
                                     Missing chunk -> a ToolResult with a
                                     'missing' payload, no crash.
    finish     {answer}            -> handled by the machine, not the executor
                                     (finish is not a tool call).

The executor never raises on a missing chunk and never swallows an
unexpected exception silently: it returns a ToolResult with an `error` field
so the trajectory shows what went wrong.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Protocol

from contextlab.config import get_config
from contextlab.retrieve import retrieve as do_retrieve
from contextlab.types import Hit, ToolResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"


class UnknownToolError(Exception):
    """Raised by the machine when the policy asks for an unregistered tool.

    The machine catches this and stops with `stop_reason=unknown_tool`,
    which is a successful machine exit (not an exception out of `run()`).
    """


class Executor(Protocol):
    """Protocol. Slice 8 implementations must match this shape."""

    def run(self, tool: str, args: dict[str, Any]) -> ToolResult: ...


class InProcessExecutor:
    """Executes Slice 1 / Slice 2 helpers directly. No sandbox in this slice."""

    def __init__(
        self,
        chunks_path: Optional[Path] = None,
        registered_tools: Optional[list[str]] = None,
    ) -> None:
        self.chunks_path = Path(chunks_path) if chunks_path else DEFAULT_CHUNKS_PATH
        self._registered = set(registered_tools or ["retrieve", "read_chunk"])

    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        if tool not in self._registered:
            # Unknown tool is a machine-level error, not a tool result —
            # propagate so the machine can set stop_reason=unknown_tool.
            raise UnknownToolError(tool)
        try:
            if tool == "retrieve":
                return self._retrieve(args)
            if tool == "read_chunk":
                return self._read_chunk(args)
        except UnknownToolError:
            raise
        except Exception as exc:  # noqa: BLE001 — tool errors must not crash the run
            return ToolResult(
                tool_id=f"{tool}:error",
                name=tool,
                content=json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                must_keep=False,
            )
        # Defensive — only reachable if a registered tool name has no handler.
        raise UnknownToolError(tool)

    # ── Tool implementations ─────────────────────────────────────────────

    def _retrieve(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(
                tool_id="retrieve:empty",
                name="retrieve",
                content=json.dumps({"error": "missing query", "hits": []}),
                must_keep=False,
            )
        # Defaults come from the retrieval config so a test that overrides
        # retrieval.yaml also influences the orchestrator.
        cfg = get_config()
        k = int(args.get("k", cfg.get("hybrid_k", 5)))
        mode = str(args.get("mode", cfg.get("mode", "hybrid")))

        resp = do_retrieve(query=query, k=k, mode=mode)
        hits_payload = [h.model_dump() for h in resp.hits]
        payload = {
            "query": resp.query,
            "mode": resp.mode,
            "hits": hits_payload,
            "n_hits": len(hits_payload),
        }
        tool_id = f"retrieve:{query[:32]}"
        return ToolResult(
            tool_id=tool_id,
            name="retrieve",
            content=json.dumps(payload),
            must_keep=False,
        )

    def _read_chunk(self, args: dict[str, Any]) -> ToolResult:
        chunk_id = str(args.get("chunk_id", "")).strip()
        if not chunk_id:
            return ToolResult(
                tool_id="read_chunk:empty",
                name="read_chunk",
                content=json.dumps({"error": "missing chunk_id", "chunk_id": ""}),
                must_keep=False,
            )
        text, meta = self._load_chunk(chunk_id)
        payload: dict[str, Any] = {
            "chunk_id": chunk_id,
            "found": text is not None,
        }
        if text is not None:
            payload["text"] = text
            payload["doc_id"] = meta.get("doc_id")
            payload["section"] = meta.get("section")
        return ToolResult(
            tool_id=f"read_chunk:{chunk_id}",
            name="read_chunk",
            content=json.dumps(payload),
            must_keep=False,
        )

    def _load_chunk(self, chunk_id: str) -> tuple[Optional[str], dict]:
        """Read chunks.jsonl and return (text, {doc_id, section}) or (None, {}).

        Linear scan over a few dozen chunks is fine — `data/chunks.jsonl`
        is small and the orchestrator only loads it on a `read_chunk` call.
        """
        if not self.chunks_path.exists():
            return None, {}
        with open(self.chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("chunk_id") == chunk_id:
                    return row.get("text", ""), {
                        "doc_id": row.get("doc_id"),
                        "section": row.get("section"),
                    }
        return None, {}