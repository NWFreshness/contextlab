"""In-process memory store for the assembler."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from contextlab.types import MemoryItem


class MemoryStore:
    """Simple in-memory list with optional JSONL persistence."""

    def __init__(self):
        self.items: list[MemoryItem] = []

    def add(
        self,
        text: str,
        memory_id: Optional[str] = None,
        priority: int = 0,
    ) -> MemoryItem:
        """Add a memory item."""
        if memory_id is None:
            memory_id = f"mem_{len(self.items):04d}"
        item = MemoryItem(
            memory_id=memory_id,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
            priority=priority,
        )
        self.items.append(item)
        return item

    def load(self, path: str | Path) -> None:
        """Load memory items from a JSONL file."""
        path = Path(path)
        if not path.exists():
            return
        self.items = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.items.append(MemoryItem.model_validate_json(line))

    def save(self, path: str | Path) -> None:
        """Save memory items to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for item in self.items:
                f.write(item.model_dump_json() + "\n")

    def sorted(self) -> list[MemoryItem]:
        """Return items sorted by (priority desc, created_at desc)."""
        return sorted(
            self.items,
            key=lambda x: (-x.priority, x.created_at),
            reverse=True,
        )

    def clear(self) -> None:
        self.items = []
