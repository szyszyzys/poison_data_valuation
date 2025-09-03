from collections import deque
from dataclasses import field, dataclass
from typing import Optional


@dataclass
class ClientState:
    seller_obj: object  # The actual seller instance
    selection_history: deque = field(default_factory=lambda: deque(maxlen=20))  # Example maxlen
    selection_rate: float = 0.0
    rounds_participated: int = 0
    phase: str = "benign"  # "benign" or "attack"
    role: Optional[str] = "hybrid"  # "attacker", "explorer", or "hybrid"
