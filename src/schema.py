from dataclasses import dataclass
from typing import Optional

@dataclass
class DataCfg:
    target: str
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelCfg:
    C: float = 1.0
    max_iter: int = 200

@dataclass
class GACfg:
    population_size: int = 40
    generations: int = 20
    tournament_k: int = 3
    crossover_prob: float = 0.8
    mutation_prob: float = 0.05
    elitism: int = 1
    max_features: Optional[int] = None
