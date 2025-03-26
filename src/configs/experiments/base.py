from dataclasses import dataclass, asdict



@dataclass
class Experiment:
    def to_dict(self):
        return asdict(self)
