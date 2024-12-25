from pathlib import Path
from dataclasses import dataclass
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_dir:Path
    split_dir:Path
    
