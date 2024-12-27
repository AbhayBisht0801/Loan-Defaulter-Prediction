from pathlib import Path
from dataclasses import dataclass
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_dir:Path
    data_dir:Path
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir:Path
    split_dir:Path
    preprocess_obj:Path
    data_dir:Path