from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class MatchStatus(Enum):
    """Status of file comparison"""
    MATCH = "match"  # SHA256 matches
    MISMATCH = "mismatch"  # SHA256 differs
    MISSING_LOCAL = "missing_local"  # File exists on HF but not locally
    MISSING_REMOTE = "missing_remote"  # File exists locally but not on HF
    NAME_MATCH_ONLY = "name_match_only"  # Name matches but no SHA256 to compare


@dataclass
class HFFileInfo:
    """File information from HuggingFace repository"""
    filename: str
    sha256: Optional[str] = None
    size: Optional[int] = None
    path: str = ""  # Full path in repo (e.g., "subfolder/file.safetensors")
    lfs: bool = False  # Whether file is stored in LFS
    
    @property
    def basename(self) -> str:
        """Get just the filename without path"""
        return self.filename.split("/")[-1] if "/" in self.filename else self.filename


@dataclass
class LocalFileInfo:
    """File information from local metadata JSON"""
    file_name: str
    sha256: Optional[str] = None
    file_path: Optional[str] = None
    size: Optional[int] = None
    model_name: Optional[str] = None
    base_model: Optional[str] = None
    metadata_path: str = ""  # Path to the JSON metadata file
    
    @property
    def basename(self) -> str:
        """Get just the filename"""
        if self.file_path:
            return self.file_path.replace("\\", "/").split("/")[-1]
        return self.file_name


@dataclass
class ComparisonResult:
    """Result of comparing a file"""
    filename: str
    status: MatchStatus
    local_sha256: Optional[str] = None
    remote_sha256: Optional[str] = None
    local_path: Optional[str] = None
    remote_path: Optional[str] = None
    local_size: Optional[int] = None
    remote_size: Optional[int] = None
    download_url: Optional[str] = None  # Direct download link (resolve URL)
    visit_url: Optional[str] = None     # Page view link (blob URL)
    notes: str = ""


@dataclass
class ComparisonSummary:
    """Summary of all comparisons"""
    total_hf_files: int = 0
    total_local_files: int = 0
    matches: List[ComparisonResult] = field(default_factory=list)
    mismatches: List[ComparisonResult] = field(default_factory=list)
    missing_local: List[ComparisonResult] = field(default_factory=list)
    missing_remote: List[ComparisonResult] = field(default_factory=list)
    name_matches_only: List[ComparisonResult] = field(default_factory=list)
    
    @property
    def match_count(self) -> int:
        return len(self.matches)
    
    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)
    
    @property
    def missing_local_count(self) -> int:
        return len(self.missing_local)
    
    @property
    def missing_remote_count(self) -> int:
        return len(self.missing_remote)