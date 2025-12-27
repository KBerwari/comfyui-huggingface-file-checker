"""Matches files by SHA256, falls back to filename if needed."""

from typing import List, Dict, Optional, Set, Literal
from pathlib import Path
from urllib.parse import quote

from models import (
    HFFileInfo, 
    LocalFileInfo, 
    ComparisonResult, 
    ComparisonSummary,
    MatchStatus
)


class Comparator:
    """Compares local files against HuggingFace repository files"""
    
    def __init__(
        self, 
        local_files: List[LocalFileInfo], 
        hf_files: List[HFFileInfo],
        repo_id: str = "",
        repo_type: Literal["model", "dataset", "space"] = "model",
        revision: str = "main",
        match_by_sha256: bool = True,
        match_by_name: bool = True
    ):
        """
        Initialize comparator.
        
        Args:
            local_files: List of local file info
            hf_files: List of HuggingFace file info
            repo_id: HuggingFace repo ID for generating download URLs
            repo_type: Type of repo (model/dataset/space)
            revision: Branch/revision
            match_by_sha256: Whether to match by SHA256 hash
            match_by_name: Whether to match by filename (fallback)
        """
        self.local_files = local_files
        self.hf_files = hf_files
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.match_by_sha256 = match_by_sha256
        self.match_by_name = match_by_name
        
        # Build indexes for fast lookup
        self._local_sha256_index: Dict[str, LocalFileInfo] = {}
        self._local_name_index: Dict[str, List[LocalFileInfo]] = {}
        self._build_local_indexes()
        
        self._hf_sha256_index: Dict[str, HFFileInfo] = {}
        self._hf_name_index: Dict[str, HFFileInfo] = {}
        self._build_hf_indexes()
    
    def _get_download_url(self, file_path: str) -> str:
        """Generate direct download URL for a HuggingFace file (resolve URL)."""
        if not self.repo_id:
            return ""
        
        # URL encode the file path (but keep slashes)
        encoded_path = "/".join(quote(part, safe="") for part in file_path.split("/"))
        
        # Different base URLs for different repo types
        if self.repo_type == "dataset":
            return f"https://huggingface.co/datasets/{self.repo_id}/resolve/{self.revision}/{encoded_path}"
        elif self.repo_type == "space":
            return f"https://huggingface.co/spaces/{self.repo_id}/resolve/{self.revision}/{encoded_path}"
        else:
            return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{encoded_path}"
    
    def _get_visit_url(self, file_path: str) -> str:
        """Generate page view URL for a HuggingFace file (blob URL)."""
        if not self.repo_id:
            return ""
        
        # URL encode the file path (but keep slashes)
        encoded_path = "/".join(quote(part, safe="") for part in file_path.split("/"))
        
        # Different base URLs for different repo types
        if self.repo_type == "dataset":
            return f"https://huggingface.co/datasets/{self.repo_id}/blob/{self.revision}/{encoded_path}"
        elif self.repo_type == "space":
            return f"https://huggingface.co/spaces/{self.repo_id}/blob/{self.revision}/{encoded_path}"
        else:
            return f"https://huggingface.co/{self.repo_id}/blob/{self.revision}/{encoded_path}"
    
    def _build_local_indexes(self):
        """Build indexes for local files."""
        for f in self.local_files:
            if f.sha256:
                self._local_sha256_index[f.sha256.lower()] = f
            
            # Index by various name forms
            basename = f.basename.lower()
            if basename not in self._local_name_index:
                self._local_name_index[basename] = []
            self._local_name_index[basename].append(f)
            
            # Also index by stem
            stem = Path(basename).stem
            if stem not in self._local_name_index:
                self._local_name_index[stem] = []
            if f not in self._local_name_index[stem]:
                self._local_name_index[stem].append(f)
    
    def _build_hf_indexes(self):
        """Build indexes for HuggingFace files."""
        for f in self.hf_files:
            if f.sha256:
                self._hf_sha256_index[f.sha256.lower()] = f
            
            # Index by basename
            basename = f.basename.lower()
            self._hf_name_index[basename] = f
            
            # Also index by stem
            stem = Path(basename).stem
            self._hf_name_index[stem] = f
    
    def compare(self) -> ComparisonSummary:
        """
        Compare all files and return a summary.
        
        The comparison checks:
        1. For each HF file, is there a matching local file (by SHA256 or name)?
        2. Reports files that exist only on HF (missing locally)
        3. Reports files that exist only locally (not on HF)
        
        Returns:
            ComparisonSummary with detailed results
        """
        summary = ComparisonSummary(
            total_hf_files=len(self.hf_files),
            total_local_files=len(self.local_files)
        )
        
        matched_local_files: Set[str] = set()  # Track which local files were matched
        
        # Check each HF file against local files
        for hf_file in self.hf_files:
            result = self._compare_hf_file(hf_file)
            
            # Track matched local files
            if result.status in (MatchStatus.MATCH, MatchStatus.MISMATCH, MatchStatus.NAME_MATCH_ONLY):
                if result.local_path:
                    matched_local_files.add(result.local_path)
            
            # Add to appropriate list
            if result.status == MatchStatus.MATCH:
                summary.matches.append(result)
            elif result.status == MatchStatus.MISMATCH:
                summary.mismatches.append(result)
            elif result.status == MatchStatus.MISSING_LOCAL:
                summary.missing_local.append(result)
            elif result.status == MatchStatus.NAME_MATCH_ONLY:
                summary.name_matches_only.append(result)
        
        # Find local files that weren't matched to any HF file
        # (These are files you have locally that aren't in the HF repo)
        # This is commented out by default as it may not be desired
        # for local_file in self.local_files:
        #     local_key = local_file.metadata_path
        #     if local_key not in matched_local_files:
        #         result = ComparisonResult(
        #             filename=local_file.basename,
        #             status=MatchStatus.MISSING_REMOTE,
        #             local_sha256=local_file.sha256,
        #             local_path=local_file.file_path,
        #             local_size=local_file.size,
        #             notes="File exists locally but not found in HuggingFace repository"
        #         )
        #         summary.missing_remote.append(result)
        
        return summary
    
    def _compare_hf_file(self, hf_file: HFFileInfo) -> ComparisonResult:
        """
        Compare a single HF file against local files.
        
        Priority:
        1. Match by SHA256 (exact match)
        2. Match by filename
        """
        # Try to match by SHA256 first (most reliable)
        if self.match_by_sha256 and hf_file.sha256:
            sha256_lower = hf_file.sha256.lower()
            if sha256_lower in self._local_sha256_index:
                local_file = self._local_sha256_index[sha256_lower]
                return ComparisonResult(
                    filename=hf_file.basename,
                    status=MatchStatus.MATCH,
                    local_sha256=local_file.sha256,
                    remote_sha256=hf_file.sha256,
                    local_path=local_file.file_path,
                    remote_path=hf_file.path,
                    local_size=local_file.size,
                    remote_size=hf_file.size,
                    notes="SHA256 match"
                )
        
        # Try to match by filename
        if self.match_by_name:
            basename = hf_file.basename.lower()
            stem = Path(basename).stem
            
            local_matches = self._local_name_index.get(basename, [])
            if not local_matches:
                local_matches = self._local_name_index.get(stem, [])
            
            if local_matches:
                local_file = local_matches[0]  # Take first match
                
                # Check if SHA256 matches
                if hf_file.sha256 and local_file.sha256:
                    if hf_file.sha256.lower() == local_file.sha256.lower():
                        return ComparisonResult(
                            filename=hf_file.basename,
                            status=MatchStatus.MATCH,
                            local_sha256=local_file.sha256,
                            remote_sha256=hf_file.sha256,
                            local_path=local_file.file_path,
                            remote_path=hf_file.path,
                            local_size=local_file.size,
                            remote_size=hf_file.size,
                            notes="SHA256 match (found by name)"
                        )
                    else:
                        return ComparisonResult(
                            filename=hf_file.basename,
                            status=MatchStatus.MISMATCH,
                            local_sha256=local_file.sha256,
                            remote_sha256=hf_file.sha256,
                            local_path=local_file.file_path,
                            remote_path=hf_file.path,
                            local_size=local_file.size,
                            remote_size=hf_file.size,
                            download_url=self._get_download_url(hf_file.path),
                            visit_url=self._get_visit_url(hf_file.path),
                            notes="Filename matches but SHA256 differs - possible different version"
                        )
                else:
                    # Name matches but can't verify SHA256
                    return ComparisonResult(
                        filename=hf_file.basename,
                        status=MatchStatus.NAME_MATCH_ONLY,
                        local_sha256=local_file.sha256,
                        remote_sha256=hf_file.sha256,
                        local_path=local_file.file_path,
                        remote_path=hf_file.path,
                        local_size=local_file.size,
                        remote_size=hf_file.size,
                        notes="Filename matches but SHA256 not available for verification"
                    )
        
        # No match found - file is missing locally
        return ComparisonResult(
            filename=hf_file.basename,
            status=MatchStatus.MISSING_LOCAL,
            remote_sha256=hf_file.sha256,
            remote_path=hf_file.path,
            remote_size=hf_file.size,
            download_url=self._get_download_url(hf_file.path),
            visit_url=self._get_visit_url(hf_file.path),
            notes="File not found in local metadata"
        )
    
    def find_local_file_by_hf_file(self, hf_file: HFFileInfo) -> Optional[LocalFileInfo]:
        """Find a local file that matches a HF file."""
        # Try SHA256 first
        if hf_file.sha256:
            sha256_lower = hf_file.sha256.lower()
            if sha256_lower in self._local_sha256_index:
                return self._local_sha256_index[sha256_lower]
        
        # Try by name
        basename = hf_file.basename.lower()
        if basename in self._local_name_index:
            return self._local_name_index[basename][0]
        
        return None