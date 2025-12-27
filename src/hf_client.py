"""HuggingFace API - fetching file lists, parsing URLs."""

from typing import List, Optional, Tuple, Literal
from huggingface_hub import HfApi, list_repo_tree, hf_hub_url
from huggingface_hub.hf_api import RepoFile
import re

from models import HFFileInfo

# Valid HuggingFace repository types
RepoType = Literal["model", "dataset", "space"]


class HuggingFaceClient:
    """Client to fetch file information from HuggingFace repositories"""
    
    def __init__(self, repo_id: str, revision: str = "main", repo_type: RepoType = "model", token: Optional[str] = None):
        """
        Initialize the HuggingFace client.
        
        Args:
            repo_id: Repository ID (e.g., "K3NK/loras-WAN")
            revision: Branch/revision to check (default: "main")
            repo_type: Type of repository ("model", "dataset", or "space")
            token: Optional HuggingFace token for private repos
        """
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
        self.token = token
        self.api = HfApi(token=token)
        
    @classmethod
    def from_url(cls, url: str, token: Optional[str] = None) -> "HuggingFaceClient":
        """
        Create client from a HuggingFace URL.
        
        Args:
            url: URL like "https://huggingface.co/K3NK/loras-WAN/tree/main"
                 or "https://huggingface.co/datasets/user/repo/tree/main"
                 or "https://huggingface.co/spaces/user/repo"
            token: Optional HuggingFace token
            
        Returns:
            HuggingFaceClient instance
        """
        repo_id, revision, repo_type = cls.parse_url(url)
        return cls(repo_id=repo_id, revision=revision, repo_type=repo_type, token=token)
    
    @staticmethod
    def parse_url(url: str) -> Tuple[str, str, RepoType]:
        """
        Parse HuggingFace URL to extract repo_id, revision, and repo_type.
        
        Supports:
        - https://huggingface.co/username/repo (model)
        - https://huggingface.co/username/repo/tree/main (model)
        - https://huggingface.co/datasets/username/repo (dataset)
        - https://huggingface.co/datasets/username/repo/tree/main (dataset)
        - https://huggingface.co/spaces/username/repo (space)
        - https://huggingface.co/spaces/username/repo/tree/main (space)
        
        Returns:
            Tuple of (repo_id, revision, repo_type)
        """
        # Remove trailing slashes
        url = url.rstrip("/")
        
        # Determine repo type from URL
        repo_type: RepoType = "model"
        if "/datasets/" in url:
            repo_type = "dataset"
        elif "/spaces/" in url:
            repo_type = "space"
        
        # Pattern for various HF URL formats
        # For datasets/spaces: huggingface.co/datasets/user/repo or huggingface.co/spaces/user/repo
        patterns = [
            # datasets/spaces tree/blob URLs with branch
            r"huggingface\.co/(?:datasets|spaces)/([^/]+/[^/]+)/(?:tree|blob)/([^/]+)",
            # datasets/spaces direct repo URL
            r"huggingface\.co/(?:datasets|spaces)/([^/]+/[^/]+)/?$",
            # model tree/blob URLs with branch
            r"huggingface\.co/([^/]+/[^/]+)/(?:tree|blob)/([^/]+)",
            # model direct repo URL
            r"huggingface\.co/([^/]+/[^/]+)/?$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                repo_id = match.group(1)
                revision = match.group(2) if len(match.groups()) > 1 and match.group(2) else "main"
                return repo_id, revision, repo_type
        
        raise ValueError(f"Could not parse HuggingFace URL: {url}")
    
    def fetch_all_files(self, path: str = "", recursive: bool = True) -> List[HFFileInfo]:
        """
        Fetch all files from the repository with their metadata.
        
        Args:
            path: Subdirectory path to start from (empty for root)
            recursive: Whether to recursively fetch all files
            
        Returns:
            List of HFFileInfo objects
        """
        files = []
        
        try:
            # Use list_repo_tree to get all files recursively
            repo_files = list_repo_tree(
                repo_id=self.repo_id,
                revision=self.revision,
                path_in_repo=path if path else None,
                recursive=recursive,
                token=self.token,
                repo_type=self.repo_type
            )
            
            for item in repo_files:
                # Only process files (not directories)
                if isinstance(item, RepoFile):
                    file_info = self._parse_repo_file(item)
                    if file_info:
                        files.append(file_info)
                        
        except Exception as e:
            raise RuntimeError(f"Failed to fetch files from {self.repo_id}: {e}")
        
        return files
    
    def fetch_safetensors_only(self, path: str = "") -> List[HFFileInfo]:
        """
        Fetch only .safetensors files from the repository.
        
        Args:
            path: Subdirectory path to start from
            
        Returns:
            List of HFFileInfo objects for safetensors files
        """
        all_files = self.fetch_all_files(path=path, recursive=True)
        return [f for f in all_files if f.filename.endswith(".safetensors")]
    
    def _parse_repo_file(self, repo_file: RepoFile) -> Optional[HFFileInfo]:
        """
        Parse a RepoFile object into HFFileInfo.
        
        The SHA256 for LFS files is stored in the blob_id (which is the LFS OID).
        """
        try:
            # For LFS files, the lfs attribute contains the SHA256
            sha256 = None
            is_lfs = False
            size = repo_file.size
            
            # Check if this is an LFS file
            if hasattr(repo_file, 'lfs') and repo_file.lfs:
                is_lfs = True
                # LFS object has sha256 as the oid
                if hasattr(repo_file.lfs, 'sha256'):
                    sha256 = repo_file.lfs.sha256
                elif isinstance(repo_file.lfs, dict):
                    sha256 = repo_file.lfs.get('sha256') or repo_file.lfs.get('oid')
                    
            return HFFileInfo(
                filename=repo_file.path,
                sha256=sha256,
                size=size,
                path=repo_file.path,
                lfs=is_lfs
            )
        except Exception as e:
            print(f"Warning: Could not parse file {repo_file.path}: {e}")
            return None
    
    def get_file_info(self, filename: str) -> Optional[HFFileInfo]:
        """
        Get info for a specific file.
        
        Args:
            filename: Path to file in repo
            
        Returns:
            HFFileInfo or None if not found
        """
        all_files = self.fetch_all_files()
        for f in all_files:
            if f.filename == filename or f.basename == filename:
                return f
        return None