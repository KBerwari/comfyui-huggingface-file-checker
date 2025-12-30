"""
Scans local directories for metadata JSON files and extracts SHA256 hashes.
Caches results so we don't have to re-parse thousands of files every run.

Can also scan actual model files directly and calculate their SHA256 hashes
(slower, but works without metadata files).
"""

import os
import json
import hashlib
import sqlite3
from typing import List, Optional, Dict, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn, SpinnerColumn

from models import LocalFileInfo


# SQLite-based cache
class SQLiteMetadataCache:
    """SQLite-based cache for file metadata."""
    
    SCHEMA_VERSION = 1
    
    def __init__(self, cache_path: Path, directory: str):
        self.cache_path = cache_path
        self.directory = directory
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                
                CREATE TABLE IF NOT EXISTS file_cache (
                    file_path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    file_name TEXT,
                    sha256 TEXT,
                    model_name TEXT,
                    base_model TEXT,
                    actual_file_path TEXT,
                    metadata_path TEXT,
                    file_size INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS idx_sha256 ON file_cache(sha256);
            """)
            conn.execute("INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('directory', ?)", (self.directory,))
            conn.execute("INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('version', ?)", (str(self.SCHEMA_VERSION),))
            conn.commit()
    
    def is_valid_for_directory(self, directory: str) -> bool:
        """Check if cache is valid for the given directory."""
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                cursor = conn.execute("SELECT value FROM cache_meta WHERE key = 'directory'")
                row = cursor.fetchone()
                return row and row[0] == directory
        except:
            return False
    
    def get_entry(self, file_path: str) -> Optional[Tuple[float, int, Optional[LocalFileInfo]]]:
        """Get cached entry for a file path."""
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM file_cache WHERE file_path = ?", (file_path,))
                row = cursor.fetchone()
                if not row:
                    return None
                
                file_info = None
                if row['file_name'] or row['sha256']:
                    file_info = LocalFileInfo(
                        file_name=row['file_name'] or '',
                        sha256=row['sha256'],
                        file_path=row['actual_file_path'],
                        size=row['file_size'],
                        model_name=row['model_name'],
                        base_model=row['base_model'],
                        metadata_path=row['metadata_path'] or ''
                    )
                return (row['mtime'], row['size'], file_info)
        except:
            return None
    
    def set_entry(self, file_path: str, mtime: float, size: int, file_info: Optional[LocalFileInfo]):
        """Store or update a cache entry."""
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO file_cache 
                    (file_path, mtime, size, file_name, sha256, model_name, base_model, actual_file_path, metadata_path, file_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path, mtime, size,
                    file_info.file_name if file_info else None,
                    file_info.sha256 if file_info else None,
                    file_info.model_name if file_info else None,
                    file_info.base_model if file_info else None,
                    file_info.file_path if file_info else None,
                    file_info.metadata_path if file_info else None,
                    file_info.size if file_info else None,
                ))
                conn.commit()
        except:
            pass
    
    def cleanup_stale(self, valid_paths: set):
        """Remove entries for files that no longer exist."""
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                cursor = conn.execute("SELECT file_path FROM file_cache")
                all_paths = [row[0] for row in cursor.fetchall()]
                stale = [p for p in all_paths if p not in valid_paths]
                if stale:
                    conn.executemany("DELETE FROM file_cache WHERE file_path = ?", [(p,) for p in stale])
                    conn.commit()
        except:
            pass
    
    def clear(self):
        """Clear all cache entries."""
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                conn.execute("DELETE FROM file_cache")
                conn.commit()
        except:
            pass


class LocalScanner:
    """Scanner for local metadata JSON files with caching support"""
    
    CACHE_FILENAME = ".hf_checker_cache.sqlite"
    
    def __init__(self, directory: str, use_cache: bool = True):
        """
        Initialize scanner with a directory path.
        
        Args:
            directory: Path to directory containing metadata JSON files
            use_cache: Whether to use caching (default: True)
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        self.use_cache = use_cache
        self._metadata_files: List[Path] = []
        self._local_files: List[LocalFileInfo] = []
        self._cache: Optional[SQLiteMetadataCache] = None
        self._cache_path = self.directory / self.CACHE_FILENAME
        
        # Stats
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'files_scanned': 0,
            'parse_errors': 0
        }
    
    def scan(self, extensions: Optional[List[str]] = None, force_rescan: bool = False) -> List[LocalFileInfo]:
        """
        Scan directory for metadata JSON files and extract file info.
        Uses cache to avoid re-parsing unchanged files.
        
        Args:
            extensions: List of extensions to look for (default: ['.json'])
            force_rescan: If True, ignore cache and rescan everything
            
        Returns:
            List of LocalFileInfo objects
        """
        if extensions is None:
            extensions = ['.json']
        
        self._metadata_files = []
        self._local_files = []
        self._stats = {'cache_hits': 0, 'cache_misses': 0, 'files_scanned': 0, 'parse_errors': 0}
        
        # Load/create cache if enabled
        if self.use_cache and not force_rescan:
            self._cache = SQLiteMetadataCache(self._cache_path, str(self.directory))
            if not self._cache.is_valid_for_directory(str(self.directory)):
                self._cache.clear()
        elif self.use_cache:
            self._cache = SQLiteMetadataCache(self._cache_path, str(self.directory))
            self._cache.clear()
        else:
            self._cache = None
        
        # Find all JSON files
        visited_dirs = set()
        for root, dirs, files in os.walk(self.directory, followlinks=True):
            
            real_root = os.path.realpath(root)
            if real_root in visited_dirs:
                dirs[:] = []  
                continue
            visited_dirs.add(real_root)
            
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    self._metadata_files.append(Path(root) / file)
        
        self._stats['files_scanned'] = len(self._metadata_files)
        
        # Parse each JSON file (using cache when possible)
        for meta_path in self._metadata_files:
            file_info = self._get_file_info_cached(meta_path)
            if file_info:
                self._local_files.append(file_info)
        
        # Clean up cache (remove entries for deleted files)
        self._cleanup_cache()
        
        return self._local_files
    
    def _get_file_info_cached(self, path: Path) -> Optional[LocalFileInfo]:
        """Get file info, using cache if available and valid."""
        path_str = str(path)
        
        try:
            stat = path.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size
        except OSError:
            return None
        
        # Check cache
        if self._cache:
            cached = self._cache.get_entry(path_str)
            if cached:
                cached_mtime, cached_size, cached_info = cached
                
                if cached_mtime == current_mtime and cached_size == current_size:
                    self._stats['cache_hits'] += 1
                    return cached_info
        
        # Cache miss - need to parse
        self._stats['cache_misses'] += 1
        file_info = self._parse_metadata_file(path)
        
        # Store in cache
        if self._cache:
            self._cache.set_entry(path_str, current_mtime, current_size, file_info)
        
        return file_info
    
    def _parse_metadata_file(self, path: Path) -> Optional[LocalFileInfo]:
        """Parse a single metadata JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both dict and list formats
            if isinstance(data, list):
                if len(data) > 0:
                    data = data[0]
                else:
                    return None
            
            # Extract relevant fields
            file_name = data.get('file_name', '')
            sha256 = data.get('sha256')
            file_path = data.get('file_path')
            size = data.get('size')
            model_name = data.get('model_name')
            base_model = data.get('base_model')
            
            # Skip if no meaningful data
            if not file_name and not file_path and not sha256:
                return None
            
            # If file_name is missing but file_path exists, extract filename
            if not file_name and file_path:
                file_name = Path(file_path).stem
            
            return LocalFileInfo(
                file_name=file_name,
                sha256=sha256.lower() if sha256 else None,  # Normalize to lowercase
                file_path=file_path,
                size=size,
                model_name=model_name,
                base_model=base_model,
                metadata_path=str(path)
            )
            
        except json.JSONDecodeError as e:
            self._stats['parse_errors'] += 1
            return None
        except Exception as e:
            self._stats['parse_errors'] += 1
            return None
    
    def _cleanup_cache(self):
        """Remove cache entries for files that no longer exist."""
        if not self._cache:
            return
        existing_paths = {str(p) for p in self._metadata_files}
        self._cache.cleanup_stale(existing_paths)
    
    def clear_cache(self):
        """Delete the cache file."""
        try:
            if self._cache_path.exists():
                self._cache_path.unlink()
        except Exception:
            pass
    
    def get_by_sha256(self, sha256: str) -> Optional[LocalFileInfo]:
        """Find a local file by SHA256 hash."""
        sha256_lower = sha256.lower()
        for f in self._local_files:
            if f.sha256 and f.sha256.lower() == sha256_lower:
                return f
        return None
    
    def get_by_filename(self, filename: str) -> List[LocalFileInfo]:
        """Find local files by filename (may return multiple matches)."""
        matches = []
        filename_lower = filename.lower()
        
        for f in self._local_files:
            # Check against file_name
            if f.file_name and f.file_name.lower() == filename_lower:
                matches.append(f)
                continue
            
            # Check against basename from file_path
            if f.basename.lower() == filename_lower:
                matches.append(f)
                continue
            
            # Check without extension
            name_no_ext = Path(filename_lower).stem
            local_no_ext = Path(f.basename.lower()).stem
            if name_no_ext == local_no_ext:
                matches.append(f)
        
        return matches
    
    def build_sha256_index(self) -> Dict[str, LocalFileInfo]:
        """Build an index of files by SHA256 for fast lookup."""
        index = {}
        for f in self._local_files:
            if f.sha256:
                index[f.sha256.lower()] = f
        return index
    
    def build_filename_index(self) -> Dict[str, List[LocalFileInfo]]:
        """Build an index of files by filename for fast lookup."""
        index: Dict[str, List[LocalFileInfo]] = {}
        for f in self._local_files:
            # Index by basename
            basename = f.basename.lower()
            if basename not in index:
                index[basename] = []
            index[basename].append(f)
            
            # Also index by stem (without extension)
            stem = Path(basename).stem
            if stem not in index:
                index[stem] = []
            if f not in index[stem]:
                index[stem].append(f)
        
        return index
    
    @property
    def file_count(self) -> int:
        """Number of files found."""
        return len(self._local_files)
    
    @property
    def files_with_sha256(self) -> int:
        """Number of files that have SHA256 hashes."""
        return sum(1 for f in self._local_files if f.sha256)
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get scanning statistics."""
        return self._stats.copy()
    
    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self._stats['cache_hits'] + self._stats['cache_misses']
        if total == 0:
            return 0.0
        return (self._stats['cache_hits'] / total) * 100


class DirectScanner:
    """
    Scans actual model files directly and calculates SHA256 hashes.
    Use this when you don't have metadata JSON files.
    
    Results are cached to speed up subsequent runs.
    """
    
    CACHE_FILENAME = ".hf_checker_direct_cache.sqlite"
    
    def __init__(self, directory: str, use_cache: bool = True):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        self.use_cache = use_cache
        self._model_files: List[Path] = []
        self._local_files: List[LocalFileInfo] = []
        self._sqlite_cache: Optional[SQLiteMetadataCache] = None
        self._cache_path = self.directory / self.CACHE_FILENAME
        
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'files_scanned': 0,
            'hash_errors': 0
        }
    
    def scan(
        self, 
        extensions: Optional[List[str]] = None, 
        force_rescan: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        show_progress: bool = True
    ) -> List[LocalFileInfo]:
        """
        Scan directory for model files and calculate their SHA256 hashes.
        
        Args:
            extensions: File extensions to scan (default: ['.safetensors', '.ckpt', '.pt'])
            force_rescan: Ignore cache and rehash everything
            progress_callback: Optional callback(filename, current, total) for progress updates
            show_progress: Show rich progress bar (default: True)
        """
        if extensions is None:
            extensions = ['.safetensors', '.ckpt', '.pt', '.bin']
        
        self._model_files = []
        self._local_files = []
        self._stats = {'cache_hits': 0, 'cache_misses': 0, 'files_scanned': 0, 'hash_errors': 0}
        
        # Initialize SQLite cache
        if self.use_cache and not force_rescan:
            self._sqlite_cache = SQLiteMetadataCache(self._cache_path, str(self.directory))
        else:
            # Clear existing cache if force_rescan
            if force_rescan and self._cache_path.exists():
                self._cache_path.unlink()
            self._sqlite_cache = SQLiteMetadataCache(self._cache_path, str(self.directory)) if self.use_cache else None
        
        # Find all model files 
        visited_dirs = set()
        for root, dirs, files in os.walk(self.directory, followlinks=True):
            
            real_root = os.path.realpath(root)
            if real_root in visited_dirs:
                dirs[:] = []  
                continue
            visited_dirs.add(real_root)
            
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    self._model_files.append(Path(root) / file)
        
        self._stats['files_scanned'] = len(self._model_files)
        
        # Check how many will need hashing (not in cache)
        files_to_hash = []
        for model_path in self._model_files:
            path_str = str(model_path)
            try:
                stat = model_path.stat()
                if self._sqlite_cache:
                    cached = self._sqlite_cache.get_entry(path_str)
                    if cached:
                        mtime, size, _ = cached
                        if mtime == stat.st_mtime and size == stat.st_size:
                            continue  # Will be a cache hit
            except OSError:
                continue
            files_to_hash.append(model_path)
        
        # Hash files with progress bar if there are files to hash
        if show_progress and files_to_hash:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Hashing {len(files_to_hash)} files...", 
                    total=len(files_to_hash)
                )
                
                hash_count = 0
                for i, model_path in enumerate(self._model_files):
                    file_info = self._get_file_info_cached(model_path)
                    if file_info:
                        self._local_files.append(file_info)
                    
                    # Update progress only for files that needed hashing
                    if model_path in files_to_hash:
                        hash_count += 1
                        progress.update(
                            task, 
                            completed=hash_count,
                            description=f"[cyan]Hashing: {model_path.name[:40]}..."
                        )
        else:
            # No progress bar needed (all cached or progress disabled)
            for i, model_path in enumerate(self._model_files):
                if progress_callback:
                    progress_callback(model_path.name, i + 1, len(self._model_files))
                
                file_info = self._get_file_info_cached(model_path)
                if file_info:
                    self._local_files.append(file_info)
        
        # Cleanup stale cache entries
        self._cleanup_cache()
        
        return self._local_files
    
    def _get_file_info_cached(self, path: Path) -> Optional[LocalFileInfo]:
        """Get file info, using cached hash if file hasn't changed."""
        path_str = str(path)
        
        try:
            stat = path.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size
        except OSError:
            return None
        
        # Check SQLite cache
        if self._sqlite_cache:
            cached = self._sqlite_cache.get_entry(path_str)
            if cached:
                mtime, size, file_info = cached
                if mtime == current_mtime and size == current_size and file_info:
                    self._stats['cache_hits'] += 1
                    return file_info
        
        # Cache miss
        self._stats['cache_misses'] += 1
        file_info = self._hash_file(path)
        
        # Store in SQLite cache
        if self._sqlite_cache and file_info:
            self._sqlite_cache.set_entry(path_str, current_mtime, current_size, file_info)
        
        return file_info
    
    def _hash_file(self, path: Path) -> Optional[LocalFileInfo]:
        """Calculate SHA256 hash of a file."""
        try:
            sha256 = hashlib.sha256()
            with open(path, 'rb') as f:
                # Read in 1MB chunks to handle large files
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    sha256.update(chunk)
            
            return LocalFileInfo(
                file_name=path.stem,
                sha256=sha256.hexdigest().lower(),
                file_path=str(path),
                size=path.stat().st_size,
                metadata_path=""  # No metadata file
            )
        except Exception as e:
            self._stats['hash_errors'] += 1
            return None
    
    def _cleanup_cache(self):
        """Remove cache entries for deleted files."""
        if self._sqlite_cache:
            valid_paths = {str(p) for p in self._model_files}
            self._sqlite_cache.cleanup_stale(valid_paths)
    
    def clear_cache(self):
        """Delete the cache file."""
        try:
            if self._cache_path.exists():
                self._cache_path.unlink()
        except Exception:
            pass
    
    @property
    def file_count(self) -> int:
        return len(self._local_files)
    
    @property
    def files_with_sha256(self) -> int:
        return sum(1 for f in self._local_files if f.sha256)
    
    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    @property
    def cache_hit_rate(self) -> float:
        total = self._stats['cache_hits'] + self._stats['cache_misses']
        return (self._stats['cache_hits'] / total * 100) if total > 0 else 0.0