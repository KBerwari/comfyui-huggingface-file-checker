"""
Local API server for HuggingFace File Checker.
Allows browser extensions to check file status against local directories.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import get_config_manager, DirectoryConfig
from local_scanner import DirectScanner, LocalScanner
from models import LocalFileInfo
from hf_client import HuggingFaceClient


app = Flask(__name__)
app.json.ensure_ascii = False  
CORS(app)  


@dataclass
class HashCache:
    """In-memory cache of all SHA256 hashes across directories."""
    sha256_index: Dict[str, List[dict]] = field(default_factory=dict)
    filename_index: Dict[str, List[dict]] = field(default_factory=dict)
    total_files: int = 0
    last_scan: Optional[str] = None
    directories_scanned: List[str] = field(default_factory=list)

_hash_cache = HashCache()


def scan_directory(dir_config: DirectoryConfig) -> List[LocalFileInfo]:
    """Scan a single directory and return file info."""
    path = dir_config.path
    
    if not os.path.isdir(path):
        print(f"  Warning: Directory does not exist: {path}")
        return []
    
    try:
        if dir_config.scan_mode == "metadata":
            scanner = LocalScanner(path, use_cache=True)
            files = scanner.scan()
            stats = scanner.stats
            if stats['cache_hits'] > 0 and stats['cache_misses'] == 0:
                print(f"  → {len(files)} files (all from cache)")
            elif stats['cache_hits'] > 0:
                print(f"  → {len(files)} files ({stats['cache_hits']} cached, {stats['cache_misses']} new)")
            else:
                print(f"  → {len(files)} metadata files parsed")
            return files
        else:
            scanner = DirectScanner(path, use_cache=True)
            # Show progress bar for direct scanning since it can be slow
            files = scanner.scan(extensions=dir_config.extensions, show_progress=True)
            stats = scanner.stats
            if stats['cache_hits'] > 0 and stats['cache_misses'] == 0:
                print(f"  → {len(files)} files (all from cache)")
            elif stats['cache_hits'] > 0:
                print(f"  → {len(files)} files ({stats['cache_hits']} cached, {stats['cache_misses']} newly hashed)")
            else:
                print(f"  → {len(files)} files hashed")
            return files
    except Exception as e:
        print(f"  Error scanning {path}: {e}")
        return []


def rebuild_cache():
    """Rebuild the in-memory hash cache from all configured directories."""
    global _hash_cache
    
    config_manager = get_config_manager()
    config = config_manager.config
    directories = config_manager.get_enabled_directories()
    
    _hash_cache = HashCache()
    _hash_cache.last_scan = datetime.now().isoformat()
    
    if not directories:
        print("No directories configured.")
        print("Add a directory with: python main.py config --add \"path\"")
        return
    
    print(f"\nScanning {len(directories)} directory(s)...\n")
    
    for dir_config in directories:
        mode_str = "metadata" if dir_config.scan_mode == "metadata" else "direct hash"
        print(f"[{mode_str}] {dir_config.path}")
        files = scan_directory(dir_config)
        _hash_cache.directories_scanned.append(dir_config.path)
        
        for f in files:
            file_info = {
                'filename': f.basename,
                'sha256': f.sha256,
                'path': f.file_path,
                'size': f.size,
                'directory': dir_config.path,
                'directory_name': dir_config.name
            }
            
            # Index by SHA256
            if f.sha256:
                sha_lower = f.sha256.lower()
                if sha_lower not in _hash_cache.sha256_index:
                    _hash_cache.sha256_index[sha_lower] = []
                _hash_cache.sha256_index[sha_lower].append(file_info)
            
            # Index by filename
            basename_lower = f.basename.lower()
            if basename_lower not in _hash_cache.filename_index:
                _hash_cache.filename_index[basename_lower] = []
            _hash_cache.filename_index[basename_lower].append(file_info)
            
            _hash_cache.total_files += 1
    
    print(f"Cache rebuilt: {_hash_cache.total_files} files indexed")



# API Routes

@app.route('/api/status', methods=['GET'])
def status():
    """Health check and status info."""
    return jsonify({
        'status': 'ok',
        'service': 'hf-file-checker',
        'version': '1.0.0',
        'cache': {
            'total_files': _hash_cache.total_files,
            'total_hashes': len(_hash_cache.sha256_index),
            'last_scan': _hash_cache.last_scan,
            'directories': len(_hash_cache.directories_scanned)
        }
    })


@app.route('/api/check/sha256/<sha256>', methods=['GET'])
def check_sha256(sha256: str):
    """Check if a SHA256 hash exists locally."""
    sha_lower = sha256.lower()
    
    if sha_lower in _hash_cache.sha256_index:
        matches = _hash_cache.sha256_index[sha_lower]
        return jsonify({
            'found': True,
            'sha256': sha256,
            'matches': matches,
            'count': len(matches)
        })
    
    return jsonify({
        'found': False,
        'sha256': sha256,
        'matches': [],
        'count': 0
    })


@app.route('/api/check/filename/<filename>', methods=['GET'])
def check_filename(filename: str):
    """Check if a filename exists locally (case-insensitive)."""
    filename_lower = filename.lower()
    
    if filename_lower in _hash_cache.filename_index:
        matches = _hash_cache.filename_index[filename_lower]
        return jsonify({
            'found': True,
            'filename': filename,
            'matches': matches,
            'count': len(matches)
        })
    
    return jsonify({
        'found': False,
        'filename': filename,
        'matches': [],
        'count': 0
    })


@app.route('/api/check/batch', methods=['POST'])
def check_batch():
    """
    Check multiple files at once.
    
    Request body:
    {
        "files": [
            {"sha256": "abc123...", "filename": "model.safetensors"},
            ...
        ]
    }
    
    Response:
    {
        "results": {
            "abc123...": {"found": true, "match_type": "sha256", "local_path": "..."},
            ...
        }
    }
    """
    data = request.get_json()
    if not data or 'files' not in data:
        return jsonify({'error': 'Missing files array'}), 400
    
    results = {}
    
    for file_info in data['files']:
        if file_info is None:
            continue
        sha256 = (file_info.get('sha256') or '').lower()
        filename = (file_info.get('filename') or '').lower()
        full_path = file_info.get('fullPath') or ''
        
        # Extract basename from fullPath if available (for files in subfolders)
        if full_path and '/' in full_path:
            basename_from_path = full_path.split('/')[-1].lower()
        else:
            basename_from_path = filename
        
        key = sha256 or filename
        
        if not key:
            continue
        
        # Try SHA256 first
        if sha256 and sha256 in _hash_cache.sha256_index:
            matches = _hash_cache.sha256_index[sha256]
            results[key] = {
                'found': True,
                'match_type': 'sha256',
                'local_paths': [m['path'] for m in matches],
                'directories': [m['directory_name'] for m in matches]
            }
        # Try filename
        elif filename and filename in _hash_cache.filename_index:
            matches = _hash_cache.filename_index[filename]
            # Check if any SHA256 matches
            if sha256:
                sha_matches = [m for m in matches if m.get('sha256', '').lower() == sha256]
                if sha_matches:
                    results[key] = {
                        'found': True,
                        'match_type': 'sha256',
                        'local_paths': [m['path'] for m in sha_matches],
                        'directories': [m['directory_name'] for m in sha_matches]
                    }
                else:
                    # Filename match but SHA256 differs
                    results[key] = {
                        'found': True,
                        'match_type': 'filename_only',
                        'mismatch': True,
                        'local_paths': [m['path'] for m in matches],
                        'directories': [m['directory_name'] for m in matches],
                        'local_sha256': matches[0].get('sha256') if matches else None
                    }
            else:
                results[key] = {
                    'found': True,
                    'match_type': 'filename',
                    'local_paths': [m['path'] for m in matches],
                    'directories': [m['directory_name'] for m in matches]
                }
        # Try basename extracted from fullPath
        elif basename_from_path and basename_from_path != filename and basename_from_path in _hash_cache.filename_index:
            matches = _hash_cache.filename_index[basename_from_path]
            if sha256:
                sha_matches = [m for m in matches if m.get('sha256', '').lower() == sha256]
                if sha_matches:
                    results[key] = {
                        'found': True,
                        'match_type': 'sha256',
                        'local_paths': [m['path'] for m in sha_matches],
                        'directories': [m['directory_name'] for m in sha_matches]
                    }
                else:
                    results[key] = {
                        'found': True,
                        'match_type': 'filename_only',
                        'mismatch': True,
                        'local_paths': [m['path'] for m in matches],
                        'directories': [m['directory_name'] for m in matches],
                        'local_sha256': matches[0].get('sha256') if matches else None
                    }
            else:
                results[key] = {
                    'found': True,
                    'match_type': 'filename',
                    'local_paths': [m['path'] for m in matches],
                    'directories': [m['directory_name'] for m in matches]
                }
        else:
            results[key] = {
                'found': False,
                'match_type': None
            }
    
    return jsonify({
        'results': results,
        'checked': len(data['files']),
        'found': sum(1 for r in results.values() if r['found'])
    })


@app.route('/api/directories', methods=['GET'])
def list_directories():
    """List all configured directories."""
    config_manager = get_config_manager()
    dirs = config_manager.config.directories
    
    return jsonify({
        'directories': [
            {
                'path': d.path,
                'name': d.name,
                'scan_mode': d.scan_mode,
                'extensions': d.extensions,
                'enabled': d.enabled,
                'added': d.added
            }
            for d in dirs
        ]
    })


@app.route('/api/directories', methods=['POST'])
def add_directory():
    """Add a new directory to scan."""
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({'error': 'Missing path'}), 400
    
    path = data['path']
    
    # Validate path exists
    if not os.path.isdir(path):
        return jsonify({'error': f'Directory does not exist: {path}'}), 400
    
    config_manager = get_config_manager()
    dir_config = config_manager.add_directory(
        path=path,
        name=data.get('name', ''),
        scan_mode=data.get('scan_mode', 'files'),
        extensions=data.get('extensions')
    )
    
    # Rescan to include new directory
    rebuild_cache()
    
    return jsonify({
        'success': True,
        'directory': {
            'path': dir_config.path,
            'name': dir_config.name,
            'scan_mode': dir_config.scan_mode,
            'enabled': dir_config.enabled
        },
        'cache': {
            'total_files': _hash_cache.total_files
        }
    })


@app.route('/api/directories', methods=['DELETE'])
def remove_directory():
    """Remove a directory from config."""
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({'error': 'Missing path'}), 400
    
    config_manager = get_config_manager()
    removed = config_manager.remove_directory(data['path'])
    
    if removed:
        rebuild_cache()
        return jsonify({'success': True, 'removed': data['path']})
    
    return jsonify({'error': 'Directory not found in config'}), 404



# API Endpoints

@app.route('/api/rescan', methods=['POST'])
def rescan():
    """Force rescan all directories."""
    rebuild_cache()
    return jsonify({
        'success': True,
        'total_files': _hash_cache.total_files,
        'directories_scanned': len(_hash_cache.directories_scanned)
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    config_manager = get_config_manager()
    return jsonify(config_manager.config.to_dict())


@app.route('/api/check/repo', methods=['POST'])
def check_repo():
    """
    Check all files in a HuggingFace repository against local cache.
    
    Request body:
    {
        "repo_url": "https://huggingface.co/user/repo",
        "repo_id": "user/repo",  // alternative to repo_url
        "repo_type": "model",    // optional: model, dataset, space
        "revision": "main"       // optional: branch/revision
    }
    
    Response:
    {
        "repo_id": "user/repo",
        "total_files": 100,
        "found": 80,
        "missing": 15,
        "mismatch": 5,
        "files": [...]
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request body'}), 400
    
    # Parse repo info
    repo_url = data.get('repo_url')
    repo_id = data.get('repo_id')
    repo_type = data.get('repo_type', 'model')
    revision = data.get('revision', 'main')
    
    if not repo_url and not repo_id:
        return jsonify({'error': 'Missing repo_url or repo_id'}), 400
    
    try:
        # Create client
        if repo_url:
            client = HuggingFaceClient.from_url(repo_url)
        else:
            client = HuggingFaceClient(repo_id=repo_id, revision=revision, repo_type=repo_type)
        
        hf_files = client.fetch_all_files()
        
        # Check each file against local cache
        results = {
            'repo_id': client.repo_id,
            'repo_type': client.repo_type,
            'revision': client.revision,
            'total_files': len(hf_files),
            'found': 0,
            'missing': 0,
            'mismatch': 0,
            'found_size': 0,
            'missing_size': 0,
            'files': []
        }
        
        for hf_file in hf_files:
            file_result = {
                'filename': hf_file.basename,
                'path': hf_file.path,
                'sha256': hf_file.sha256,
                'size': hf_file.size,
                'status': 'missing',
                'local_paths': [],
                'directories': []
            }
            
            sha_lower = (hf_file.sha256 or '').lower()
            filename_lower = hf_file.basename.lower()
            
            # Check by SHA256 first
            if sha_lower and sha_lower in _hash_cache.sha256_index:
                matches = _hash_cache.sha256_index[sha_lower]
                file_result['status'] = 'found'
                file_result['local_paths'] = [m['path'] for m in matches]
                file_result['directories'] = list(set(m['directory_name'] for m in matches))
                results['found'] += 1
                results['found_size'] += hf_file.size or 0
            # Check by filename
            elif filename_lower in _hash_cache.filename_index:
                matches = _hash_cache.filename_index[filename_lower]
                # Check if any SHA matches
                if sha_lower:
                    sha_matches = [m for m in matches if (m.get('sha256') or '').lower() == sha_lower]
                    if sha_matches:
                        file_result['status'] = 'found'
                        file_result['local_paths'] = [m['path'] for m in sha_matches]
                        file_result['directories'] = list(set(m['directory_name'] for m in sha_matches))
                        results['found'] += 1
                        results['found_size'] += hf_file.size or 0
                    else:
                        file_result['status'] = 'mismatch'
                        file_result['local_paths'] = [m['path'] for m in matches]
                        file_result['directories'] = list(set(m['directory_name'] for m in matches))
                        file_result['local_sha256'] = matches[0].get('sha256') if matches else None
                        results['mismatch'] += 1
                        results['missing_size'] += hf_file.size or 0
                else:
                    # No SHA to compare, consider found by filename
                    file_result['status'] = 'found'
                    file_result['local_paths'] = [m['path'] for m in matches]
                    file_result['directories'] = list(set(m['directory_name'] for m in matches))
                    results['found'] += 1
                    results['found_size'] += hf_file.size or 0
            else:
                results['missing'] += 1
                results['missing_size'] += hf_file.size or 0
            
            results['files'].append(file_result)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_server(host: str = "127.0.0.1", port: int = 7860, auto_scan: bool = True):
    """Run the API server."""
    print(f"\n{'='*60}")
    print("HuggingFace File Checker - Local API Server")
    print(f"{'='*60}")
    print(f"Server: http://{host}:{port}")
    print(f"Config: {get_config_manager().config_path}")
    print(f"{'='*60}\n")
    
    if auto_scan:
        print("Scanning directories...")
        rebuild_cache()
        print()
    
    print("API Endpoints:")
    print(f"  GET  /api/status              - Server status")
    print(f"  GET  /api/check/sha256/<hash> - Check if SHA256 exists")
    print(f"  GET  /api/check/filename/<n>  - Check if filename exists")
    print(f"  POST /api/check/batch         - Check multiple files")
    print(f"  GET  /api/directories         - List directories")
    print(f"  POST /api/directories         - Add directory")
    print(f"  DEL  /api/directories         - Remove directory")
    print(f"  POST /api/rescan              - Rescan all directories")
    print()
    print("Press Ctrl+C to stop\n")
    
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    run_server()
