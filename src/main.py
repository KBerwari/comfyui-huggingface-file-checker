#!/usr/bin/env python3
"""
CLI for comparing local files against HuggingFace repos.
Run with --help for options.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import fnmatch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich import print as rprint

from hf_client import HuggingFaceClient
from local_scanner import LocalScanner, DirectScanner
from comparator import Comparator
from models import MatchStatus, ComparisonSummary


console = Console()

# CLI Group

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """
    HuggingFace File Checker - Compare local files against HuggingFace repos.
    
    \b
    Commands:
      check   - Compare local files against a HuggingFace repository
      server  - Run API server for browser extension
      config  - Manage configuration (directories, etc.)
    
    \b
    Quick start:
      python main.py check --hf-repo "user/repo" --local-dir "./models"
      python main.py server --add-dir "./models"
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Helper Functions


def print_summary(summary: ComparisonSummary):
    """Print a formatted summary of the comparison."""
    
    # Create summary panel
    summary_text = f"""
[bold]Total HuggingFace files:[/bold] {summary.total_hf_files}
[bold]Total local metadata files:[/bold] {summary.total_local_files}

[green]✓ Matches (SHA256 verified):[/green] {summary.match_count}
[yellow]⚠ Name matches only:[/yellow] {len(summary.name_matches_only)}
[red]✗ SHA256 mismatches:[/red] {summary.mismatch_count}
[blue]↓ Missing locally:[/blue] {summary.missing_local_count}
"""
    
    console.print(Panel(summary_text, title="[bold]Comparison Summary[/bold]", border_style="cyan"))


def print_matches(summary: ComparisonSummary, show_all: bool = False):
    """Print matched files."""
    if not summary.matches:
        return
    
    if not show_all and len(summary.matches) > 10:
        console.print(f"\n[green]✓ {len(summary.matches)} files matched by SHA256[/green] (use --verbose to see all)")
        return
    
    table = Table(title="[green]Matched Files (SHA256 Verified)[/green]")
    table.add_column("Filename", style="green")
    table.add_column("SHA256", style="dim")
    
    for match in summary.matches:
        sha_display = match.remote_sha256[:16] + "..." if match.remote_sha256 else "N/A"
        table.add_row(match.filename, sha_display)
    
    console.print(table)


def print_missing(summary: ComparisonSummary):
    """Print files missing locally."""
    if not summary.missing_local:
        return
    
    table = Table(title="[blue]Files Missing Locally[/blue]")
    table.add_column("Filename", style="blue")
    table.add_column("Remote Path", style="dim")
    table.add_column("SHA256", style="dim")
    table.add_column("Size", style="dim")
    
    for item in summary.missing_local:
        size_str = format_size(item.remote_size) if item.remote_size else "Unknown"
        sha_display = item.remote_sha256[:16] + "..." if item.remote_sha256 else "N/A"
        table.add_row(
            item.filename,
            item.remote_path or "",
            sha_display,
            size_str
        )
    
    console.print(table)


def print_mismatches(summary: ComparisonSummary):
    """Print files with SHA256 mismatches."""
    if not summary.mismatches:
        return
    
    table = Table(title="[red]SHA256 Mismatches (Different Versions?)[/red]")
    table.add_column("Filename", style="red")
    table.add_column("Local SHA256", style="yellow")
    table.add_column("Remote SHA256", style="cyan")
    
    for item in summary.mismatches:
        local_sha = item.local_sha256[:16] + "..." if item.local_sha256 else "N/A"
        remote_sha = item.remote_sha256[:16] + "..." if item.remote_sha256 else "N/A"
        table.add_row(item.filename, local_sha, remote_sha)
    
    console.print(table)


def print_name_matches(summary: ComparisonSummary):
    """Print files matched by name only."""
    if not summary.name_matches_only:
        return
    
    table = Table(title="[yellow]Name Matches (SHA256 Not Verified)[/yellow]")
    table.add_column("Filename", style="yellow")
    table.add_column("Local Path", style="dim")
    table.add_column("Notes", style="dim")
    
    for item in summary.name_matches_only:
        table.add_row(
            item.filename,
            item.local_path or "Unknown",
            item.notes
        )
    
    console.print(table)


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes is None:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def export_missing(summary: ComparisonSummary, output_file: str):
    """Export missing files to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Files missing locally from HuggingFace repository\n")
        f.write(f"# Total: {summary.missing_local_count} files\n\n")
        
        for item in summary.missing_local:
            f.write(f"{item.remote_path or item.filename}\n")
            if item.visit_url:
                f.write(f"  View: {item.visit_url}\n")
            if item.download_url:
                f.write(f"  Download: {item.download_url}\n")
            if item.remote_sha256:
                f.write(f"  SHA256: {item.remote_sha256}\n")
            if item.remote_size:
                f.write(f"  Size: {format_size(item.remote_size)}\n")
            f.write("\n")
    
    console.print(f"[green]Exported missing files list to: {output_file}[/green]")


def export_urls(summary: ComparisonSummary, output_file: str):
    """Export just download URLs for missing files (one per line, for wget/aria2c)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in summary.missing_local:
            if item.download_url:
                f.write(f"{item.download_url}\n")
    
    console.print(f"[green]Exported {summary.missing_local_count} download URLs to: {output_file}[/green]")


def export_matches(summary: ComparisonSummary, output_file: str):
    """Export matched files to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Files you have that match the HuggingFace repository (SHA256 verified)\n")
        f.write(f"# Total: {summary.match_count} files\n\n")
        
        for item in summary.matches:
            f.write(f"{item.filename}\n")
            if item.local_path:
                f.write(f"  Local path: {item.local_path}\n")
            if item.remote_sha256:
                f.write(f"  SHA256: {item.remote_sha256}\n")
            if item.remote_size:
                f.write(f"  Size: {format_size(item.remote_size)}\n")
            f.write("\n")
    
    console.print(f"[green]Exported matched files list to: {output_file}[/green]")


def export_mismatches(summary: ComparisonSummary, output_file: str):
    """Export mismatched files to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Files with SHA256 mismatches (different versions)\n")
        f.write(f"# Total: {summary.mismatch_count} files\n\n")
        
        for item in summary.mismatches:
            f.write(f"{item.filename}\n")
            if item.local_path:
                f.write(f"  Local path: {item.local_path}\n")
            f.write(f"  Local SHA256:  {item.local_sha256 or 'N/A'}\n")
            f.write(f"  Remote SHA256: {item.remote_sha256 or 'N/A'}\n")
            if item.visit_url:
                f.write(f"  View new version: {item.visit_url}\n")
            if item.download_url:
                f.write(f"  Download new version: {item.download_url}\n")
            if item.remote_size:
                f.write(f"  Remote size: {format_size(item.remote_size)}\n")
            if item.local_size:
                f.write(f"  Local size: {format_size(item.local_size)}\n")
            f.write("\n")
    
    console.print(f"[green]Exported mismatched files list to: {output_file}[/green]")


def export_all(summary: ComparisonSummary, output_dir: str):
    """Export all lists to separate files in a directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if summary.matches:
        export_matches(summary, os.path.join(output_dir, "matched_files.txt"))
    if summary.missing_local:
        export_missing(summary, os.path.join(output_dir, "missing_files.txt"))
        export_urls(summary, os.path.join(output_dir, "download_urls.txt"))
    if summary.mismatches:
        export_mismatches(summary, os.path.join(output_dir, "mismatched_files.txt"))
    
    # Also create a summary file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# HuggingFace File Checker Summary\n\n")
        f.write(f"Total HuggingFace files: {summary.total_hf_files}\n")
        f.write(f"Total local metadata files: {summary.total_local_files}\n\n")
        f.write(f"Matches (SHA256 verified): {summary.match_count}\n")
        f.write(f"Name matches only: {len(summary.name_matches_only)}\n")
        f.write(f"SHA256 mismatches: {summary.mismatch_count}\n")
        f.write(f"Missing locally: {summary.missing_local_count}\n")
    
    console.print(f"[green]Exported all results to: {output_dir}/[/green]")


# CHECK Command (main comparison functionality)

@cli.command()
@click.option('--hf-url', help='HuggingFace repository URL (e.g., https://huggingface.co/K3NK/loras-WAN/tree/main or https://huggingface.co/datasets/user/repo)')
@click.option('--hf-repo', help='HuggingFace repository ID (e.g., K3NK/loras-WAN)')
@click.option('--repo-type', type=click.Choice(['model', 'dataset', 'space']), default='model', help='Repository type (default: model). Auto-detected from URL if using --hf-url')
@click.option('--local-dir', required=True, help='Path to directory containing metadata JSON files or model files')
@click.option('--scan-files', is_flag=True, help='Scan actual model files and calculate SHA256 (slow, but works without metadata files)')
@click.option('--safetensors-only', is_flag=True, help='Only check .safetensors files')
@click.option('--filter', 'filter_pattern', help='Only check HF files matching this pattern (e.g., "*wan22*" or "*.safetensors")')
@click.option('--verbose', '-v', is_flag=True, help='Show all matched files')
@click.option('--export-missing', 'export_file', help='Export missing files list to a file')
@click.option('--export-urls', 'export_urls_file', help='Export just the download URLs for missing files (one per line, for wget/aria2c)')
@click.option('--export-matches', 'export_matches_file', help='Export matched files list to a file')
@click.option('--export-mismatches', 'export_mismatches_file', help='Export mismatched files list to a file')
@click.option('--export-all', 'export_all_dir', help='Export all lists to a directory')
@click.option('--token', envvar='HF_TOKEN', help='HuggingFace API token (or set HF_TOKEN env var)')
@click.option('--branch', default='main', help='Repository branch to check (default: main)')
@click.option('--no-cache', is_flag=True, help='Disable caching (rescan all files every time)')
@click.option('--clear-cache', is_flag=True, help='Clear the cache before scanning')
def check(hf_url, hf_repo, repo_type, local_dir, scan_files, safetensors_only, filter_pattern, verbose, export_file, export_urls_file, export_matches_file, export_mismatches_file, export_all_dir, token, branch, no_cache, clear_cache):
    """
    Check if you have files from a HuggingFace repository by comparing SHA256 hashes.
    
    Examples:
    
        python main.py --hf-url "https://huggingface.co/K3NK/loras-WAN" --local-dir "./metadata"
        
        python main.py --hf-repo "K3NK/loras-WAN" --local-dir "./metadata" --safetensors-only
    """
    
    # Validate inputs
    if not hf_url and not hf_repo:
        console.print("[red]Error: Must provide either --hf-url or --hf-repo[/red]")
        sys.exit(1)
    
    if not os.path.isdir(local_dir):
        console.print(f"[red]Error: Local directory does not exist: {local_dir}[/red]")
        sys.exit(1)
    
    console.print(Panel.fit(
        "[bold cyan]HuggingFace File Checker[/bold cyan]\n"
        "Comparing local files against HuggingFace repository",
        border_style="cyan"
    ))
    
    # Initialize HuggingFace client
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Create HF client
        task = progress.add_task("Connecting to HuggingFace...", total=None)
        try:
            if hf_url:
                hf_client = HuggingFaceClient.from_url(hf_url, token=token)
            else:
                hf_client = HuggingFaceClient(repo_id=hf_repo, revision=branch, repo_type=repo_type, token=token)
            
            # Show repo type if not model
            type_info = f" [dim]({hf_client.repo_type})[/dim]" if hf_client.repo_type != "model" else ""
            console.print(f"[green]✓[/green] Connected to repository: [cyan]{hf_client.repo_id}[/cyan]{type_info}")
        except Exception as e:
            console.print(f"[red]Error connecting to HuggingFace: {e}[/red]")
            sys.exit(1)
        
        # Fetch HuggingFace files
        progress.update(task, description="Fetching files from HuggingFace...")
        try:
            if safetensors_only:
                hf_files = hf_client.fetch_safetensors_only()
                console.print(f"[green]✓[/green] Found [cyan]{len(hf_files)}[/cyan] .safetensors files on HuggingFace")
            else:
                hf_files = hf_client.fetch_all_files()
                console.print(f"[green]✓[/green] Found [cyan]{len(hf_files)}[/cyan] files on HuggingFace")
            
            # Apply filter if specified
            if filter_pattern:
                original_count = len(hf_files)
                hf_files = [f for f in hf_files if fnmatch.fnmatch(f.path.lower(), filter_pattern.lower())]
                console.print(f"[green]✓[/green] Filter [cyan]{filter_pattern}[/cyan]: {len(hf_files)}/{original_count} files match")
        except Exception as e:
            console.print(f"[red]Error fetching HuggingFace files: {e}[/red]")
            sys.exit(1)
        
        # Scan local files
        if scan_files:
            # Direct file scanning mode - hash actual model files
            progress.update(task, description="Finding model files...")
            try:
                scanner = DirectScanner(local_dir, use_cache=not no_cache)
                
                if clear_cache:
                    scanner.clear_cache()
                
                # Determine extensions to scan
                extensions = ['.safetensors'] if safetensors_only else ['.safetensors', '.ckpt', '.pt', '.bin']
                
                local_files = scanner.scan(extensions=extensions)
                
                stats = scanner.stats
                
                # Show clear cache status
                if not no_cache and stats['cache_hits'] > 0:
                    if stats['cache_misses'] == 0:
                        cache_info = f" [dim](all {stats['cache_hits']} loaded from cache)[/dim]"
                    else:
                        cache_info = f" [dim](cache: {scanner.cache_hit_rate:.0f}% hit, {stats['cache_misses']} newly hashed)[/dim]"
                elif stats['cache_misses'] > 0:
                    cache_info = f" [dim]({stats['cache_misses']} files hashed)[/dim]"
                else:
                    cache_info = ""
                
                console.print(f"[green]✓[/green] Scanned [cyan]{scanner.file_count}[/cyan] model files{cache_info}")
            except Exception as e:
                console.print(f"[red]Error scanning model files: {e}[/red]")
                sys.exit(1)
        else:
            # Metadata JSON mode (default)
            progress.update(task, description="Scanning local metadata files...")
            try:
                scanner = LocalScanner(local_dir, use_cache=not no_cache)
                
                if clear_cache:
                    scanner.clear_cache()
                
                local_files = scanner.scan()
                
                stats = scanner.stats
                cache_info = ""
                if not no_cache and stats['cache_hits'] > 0:
                    cache_info = f" [dim](cache: {scanner.cache_hit_rate:.0f}% hit rate)[/dim]"
                
                console.print(f"[green]✓[/green] Found [cyan]{scanner.file_count}[/cyan] local metadata files ({scanner.files_with_sha256} with SHA256){cache_info}")
                
                # Hint if no files found
                if scanner.file_count == 0:
                    console.print("[yellow]Tip: No metadata files found. If you have model files but no JSON metadata, try --scan-files[/yellow]")
            except Exception as e:
                console.print(f"[red]Error scanning local files: {e}[/red]")
                sys.exit(1)
        
        # Compare
        progress.update(task, description="Comparing files...")
        comparator = Comparator(
            local_files, 
            hf_files,
            repo_id=hf_client.repo_id,
            repo_type=hf_client.repo_type,
            revision=hf_client.revision
        )
        summary = comparator.compare()
        
        progress.update(task, description="Done!", completed=True)
    
    # Print results
    console.print()
    print_summary(summary)
    
    if summary.mismatches:
        console.print()
        print_mismatches(summary)
    
    if summary.name_matches_only:
        console.print()
        print_name_matches(summary)
    
    if summary.missing_local:
        console.print()
        print_missing(summary)
    
    if verbose:
        console.print()
        print_matches(summary, show_all=True)
    
    # Export if requested
    if export_all_dir:
        export_all(summary, export_all_dir)
    else:
        if export_file and summary.missing_local:
            export_missing(summary, export_file)
        if export_urls_file and summary.missing_local:
            export_urls(summary, export_urls_file)
        if export_matches_file and summary.matches:
            export_matches(summary, export_matches_file)
        if export_mismatches_file and summary.mismatches:
            export_mismatches(summary, export_mismatches_file)
    
    # Exit code based on results
    if summary.missing_local_count > 0:
        msg = f"You are missing {summary.missing_local_count} files from the HuggingFace repository."
        if summary.mismatch_count > 0:
            msg += f" Also, {summary.mismatch_count} files have different SHA256."
        console.print(f"\n[yellow]{msg}[/yellow]")
        sys.exit(2)
    elif summary.mismatch_count > 0:
        console.print(f"\n[yellow]{summary.mismatch_count} files have different SHA256 (possibly different versions).[/yellow]")
        sys.exit(1)
    else:
        console.print("\n[green]All HuggingFace files are present locally![/green]")
        sys.exit(0)



# SERVER Command

@cli.command()
@click.option('--port', default=7860, help='Port to run server on (default: 7860)')
@click.option('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
@click.option('--add-dir', multiple=True, help='Add directory to scan (can be used multiple times)')
@click.option('--no-scan', is_flag=True, help='Start server without initial scan')
def server(port, host, add_dir, no_scan):
    """
    Run API server for browser extension integration.
    
    The server provides a local API that the Tampermonkey script can query
    to check if you have files from HuggingFace repositories.
    
    \b
    Examples:
        python main.py server
        python main.py server --port 8080
        python main.py server --add-dir "H:/AI/models" --add-dir "D:/loras"
    """
    from config import get_config_manager
    from server import run_server
    
    # Add any directories specified on command line
    if add_dir:
        config_manager = get_config_manager()
        for directory in add_dir:
            if os.path.isdir(directory):
                config_manager.add_directory(directory)
                console.print(f"[green]✓[/green] Added directory: {directory}")
            else:
                console.print(f"[yellow]Warning: Directory does not exist: {directory}[/yellow]")
    
    # Run the server
    run_server(host=host, port=port, auto_scan=not no_scan)



# CONFIG Command

@cli.command()
@click.option('--list', 'list_dirs', is_flag=True, help='List all configured directories')
@click.option('--add', 'add_dir', help='Add a directory')
@click.option('--remove', 'remove_dir', help='Remove a directory')
@click.option('--mode', type=click.Choice(['files', 'metadata']), default='metadata', help='Scan mode: "metadata" (read JSON metadata, fast) or "files" (hash model files directly, slow)')
@click.option('--show-path', is_flag=True, help='Show config file path')
def config(list_dirs, add_dir, remove_dir, mode, show_path):
    """
    Manage configuration (directories, settings).
    
    \b
    Scan modes:
      metadata - Read SHA256 from JSON metadata files (fast, needs ComfyUI-Lora-Manager)
      files    - Hash actual model files (slower first run, but works without metadata)
    
    \b
    Examples:
        python main.py config --list
        python main.py config --add "H:/AI/models"
        python main.py config --add "H:/AI/loras" --mode files
        python main.py config --remove "H:/AI/models"
    """
    from config import get_config_manager
    
    config_manager = get_config_manager()
    
    if show_path:
        console.print(f"Config file: {config_manager.config_path}")
        return
    
    if add_dir:
        if os.path.isdir(add_dir):
            dir_config = config_manager.add_directory(add_dir, scan_mode=mode)
            console.print(f"[green]✓[/green] Added: {dir_config.name} ({dir_config.path})")
            console.print(f"  Scan mode: [cyan]{mode}[/cyan]")
        else:
            console.print(f"[red]Error: Directory does not exist: {add_dir}[/red]")
            sys.exit(1)
        return
    
    if remove_dir:
        if config_manager.remove_directory(remove_dir):
            console.print(f"[green]✓[/green] Removed: {remove_dir}")
        else:
            console.print(f"[yellow]Directory not found in config: {remove_dir}[/yellow]")
        return
    
    # Default: list directories
    dirs = config_manager.config.directories
    if not dirs:
        console.print("[dim]No directories configured.[/dim]")
        console.print("Add one with: python main.py config --add \"path/to/models\"")
        return
    
    table = Table(title="Configured Directories")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Mode", style="green")
    table.add_column("Enabled", style="yellow")
    
    for d in dirs:
        table.add_row(
            d.name,
            d.path,
            d.scan_mode,
            "✓" if d.enabled else "✗"
        )
    
    console.print(table)

if __name__ == "__main__":
    cli()