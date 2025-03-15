#!/usr/bin/env python
"""
Script to batch process packages.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from sqlmodel import Session, select

from llm_docs.distillation import DocumentationDistiller
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.database import engine, init_db
from llm_docs.storage.models import Package, PackageStatus

# Create rich console
console = Console()


async def process_package(package: Package) -> bool:
    """
    Process a package.
    
    Args:
        package: Package to process
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Extract documentation
        extractor = DocumentationExtractor()
        doc_path = await extractor.process_package_documentation(package)
        
        if not doc_path:
            return False
            
        # Update package in session
        session = Session(engine)
        db_package = session.exec(select(Package).where(Package.id == package.id)).one()
        db_package.original_doc_path = doc_path
        db_package.status = PackageStatus.DOCS_EXTRACTED
        session.add(db_package)
        session.commit()
        
        # Distill documentation
        distiller = DocumentationDistiller()
        distilled_path = await distiller.distill_documentation(db_package, doc_path)
        
        if not distilled_path:
            return False
            
        # Update package again
        db_package = session.exec(select(Package).where(Package.id == package.id)).one()
        db_package.distilled_doc_path = distilled_path
        db_package.status = PackageStatus.DISTILLATION_COMPLETED
        session.add(db_package)
        session.commit()
        session.close()
        
        return True
    except Exception as e:
        console.print(f"[red]Error processing {package.name}: {e}[/red]")
        return False


async def batch_process(package_names: List[str], max_parallel: int = 1) -> None:
    """
    Process multiple packages.
    
    Args:
        package_names: List of package names to process
        max_parallel: Maximum number of packages to process in parallel
    """
    session = Session(engine)
    
    # Find packages
    packages = []
    for name in package_names:
        package = session.exec(select(Package).where(Package.name == name)).first()
        
        if package:
            packages.append(package)
        else:
            console.print(f"[yellow]Package '{name}' not found, creating...[/yellow]")
            package = Package(name=name)
            session.add(package)
            session.commit()
            session.refresh(package)
            packages.append(package)
    
    session.close()
    
    # Process packages
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing packages...", total=len(packages))
        
        # Semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_with_semaphore(pkg):
            async with semaphore:
                progress.update(task, description=f"Processing {pkg.name}...")
                result = await process_package(pkg)
                progress.update(task, completed=task.completed + 1)
                return result
                
        # Create tasks for all packages
        tasks = [process_with_semaphore(pkg) for pkg in packages]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Report results
        successes = sum(1 for r in results if r)
        failures = len(results) - successes
        
        progress.update(task, description=f"Completed: {successes} succeeded, {failures} failed")


def main():
    """Run the batch processing."""
    parser = argparse.ArgumentParser(description="Batch process packages")
    parser.add_argument(
        "packages",
        nargs="*",
        help="Package names to process (if empty, will use --top)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Process top N packages by download count"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum number of packages to process in parallel"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the database if it doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Initialize database if needed
    if args.init_db or not os.path.exists("llm_docs.db"):
        print("Initializing database...")
        init_db()
    
    # Get packages to process
    package_names = args.packages
    
    if not package_names and args.top > 0:
        # Get top packages from PyPI
        session = Session(engine)
        discovery = PackageDiscovery(session)
        
        async def discover_top():
            console.print(f"Discovering top {args.top} packages...")
            packages = await discovery.discover_and_store_packages(args.top)
            await discovery.close()
            return [p.name for p in packages]
            
        package_names = asyncio.run(discover_top())
    
    if not package_names:
        console.print("[red]No packages specified. Use positional arguments or --top.[/red]")
        sys.exit(1)
    
    # Process packages
    console.print(f"Processing {len(package_names)} packages...")
    asyncio.run(batch_process(package_names, args.parallel))


if __name__ == "__main__":
    main()
