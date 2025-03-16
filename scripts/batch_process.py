#!/usr/bin/env python
"""
Script to batch process packages.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sqlalchemy.future import select

from llm_docs.config import config
from llm_docs.distillation import DocumentationDistiller
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.database import init_db, transaction
from llm_docs.storage.models import DistillationJob, DistillationJobStatus, Package, PackageStatus

db_path = config.database.url.replace("sqlite+aiosqlite:///", "")

# Create rich console
console = Console()


async def process_package(package: Package, extract_only: bool = False) -> bool:
    """
    Process a package.
    
    Args:
        package: Package to process
        extract_only: If True, only extract documentation without distilling
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        console.print(f"[cyan]Processing {package.name}...[/cyan]")
        start_time = datetime.now()
        
        # Extract documentation
        extractor = DocumentationExtractor()
        doc_path = await extractor.process_package_documentation(package)
        
        if not doc_path:
            console.print(f"[red]Failed to extract documentation for {package.name}[/red]")
            return False
        
        # Update package in session
        async with transaction() as session:

            # Get fresh package from database
            result = await session.execute(select(Package).where(Package.id == package.id))
            db_package = result.scalar_one()
            
            # Update with documentation path
            db_package.original_doc_path = doc_path
            db_package.docs_extraction_date = datetime.now()
            db_package.status = PackageStatus.DOCS_EXTRACTED
            session.add(db_package)
            await session.commit()
            
            # Create distillation job if not extract_only
            if not extract_only:
                job = DistillationJob(
                    package_id=db_package.id,
                    status=DistillationJobStatus.IN_PROGRESS,
                    started_at=datetime.now(),
                    input_file_path=doc_path
                )
                session.add(job)
                
                # Update package status
                db_package.status = PackageStatus.DISTILLATION_IN_PROGRESS
                db_package.distillation_start_date = datetime.now()
                session.add(db_package)
                await session.commit()
                await session.refresh(job)
                
                # Distill documentation
                distiller = DocumentationDistiller()
                distilled_path = await distiller.distill_documentation(db_package, doc_path)
                
                if not distilled_path:
                    console.print(f"[red]Distillation failed for {package.name}[/red]")
                    
                    # Update job and package with failure
                    result = await session.execute(select(Package).where(Package.id == package.id))
                    db_package = result.scalar_one()
                    db_package.status = PackageStatus.DISTILLATION_FAILED
                    session.add(db_package)
                    
                    result = await session.execute(
                        select(DistillationJob).where(DistillationJob.id == job.id)
                    )
                    db_job = result.scalar_one()
                    db_job.status = DistillationJobStatus.FAILED
                    db_job.error_message = "Distillation failed"
                    db_job.completed_at = datetime.now()
                    db_job.chunks_processed = 0  # Set to 0 to indicate processing failed
                    session.add(db_job)
                    await session.commit()
                    
                    return False
                
                # Update package and job with success
                result = await session.execute(select(Package).where(Package.id == package.id))
                db_package = result.scalar_one()
                db_package.distilled_doc_path = distilled_path
                db_package.status = PackageStatus.DISTILLATION_COMPLETED
                db_package.distillation_end_date = datetime.now()
                session.add(db_package)
                
                result = await session.execute(
                    select(DistillationJob).where(DistillationJob.id == job.id)
                )
                db_job = result.scalar_one()
                db_job.status = DistillationJobStatus.COMPLETED
                db_job.completed_at = datetime.now()
                db_job.output_file_path = distilled_path
                db_job.chunks_processed = db_job.num_chunks
                session.add(db_job)
                await session.commit()
        
        # Calculate elapsed time
        elapsed = datetime.now() - start_time
        console.print(f"[green]Successfully processed {package.name} in {elapsed.total_seconds():.1f} seconds[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Error processing {package.name}: {e}[/red]")
        
        # Update package status to failed
        try:
            async with transaction() as session:
                result = await session.execute(select(Package).where(Package.id == package.id))
                db_package = result.scalar_one_or_none()
                if db_package:
                    db_package.status = PackageStatus.DISTILLATION_FAILED
                    session.add(db_package)
                    await session.commit()
        except Exception as db_err:
            console.print(f"[red]Failed to update error status: {db_err}[/red]")
            
        return False


async def batch_process(package_names: List[str], max_parallel: int = 1, extract_only: bool = False) -> None:
    """
    Process multiple packages.
    
    Args:
        package_names: List of package names to process
        max_parallel: Maximum number of packages to process in parallel
        extract_only: If True, only extract documentation without distilling
    """
    # Find or create packages
    packages = []
    
    async with transaction() as session:
        for name in package_names:
            result = await session.execute(select(Package).where(Package.name == name))
            package = result.scalar_one_or_none()
            
            if package:
                packages.append(package)
            else:
                console.print(f"[yellow]Package '{name}' not found, creating...[/yellow]")
                package = Package(name=name)
                session.add(package)
                await session.commit()
                await session.refresh(package)
                packages.append(package)
    
    # Process packages
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task(
            f"Processing packages{' (extract only)' if extract_only else ''}...", 
            total=len(packages)
        )
        
        # Semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_with_semaphore(pkg):
            async with semaphore:
                progress.update(task, description=f"Processing {pkg.name}...")
                result = await process_package(pkg, extract_only)
                progress.update(task, advance=1)
                return pkg.name, result
                
        # Create tasks for all packages
        tasks = [process_with_semaphore(pkg) for pkg in packages]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Report results
        successes = [name for name, success in results if success]
        failures = [name for name, success in results if not success]
        
        progress.update(task, description=f"Completed: {len(successes)} succeeded, {len(failures)} failed")
    
    # Print a more detailed summary
    console.print("\n[bold]Processing Summary:[/bold]")
    console.print(f"Total packages processed: {len(packages)}")
    console.print(f"Successfully processed: {len(successes)} ({len(successes)/len(packages)*100:.1f}%)")
    console.print(f"Failed: {len(failures)} ({len(failures)/len(packages)*100:.1f}%)")
    
    if failures:
        console.print("\n[bold red]Failed packages:[/bold red]")
        for name in failures:
            console.print(f"  - {name}")


async def async_main(args):
    """Run the batch processing asynchronously."""
    # Initialize database if needed
    if args.init_db or not os.path.exists("llm_docs.db"):
        console.print("[yellow]Initializing database...[/yellow]")
        await init_db()
    
    # Get packages to process
    package_names = args.packages
    
    if not package_names and args.top > 0:
        # Get top packages from PyPI
        async with transaction() as session:
            discovery = PackageDiscovery(session)
            console.print(f"[cyan]Discovering top {args.top} packages...[/cyan]")
            packages = await discovery.discover_and_store_packages(args.top)
            await discovery.close()
            package_names = [p.name for p in packages]
    
    if not package_names:
        console.print("[red]No packages specified. Use positional arguments or --top.[/red]")
        return
    
    # Process packages
    console.print(f"[green]Processing {len(package_names)} packages with parallelism {args.parallel}...[/green]")
    await batch_process(package_names, args.parallel, args.extract_only)


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
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract documentation, don't distill it"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing package names, one per line"
    )
    
    args = parser.parse_args()
    
    # Handle package list from file
    if args.file:
        try:
            with open(args.file, 'r') as f:
                file_packages = [line.strip() for line in f if line.strip()]
                args.packages = list(args.packages) + file_packages
        except Exception as e:
            console.print(f"[red]Error reading package file: {e}[/red]")
            sys.exit(1)
    
    # Run the async main function
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()