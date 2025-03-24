"""
Command-line interface for llm_docs.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sqlalchemy.future import select

from llm_docs.config import config
from llm_docs.distillation import DocumentationDistiller
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.database import init_db, reset_db, transaction
from llm_docs.storage.models import DistillationJob, DistillationJobStatus, Package, PackageStatus
from llm_docs.utils.token_tracking import print_usage_report

db_path = config.database.url.replace("sqlite+aiosqlite:///", "")

# Create Typer app
app = typer.Typer(
    name="llm_docs",
    help="llm_docs: Optimized documentation for LLMs",
    add_completion=False
)

# Create rich console for pretty output
console = Console()


@app.command()
def init(
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Reset the database (WARNING: This will delete all data)"
    )
):
    """Initialize the database."""
    async def async_init():
        if reset:
            if not typer.confirm(
                "WARNING: This will delete all data in the database. Are you sure?",
                default=False
            ):
                console.print("[yellow]Operation cancelled.[/yellow]")
                return
                
            await reset_db()
            console.print("[green]Database reset and initialized.[/green]")
        else:
            await init_db()
            console.print("[green]Database initialized.[/green]")
    
    asyncio.run(async_init())


@app.command()
def discover(
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum number of packages to discover"
    ),
    process_top: int = typer.Option(
        0,
        "--process",
        "-p",
        help="Number of top packages to process immediately"
    ),
    concurrency: int = typer.Option(
        5,
        "--concurrency",
        "-c",
        help="Maximum number of concurrent API calls"
    ),
    no_retry: bool = typer.Option(
        False,
        "--no-retry",
        help="Disable retry for failed stats retrievals"
    ),
    rate_limit: float = typer.Option(
        0.5,
        "--rate-limit",
        help="Maximum rate for API requests (requests per second)"
    ),
    process_concurrency: int = typer.Option(
        2,
        "--process-concurrency",
        help="Maximum number of packages to process concurrently"
    )
):
    """Discover packages from PyPI with improved rate limiting."""
    async def run():
        async with transaction() as session:
            # Initialize the package discovery with custom rate limiter
            discovery = PackageDiscovery(session)
            
            # Adjust rate limiter based on parameters
            discovery.stats_rate_limiter.rate = rate_limit
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold]{task.completed}/{task.total}"),
                console=console
            ) as progress:
                task = progress.add_task("Discovering packages...", total=limit)
                
                packages = await discovery.discover_and_store_packages(
                    limit=limit,
                    concurrency=concurrency,
                    retry_failed=not no_retry
                )
                progress.update(task, completed=len(packages))
                
                if process_top > 0:
                    top_packages = packages[:min(process_top, len(packages))]
                    
                    process_task = progress.add_task(
                        f"Processing top {len(top_packages)} packages...",
                        total=len(top_packages)
                    )
                    
                    # Use semaphore to limit concurrent processing
                    semaphore = asyncio.Semaphore(process_concurrency)
                    
                    async def process_with_semaphore(pkg, idx):
                        async with semaphore:
                            progress.update(process_task, description=f"Processing {pkg.name} ({idx+1}/{len(top_packages)})...")
                            
                            # Extract documentation
                            extractor = DocumentationExtractor()
                            doc_path = await extractor.process_package_documentation(pkg)
                            
                            if doc_path:
                                # Update package
                                pkg.original_doc_path = doc_path
                                pkg.docs_extraction_date = datetime.now()
                                pkg.status = PackageStatus.DOCS_EXTRACTED
                                session.add(pkg)
                                await session.commit()
                                
                                # Create distillation job
                                job = DistillationJob(
                                    package_id=pkg.id,
                                    status="pending",
                                    input_file_path=doc_path
                                )
                                session.add(job)
                                await session.commit()
                            
                            progress.update(process_task, advance=1)
                    
                    # Process top packages concurrently
                    tasks = [process_with_semaphore(pkg, i) for i, pkg in enumerate(top_packages)]
                    await asyncio.gather(*tasks)
                
                await discovery.close()
                
        # Show summary after completion
        if discovery.failed_stats_packages:
            console.print(f"[yellow]Note: Failed to get stats for {len(discovery.failed_stats_packages)} packages.[/yellow]")
            if len(discovery.failed_stats_packages) <= 10:
                console.print(f"[yellow]Failed packages: {', '.join(discovery.failed_stats_packages)}[/yellow]")
                
        console.print(f"[green]Discovered {len(packages)} packages successfully.[/green]")
        if process_top > 0:
            console.print(f"[green]Processed top {min(process_top, len(packages))} packages.[/green]")
    
    asyncio.run(run())


@app.command()
def process(
    package_name: str = typer.Argument(..., help="Name of the package to process"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force processing even if the package has already been processed"
    ),
    extract_only: bool = typer.Option(
        False,
        "--extract-only",
        help="Only extract documentation, don't distill it"
    ),
    distill_only: bool = typer.Option(
        False,
        "--distill-only",
        help="Only distill documentation if already extracted"
    )
):
    """Process a specific package."""
    async def run():
        async with transaction() as session:            # Find the package
            result = await session.execute(
                select(Package).where(Package.name == package_name)
            )
            package = result.scalar_one_or_none()
            
            if not package:
                console.print(f"[yellow]Package '{package_name}' not found. Creating it...[/yellow]")
                package = Package(name=package_name)
                session.add(package)
                await session.commit()
                await session.refresh(package)
            
            # Check if already processed
            if not force and not distill_only and package.status == PackageStatus.DISTILLATION_COMPLETED:
                console.print(f"[yellow]Package '{package_name}' already processed.[/yellow]")
                console.print("Use --force to reprocess.")
                return
            
            # Check if extract_only and distill_only are both set
            if extract_only and distill_only:
                console.print("[red]Cannot use both --extract-only and --distill-only at the same time.[/red]")
                return
            
            # Check for distill_only without extracted docs
            if distill_only and not package.original_doc_path:
                console.print(f"[red]Cannot use --distill-only for '{package_name}' - no documentation has been extracted yet.[/red]")
                return
                
            # Reset status if needed
            if force:
                if not distill_only:
                    package.status = PackageStatus.DISCOVERED
                    package.original_doc_path = None
                    package.docs_extraction_date = None
                
                if not extract_only:
                    if distill_only and package.status == PackageStatus.DISTILLATION_COMPLETED:
                        package.status = PackageStatus.DOCS_EXTRACTED
                    package.distilled_doc_path = None
                    package.distillation_start_date = None
                    package.distillation_end_date = None
                
                session.add(package)
                await session.commit()
                await session.refresh(package)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing {package_name}...", total=None)
                
                doc_path = package.original_doc_path
                
                # Extract documentation if needed
                if not doc_path and not distill_only:
                    progress.update(task, description=f"Extracting documentation for {package_name}...")
                    extractor = DocumentationExtractor()
                    doc_path = await extractor.process_package_documentation(package)
                    
                    if not doc_path:
                        progress.update(task, description="[red]Documentation extraction failed.[/red]")
                        return
                        
                    # Update package
                    package.original_doc_path = doc_path
                    package.docs_extraction_date = datetime.now()
                    package.status = PackageStatus.DOCS_EXTRACTED
                    session.add(package)
                    await session.commit()
                    await session.refresh(package)
                    
                    progress.update(task, description="[green]Documentation extracted successfully.[/green]")
                
                # Stop here if extract_only
                if extract_only:
                    progress.update(task, description="[green]Documentation extraction complete.[/green]")
                    return
                
                # Create distillation job if not extract_only
                progress.update(task, description=f"Distilling documentation for {package_name}...")
                
                job = DistillationJob(
                    package_id=package.id,
                    status=DistillationJobStatus.IN_PROGRESS,
                    started_at=datetime.now(),
                    input_file_path=doc_path
                )
                session.add(job)
                
                # Update package status
                package.status = PackageStatus.DISTILLATION_IN_PROGRESS
                package.distillation_start_date = datetime.now()
                session.add(package)
                await session.commit()
                await session.refresh(job)
                
                # Distill documentation
                distiller = DocumentationDistiller()
                distilled_path = await distiller.distill_documentation(package, doc_path)
                
                if not distilled_path:
                    progress.update(task, description="[red]Distillation failed.[/red]")
                    
                    # Update job
                    job.status = DistillationJobStatus.FAILED
                    job.error_message = "Distillation failed"
                    job.completed_at = datetime.now()
                    session.add(job)
                    
                    # Update package
                    package.status = PackageStatus.DISTILLATION_FAILED
                    session.add(package)
                    await session.commit()
                    return
                    
                # Update package
                package.distilled_doc_path = distilled_path
                package.status = PackageStatus.DISTILLATION_COMPLETED
                package.distillation_end_date = datetime.now()
                session.add(package)
                
                # Update job
                job.status = DistillationJobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.output_file_path = distilled_path
                job.chunks_processed = job.num_chunks
                session.add(job)
                await session.commit()
                
                progress.update(task, description=f"[green]Processing of {package_name} completed.[/green]")
            
            # Print token usage statistics
            console.print("\n[bold]LLM API Usage Statistics:[/bold]")
            print_usage_report()

            # Get the final status
            result = await session.execute(
                select(Package).where(Package.id == package.id)
            )
            package = result.scalar_one()
            
            # Print token usage statistics
            console.print("\n[bold]LLM API Usage Statistics:[/bold]")
            print_usage_report()

            if package.status == PackageStatus.DISTILLATION_COMPLETED:
                console.print(f"[green]Processing of '{package_name}' completed successfully.[/green]")
                console.print(f"Original documentation: {package.original_doc_path}")
                console.print(f"Distilled documentation: {package.distilled_doc_path}")
            else:
                console.print(f"[red]Processing of '{package_name}' failed with status: {package.status}[/red]")
    

    asyncio.run(run())


@app.command()
def list_packages(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status"
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of packages to show"
    )
):
    """List packages in the database."""
    async def run():
        async with transaction() as session:            # Build query
            query = select(Package)
            
            if status:
                try:
                    enum_status = PackageStatus(status)
                    query = query.where(Package.status == enum_status)
                except ValueError:
                    console.print(f"[red]Invalid status: {status}[/red]")
                    valid_statuses = [s.value for s in PackageStatus]
                    console.print(f"Valid statuses: {', '.join(valid_statuses)}")
                    return
            
            # Execute query
            result = await session.execute(
                query.order_by(Package.priority.desc()).limit(limit)
            )
            packages = result.scalars().all()
            
            # Get total count
            count_result = await session.execute(select(Package))
            total_count = len(count_result.scalars().all())
            
            # Create table
            table = Table(show_header=True, header_style="bold")
            table.add_column("ID")
            table.add_column("Name")
            table.add_column("Status")
            table.add_column("Priority")
            table.add_column("Docs Extracted")
            table.add_column("Distilled")
            
            for pkg in packages:
                # Set color based on status
                if pkg.status == PackageStatus.DISTILLATION_COMPLETED:
                    status_color = "green"
                elif pkg.status == PackageStatus.DISTILLATION_FAILED:
                    status_color = "red"
                elif pkg.status == PackageStatus.DISTILLATION_IN_PROGRESS:
                    status_color = "blue"
                else:
                    status_color = "yellow"
                
                table.add_row(
                    str(pkg.id),
                    pkg.name,
                    f"[{status_color}]{pkg.status}[/{status_color}]",
                    str(pkg.priority),
                    "✅" if pkg.original_doc_path else "❌",
                    "✅" if pkg.distilled_doc_path else "❌"
                )
            
            console.print(table)
            console.print(f"Showing {len(packages)} of {total_count} packages.")
    
    asyncio.run(run())


@app.command()
def show_package(
    package_name: str = typer.Argument(..., help="Name of the package to show details for")
):
    """Show detailed information about a specific package."""
    async def run():
        async with transaction() as session:            # Find the package
            result = await session.execute(
                select(Package).where(Package.name == package_name)
            )
            package = result.scalar_one_or_none()
            
            if not package:
                console.print(f"[red]Package '{package_name}' not found.[/red]")
                return
            
            # Display package information
            console.print(f"[bold]Package:[/bold] {package.name}")
            console.print(f"[bold]ID:[/bold] {package.id}")
            console.print(f"[bold]Status:[/bold] {package.status}")
            console.print(f"[bold]Priority:[/bold] {package.priority}")
            console.print(f"[bold]Description:[/bold] {package.description or 'N/A'}")
            console.print(f"[bold]Discovery Date:[/bold] {package.discovery_date}")
            console.print(f"[bold]Documentation URL:[/bold] {package.docs_url or 'N/A'}")
            console.print(f"[bold]Original Documentation:[/bold] {package.original_doc_path or 'Not extracted'}")
            console.print(f"[bold]Docs Extraction Date:[/bold] {package.docs_extraction_date or 'N/A'}")
            console.print(f"[bold]Distilled Documentation:[/bold] {package.distilled_doc_path or 'Not distilled'}")
            console.print(f"[bold]Distillation Start Date:[/bold] {package.distillation_start_date or 'N/A'}")
            console.print(f"[bold]Distillation End Date:[/bold] {package.distillation_end_date or 'N/A'}")
            
            # Get distillation jobs
            jobs_result = await session.execute(
                select(DistillationJob)
                .where(DistillationJob.package_id == package.id)
                .order_by(DistillationJob.created_at.desc())
            )
            jobs = jobs_result.scalars().all()
            
            if jobs:
                console.print("\n[bold]Distillation Jobs:[/bold]")
                for job in jobs:
                    console.print(f"  Job ID: {job.id}")
                    console.print(f"  Status: {job.status}")
                    console.print(f"  Created: {job.created_at}")
                    console.print(f"  Started: {job.started_at or 'N/A'}")
                    console.print(f"  Completed: {job.completed_at or 'N/A'}")
                    if job.error_message:
                        console.print(f"  Error: {job.error_message}")
                    console.print("")
    
    asyncio.run(run())


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to"
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind to"
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload"
    )
):
    """Start the API server."""
    import uvicorn
    
    # Initialize database if it doesn't exist
    async def async_init():
        if not os.path.exists(db_path):
            console.print("[yellow]Database not initialized. Initializing now...[/yellow]")
            await init_db()
    
    asyncio.run(async_init())
    
    # Start server
    console.print(f"[green]Starting API server at http://{host}:{port}[/green]")
    console.print("Press Ctrl+C to stop.")
    
    uvicorn.run(
        "llm_docs.api.app:app",
        host=host,
        port=port,
        reload=reload
    )


@app.command()
def batch_process(
    package_file: str = typer.Option(
        None,
        "--file",
        "-f",
        help="File containing package names, one per line"
    ),
    packages: List[str] = typer.Argument(
        None,
        help="Package names to process"
    ),
    parallel: int = typer.Option(
        1,
        "--parallel",
        "-p",
        help="Maximum number of packages to process in parallel"
    )
):
    """Process multiple packages in batch."""
    # Get package names from file if specified
    package_names = list(packages) if packages else []
    
    if package_file:
        try:
            with open(package_file, 'r') as f:
                file_packages = [line.strip() for line in f if line.strip()]
                package_names.extend(file_packages)
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            return
    
    if not package_names:
        console.print("[red]No packages specified. Use positional arguments or --file.[/red]")
        return
    
    # Remove duplicates while preserving order
    seen = set()
    package_names = [p for p in package_names if not (p in seen or seen.add(p))]
    
    console.print(f"[green]Processing {len(package_names)} packages with parallelism {parallel}...[/green]")
    
    async def run():
        # Process packages in parallel
        semaphore = asyncio.Semaphore(parallel)
        
        async def process_one(name):
            async with semaphore:
                async with transaction() as session:                    # Get or create package
                    result = await session.execute(
                        select(Package).where(Package.name == name)
                    )
                    package = result.scalar_one_or_none()
                    
                    if not package:
                        console.print(f"[yellow]Package '{name}' not found, creating...[/yellow]")
                        package = Package(name=name)
                        session.add(package)
                        await session.commit()
                        await session.refresh(package)
                    
                    # Extract documentation
                    console.print(f"[cyan]Processing {name}...[/cyan]")
                    extractor = DocumentationExtractor()
                    doc_path = await extractor.process_package_documentation(package)
                    
                    if not doc_path:
                        console.print(f"[red]Failed to extract documentation for {name}[/red]")
                        return False
                    
                    # Update package
                    package.original_doc_path = doc_path
                    package.docs_extraction_date = datetime.now()
                    package.status = PackageStatus.DOCS_EXTRACTED
                    session.add(package)
                    await session.commit()
                    
                    # Create distillation job
                    job = DistillationJob(
                        package_id=package.id,
                        status=DistillationJobStatus.IN_PROGRESS,
                        started_at=datetime.now(),
                        input_file_path=doc_path
                    )
                    session.add(job)
                    
                    # Update package
                    package.status = PackageStatus.DISTILLATION_IN_PROGRESS
                    package.distillation_start_date = datetime.now()
                    session.add(package)
                    await session.commit()
                    
                    # Distill documentation
                    distiller = DocumentationDistiller()
                    distilled_path = await distiller.distill_documentation(package, doc_path)
                    
                    if not distilled_path:
                        console.print(f"[red]Distillation failed for {name}[/red]")
                        
                        # Update job and package
                        job.status = DistillationJobStatus.FAILED
                        job.error_message = "Distillation failed"
                        job.completed_at = datetime.now()
                        session.add(job)
                        
                        package.status = PackageStatus.DISTILLATION_FAILED
                        session.add(package)
                        await session.commit()
                        return False
                    
                    # Update job and package
                    job.status = DistillationJobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    job.output_file_path = distilled_path
                    job.chunks_processed = job.num_chunks
                    session.add(job)
                    
                    package.distilled_doc_path = distilled_path
                    package.status = PackageStatus.DISTILLATION_COMPLETED
                    package.distillation_end_date = datetime.now()
                    session.add(package)
                    await session.commit()
                    
                    console.print(f"[green]Successfully processed {name}[/green]")
                    return True
        
        # Process all packages with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing packages...", total=len(package_names))
            
            tasks = []
            for name in package_names:
                async def process_with_progress(pkg_name):
                    result = await process_one(pkg_name)
                    progress.update(task, advance=1)
                    return pkg_name, result
                
                tasks.append(asyncio.create_task(process_with_progress(name)))
            
            results = await asyncio.gather(*tasks)
            
            # Count successful and failed packages
            successful = [name for name, success in results if success]
            failed = [name for name, success in results if not success]
            
            progress.update(task, description=f"Completed: {len(successful)} succeeded, {len(failed)} failed")
            
        # Print summary
        console.print("\n[bold]Batch Processing Summary[/bold]")
        console.print(f"Total packages: {len(package_names)}")
        console.print(f"Successfully processed: {len(successful)}")
        console.print(f"Failed: {len(failed)}")
        
        if failed:
            console.print("\n[bold red]Failed packages:[/bold red]")
            for name in failed:
                console.print(f"  - {name}")
    
    asyncio.run(run())


@app.command()
def stats():
    """Show statistics about processed packages and documentation."""
    async def run():
        async with transaction() as session:            # Count packages by status
            total_result = await session.execute(select(Package))
            total_packages = len(total_result.scalars().all())
            
            extracted_result = await session.execute(
                select(Package).where(Package.original_doc_path.isnot(None))
            )
            extracted_packages = len(extracted_result.scalars().all())
            
            distilled_result = await session.execute(
                select(Package).where(Package.distilled_doc_path.isnot(None))
            )
            distilled_packages = len(distilled_result.scalars().all())
            
            failed_result = await session.execute(
                select(Package).where(Package.status == PackageStatus.DISTILLATION_FAILED)
            )
            failed_packages = len(failed_result.scalars().all())
            
            in_progress_result = await session.execute(
                select(Package).where(Package.status == PackageStatus.DISTILLATION_IN_PROGRESS)
            )
            in_progress_packages = len(in_progress_result.scalars().all())
            
            # Create stats table
            table = Table(title="Package Statistics", show_header=True)
            table.add_column("Metric")
            table.add_column("Count")
            table.add_column("Percentage")
            
            if total_packages > 0:
                table.add_row("Total Packages", str(total_packages), "100%")
                table.add_row(
                    "With Extracted Docs", 
                    str(extracted_packages), 
                    f"{extracted_packages/total_packages*100:.1f}%"
                )
                table.add_row(
                    "With Distilled Docs", 
                    str(distilled_packages), 
                    f"{distilled_packages/total_packages*100:.1f}%"
                )
                table.add_row(
                    "Failed Distillation", 
                    str(failed_packages), 
                    f"{failed_packages/total_packages*100:.1f}%"
                )
                table.add_row(
                    "In Progress", 
                    str(in_progress_packages), 
                    f"{in_progress_packages/total_packages*100:.1f}%"
                )
            else:
                table.add_row("Total Packages", "0", "0%")
                table.add_row("With Extracted Docs", "0", "0%")
                table.add_row("With Distilled Docs", "0", "0%")
                table.add_row("Failed Distillation", "0", "0%")
                table.add_row("In Progress", "0", "0%")
            
            console.print(table)
            
            # Storage stats
            docs_dir = Path("docs")
            distilled_dir = Path("distilled_docs")
            
            storage_table = Table(title="Storage Statistics", show_header=True)
            storage_table.add_column("Metric")
            storage_table.add_column("Value")
            
            if docs_dir.exists():
                orig_docs = list(docs_dir.glob("*.md"))
                orig_docs_count = len(orig_docs)
                orig_docs_size = sum(p.stat().st_size for p in orig_docs if p.is_file())
                storage_table.add_row("Original Documentation Files", str(orig_docs_count))
                storage_table.add_row("Original Documentation Size", f"{orig_docs_size / (1024*1024):.2f} MB")
            else:
                storage_table.add_row("Original Documentation Files", "0")
                storage_table.add_row("Original Documentation Size", "0 MB")
            
            if distilled_dir.exists():
                distilled_docs = list(distilled_dir.glob("*.md"))
                distilled_docs_count = len(distilled_docs)
                distilled_docs_size = sum(p.stat().st_size for p in distilled_docs if p.is_file())
                storage_table.add_row("Distilled Documentation Files", str(distilled_docs_count))
                storage_table.add_row("Distilled Documentation Size", f"{distilled_docs_size / (1024*1024):.2f} MB")
                
                if orig_docs_size > 0 and distilled_docs_size > 0:
                    ratio = orig_docs_size / distilled_docs_size
                    storage_table.add_row("Compression Ratio", f"{ratio:.2f}x")
            else:
                storage_table.add_row("Distilled Documentation Files", "0")
                storage_table.add_row("Distilled Documentation Size", "0 MB")
                storage_table.add_row("Compression Ratio", "N/A")
            
            console.print(storage_table)
            
            # Recent activity
            recent_table = Table(title="Recent Activity", show_header=True)
            recent_table.add_column("Package")
            recent_table.add_column("Status")
            recent_table.add_column("Date")
            
            recent_result = await session.execute(
                select(Package)
                .where(Package.status.in_([
                    PackageStatus.DISTILLATION_COMPLETED, 
                    PackageStatus.DISTILLATION_FAILED
                ]))
                .order_by(Package.distillation_end_date.desc())
                .limit(5)
            )
            recent_packages = recent_result.scalars().all()
            
            for pkg in recent_packages:
                status_color = "green" if pkg.status == PackageStatus.DISTILLATION_COMPLETED else "red"
                recent_table.add_row(
                    pkg.name,
                    f"[{status_color}]{pkg.status}[/{status_color}]",
                    str(pkg.distillation_end_date) if pkg.distillation_end_date else "N/A"
                )
            
            if recent_packages:
                console.print(recent_table)
    
    asyncio.run(run())


if __name__ == "__main__":
    app()