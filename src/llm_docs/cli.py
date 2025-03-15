"""
Command-line interface for llm_docs.
"""

import asyncio
import os
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sqlmodel import Session, select

from llm_docs.distillation import DocumentationDistiller
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.database import engine, init_db
from llm_docs.storage.models import DistillationJob, Package, PackageStatus

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
    if reset:
        if not typer.confirm(
            "WARNING: This will delete all data in the database. Are you sure?",
            default=False
        ):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
            
        from llm_docs.storage.database import reset_db
        reset_db()
        console.print("[green]Database reset and initialized.[/green]")
    else:
        init_db()
        console.print("[green]Database initialized.[/green]")


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
    )
):
    """Discover packages from PyPI."""
    session = Session(engine)
    
    # Initialize the package discovery
    discovery = PackageDiscovery(session)
    
    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            console=console
        ) as progress:
            task = progress.add_task("Discovering packages...", total=limit)
            
            packages = await discovery.discover_and_store_packages(limit)
            progress.update(task, completed=len(packages))
            
            if process_top > 0:
                top_packages = packages[:min(process_top, len(packages))]
                
                process_task = progress.add_task(
                    f"Processing top {len(top_packages)} packages...",
                    total=len(top_packages)
                )
                
                for i, package in enumerate(top_packages):
                    progress.update(task, description=f"Processing {package.name}...")
                    
                    # Extract documentation
                    extractor = DocumentationExtractor()
                    doc_path = await extractor.process_package_documentation(package)
                    
                    if doc_path:
                        # Update package
                        package.original_doc_path = doc_path
                        package.docs_extraction_date = datetime.now()
                        package.status = PackageStatus.DOCS_EXTRACTED
                        session.add(package)
                        session.commit()
                        
                        # Create distillation job
                        job = DistillationJob(
                            package_id=package.id,
                            status="pending",
                            input_file_path=doc_path
                        )
                        session.add(job)
                        session.commit()
                    
                    progress.update(process_task, completed=i+1)
            
            await discovery.close()
            
    asyncio.run(run())
    console.print(f"[green]Discovered {limit} packages.[/green]")
    if process_top > 0:
        console.print(f"[green]Processed top {min(process_top, limit)} packages.[/green]")


@app.command()
def process(
    package_name: str = typer.Argument(..., help="Name of the package to process"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force processing even if the package has already been processed"
    )
):
    """Process a specific package."""
    session = Session(engine)
    
    # Find the package
    package = session.exec(
        select(Package).where(Package.name == package_name)
    ).first()
    
    if not package:
        console.print(f"[red]Package '{package_name}' not found.[/red]")
        console.print("Use 'llm_docs discover' to discover packages or create it manually.")
        return
        
    # Check if already processed
    if not force and package.status == PackageStatus.DISTILLATION_COMPLETED:
        console.print(f"[yellow]Package '{package_name}' already processed.[/yellow]")
        console.print("Use --force to reprocess.")
        return
        
    # Reset status if needed
    if force and package.status in [
        PackageStatus.DOCS_EXTRACTED,
        PackageStatus.DISTILLATION_COMPLETED,
        PackageStatus.DISTILLATION_FAILED
    ]:
        package.status = PackageStatus.DISCOVERED
        session.add(package)
        session.commit()
    
    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing {package_name}...", total=None)
            
            # Extract documentation
            progress.update(task, description=f"Extracting documentation for {package_name}...")
            extractor = DocumentationExtractor()
            doc_path = await extractor.process_package_documentation(package)
            
            if not doc_path:
                progress.update(task, description="Documentation extraction failed.")
                return
                
            # Update package
            package.original_doc_path = doc_path
            package.docs_extraction_date = datetime.now()
            package.status = PackageStatus.DOCS_EXTRACTED
            session.add(package)
            session.commit()
            
            # Create distillation job
            job = DistillationJob(
                package_id=package.id,
                status="pending",
                input_file_path=doc_path
            )
            session.add(job)
            session.commit()
            
            # Distill documentation
            progress.update(task, description=f"Distilling documentation for {package_name}...")
            
            distiller = DocumentationDistiller()
            distilled_path = await distiller.distill_documentation(package, doc_path)
            
            if not distilled_path:
                progress.update(task, description="Distillation failed.")
                return
                
            # Update package
            package.distilled_doc_path = distilled_path
            package.status = PackageStatus.DISTILLATION_COMPLETED
            package.distillation_end_date = datetime.now()
            session.add(package)
            
            # Update job
            job.status = "completed"
            job.completed_at = datetime.now()
            job.output_file_path = distilled_path
            session.add(job)
            session.commit()
            
            progress.update(task, description=f"Processing of {package_name} completed.")
    
    asyncio.run(run())
    
    # Get the final status
    package = session.exec(
        select(Package).where(Package.id == package.id)
    ).first()
    
    if package.status == PackageStatus.DISTILLATION_COMPLETED:
        console.print(f"[green]Processing of '{package_name}' completed successfully.[/green]")
        console.print(f"Original documentation: {package.original_doc_path}")
        console.print(f"Distilled documentation: {package.distilled_doc_path}")
    else:
        console.print(f"[red]Processing of '{package_name}' failed with status: {package.status}[/red]")


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
    session = Session(engine)
    
    # Build query
    query = select(Package)
    
    if status:
        try:
            query = query.where(Package.status == status)
        except ValueError:
            console.print(f"[red]Invalid status: {status}[/red]")
            valid_statuses = [s.value for s in PackageStatus]
            console.print(f"Valid statuses: {', '.join(valid_statuses)}")
            return
    
    # Execute query
    packages = session.exec(
        query.order_by(Package.priority.desc()).limit(limit)
    ).all()
    
    # Create table
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Priority")
    table.add_column("Docs Extracted")
    table.add_column("Distilled")
    
    for pkg in packages:
        table.add_row(
            str(pkg.id),
            pkg.name,
            pkg.status,
            str(pkg.priority),
            "✅" if pkg.original_doc_path else "❌",
            "✅" if pkg.distilled_doc_path else "❌"
        )
    
    console.print(table)
    console.print(f"Showing {len(packages)} of {session.exec(select(Package)).count()} packages.")


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
    if not os.path.exists("llm_docs.db"):
        console.print("[yellow]Database not initialized. Initializing now...[/yellow]")
        init_db()
    
    # Start server
    console.print(f"[green]Starting API server at http://{host}:{port}[/green]")
    console.print("Press Ctrl+C to stop.")
    
    uvicorn.run(
        "llm_docs.api.app:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    app()
