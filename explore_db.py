#!/usr/bin/env python
"""
explore_db.py

Database explorer script for llm_docs.

This script explores the SQLite database to provide an overview of the current progress
of package discovery, documentation extraction, and distillation.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree
from sqlalchemy import func, text
from sqlalchemy.future import select

from llm_docs.storage.database import transaction
from llm_docs.storage.models import (
    DistillationJob,
    DistillationJobStatus,
    DocumentationPage,
    Package,
    PackageStats,
    PackageStatus,
)

# Initialize rich console for pretty output
console = Console()

# Database file path (default from config)
DB_PATH = "llm_docs.db"

# Configuration
SHOW_TOP_PACKAGES = 10
SHOW_RECENT_ACTIVITY = 10
SHOW_FAILED_PACKAGES = 10


async def get_database_info() -> Dict[str, Any]:
    """Get basic information about the database and tables."""
    async with transaction() as session:
        # Get table names
        result = await session.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        tables = [row[0] for row in result.fetchall()]
        
        # Get database file info
        db_size = Path(DB_PATH).stat().st_size if Path(DB_PATH).exists() else 0
        db_modified = datetime.fromtimestamp(Path(DB_PATH).stat().st_mtime) if Path(DB_PATH).exists() else None
        
        return {
            "tables": tables,
            "db_size": db_size,
            "db_modified": db_modified
        }


async def get_package_stats() -> Dict[str, Any]:
    """Get statistics about packages."""
    async with transaction() as session:
        # Count packages
        result = await session.execute(select(func.count()).select_from(Package))
        total_packages = result.scalar() or 0
        
        # Count packages by status
        status_counts = {}
        for status in PackageStatus:
            result = await session.execute(
                select(func.count()).select_from(Package).where(Package.status == status)
            )
            status_counts[status.value] = result.scalar() or 0
        
        # Get packages with documentation extracted
        result = await session.execute(
            select(func.count()).select_from(Package).where(Package.original_doc_path.is_not(None))
        )
        docs_extracted = result.scalar() or 0
        
        # Get packages with distilled documentation
        result = await session.execute(
            select(func.count()).select_from(Package).where(Package.distilled_doc_path.is_not(None))
        )
        docs_distilled = result.scalar() or 0
        
        # Top packages by priority (downloads)
        result = await session.execute(
            select(Package).order_by(Package.priority.desc()).limit(SHOW_TOP_PACKAGES)
        )
        top_packages = result.scalars().all()
        
        # Recent packages by discovery date
        result = await session.execute(
            select(Package).order_by(Package.discovery_date.desc()).limit(SHOW_RECENT_ACTIVITY)
        )
        recent_discoveries = result.scalars().all()
        
        # Recent distillations
        result = await session.execute(
            select(Package)
            .where(Package.distillation_end_date.is_not(None))
            .order_by(Package.distillation_end_date.desc())
            .limit(SHOW_RECENT_ACTIVITY)
        )
        recent_distillations = result.scalars().all()
        
        # Failed packages
        result = await session.execute(
            select(Package).where(Package.status == PackageStatus.DISTILLATION_FAILED).limit(SHOW_FAILED_PACKAGES)
        )
        failed_packages = result.scalars().all()
        
        # Distillation time statistics
        result = await session.execute(
            select(
                func.avg(Package.distillation_end_date - Package.distillation_start_date)
            ).where(
                Package.distillation_end_date.is_not(None),
                Package.distillation_start_date.is_not(None)
            )
        )
        avg_distillation_time = result.scalar()
        
        return {
            "total_packages": total_packages,
            "status_counts": status_counts,
            "docs_extracted": docs_extracted,
            "docs_distilled": docs_distilled,
            "top_packages": top_packages,
            "recent_discoveries": recent_discoveries,
            "recent_distillations": recent_distillations,
            "failed_packages": failed_packages,
            "avg_distillation_time": avg_distillation_time
        }


async def get_distillation_job_stats() -> Dict[str, Any]:
    """Get statistics about distillation jobs."""
    async with transaction() as session:
        # Count jobs
        result = await session.execute(select(func.count()).select_from(DistillationJob))
        total_jobs = result.scalar() or 0
        
        # Count jobs by status
        status_counts = {}
        for status in DistillationJobStatus:
            result = await session.execute(
                select(func.count()).select_from(DistillationJob).where(DistillationJob.status == status)
            )
            status_counts[status.value] = result.scalar() or 0
        
        # Recent jobs
        result = await session.execute(
            select(DistillationJob).order_by(DistillationJob.created_at.desc()).limit(SHOW_RECENT_ACTIVITY)
        )
        recent_jobs = result.scalars().all()
        
        # Failed jobs
        result = await session.execute(
            select(DistillationJob).where(DistillationJob.status == DistillationJobStatus.FAILED).limit(SHOW_FAILED_PACKAGES)
        )
        failed_jobs = result.scalars().all()

        # Job completion time statistics
        result = await session.execute(
            select(
                func.avg(DistillationJob.completed_at - DistillationJob.started_at)
            ).where(
                DistillationJob.completed_at.is_not(None),
                DistillationJob.started_at.is_not(None)
            )
        )
        avg_job_time = result.scalar()
        
        return {
            "total_jobs": total_jobs,
            "status_counts": status_counts,
            "recent_jobs": recent_jobs,
            "failed_jobs": failed_jobs,
            "avg_job_time": avg_job_time
        }


async def get_documentation_stats() -> Dict[str, Any]:
    """Get statistics about documentation pages."""
    async with transaction() as session:
        # Count documentation pages
        result = await session.execute(select(func.count()).select_from(DocumentationPage))
        total_pages = result.scalar() or 0
        
        # Get documentation storage information
        docs_dir = Path("docs")
        distilled_dir = Path("distilled_docs")
        
        docs_size = sum(f.stat().st_size for f in docs_dir.glob("*") if f.is_file()) if docs_dir.exists() else 0
        distilled_size = sum(f.stat().st_size for f in distilled_dir.glob("*") if f.is_file()) if distilled_dir.exists() else 0
        
        docs_count = len(list(docs_dir.glob("*"))) if docs_dir.exists() else 0
        distilled_count = len(list(distilled_dir.glob("*"))) if distilled_dir.exists() else 0
        
        return {
            "total_pages": total_pages,
            "docs_size": docs_size,
            "distilled_size": distilled_size,
            "docs_count": docs_count,
            "distilled_count": distilled_count
        }


def print_database_info(db_info: Dict[str, Any]):
    """Print database information."""
    # Database summary panel
    db_size_mb = db_info["db_size"] / (1024 * 1024)
    db_modified_str = db_info["db_modified"].strftime("%Y-%m-%d %H:%M:%S") if db_info["db_modified"] else "Unknown"
    
    console.print(Panel(
        f"Database Path: [bold]{DB_PATH}[/bold]\n"
        f"Database Size: [bold]{db_size_mb:.2f} MB[/bold]\n"
        f"Last Modified: [bold]{db_modified_str}[/bold]\n"
        f"Tables: [bold]{', '.join(db_info['tables'])}[/bold]",
        title="Database Information",
        expand=False
    ))


def print_package_stats(pkg_stats: Dict[str, Any]):
    """Print package statistics."""
    # Create status table
    status_table = Table(title="Package Status Breakdown")
    status_table.add_column("Status", style="cyan")
    status_table.add_column("Count", style="magenta")
    status_table.add_column("Percentage", style="green")
    
    for status, count in pkg_stats["status_counts"].items():
        percentage = (count / pkg_stats["total_packages"] * 100) if pkg_stats["total_packages"] > 0 else 0
        status_table.add_row(status, str(count), f"{percentage:.1f}%")
    
    # Create top packages table
    top_table = Table(title=f"Top {len(pkg_stats['top_packages'])} Packages by Downloads")
    top_table.add_column("Package", style="cyan")
    top_table.add_column("Downloads", style="magenta")
    top_table.add_column("Status", style="green")
    top_table.add_column("Description", style="dim")
    
    for pkg in pkg_stats["top_packages"]:
        top_table.add_row(
            pkg.name,
            f"{pkg.priority:,}",
            pkg.status,
            (pkg.description[:50] + "...") if pkg.description and len(pkg.description) > 50 else (pkg.description or "")
        )
    
    # Create recent distillations table
    if pkg_stats["recent_distillations"]:
        recent_table = Table(title="Recent Distillations")
        recent_table.add_column("Package", style="cyan")
        recent_table.add_column("Completed", style="magenta")
        recent_table.add_column("Duration", style="green")
        
        for pkg in pkg_stats["recent_distillations"]:
            if pkg.distillation_start_date and pkg.distillation_end_date:
                duration = pkg.distillation_end_date - pkg.distillation_start_date
                duration_str = f"{duration.total_seconds() / 60:.1f} min"
            else:
                duration_str = "Unknown"
                
            recent_table.add_row(
                pkg.name,
                pkg.distillation_end_date.strftime("%Y-%m-%d %H:%M") if pkg.distillation_end_date else "Unknown",
                duration_str
            )
    # Print overall stats
    extraction_percentage = (pkg_stats['docs_extracted'] / pkg_stats['total_packages'] * 100) if pkg_stats['total_packages'] > 0 else 0
    distillation_percentage = (pkg_stats['docs_distilled'] / pkg_stats['total_packages'] * 100) if pkg_stats['total_packages'] > 0 else 0
    
    console.print(Panel(
        f"Total Packages: [bold]{pkg_stats['total_packages']}[/bold]\n"
        f"With Extracted Docs: [bold]{pkg_stats['docs_extracted']} ({extraction_percentage:.1f}%)[/bold]\n"
        f"With Distilled Docs: [bold]{pkg_stats['docs_distilled']} ({distillation_percentage:.1f}%)[/bold]",
        title="Package Statistics",
        expand=False
    ))

    # Print tables
    console.print(status_table)
    console.print(top_table)
    if pkg_stats["recent_distillations"]:
        console.print(recent_table)
    
    # Print failed packages if any
    if pkg_stats["failed_packages"]:
        failed_table = Table(title="Failed Packages")
        failed_table.add_column("Package", style="cyan")
        failed_table.add_column("Discovery Date", style="magenta")
        
        for pkg in pkg_stats["failed_packages"]:
            failed_table.add_row(
                pkg.name,
                pkg.discovery_date.strftime("%Y-%m-%d") if pkg.discovery_date else "Unknown"
            )
        
        console.print(failed_table)


def print_distillation_job_stats(job_stats: Dict[str, Any]):
    """Print distillation job statistics."""
    # Create status table
    status_table = Table(title="Distillation Job Status Breakdown")
    status_table.add_column("Status", style="cyan")
    status_table.add_column("Count", style="magenta")
    status_table.add_column("Percentage", style="green")
    
    for status, count in job_stats["status_counts"].items():
        percentage = (count / job_stats["total_jobs"] * 100) if job_stats["total_jobs"] > 0 else 0
        status_table.add_row(status, str(count), f"{percentage:.1f}%")
    
    # Print overall stats
    console.print(Panel(
        f"Total Jobs: [bold]{job_stats['total_jobs']}[/bold]\n"
        f"Average Job Time: [bold]{job_stats['avg_job_time']}[/bold]",
        title="Distillation Job Statistics",
        expand=False
    ))
    
    # Print table
    console.print(status_table)
    
    # Print recent jobs if any
    if job_stats["recent_jobs"]:
        recent_table = Table(title="Recent Distillation Jobs")
        recent_table.add_column("ID", style="cyan")
        recent_table.add_column("Package ID", style="magenta")
        recent_table.add_column("Status", style="green")
        recent_table.add_column("Created", style="dim")
        
        for job in job_stats["recent_jobs"]:
            recent_table.add_row(
                str(job.id),
                str(job.package_id),
                job.status,
                job.created_at.strftime("%Y-%m-%d %H:%M") if job.created_at else "Unknown"
            )
        
        console.print(recent_table)


def print_documentation_stats(doc_stats: Dict[str, Any]):
    """Print documentation statistics."""
    docs_size_mb = doc_stats["docs_size"] / (1024 * 1024)
    distilled_size_mb = doc_stats["distilled_size"] / (1024 * 1024)
    
    reduction_percentage = ((doc_stats["docs_size"] - doc_stats["distilled_size"]) / doc_stats["docs_size"] * 100) if doc_stats["docs_size"] > 0 else 0
    
    console.print(Panel(
        f"Documentation Pages: [bold]{doc_stats['total_pages']}[/bold]\n"
        f"Original Docs Count: [bold]{doc_stats['docs_count']}[/bold]\n"
        f"Distilled Docs Count: [bold]{doc_stats['distilled_count']}[/bold]\n"
        f"Original Docs Size: [bold]{docs_size_mb:.2f} MB[/bold]\n"
        f"Distilled Docs Size: [bold]{distilled_size_mb:.2f} MB[/bold]\n"
        f"Size Reduction: [bold]{reduction_percentage:.1f}%[/bold]",
        title="Documentation Statistics",
        expand=False
    ))


async def explore_specific_package(package_name: str) -> None:
    """Explore a specific package in detail."""
    async with transaction() as session:
        # Get package
        result = await session.execute(
            select(Package).where(Package.name == package_name)
        )
        package = result.scalar_one_or_none()
        
        if not package:
            console.print(f"[red]Package '{package_name}' not found in database[/red]")
            return
        
        # Get stats for the package
        result = await session.execute(
            select(PackageStats).where(PackageStats.package_id == package.id).order_by(PackageStats.recorded_at)
        )
        stats = result.scalars().all()
        
        # Get distillation jobs for the package
        result = await session.execute(
            select(DistillationJob).where(DistillationJob.package_id == package.id).order_by(DistillationJob.created_at)
        )
        jobs = result.scalars().all()
        
        # Get documentation pages for the package
        result = await session.execute(
            select(DocumentationPage).where(DocumentationPage.package_id == package.id)
        )
        pages = result.scalars().all()
        
        # Print package details
        console.print(Panel(
            f"Package: [bold]{package.name}[/bold] (ID: {package.id})\n"
            f"Status: [bold]{package.status}[/bold]\n"
            f"Priority: [bold]{package.priority:,}[/bold]\n"
            f"Description: [dim]{package.description or 'None'}[/dim]\n"
            f"Discovery Date: [bold]{package.discovery_date}[/bold]\n"
            f"Docs URL: [link={package.docs_url or ''}]{package.docs_url or 'None'}[/link]\n"
            f"Original Doc Path: [green]{package.original_doc_path or 'None'}[/green]\n"
            f"Distilled Doc Path: [green]{package.distilled_doc_path or 'None'}[/green]\n"
            f"Docs Extraction Date: [bold]{package.docs_extraction_date or 'None'}[/bold]\n"
            f"Distillation Start: [bold]{package.distillation_start_date or 'None'}[/bold]\n"
            f"Distillation End: [bold]{package.distillation_end_date or 'None'}[/bold]",
            title=f"Package Details: {package.name}",
            expand=False
        ))
        
        if stats:
            stats_table = Table(title="Download Statistics History")
            stats_table.add_column("Date", style="cyan")
            stats_table.add_column("Monthly Downloads", style="magenta")
            
            for stat in stats:
                stats_table.add_row(
                    stat.recorded_at.strftime("%Y-%m-%d") if stat.recorded_at else "Unknown",
                    f"{stat.monthly_downloads:,}"
                )
            
            console.print(stats_table)
        
        if jobs:
            jobs_table = Table(title="Distillation Jobs")
            jobs_table.add_column("ID", style="cyan")
            jobs_table.add_column("Status", style="magenta")
            jobs_table.add_column("Created", style="green")
            jobs_table.add_column("Started", style="green")
            jobs_table.add_column("Completed", style="green")
            
            for job in jobs:
                jobs_table.add_row(
                    str(job.id),
                    job.status,
                    job.created_at.strftime("%Y-%m-%d %H:%M") if job.created_at else "Unknown",
                    job.started_at.strftime("%Y-%m-%d %H:%M") if job.started_at else "None",
                    job.completed_at.strftime("%Y-%m-%d %H:%M") if job.completed_at else "None"
                )
            
            console.print(jobs_table)
        
        if pages:
            pages_table = Table(title="Documentation Pages")
            pages_table.add_column("URL", style="cyan")
            pages_table.add_column("Title", style="magenta")
            pages_table.add_column("Extraction Date", style="green")
            
            for page in pages:
                pages_table.add_row(
                    page.url,
                    page.title or "No title",
                    page.extraction_date.strftime("%Y-%m-%d") if page.extraction_date else "Unknown"
                )
            
            console.print(pages_table)
        
        # Show file sizes if available
        if package.original_doc_path and Path(package.original_doc_path).exists():
            orig_size = Path(package.original_doc_path).stat().st_size
            console.print(f"Original documentation file size: [bold]{orig_size / 1024:.1f} KB[/bold]")
        
        if package.distilled_doc_path and Path(package.distilled_doc_path).exists():
            distilled_size = Path(package.distilled_doc_path).stat().st_size
            console.print(f"Distilled documentation file size: [bold]{distilled_size / 1024:.1f} KB[/bold]")
            
            if package.original_doc_path and Path(package.original_doc_path).exists():
                orig_size = Path(package.original_doc_path).stat().st_size
                reduction = ((orig_size - distilled_size) / orig_size * 100)
                console.print(f"Size reduction: [bold]{reduction:.1f}%[/bold]")


async def get_processing_pipeline_status() -> None:
    """Get status of the entire processing pipeline."""
    async with transaction() as session:
        # Count packages by status
        status_counts = {}
        for status in PackageStatus:
            result = await session.execute(
                select(func.count()).select_from(Package).where(Package.status == status)
            )
            status_counts[status.value] = result.scalar() or 0
        
        # Count distillation jobs by status
        job_status_counts = {}
        for status in DistillationJobStatus:
            result = await session.execute(
                select(func.count()).select_from(DistillationJob).where(DistillationJob.status == status)
            )
            job_status_counts[status.value] = result.scalar() or 0
        
        # Draw pipeline tree
        pipeline = Tree("📦 Processing Pipeline")
        
        discovered = pipeline.add("🔍 [bold]Discovery[/bold]")
        discovered.add(f"Discovered: {status_counts.get('discovered', 0):,}")
        
        extraction = pipeline.add("📄 [bold]Documentation Extraction[/bold]")
        extraction.add(f"Extracted: {status_counts.get('docs_extracted', 0):,}")
        extraction.add(f"Failed: {status_counts.get('extraction_failed', 0):,}")
        
        distillation = pipeline.add("🧠 [bold]Distillation[/bold]")
        distillation.add(f"Pending: {job_status_counts.get('pending', 0):,}")
        distillation.add(f"In Progress: {job_status_counts.get('in_progress', 0):,}")
        distillation.add(f"Completed: {job_status_counts.get('completed', 0):,}")
        distillation.add(f"Failed: {job_status_counts.get('failed', 0):,}")
        
        # Calculate overall progress
        result = await session.execute(select(func.count()).select_from(Package))
        total_packages = result.scalar() or 0
        
        completed_count = status_counts.get("distillation_completed", 0)
        overall_progress = (completed_count / total_packages * 100) if total_packages > 0 else 0
        
        console.print(Panel(
            f"Overall Progress: [bold]{overall_progress:.1f}%[/bold] ({completed_count:,}/{total_packages:,} packages fully processed)",
            title="Processing Pipeline Status",
            expand=False
        ))
        
        console.print(pipeline)


async def main(package_name: Optional[str] = None):
    """Main function to run the database explorer."""
    console.print("[bold]LLM-Docs Database Explorer[/bold]")
    
    # Check if database exists
    if not Path(DB_PATH).exists():
        console.print(f"[red]Database file '{DB_PATH}' not found.[/red]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Exploring database...", total=None)
        
        if package_name:
            # Explore specific package
            progress.update(task, description=f"Exploring package: {package_name}...")
            await explore_specific_package(package_name)
        else:
            # Get database statistics
            progress.update(task, description="Getting database info...")
            db_info = await get_database_info()
            
            progress.update(task, description="Getting package stats...")
            pkg_stats = await get_package_stats()
            
            progress.update(task, description="Getting distillation job stats...")
            job_stats = await get_distillation_job_stats()
            
            progress.update(task, description="Getting documentation stats...")
            doc_stats = await get_documentation_stats()
            
            progress.update(task, description="Getting pipeline status...")
            
            # Print statistics
            console.print("\n")
            print_database_info(db_info)
            console.print("\n")
            await get_processing_pipeline_status()
            console.print("\n")
            print_package_stats(pkg_stats)
            console.print("\n")
            print_distillation_job_stats(job_stats)
            console.print("\n")
            print_documentation_stats(doc_stats)


if __name__ == "__main__":
    package_name = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(package_name))