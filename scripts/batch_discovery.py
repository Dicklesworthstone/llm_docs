#!/usr/bin/env python
"""
Script to discover packages in batches and store them in the database.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sqlalchemy.future import select

from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.database import init_db, transaction
from llm_docs.storage.models import Package, PackageStats

# Initialize console
console = Console()

async def discover_batch(
    start_index: int, 
    batch_size: int, 
    rate_limit: float = 0.5,
    concurrency: int = 5,
    parent_progress = None
) -> Tuple[List[Package], int]:
    """
    Discover a batch of packages starting from a specific index.
    
    Args:
        start_index: Starting index in the top packages list
        batch_size: Number of packages to discover
        rate_limit: Maximum rate for API requests (requests per second)
        concurrency: Maximum number of concurrent API calls
        parent_progress: Optional parent progress object
        
    Returns:
        Tuple of (discovered packages, number of attempted packages)
    """
    async with transaction() as session:
        # Create discovery instance
        discovery = PackageDiscovery(session)
        
        # Adjust rate limiter based on parameters
        discovery.stats_rate_limiter.rate = rate_limit
        
        # Get the full list of top packages first
        console.print("Fetching top packages from PyPI...")
        top_packages_data = await discovery.get_top_packages_from_pypi(15000)
        
        # Extract just the batch we want
        batch_packages = top_packages_data[start_index:start_index + batch_size]
        package_names = [p.get("name", p) if isinstance(p, dict) else p for p in batch_packages if p]
        
        console.print(f"Processing {len(package_names)} packages in this batch")
        
        # Process packages
        discovered_packages = []
        
        # Set up batch progress tracking
        if parent_progress:
            batch_task = parent_progress.add_task(
                f"Batch {start_index//batch_size + 1} (positions {start_index+1}-{start_index+batch_size})...",
                total=len(package_names)
            )
            
            def progress_callback(completed):
                parent_progress.update(batch_task, completed=completed)
        else:
            # No progress tracking if no parent progress provided
            progress_callback = None
        
        # Process packages with manual implementation to ensure stats are created
        # Use semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0
        total = len(package_names)
        
        async def process_package(package_name: str) -> Optional[Package]:
            async with semaphore:
                try:
                    # Get package information
                    monthly_downloads, description = await discovery.get_package_info(package_name)
                    
                    # Store in database with explicit stats creation
                    async with transaction() as inner_session:
                        # Check if package already exists
                        result = await inner_session.execute(
                            select(Package).where(Package.name == package_name)
                        )
                        existing_package = result.scalar_one_or_none()
                        
                        if existing_package:
                            # Update existing package
                            package_stats = PackageStats(
                                package_id=existing_package.id,
                                monthly_downloads=monthly_downloads,
                                recorded_at=datetime.now()
                            )
                            inner_session.add(package_stats)
                            
                            # Update priority if downloads are higher
                            if monthly_downloads > existing_package.priority:
                                existing_package.priority = monthly_downloads
                                inner_session.add(existing_package)
                                
                            await inner_session.commit()
                            await inner_session.refresh(existing_package)
                            return existing_package
                        else:
                            # Create new package
                            new_package = Package(
                                name=package_name,
                                description=description,
                                discovery_date=datetime.now(),
                                priority=monthly_downloads  # Use downloads as initial priority
                            )
                            inner_session.add(new_package)
                            await inner_session.flush()  # To get the ID
                            
                            # Create stats record separately
                            package_stats = PackageStats(
                                package_id=new_package.id,
                                monthly_downloads=monthly_downloads,
                                recorded_at=datetime.now()
                            )
                            inner_session.add(package_stats)
                            await inner_session.commit()
                            await inner_session.refresh(new_package)
                            
                            return new_package
                except Exception as e:
                    console.print(f"[red]Error processing {package_name}: {str(e)}[/red]")
                    return None
        
        # Process packages in smaller batches for better progress reporting
        batch_size = min(10, max(1, concurrency * 2))
        
        for i in range(0, total, batch_size):
            batch = package_names[i:i+batch_size]
            tasks = [process_package(name) for name in batch]
            results = await asyncio.gather(*tasks)
            
            valid_results = [r for r in results if r is not None]
            discovered_packages.extend(valid_results)
            
            # Update progress if callback provided
            completed += len(batch)
            if progress_callback:
                progress_callback(completed)
        
        # Retry failed packages if needed
        if discovery.failed_stats_packages and len(discovery.failed_stats_packages) > 0:
            console.print(f"[yellow]Retrying {len(discovery.failed_stats_packages)} failed packages...[/yellow]")
            await asyncio.sleep(5)  # Give the API a small break before retrying
            retry_count = await discovery.retry_failed_packages()
            console.print(f"[green]Successfully retrieved stats for {retry_count} retry packages[/green]")
        
        await discovery.close()
        
        return discovered_packages, len(package_names)

async def async_main(args):
    """Run the batch discovery asynchronously."""
    # Initialize database if needed
    console.print("[yellow]Ensuring database is initialized...[/yellow]")
    await init_db()  # Always call init_db to ensure PRAGMAs are set
    
    total_batches = args.total // args.batch_size
    if args.total % args.batch_size > 0:
        total_batches += 1
    
    console.print(f"Discovering {args.total} packages in {total_batches} batches of {args.batch_size}")
    console.print(f"Rate limit: {args.rate_limit} requests/second, Concurrency: {args.concurrency}")
    
    all_discovered = []
    
    # Use a single Progress object for all progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        console=console
    ) as progress:
        main_task = progress.add_task("Processing batches...", total=total_batches)
        
        for batch_idx in range(total_batches):
            start_index = batch_idx * args.batch_size
            batch_size = min(args.batch_size, args.total - start_index)
            
            # Update the main task description
            progress.update(main_task, description=f"Processing batch {batch_idx+1}/{total_batches}...")
            
            # Pass the progress object to avoid nesting
            discovered, attempted = await discover_batch(
                start_index=start_index,
                batch_size=batch_size,
                rate_limit=args.rate_limit,
                concurrency=args.concurrency,
                parent_progress=progress  # Pass the progress object
            )
            
            all_discovered.extend(discovered)
            
            progress.update(main_task, advance=1)
            
            # Apply pause between batches if specified
            if batch_idx < total_batches - 1 and args.pause > 0:
                pause_msg = f"Pausing for {args.pause} seconds before next batch..."
                progress.update(main_task, description=pause_msg)
                await asyncio.sleep(args.pause)
    
    # Report final statistics
    console.print(f"[green]Successfully discovered {len(all_discovered)} packages out of {args.total} attempted.[/green]")
    
    # Verify package stats were created
    async with transaction() as session:
        pkg_count = await session.execute(select(Package))
        pkg_count = len(pkg_count.scalars().all())
        
        stats_count = await session.execute(select(PackageStats))
        stats_count = len(stats_count.scalars().all())
        
        console.print(f"[cyan]Database contains {pkg_count} packages with {stats_count} package stats records.[/cyan]")

def main():
    """Run the batch discovery."""
    parser = argparse.ArgumentParser(description="Discover packages in batches")
    parser.add_argument(
        "--total",
        type=int,
        default=100,
        help="Total number of packages to discover"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of packages per batch"
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=0,
        help="Seconds to pause between batches"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Maximum rate for API requests (requests per second)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent API calls"
    )
    
    args = parser.parse_args()
    
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()