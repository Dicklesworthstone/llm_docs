#!/usr/bin/env python
"""
Script to retry fetching stats for packages that previously failed.
"""

import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from sqlalchemy.future import select

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_docs.storage.database import transaction
from llm_docs.storage.models import Package, PackageStats

# Initialize console
console = Console()

class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, rate: float = 1.0, burst: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate: Requests per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_refill = datetime.now()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            while True:
                # Refill tokens based on time elapsed
                now = datetime.now()
                elapsed = (now - self.last_refill).total_seconds()
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_refill = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                
                # No tokens available, wait until next token is available
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)

async def retry_package_stats(package_name, rate_limiter, cache_dir=None):
    """Retry fetching stats for a package with improved rate limiting."""
    import httpx
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path("cache/package_discovery")
        cache_dir.mkdir(exist_ok=True, parents=True)
    
    cache_file = cache_dir / f"{package_name}_stats.json"
    
    # Apply rate limiting before making API request
    await rate_limiter.acquire()
    
    # Add jitter to avoid burst requests
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    # Fetch from API
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            url = f"https://pypistats.org/api/packages/{package_name}/recent"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache the successful results
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                    
                monthly_downloads = data.get("data", {}).get("last_month", 0)
                return package_name, monthly_downloads, True
            else:
                return package_name, 0, False
    except Exception as e:
        console.print(f"[yellow]Error fetching stats for {package_name}: {str(e)}[/yellow]")
        return package_name, 0, False

async def find_packages_without_stats():
    """Find packages that have 0 priority (indicating they likely have no stats)."""
    async with transaction() as session:
        # Find packages with priority 0
        result = await session.execute(
            select(Package).where(Package.priority == 0)
        )
        packages = result.scalars().all()
        
        return [pkg.name for pkg in packages]

async def update_package_priority(package_name, monthly_downloads):
    """Update package priority based on new statistics."""
    async with transaction() as session:
        # Find the package
        result = await session.execute(
            select(Package).where(Package.name == package_name)
        )
        package = result.scalar_one_or_none()
        
        if package:
            # Update priority and add new stats entry
            package.priority = monthly_downloads
            package_stats = PackageStats(
                package_id=package.id,
                monthly_downloads=monthly_downloads,
                recorded_at=datetime.now()
            )
            session.add(package_stats)
            session.add(package)
            await session.commit()
            return True
        return False

async def main():
    """Retry fetching stats for packages that failed previously."""
    # Find packages without stats
    console.print("[cyan]Finding packages without stats...[/cyan]")
    packages_without_stats = await find_packages_without_stats()
    
    if not packages_without_stats:
        console.print("[green]No packages without stats found.[/green]")
        return
    
    console.print(f"[cyan]Found {len(packages_without_stats)} packages without stats.[/cyan]")
    
    # Create rate limiter - very conservative to avoid 429s
    rate_limiter = RateLimiter(rate=0.3, burst=1)  # One request per 3.33 seconds
    
    # Process packages with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Retrying package stats...", total=len(packages_without_stats))
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)
        
        async def process_with_rate_limit(pkg_name):
            async with semaphore:
                progress.update(task, description=f"Processing {pkg_name}...")
                pkg_name, downloads, success = await retry_package_stats(pkg_name, rate_limiter)
                
                if success and downloads > 0:
                    await update_package_priority(pkg_name, downloads)
                
                # Add random delay between requests
                delay = random.uniform(0.5, 2.0)
                await asyncio.sleep(delay)
                
                progress.update(task, advance=1)
                return pkg_name, downloads, success
        
        # Process all packages concurrently (but with rate limiting)
        tasks = [process_with_rate_limit(name) for name in packages_without_stats]
        results = await asyncio.gather(*tasks)
        
        # Count successes and failures
        successes = [name for name, downloads, success in results if success and downloads > 0]
        failures = [name for name, downloads, success in results if not success or downloads == 0]
        
        progress.update(task, description=f"Completed: {len(successes)} succeeded, {len(failures)} failed")
    
    # Show summary
    console.print("\n[bold]Stats Retry Summary[/bold]")
    console.print(f"Total packages processed: {len(packages_without_stats)}")
    console.print(f"Successfully retrieved stats: {len(successes)} packages")
    console.print(f"Failed to retrieve stats: {len(failures)} packages")
    
    if len(failures) > 0:
        console.print("\n[yellow]Failed packages:[/yellow]")
        if len(failures) <= 20:
            for name in failures:
                console.print(f"  - {name}")
        else:
            console.print(f"  {len(failures)} packages failed (too many to display)")

if __name__ == "__main__":
    asyncio.run(main())