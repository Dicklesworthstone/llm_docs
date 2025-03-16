import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from tqdm import tqdm

from llm_docs.storage.database import transaction
from llm_docs.storage.models import Package, PackageStats

# Initialize console for rich output
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

class PackageDiscovery:
    """Discovers and prioritizes Python packages from PyPI."""
    
    def __init__(self, db_session: AsyncSession, cache_dir: Optional[Path] = None):
        """
        Initialize the package discovery module.
        
        Args:
            db_session: Async database session
            cache_dir: Directory to cache package data (optional)
        """
        self.db_session = db_session
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        
        # Set up cache directory
        self.cache_dir = cache_dir or Path("cache/package_discovery")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for top packages
        self.top_packages_cache = self.cache_dir / "top_packages.json"
        
        # Rate limiters for different API endpoints
        self.stats_rate_limiter = RateLimiter(rate=0.5, burst=1)  # 2 requests per second max
        self.pypi_rate_limiter = RateLimiter(rate=2.0, burst=5)   # More generous for PyPI API
        
        # Track failed packages for retry
        self.failed_stats_packages = set()
        
    async def get_top_packages_from_pypi(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get the top packages from PyPI using the top-packages JSON data.
        
        Args:
            limit: Maximum number of packages to retrieve
            
        Returns:
            List of package dictionaries with name and download count
        """
        console.print(f"[cyan]Fetching top {limit} packages from PyPI...[/cyan]")
        
        # Check cache first (if less than 24 hours old)
        if self.top_packages_cache.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(self.top_packages_cache.stat().st_mtime)
            if cache_age < timedelta(hours=24):
                try:
                    with open(self.top_packages_cache, 'r') as f:
                        cached_data = json.load(f)
                        console.print(f"[green]Using cached package data ({len(cached_data)} packages)[/green]")
                        return cached_data[:limit]
                except (json.JSONDecodeError, KeyError):
                    console.print("[yellow]Cache file corrupt, fetching fresh data[/yellow]")
        
        try:
            # Directly fetch the JSON data from hugovk's GitHub page
            # This source provides regularly updated top PyPI packages data
            url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json"
            
            # Apply rate limiting
            await self.pypi_rate_limiter.acquire()
            
            response = await self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                # Extract the packages data
                packages = [
                    {"name": pkg["project"], "downloads": pkg["download_count"]}
                    for pkg in data.get("rows", [])
                ]
                
                # Cache the results
                if packages:
                    with open(self.top_packages_cache, 'w') as f:
                        json.dump(packages, f)
                        
                return packages[:limit]
            else:
                console.print(f"[yellow]Failed to get top packages: {response.status_code}[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error fetching top packages: {str(e)}[/bold red]")
        
        # Fall back to cache or hardcoded list as before
        if self.top_packages_cache.exists():
            console.print("[yellow]Using cached package data despite age[/yellow]")
            try:
                with open(self.top_packages_cache, 'r') as f:
                    return json.load(f)[:limit]
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Last resort: hardcoded top packages
        console.print("[yellow]Using fallback hardcoded top packages list[/yellow]")
        return [
            {"name": "numpy", "downloads": 50000000},
            {"name": "pandas", "downloads": 45000000},
            # Rest of hardcoded packages...
        ][:limit]
    
    async def get_package_stats(self, package_name: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Get download statistics for a specific package with retry logic.
        
        Args:
            package_name: Name of the package
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing download statistics
        """
        # Check cache first
        cache_file = self.cache_dir / f"{package_name}_stats.json"
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(days=7):  # Cache for a week
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, KeyError):
                    pass

        # Apply rate limiting before making API request
        await self.stats_rate_limiter.acquire()

        # Fetch from API with retry logic
        retries = 0
        backoff_time = 2.0  # Start with 2-second backoff
        
        while retries <= max_retries:
            try:
                url = f"https://pypistats.org/api/packages/{package_name}/recent"
                response = await self.client.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Cache the successful results
                    with open(cache_file, 'w') as f:
                        json.dump(data, f)
                        
                    # Remove from failed packages if it was there
                    if package_name in self.failed_stats_packages:
                        self.failed_stats_packages.remove(package_name)
                        
                    return data
                elif response.status_code == 429:  # Rate limited
                    retries += 1
                    if retries <= max_retries:
                        # Add jitter to backoff (±20%)
                        jitter = backoff_time * 0.2
                        sleep_time = backoff_time + random.uniform(-jitter, jitter)
                        console.print(f"[yellow]Rate limited for {package_name}, retrying in {sleep_time:.1f}s (attempt {retries}/{max_retries})[/yellow]")
                        await asyncio.sleep(sleep_time)
                        backoff_time *= 2  # Exponential backoff
                    else:
                        # Track failed packages
                        self.failed_stats_packages.add(package_name)
                        console.print(f"[yellow]Failed to get stats for {package_name} after {max_retries} retries: {response.status_code}[/yellow]")
                        return {"data": {"last_month": 0}}
                else:
                    console.print(f"[yellow]Failed to get stats for {package_name}: {response.status_code}[/yellow]")
                    return {"data": {"last_month": 0}}
                    
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    console.print(f"[yellow]Error fetching stats for {package_name}: {str(e)}[/yellow]")
                    return {"data": {"last_month": 0}}
        
        return {"data": {"last_month": 0}}
        
    async def _store_package_db(self, package_name: str, monthly_downloads: int, description: Optional[str] = None, session=None) -> Package:
        """Store package in database using async operations."""
        # Use provided session or create a transaction
        if session is not None:
            # Use the provided session directly
            return await self._store_package_with_session(
                session, package_name, monthly_downloads, description
            )
        else:
            # Create a new transaction
            async with transaction() as session:
                return await self._store_package_with_session(
                    session, package_name, monthly_downloads, description
                )
                
    async def _store_package_with_session(self, session, package_name: str, monthly_downloads: int, description: Optional[str] = None) -> Package:
        """Internal helper to store package using a specific session."""
        # Check if package already exists in database
        result = await session.execute(
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
            session.add(package_stats)
            await session.refresh(existing_package)
            
            return existing_package
        else:
            # Create new package
            new_package = Package(
                name=package_name,
                description=description,
                discovery_date=datetime.now(),
                priority=monthly_downloads,  # Use downloads as initial priority
            )
            session.add(new_package)
            await session.flush()  # To get the ID
            
            # Add stats
            package_stats = PackageStats(
                package_id=new_package.id,
                monthly_downloads=monthly_downloads,
                recorded_at=datetime.now()
            )
            session.add(package_stats)
            await session.refresh(new_package)
            
            return new_package
    
    async def get_package_info(self, package_name: str) -> Tuple[int, Optional[str]]:
        """
        Get package information including downloads and description.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Tuple of (monthly_downloads, description)
        """
        downloads = 0
        description = None
        
        # Get download stats
        stats = await self.get_package_stats(package_name)
        downloads = stats.get("data", {}).get("last_month", 0)
        
        # Try to get package description from PyPI
        try:
            # Apply rate limiting
            await self.pypi_rate_limiter.acquire()
            
            url = f"https://pypi.org/pypi/{package_name}/json"
            response = await self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                description = data.get("info", {}).get("summary", "")
                
        except Exception as e:
            console.print(f"[yellow]Error fetching description for {package_name}: {str(e)}[/yellow]")
        
        return downloads, description

    async def retry_failed_packages(self) -> int:
        """
        Retry getting stats for packages that previously failed.
        
        Returns:
            Number of successfully retrieved package stats
        """
        if not self.failed_stats_packages:
            return 0
            
        console.print(f"[cyan]Retrying stats retrieval for {len(self.failed_stats_packages)} failed packages...[/cyan]")
        
        # Make a copy since we'll be modifying the set during iteration
        packages_to_retry = self.failed_stats_packages.copy()
        success_count = 0
        
        for package_name in packages_to_retry:
            # Use a longer backoff for retries
            stats = await self.get_package_stats(package_name, max_retries=5)
            monthly_downloads = stats.get("data", {}).get("last_month", 0)
            
            if monthly_downloads > 0 or package_name not in self.failed_stats_packages:
                success_count += 1
                
            # Add a small delay between retries
            await asyncio.sleep(2.0)
        
        console.print(f"[green]Successfully retrieved stats for {success_count}/{len(packages_to_retry)} retry packages[/green]")
        return success_count

    async def discover_and_store_packages(self, limit: int = 1000, concurrency: int = 5, retry_failed: bool = True) -> List[Package]:
        """
        Discover top packages and store them in the database with improved rate limiting.
        
        Args:
            limit: Maximum number of packages to retrieve
            concurrency: Maximum number of concurrent API calls
            retry_failed: Whether to retry failed stats retrievals
            
        Returns:
            List of Package objects stored in the database
        """
        
        top_packages_data = await self.get_top_packages_from_pypi(limit)
        
        # More robust handling of top_packages_data
        top_packages = []
        for p in top_packages_data:
            try:
                if isinstance(p, dict) and "name" in p and p["name"]:
                    # Ensure package name is a non-empty string
                    package_name = str(p["name"]).strip()
                    if package_name:
                        top_packages.append(package_name)
                elif isinstance(p, str) and p.strip():
                    # If it's already a string, just make sure it's not empty
                    top_packages.append(p.strip())
                else:
                    console.print(f"[yellow]Skipping invalid package data: {p}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Error processing package data entry: {e}[/yellow]")
        
        # Check if we have any valid packages
        if not top_packages:
            console.print("[red]No valid package names found in the data[/red]")
            return []
        
        stored_packages = []
        
        with tqdm(total=len(top_packages), desc="Processing packages") as pbar:
            # Use a semaphore to limit concurrent API calls - reduced from 10 to a more conservative value
            semaphore = asyncio.Semaphore(concurrency)
            
            async def process_package(package_name: str) -> Optional[Package]:
                async with semaphore:
                    try:
                        # Get package information
                        monthly_downloads, description = await self.get_package_info(package_name)
                        
                        # Store in database using transaction
                        async with transaction() as session:
                            # Check if package already exists in database
                            result = await session.execute(
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
                                session.add(package_stats)
                                await session.refresh(existing_package)
                                package = existing_package
                            else:
                                # Create new package
                                new_package = Package(
                                    name=package_name,
                                    description=description,
                                    discovery_date=datetime.now(),
                                    priority=monthly_downloads,  # Use downloads as initial priority
                                )
                                session.add(new_package)
                                await session.flush()  # To get the ID
                                
                                # Add stats
                                package_stats = PackageStats(
                                    package_id=new_package.id,
                                    monthly_downloads=monthly_downloads,
                                    recorded_at=datetime.now()
                                )
                                session.add(package_stats)
                                await session.refresh(new_package)
                                package = new_package
                        
                        pbar.update(1)
                        return package
                    except Exception as e:
                        console.print(f"[red]Error processing {package_name}: {str(e)}[/red]")
                        pbar.update(1)
                        return None
            
            # Process packages concurrently
            tasks = [process_package(name) for name in top_packages]
            results = await asyncio.gather(*tasks)
            
            # Filter out None results
            stored_packages = [p for p in results if p is not None]
        
        # Retry failed packages if requested
        if retry_failed and self.failed_stats_packages:
            await asyncio.sleep(5)  # Give the API a small break before retrying
            await self.retry_failed_packages()
        
        console.print(f"[green]Discovered and stored {len(stored_packages)} packages[/green]")
        if self.failed_stats_packages:
            console.print(f"[yellow]Failed to get stats for {len(self.failed_stats_packages)} packages[/yellow]")
        
        return stored_packages

    async def get_next_packages_to_process(self, limit: int = 10) -> List[Package]:
        """
        Get the next batch of packages to process based on priority and processing status.
        
        Args:
            limit: Maximum number of packages to retrieve
            
        Returns:
            List of Package objects to process next
        """
        from llm_docs.storage.database import transaction
        
        # Get packages that haven't been processed yet, ordered by priority
        async with transaction() as session:
            result = await session.execute(
                select(Package)
                .where(Package.original_doc_path.is_(None))  # Fixed syntax: is_(None) instead of is None
                .order_by(Package.priority.desc())
                .limit(limit)
            )
            packages = result.scalars().all()
        
        return packages

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def discover_and_store_packages_batch(self, 
                                    package_names: List[str], 
                                    concurrency: int = 5, 
                                    retry_failed: bool = True,
                                    progress_callback = None) -> List[Package]:
        """
        Discover and store a specific batch of packages by name with progress reporting.
        
        Args:
            package_names: List of package names to discover and store
            concurrency: Maximum number of concurrent API calls
            retry_failed: Whether to retry failed stats retrievals
            progress_callback: Optional callback function to report progress
            
        Returns:
            List of Package objects stored in the database
        """
        stored_packages = []
        
        # Use a semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_package(package_name: str) -> Optional[Package]:
            async with semaphore:
                try:
                    # Get package information
                    monthly_downloads, description = await self.get_package_info(package_name)
                    
                    # Store in database using transaction
                    async with transaction() as session:
                        # Check if package already exists in database
                        result = await session.execute(
                            select(Package).where(Package.name == package_name)
                        )
                        existing_package = result.scalar_one_or_none()
                        
                        if existing_package:
                            # Update existing package with new stats
                            package_stats = PackageStats(
                                package_id=existing_package.id,
                                monthly_downloads=monthly_downloads,
                                recorded_at=datetime.now()
                            )
                            session.add(package_stats)
                            
                            # Update priority if downloads are higher
                            if monthly_downloads > existing_package.priority:
                                existing_package.priority = monthly_downloads
                                session.add(existing_package)
                                
                            await session.commit()
                            await session.refresh(existing_package)
                            return existing_package
                        else:
                            # Create new package
                            new_package = Package(
                                name=package_name,
                                description=description,
                                discovery_date=datetime.now(),
                                priority=monthly_downloads  # Use downloads as initial priority
                            )
                            session.add(new_package)
                            await session.commit()
                            await session.refresh(new_package)
                            
                            # Add stats
                            package_stats = PackageStats(
                                package_id=new_package.id,
                                monthly_downloads=monthly_downloads,
                                recorded_at=datetime.now()
                            )
                            session.add(package_stats)
                            await session.commit()
                            
                            return new_package
                    
                except Exception as e:
                    console.print(f"[red]Error processing {package_name}: {str(e)}[/red]")
                    return None
        
        # Process packages in batches to enable progress reporting
        completed = 0
        total = len(package_names)
        batch_size = min(10, max(1, concurrency * 2))  # Process in reasonable batches
        
        for i in range(0, total, batch_size):
            batch = package_names[i:i+batch_size]
            tasks = [process_package(name) for name in batch]
            results = await asyncio.gather(*tasks)
            
            valid_results = [r for r in results if r is not None]
            stored_packages.extend(valid_results)
            
            # Update progress if callback provided
            completed += len(batch)
            if progress_callback:
                # Check if this is a coroutine function that needs awaiting
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(completed)
                else:
                    # Regular function, just call it
                    progress_callback(completed)
        
        # Retry failed packages if requested
        if retry_failed and self.failed_stats_packages and len(self.failed_stats_packages) > 0:
            console.print(f"[yellow]Retrying {len(self.failed_stats_packages)} failed packages...[/yellow]")
            await asyncio.sleep(5)  # Give the API a small break before retrying
            retry_count = await self.retry_failed_packages()
            console.print(f"[green]Successfully retrieved stats for {retry_count} retry packages[/green]")
        
        console.print(f"[green]Discovered and stored {len(stored_packages)} packages[/green]")
        if self.failed_stats_packages and len(self.failed_stats_packages) > 0:
            console.print(f"[yellow]Failed to get stats for {len(self.failed_stats_packages)} packages[/yellow]")
        
        return stored_packages