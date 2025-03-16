"""
Module for discovering and prioritizing Python packages from PyPI based on popularity.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from browser_use import Browser
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from tqdm import tqdm

from llm_docs.storage.database import transaction
from llm_docs.storage.models import Package, PackageStats

# Initialize console for rich output
console = Console()

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
        
    async def get_top_packages_from_pypi(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get the top packages directly from PyPI stats.
        
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
        
        packages = []
        try:
            # Using browser-use without async context manager
            browser = Browser()
            try:
                # Initialize the browser
                await browser.get_session()
                # Get the current page
                page = await browser.get_current_page()
                
                # Configure page
                await page.set_viewport_size({"width": 1280, "height": 800})
                
                # Go to the PyPI stats page
                await page.goto("https://pypistats.org/top-packages/")
                
                # Wait for the table to load
                await page.wait_for_selector('table.table-hover')
                
                # Extract package data
                packages_data = await page.evaluate("""
                    () => {
                        const rows = Array.from(document.querySelectorAll('table.table-hover tbody tr'));
                        return rows.map(row => {
                            const cells = row.querySelectorAll('td');
                            if (cells.length >= 2) {
                                return {
                                    name: cells[0].textContent.trim(),
                                    downloads: parseInt(cells[1].textContent.replace(/[^0-9]/g, ''), 10)
                                };
                            }
                            return null;
                        }).filter(p => p !== null);
                    }
                """)
                
                if not packages_data:
                    # Try an alternative approach if we couldn't get data from the primary source
                    console.print("[yellow]Couldn't get data from pypistats.org, trying alternative source...[/yellow]")
                    
                    # Use GitHub's top PyPI packages list as a fallback
                    await page.goto("https://hugovk.github.io/top-pypi-packages/")
                    await page.wait_for_selector('table#top-packages')
                    
                    packages_data = await page.evaluate("""
                        () => {
                            const rows = Array.from(document.querySelectorAll('table#top-packages tbody tr'));
                            return rows.map(row => {
                                const cells = row.querySelectorAll('td');
                                if (cells.length >= 3) {
                                    return {
                                        name: cells[1].textContent.trim(),
                                        downloads: parseInt(cells[2].textContent.replace(/[^0-9]/g, ''), 10)
                                    };
                                }
                                return null;
                            }).filter(p => p !== null);
                        }
                    """)
                
                packages = packages_data
                
            finally:
                # Always close the browser properly
                await browser.close()
                
            # Cache the results
            if packages:
                with open(self.top_packages_cache, 'w') as f:
                    json.dump(packages, f)
                
            return packages[:limit]
            
        except Exception as e:
            console.print(f"[bold red]Error fetching top packages: {str(e)}[/bold red]")
            
            # If we have a cache file, use it even if it's old
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
                {"name": "requests", "downloads": 42000000},
                {"name": "matplotlib", "downloads": 40000000},
                {"name": "scipy", "downloads": 38000000},
                {"name": "beautifulsoup4", "downloads": 35000000},
                {"name": "pillow", "downloads": 33000000},
                {"name": "scikit-learn", "downloads": 30000000},
                {"name": "tensorflow", "downloads": 28000000},
                {"name": "django", "downloads": 25000000},
                # Add more fallback packages here...
            ][:limit]
    
    async def get_package_stats(self, package_name: str) -> Dict[str, Any]:
        """
        Get download statistics for a specific package.
        
        Args:
            package_name: Name of the package
            
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

        # Fetch from API
        try:
            url = f"https://pypistats.org/api/packages/{package_name}/recent"
            response = await self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache the results
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                    
                return data
            else:
                console.print(f"[yellow]Failed to get stats for {package_name}: {response.status_code}[/yellow]")
                return {"data": {"last_month": 0}}
                
        except Exception as e:
            console.print(f"[yellow]Error fetching stats for {package_name}: {str(e)}[/yellow]")
            return {"data": {"last_month": 0}}
        
    async def _store_package_db(self, package_name: str, monthly_downloads: int, description: Optional[str] = None, session=None) -> Package:
        """Store package in database using async operations."""
        from llm_docs.storage.database import transaction
        
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
            url = f"https://pypi.org/pypi/{package_name}/json"
            response = await self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                description = data.get("info", {}).get("summary", "")
                
        except Exception as e:
            console.print(f"[yellow]Error fetching description for {package_name}: {str(e)}[/yellow]")
        
        return downloads, description

    async def discover_and_store_packages(self, limit: int = 1000) -> List[Package]:
        """
        Discover top packages and store them in the database.
        
        Args:
            limit: Maximum number of packages to retrieve
            
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
            # Use a semaphore to limit concurrent API calls
            semaphore = asyncio.Semaphore(10)
            
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
        
        console.print(f"[green]Discovered and stored {len(stored_packages)} packages[/green]")
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