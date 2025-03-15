"""
Module for discovering and prioritizing Python packages from PyPI based on popularity.
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
from termcolor import colored
from sqlmodel import Session, select
from browser_use import Browser

from llm_docs.storage.models import Package, PackageStats

class PackageDiscovery:
    """Discovers and prioritizes Python packages from PyPI."""
    
    def __init__(self, db_session: Session):
        """Initialize the package discovery module."""
        self.db_session = db_session
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_top_packages(self, limit: int = 1000) -> List[str]:
        """
        Get the top packages from PyPI stats.
        
        Args:
            limit: Maximum number of packages to retrieve
            
        Returns:
            List of package names ordered by popularity
        """
        print(colored(f"Fetching top {limit} packages from PyPI stats...", "cyan"))
        
        # In a real implementation, we would need to scrape this data
        # as PyPI Stats doesn't directly provide a "top packages" API
        # This is a placeholder for the actual implementation
        async with Browser() as browser:
            page = await browser.new_page()
            await page.goto("https://pypistats.org/top-packages")
            
            # Extract package names from the top packages page
            packages = await page.evaluate("""
                () => {
                    const packageElements = document.querySelectorAll('.package-name');
                    return Array.from(packageElements).map(el => el.textContent.trim());
                }
            """)
            
            return packages[:limit]
    
    async def get_package_stats(self, package_name: str) -> Dict[str, Any]:
        """
        Get download statistics for a specific package.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Dictionary containing download statistics
        """
        url = f"https://pypistats.org/api/packages/{package_name}/recent"
        response = await self.client.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(colored(f"Failed to get stats for {package_name}: {response.status_code}", "yellow"))
            return {"data": {"last_month": 0}}

    async def discover_and_store_packages(self, limit: int = 1000) -> List[Package]:
        """
        Discover top packages and store them in the database.
        
        Args:
            limit: Maximum number of packages to retrieve
            
        Returns:
            List of Package objects stored in the database
        """
        top_packages = await self.get_top_packages(limit)
        
        stored_packages = []
        for package_name in tqdm(top_packages, desc="Fetching package stats"):
            # Check if package already exists in database
            existing_package = self.db_session.exec(
                select(Package).where(Package.name == package_name)
            ).first()
            
            if existing_package:
                # Update existing package
                stats = await self.get_package_stats(package_name)
                monthly_downloads = stats["data"]["last_month"]
                
                # Update stats
                package_stats = PackageStats(
                    package_id=existing_package.id,
                    monthly_downloads=monthly_downloads,
                    recorded_at=datetime.now()
                )
                self.db_session.add(package_stats)
                
                stored_packages.append(existing_package)
            else:
                # Create new package
                stats = await self.get_package_stats(package_name)
                monthly_downloads = stats["data"]["last_month"]
                
                new_package = Package(
                    name=package_name,
                    discovery_date=datetime.now(),
                    priority=monthly_downloads,  # Use downloads as initial priority
                )
                self.db_session.add(new_package)
                self.db_session.flush()  # To get the ID
                
                # Add stats
                package_stats = PackageStats(
                    package_id=new_package.id,
                    monthly_downloads=monthly_downloads,
                    recorded_at=datetime.now()
                )
                self.db_session.add(package_stats)
                
                stored_packages.append(new_package)
                
        self.db_session.commit()
        return stored_packages
        
    async def get_next_packages_to_process(self, limit: int = 10) -> List[Package]:
        """
        Get the next batch of packages to process based on priority and processing status.
        
        Args:
            limit: Maximum number of packages to retrieve
            
        Returns:
            List of Package objects to process next
        """
        # Get packages that haven't been processed yet, ordered by priority
        packages = self.db_session.exec(
            select(Package)
            .where(Package.original_doc_path == None)
            .order_by(Package.priority.desc())
            .limit(limit)
        ).all()
        
        return packages

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
