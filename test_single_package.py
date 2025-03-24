#!/usr/bin/env python
"""
test_single_package.py

Test script for processing a single package with llm_docs with enhanced token tracking.
This script demonstrates the complete pipeline from discovery to distillation
for a single package, with detailed logging and token usage monitoring.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from llm_docs.distillation import DocumentationDistiller
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.database import init_db, transaction
from llm_docs.storage.models import Package, PackageStatus

# Import the token tracking adapter
from llm_docs.utils.token_tracking import print_usage_report, update_aisuite_response_tracking

# Set up rich console for pretty output
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("llm_docs_test")

# Configuration
PACKAGE_NAME = "httpx"  # The package to process
INIT_DB = True          # Whether to initialize the database
FORCE_DISCOVERY = True  # Force discovery even if package exists
FORCE_EXTRACTION = True # Force extraction even if docs already exist
FORCE_DISTILLATION = True # Force distillation even if already done
VERBOSE = True          # Enable verbose logging

# Monkey patch the DocumentationDistiller.distill_chunk method to add token tracking
original_distill_chunk = DocumentationDistiller.distill_chunk

async def patched_distill_chunk(self, package_name, chunk_content, part_num, num_parts, 
                               package_description="", doc_type="general"):
    """Patched version of distill_chunk that adds token tracking."""
    # Call the original method
    response = await original_distill_chunk(self, package_name, chunk_content, part_num, 
                                          num_parts, package_description, doc_type)
    
    # Get token counts (for demo purposes - in production you'd get this from the response)
    estimated_prompt_tokens = self.estimate_tokens(chunk_content)
    estimated_completion_tokens = self.estimate_tokens(response)
    
    # Track token usage
    update_aisuite_response_tracking(
        response=None,  # No actual response object to pass
        task=f"distill_{doc_type}_part{part_num}",
        prompt_tokens=estimated_prompt_tokens,
        model=self.model
    )
    
    # Simulate output token tracking
    from llm_docs.utils.token_tracking import tracker

    tracker.update(0, estimated_completion_tokens, f"distill_{doc_type}_completion_part{part_num}")
    
    return response

# Apply the monkey patch
DocumentationDistiller.distill_chunk = patched_distill_chunk

async def discover_package(package_name):
    """Discover a single package and return it."""
    logger.info(f"🔍 Discovering package: {package_name}")
    
    async with transaction() as session:
        # First check if the package already exists in the database
        # and we're not forcing rediscovery
        if not FORCE_DISCOVERY:
            from sqlalchemy.future import select
            result = await session.execute(
                select(Package).where(Package.name == package_name)
            )
            existing_package = result.scalar_one_or_none()
            if existing_package:
                logger.info(f"📦 Package {package_name} already exists in database (ID: {existing_package.id})")
                return existing_package
        
        # Initialize the package discovery
        discovery = PackageDiscovery(session)
        
        try:
            # Get package info
            logger.info(f"📊 Fetching package info for {package_name}...")
            
            # Start token tracking for API call
            from llm_docs.utils.token_tracking import tracker

            tracker.update(500, 100, "discover_package_info")  # Estimate tokens for API call
            
            monthly_downloads, description = await discovery.get_package_info(package_name)
            logger.info(f"📈 {package_name} has {monthly_downloads} monthly downloads")
            logger.info(f"📝 Description: {description}")
            
            # Store in database
            logger.info(f"💾 Storing {package_name} in database...")
            package = await discovery._store_package_db(
                package_name, 
                monthly_downloads,
                description,
                session
            )
            
            logger.info(f"✅ Package stored successfully (ID: {package.id})")
            return package
        finally:
            await discovery.close()


async def extract_documentation(package):
    """Extract documentation for a package."""
    logger.info(f"📚 Extracting documentation for {package.name}")
    
    # Skip if already extracted and not forcing
    if not FORCE_EXTRACTION and package.original_doc_path:
        logger.info(f"📄 Documentation already extracted: {package.original_doc_path}")
        return package.original_doc_path
    
    extractor = DocumentationExtractor()
    
    try:
        # Find documentation URL
        logger.info(f"🔗 Finding documentation URL for {package.name}...")
        
        # Track token usage for any API calls made during URL discovery
        from llm_docs.utils.token_tracking import tracker

        tracker.update(200, 50, "find_documentation_url")  # Estimate tokens for API call
        
        doc_url = await extractor.find_documentation_url(package.name)
        
        if not doc_url:
            logger.error(f"❌ Could not find documentation URL for {package.name}")
            return None
            
        logger.info(f"🌐 Documentation URL: {doc_url}")
        
        # Map documentation site
        logger.info("🗺️ Mapping documentation site structure...")
        doc_pages = await extractor.map_documentation_site(doc_url)
        logger.info(f"📋 Found {len(doc_pages)} documentation pages")
        
        # Track token usage for browser automation
        tracker.update(100, 30, "map_documentation_site")  # Minimal token usage for browser automation
        
        if VERBOSE:
            for i, page in enumerate(doc_pages[:5]):  # Show first 5 pages
                logger.info(f"  - Page {i+1}: {page['title']} ({page['url']})")
            if len(doc_pages) > 5:
                logger.info(f"  - ... and {len(doc_pages) - 5} more pages")
        
        # Process documentation
        logger.info("🔄 Processing complete documentation...")
        doc_path = await extractor.process_package_documentation(package)
        
        if not doc_path:
            logger.error(f"❌ Failed to process documentation for {package.name}")
            return None
            
        logger.info(f"📄 Documentation saved to: {doc_path}")
        
        # Update package in database
        async with transaction() as session:
            package.original_doc_path = doc_path
            package.docs_extraction_date = datetime.now()
            package.status = PackageStatus.DOCS_EXTRACTED
            session.add(package)
            await session.commit()
            await session.refresh(package)
            
        logger.info("✅ Documentation extraction completed successfully")
        return doc_path
    finally:
        await extractor.close()


async def distill_documentation(package, doc_path):
    """Distill documentation using LLM."""
    logger.info(f"🧠 Distilling documentation for {package.name}")
    
    # Skip if already distilled and not forcing
    if not FORCE_DISTILLATION and package.distilled_doc_path:
        logger.info(f"📝 Documentation already distilled: {package.distilled_doc_path}")
        return package.distilled_doc_path
    
    # Initialize distiller
    distiller = DocumentationDistiller()
    
    logger.info("🔄 Starting distillation process...")
    logger.info(f"📊 Using model: {distiller.model}")
    
    # Start distillation
    distilled_path = await distiller.distill_documentation(package, doc_path)
    
    if not distilled_path:
        logger.error(f"❌ Distillation failed for {package.name}")
        return None
        
    logger.info(f"📄 Distilled documentation saved to: {distilled_path}")
    
    # Update package in database
    async with transaction() as session:
        package.distilled_doc_path = distilled_path
        package.status = PackageStatus.DISTILLATION_COMPLETED
        package.distillation_end_date = datetime.now()
        session.add(package)
        await session.commit()
        await session.refresh(package)
        
    logger.info("✅ Distillation completed successfully")
    return distilled_path


async def process_package(package_name):
    """Process a single package through the complete pipeline."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        main_task = progress.add_task(f"Processing {package_name}", total=3)
        
        # Initialize database if needed
        if INIT_DB:
            progress.update(main_task, description="Initializing database...")
            await init_db()
        
        # Stage 1: Discover package
        progress.update(main_task, description=f"Discovering {package_name}...")
        package = await discover_package(package_name)
        if not package:
            progress.update(main_task, description=f"❌ Failed to discover {package_name}")
            return
        progress.update(main_task, advance=1)
        
        # Stage 2: Extract documentation
        progress.update(main_task, description=f"Extracting documentation for {package_name}...")
        doc_path = await extract_documentation(package)
        if not doc_path:
            progress.update(main_task, description=f"❌ Failed to extract documentation for {package_name}")
            return
        progress.update(main_task, advance=1)
        
        # Stage 3: Distill documentation
        progress.update(main_task, description=f"Distilling documentation for {package_name}...")
        distilled_path = await distill_documentation(package, doc_path)
        if not distilled_path:
            progress.update(main_task, description=f"❌ Failed to distill documentation for {package_name}")
            return
        progress.update(main_task, advance=1)
        
        # Complete
        progress.update(main_task, description=f"✅ {package_name} processed successfully")
    
    # Final report
    console.print("\n[bold green]Processing Complete![/bold green]")
    console.print(f"Package: [bold]{package_name}[/bold] (ID: {package.id})")
    console.print(f"Status: [bold]{package.status}[/bold]")
    console.print(f"Original Documentation: {package.original_doc_path}")
    console.print(f"Distilled Documentation: {package.distilled_doc_path}")
    
    # Show stats
    if package.original_doc_path and package.distilled_doc_path:
        orig_size = Path(package.original_doc_path).stat().st_size
        dist_size = Path(package.distilled_doc_path).stat().st_size
        reduction = (1 - dist_size / orig_size) * 100
        
        console.print("\n[bold]Documentation Size:[/bold]")
        console.print(f"Original: {orig_size / 1024:.1f} KB")
        console.print(f"Distilled: {dist_size / 1024:.1f} KB")
        console.print(f"Reduction: [bold]{reduction:.1f}%[/bold]")


async def main():
    """Main function to run the test script."""
    console.print("[bold]LLM-Docs Test Script[/bold]")
    console.print(f"Processing package: [bold]{PACKAGE_NAME}[/bold]")
    
    try:
        await process_package(PACKAGE_NAME)
        
        # Print token usage statistics
        console.print("\n[bold]LLM API Usage Statistics:[/bold]")
        print_usage_report()
        
    except Exception as e:
        logger.exception(f"Error processing {PACKAGE_NAME}: {e}")
        console.print(f"[bold red]Error processing {PACKAGE_NAME}:[/bold red] {str(e)}")


if __name__ == "__main__":
    # Allow command-line override of the package name
    if len(sys.argv) > 1:
        PACKAGE_NAME = sys.argv[1]
    
    # Run the main function
    asyncio.run(main())