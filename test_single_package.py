#!/usr/bin/env python
"""
test_single_package.py

Test script for processing a single package with llm_docs.
This script demonstrates the complete pipeline from discovery to distillation
for a single package, properly integrated with:

1) The new Agent caching system
2) The vision quality control stuff to reduce token usage
3) The token tracking system
4) The logging system
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sqlalchemy.future import select

from llm_docs.distillation import DocumentationDistiller
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.database import init_db, transaction
from llm_docs.storage.models import DistillationJob, DistillationJobStatus, Package, PackageStatus
from llm_docs.utils.browser.vision_config import low_cost_vision_config
from llm_docs.utils.browser.vision_integration import (
    create_browser_with_vision_config,
    enable_vision_config,
    patch_documentation_extractor_with_vision_config,
    run_agent_with_vision_config,
)
from llm_docs.utils.cache import enable_caching
from llm_docs.utils.llm_docs_logger import logger
from llm_docs.utils.token_tracking import (
    get_usage_summary,
    print_usage_report,
    tracker,
    update_aisuite_response_tracking,
)

# Set up rich console for pretty output
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)

# Configuration
PACKAGE_NAME = "httpx"  # The package to process
INIT_DB = True          # Whether to initialize the database
FORCE_DISCOVERY = True  # Force discovery even if package exists
FORCE_EXTRACTION = True # Force extraction even if docs already exist
FORCE_DISTILLATION = True # Force distillation even if already done
VERBOSE = True          # Enable verbose logging

# Enable caching to improve performance and reduce API calls
enable_caching()

# Enable vision configuration with low cost settings to reduce token usage
enable_vision_config(low_cost_vision_config)

# Patch DocumentationExtractor to use vision configuration
patch_documentation_extractor_with_vision_config()


async def discover_package(package_name):
    """Discover a single package and return it."""
    logger.info(f"🔍 Discovering package: {package_name}")
    
    async with transaction() as session:
        # First check if the package already exists in the database
        # and we're not forcing rediscovery
        if not FORCE_DISCOVERY:
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
            
            # The token tracking is handled internally by the tracker module
            # for API calls that use LLMs
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
    """Extract documentation for a package with integrated vision quality control."""
    logger.info(f"📚 Extracting documentation for {package.name}")
    
    # Skip if already extracted and not forcing
    if not FORCE_EXTRACTION and package.original_doc_path:
        logger.info(f"📄 Documentation already extracted: {package.original_doc_path}")
        return package.original_doc_path
    
    # Initialize the extractor - token tracking and vision config happen automatically
    # because we've patched the DocumentationExtractor with patch_documentation_extractor_with_vision_config
    extractor = DocumentationExtractor(
        concurrency_limit=3,  # Limit concurrent browser operations
        use_vision=True       # Enable vision capabilities
    )
    
    try:
        # Find documentation URL
        logger.info(f"🔗 Finding documentation URL for {package.name}...")
        
        # Track token usage before URL discovery
        initial_token_count = tracker.get_total_tokens()
        
        doc_url = await extractor.find_documentation_url(package.name)
        
        # Calculate token usage for URL discovery
        url_discovery_tokens = tracker.get_total_tokens() - initial_token_count
        logger.info(f"🔢 URL discovery used {url_discovery_tokens} tokens")
        
        if not doc_url:
            logger.error(f"❌ Could not find documentation URL for {package.name}")
            return None
            
        logger.info(f"🌐 Documentation URL: {doc_url}")
        
        # Analyze the documentation site structure with run_agent_with_vision_config
        # This is useful for complex documentation sites
        if VERBOSE and doc_url:
            logger.info("🔍 Running detailed analysis of documentation structure...")

            # Create a browser with our vision config
            browser = await create_browser_with_vision_config(low_cost_vision_config)
            
            try:
                # Run an agent to analyze the documentation structure
                result = await run_agent_with_vision_config(
                    task=f"Analyze the structure of {package.name} documentation site. Identify main sections, navigation structure, and content organization. Do not extract content yet.",
                    llm=ChatOpenAI(model="gpt-4o"),
                    vision_config=low_cost_vision_config,
                    browser=browser,
                    initial_actions=[{'open_tab': {'url': doc_url}}]
                )
                
                # Get analysis result
                site_analysis = result.final_result()
                if site_analysis:
                    logger.info(f"📊 Documentation structure analysis: {site_analysis[:200]}...")
                    
                    # Track token usage with update_aisuite_response_tracking
                    for action in result.model_actions():
                        if hasattr(action, 'response'):
                            update_aisuite_response_tracking(
                                response=action.response,
                                task=f"doc_structure_analysis_{package.name}",
                                model="gpt-4o"
                            )
            finally:
                await browser.close()
        
        # Map documentation site
        # Token tracking happens automatically through the patched methods
        logger.info("🗺️ Mapping documentation site structure...")
        token_count_before_mapping = tracker.get_total_tokens()
        
        doc_pages = await extractor.map_documentation_site(doc_url)
        
        # Calculate tokens used for site mapping
        mapping_tokens = tracker.get_total_tokens() - token_count_before_mapping
        logger.info(f"🔢 Documentation mapping used {mapping_tokens} tokens")
        
        logger.info(f"📋 Found {len(doc_pages)} documentation pages")
        
        if VERBOSE:
            for i, page in enumerate(doc_pages[:5]):  # Show first 5 pages
                logger.info(f"  - Page {i+1}: {page['title']} ({page['url']})")
            if len(doc_pages) > 5:
                logger.info(f"  - ... and {len(doc_pages) - 5} more pages")
        
        # Process documentation
        logger.info("🔄 Processing complete documentation...")
        token_count_before_processing = tracker.get_total_tokens()
        
        doc_path = await extractor.process_package_documentation(package, force_extract=FORCE_EXTRACTION)
        
        # Calculate tokens used for processing
        processing_tokens = tracker.get_total_tokens() - token_count_before_processing
        logger.info(f"🔢 Documentation processing used {processing_tokens} tokens")
        
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
        
        # Log total token usage for the entire extraction process
        total_tokens = url_discovery_tokens + mapping_tokens + processing_tokens
        logger.info(f"💰 Total token usage for {package.name} documentation extraction: {total_tokens}")
        
        return doc_path
    finally:
        await extractor.close()

async def distill_documentation(package, doc_path):
    """Distill documentation using LLM with token tracking."""
    logger.info(f"🧠 Distilling documentation for {package.name}")
    
    # Skip if already distilled and not forcing
    if not FORCE_DISTILLATION and package.distilled_doc_path:
        logger.info(f"📝 Documentation already distilled: {package.distilled_doc_path}")
        return package.distilled_doc_path
    
    # Create distillation job
    async with transaction() as session:
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
    
    # Initialize distiller - token tracking is handled by update_aisuite_response_tracking
    # through monkey-patching in the token_tracking module
    distiller = DocumentationDistiller()
    
    logger.info("🔄 Starting distillation process...")
    logger.info(f"📊 Using model: {distiller.model}")
    
    # Perform distillation - token tracking happens automatically
    distilled_path = await distiller.distill_documentation(package, doc_path)
    
    if not distilled_path:
        logger.error(f"❌ Distillation failed for {package.name}")
        
        # Update job
        async with transaction() as session:
            result = await session.execute(
                select(DistillationJob).where(DistillationJob.id == job.id)
            )
            db_job = result.scalar_one()
            
            db_job.status = DistillationJobStatus.FAILED
            db_job.error_message = "Distillation failed"
            db_job.completed_at = datetime.now()
            session.add(db_job)
            
            # Update package
            package.status = PackageStatus.DISTILLATION_FAILED
            session.add(package)
            await session.commit()
            
        return None
        
    logger.info(f"📄 Distilled documentation saved to: {distilled_path}")
    
    # Update job and package in database
    async with transaction() as session:
        result = await session.execute(
            select(DistillationJob).where(DistillationJob.id == job.id)
        )
        db_job = result.scalar_one()
        
        db_job.status = DistillationJobStatus.COMPLETED
        db_job.completed_at = datetime.now()
        db_job.output_file_path = distilled_path
        db_job.chunks_processed = db_job.num_chunks
        session.add(db_job)
        
        # Update package
        package.distilled_doc_path = distilled_path
        package.status = PackageStatus.DISTILLATION_COMPLETED
        package.distillation_end_date = datetime.now()
        session.add(package)
        await session.commit()
        
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
    
    # Print token usage report from the system tracker
    console.print("\n[bold]Token Usage Report:[/bold]")
    print_usage_report()
    
    # Get detailed usage summary
    usage_summary = get_usage_summary()
    
    # Display cost by category if available
    if "categories" in usage_summary:
        console.print("\n[bold]Token Usage by Category:[/bold]")
        for category, data in usage_summary["categories"].items():
            if data.get("calls", 0) > 0:
                console.print(f"[bold]{category}[/bold]:")
                console.print(f"  - Calls: {data.get('calls', 0)}")
                console.print(f"  - Tokens: {data.get('tokens', {}).get('input', 0) + data.get('tokens', {}).get('output', 0):,}")
                console.print(f"  - Cost: ${data.get('cost_usd', 0):.6f}")


async def main():
    """Main function to run the test script."""
    console.print("[bold]LLM-Docs Test Script[/bold]")
    console.print(f"Processing package: [bold]{PACKAGE_NAME}[/bold]")
    
    try:
        await process_package(PACKAGE_NAME)
    except Exception as e:
        logger.exception(f"Error processing {PACKAGE_NAME}: {e}")
        console.print(f"[bold red]Error processing {PACKAGE_NAME}:[/bold red] {str(e)}")


if __name__ == "__main__":
    # Allow command-line override of the package name
    if len(sys.argv) > 1:
        PACKAGE_NAME = sys.argv[1]
    
    # Run the main function
    asyncio.run(main())