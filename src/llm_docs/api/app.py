"""
FastAPI application for the llm_docs service.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from llm_docs.distillation import DocumentationDistiller
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.database import get_async_session, transaction
from llm_docs.storage.models import (
    DistillationJob,
    DistillationJobStatus,
    Package,
    PackageStatus,
)

# Initialize console
console = Console()

# Initialize Redis client for distributed locking
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url, decode_responses=True)

# FastAPI models for requests and responses
class PackageCreate(BaseModel):
    name: str = Field(..., pattern=r'^[a-z0-9\-_]+$')
    priority: Optional[int] = Field(None, ge=0)  # Must be non-negative
    description: Optional[str] = None


class PackageResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    status: str
    docs_url: Optional[str] = None
    original_doc_path: Optional[str] = None
    distilled_doc_path: Optional[str] = None
    discovery_date: datetime
    docs_extraction_date: Optional[datetime] = None
    distillation_start_date: Optional[datetime] = None
    distillation_end_date: Optional[datetime] = None


class DistillationJobResponse(BaseModel):
    id: int
    package_id: int
    package_name: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    num_chunks: int
    chunks_processed: int
    input_file_path: Optional[str] = None
    output_file_path: Optional[str] = None
    error_message: Optional[str] = None


class StatsResponse(BaseModel):
    total_packages: int
    packages_with_docs: int
    packages_distilled: int
    recent_distillations: List[Dict[str, Any]]
    popular_packages: List[Dict[str, Any]]


# Create the FastAPI app
app = FastAPI(
    title="llm_docs API",
    description="API for accessing and managing LLM-optimized documentation",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount static files for the docs
app.mount("/docs/original", StaticFiles(directory="docs"), name="original_docs")
app.mount("/docs/distilled", StaticFiles(directory="distilled_docs"), name="distilled_docs")

async def process_package(package_id: int):
    """Background task to extract and distill documentation for a package."""
    # Create lock key for this package
    lock_key = f"processing:package:{package_id}"
    
    # Create a Redis client for this task
    redis_instance = redis.from_url(redis_url, decode_responses=True)
    
    try:
        # Try to acquire lock
        lock_acquired = await redis_instance.set(lock_key, "1", nx=True, ex=3600)  # 1 hour timeout
        
        if not lock_acquired:
            console.print(f"[yellow]Package {package_id} is already being processed by another worker[/yellow]")
            return
        
        try:
            async with transaction() as session:
                try:
                    # Get the package
                    result = await session.execute(select(Package).where(Package.id == package_id))
                    package = result.scalar_one()
                    
                    # Extract documentation
                    extractor = DocumentationExtractor()
                    doc_path = await extractor.process_package_documentation(package)
                    
                    if doc_path:
                        # Update package with documentation path
                        package.original_doc_path = doc_path
                        package.docs_extraction_date = datetime.now()
                        package.status = PackageStatus.DOCS_EXTRACTED
                        session.add(package)
                        await session.commit()
                        
                        # Create distillation job
                        job = DistillationJob(
                            package_id=package.id,
                            status=DistillationJobStatus.PENDING,
                            input_file_path=doc_path
                        )
                        session.add(job)
                        await session.commit()
                        
                        # Start distillation
                        package.status = PackageStatus.DISTILLATION_IN_PROGRESS
                        package.distillation_start_date = datetime.now()
                        session.add(package)
                        
                        job.status = DistillationJobStatus.IN_PROGRESS
                        job.started_at = datetime.now()
                        session.add(job)
                        await session.commit()
                        
                        # Run distillation
                        distiller = DocumentationDistiller()
                        distilled_path = await distiller.distill_documentation(package, doc_path)
                        
                        # Update package and job
                        if distilled_path:
                            package.distilled_doc_path = distilled_path
                            package.status = PackageStatus.DISTILLATION_COMPLETED
                            package.distillation_end_date = datetime.now()
                            
                            job.status = DistillationJobStatus.COMPLETED
                            job.completed_at = datetime.now()
                            job.output_file_path = distilled_path
                            job.chunks_processed = job.num_chunks
                        else:
                            package.status = PackageStatus.DISTILLATION_FAILED
                            
                            job.status = DistillationJobStatus.FAILED
                            job.error_message = "Distillation failed"
                            job.completed_at = datetime.now()
                            
                        session.add(package)
                        session.add(job)
                        await session.commit()
                        
                except Exception as e:
                    # Log the exception
                    console.print(f"[red]Error processing package {package_id}: {e}[/red]")
                    
                    # Update package status
                    try:
                        result = await session.execute(select(Package).where(Package.id == package_id))
                        package = result.scalar_one_or_none()
                        
                        if package:
                            # Only update if not already in a final state
                            if package.status not in [PackageStatus.DISTILLATION_COMPLETED, PackageStatus.DISTILLATION_FAILED]:
                                package.status = PackageStatus.DISTILLATION_FAILED
                                session.add(package)
                            
                            # Update job if exists and not in final state
                            job_result = await session.execute(
                                select(DistillationJob)
                                .where(DistillationJob.package_id == package_id)
                                .order_by(DistillationJob.id.desc())
                            )
                            job = job_result.scalar_first()
                            
                            if job and job.status not in [DistillationJobStatus.COMPLETED, DistillationJobStatus.FAILED]:
                                job.status = DistillationJobStatus.FAILED
                                job.error_message = str(e)
                                job.completed_at = datetime.now()
                                session.add(job)
                                
                            await session.commit()
                    except Exception as inner_error:
                        console.print(f"[bold red]Failed to update package status for {package_id}: {inner_error}[/bold red]")
        except Exception as outer_error:
            console.print(f"[bold red]Unhandled error in process_package for {package_id}: {outer_error}[/bold red]")
    finally:
        # Always release the lock and close the Redis client, even if processing fails
        try:
            await redis_instance.delete(lock_key)
        finally:
            await redis_instance.close()

# API routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page."""
    landing_page = Path("static/landing-page.html")
    if landing_page.exists():
        return FileResponse("static/landing-page.html")
    else:
        # Fallback if the file doesn't exist
        return HTMLResponse("<html><body><h1>LLM-Docs</h1><p>Welcome to LLM-Docs API</p></body></html>")

@app.get("/stats", response_model=StatsResponse)
async def get_stats(session: AsyncSession = Depends(get_async_session)):
    """Get basic statistics and info about the system."""
    # Count packages
    result = await session.execute(select(Package))
    total_packages = len(result.scalars().all())
    
    # Count packages with docs
    result = await session.execute(
        select(Package).where(Package.original_doc_path.isnot(None))
    )
    packages_with_docs = len(result.scalars().all())
    
    # Count distilled packages
    result = await session.execute(
        select(Package).where(Package.distilled_doc_path.isnot(None))
    )
    packages_distilled = len(result.scalars().all())
    
    # Get recent distillations
    result = await session.execute(
        select(Package)
        .where(Package.status == PackageStatus.DISTILLATION_COMPLETED)
        .order_by(Package.distillation_end_date.desc())
        .limit(5)
    )
    recent_distillations = result.scalars().all()
    
    recent = []
    for pkg in recent_distillations:
        recent.append({
            "id": pkg.id,
            "name": pkg.name,
            "distillation_date": pkg.distillation_end_date
        })
    
    # Get popular packages
    result = await session.execute(
        select(Package)
        .order_by(Package.priority.desc())
        .limit(10)
    )
    popular_packages = result.scalars().all()
    
    popular = []
    for pkg in popular_packages:
        popular.append({
            "id": pkg.id,
            "name": pkg.name,
            "priority": pkg.priority,
            "status": pkg.status
        })
    
    return StatsResponse(
        total_packages=total_packages,
        packages_with_docs=packages_with_docs,
        packages_distilled=packages_distilled,
        recent_distillations=recent,
        popular_packages=popular
    )


@app.get("/packages", response_model=List[PackageResponse])
async def list_packages(
    status: Optional[str] = None, 
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_async_session)
):
    """List packages with optional filtering by status."""
    query = select(Package)
    
    if status:
        query = query.where(Package.status == status)
        
    query = query.order_by(Package.priority.desc()).offset(offset).limit(limit)
    result = await session.execute(query)
    packages = result.scalars().all()
    
    return [
        PackageResponse(
            id=pkg.id,
            name=pkg.name,
            description=pkg.description,
            status=pkg.status,
            docs_url=pkg.docs_url,
            original_doc_path=pkg.original_doc_path,
            distilled_doc_path=pkg.distilled_doc_path,
            discovery_date=pkg.discovery_date,
            docs_extraction_date=pkg.docs_extraction_date,
            distillation_start_date=pkg.distillation_start_date,
            distillation_end_date=pkg.distillation_end_date
        )
        for pkg in packages
    ]


@app.get("/packages/{package_id}", response_model=PackageResponse)
async def get_package(
    package_id: int = Path(..., description="The ID of the package"),
    session: AsyncSession = Depends(get_async_session)
):
    """Get a specific package by ID."""
    result = await session.execute(select(Package).where(Package.id == package_id))
    package = result.scalar_one_or_none()
    
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
        
    return PackageResponse(
        id=package.id,
        name=package.name,
        description=package.description,
        status=package.status,
        docs_url=package.docs_url,
        original_doc_path=package.original_doc_path,
        distilled_doc_path=package.distilled_doc_path,
        discovery_date=package.discovery_date,
        docs_extraction_date=package.docs_extraction_date,
        distillation_start_date=package.distillation_start_date,
        distillation_end_date=package.distillation_end_date
    )


@app.post("/packages", response_model=PackageResponse)
async def create_package(
    package: PackageCreate,
    background_tasks: BackgroundTasks,
    process: bool = Query(False, description="Whether to process the package immediately"),
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new package."""
    # Check if package already exists
    result = await session.execute(select(Package).where(Package.name == package.name))
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=400, 
            detail=f"Package '{package.name}' already exists"
        )
        
    # Create the package
    new_package = Package(
        name=package.name,
        description=package.description,
        priority=package.priority or 0
    )
    
    session.add(new_package)
    await session.commit()
    await session.refresh(new_package)
    
    # Start processing if requested
    if process:
        background_tasks.add_task(process_package, new_package.id)
        
    return PackageResponse(
        id=new_package.id,
        name=new_package.name,
        description=new_package.description,
        status=new_package.status,
        docs_url=new_package.docs_url,
        original_doc_path=new_package.original_doc_path,
        distilled_doc_path=new_package.distilled_doc_path,
        discovery_date=new_package.discovery_date,
        docs_extraction_date=new_package.docs_extraction_date,
        distillation_start_date=new_package.distillation_start_date,
        distillation_end_date=new_package.distillation_end_date
    )


@app.post("/packages/{package_id}/process")
async def trigger_package_processing(
    package_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session)
):
    """Trigger processing for a specific package."""
    result = await session.execute(select(Package).where(Package.id == package_id))
    package = result.scalar_one_or_none()
    
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
        
    # Check if already processing
    if package.status == PackageStatus.DISTILLATION_IN_PROGRESS:
        raise HTTPException(
            status_code=400,
            detail=f"Package '{package.name}' is already being processed"
        )
        
    # Reset status if previously failed
    if package.status == PackageStatus.DISTILLATION_FAILED:
        package.status = PackageStatus.DISCOVERED
        session.add(package)
        await session.commit()
        
    # Start processing
    background_tasks.add_task(process_package, package.id)
    
    return {"message": f"Processing started for package '{package.name}'"}


@app.get("/packages/{package_id}/original")
async def get_original_documentation(
    package_id: int,
    session: AsyncSession = Depends(get_async_session)
):
    """Get the original documentation for a package."""
    result = await session.execute(select(Package).where(Package.id == package_id))
    package = result.scalar_one_or_none()
    
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
        
    if not package.original_doc_path:
        raise HTTPException(
            status_code=404,
            detail=f"No original documentation available for package '{package.name}'"
        )
    
    # Sanitize the path to prevent directory traversal attacks
    try:
        # Normalize path and resolve symlinks
        doc_path = Path(package.original_doc_path).resolve(strict=True)
        docs_dir = Path("docs").resolve()
        current_dir = Path.cwd().resolve()
        
        # Verify file exists
        if not doc_path.exists() or not doc_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Documentation file not found for package '{package.name}'"
            )
            
        # Check if the file is within allowed directories using os.path.commonpath for safer comparison
        try:
            # Use os.path for safer absolute path comparison
            import os
            doc_path_str = str(doc_path)
            docs_dir_str = str(docs_dir)
            current_dir_str = str(current_dir)
            
            # Check if doc_path is inside docs_dir or current_dir
            in_docs_dir = os.path.commonpath([doc_path_str, docs_dir_str]) == docs_dir_str
            in_current_dir = os.path.commonpath([doc_path_str, current_dir_str]) == current_dir_str
            
            if not (in_docs_dir or in_current_dir):
                raise HTTPException(
                    status_code=403,
                    detail="Access to this file is not allowed"
                )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=403,
                detail="Invalid file path"
            ) from None
        
        return FileResponse(
            str(doc_path),
            media_type="text/markdown",
            filename=f"{package.name}_original_docs.md"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Documentation file not found for package '{package.name}'"
        ) from None
    except (ValueError, OSError) as e:
        # Handle path resolution errors
        raise HTTPException(
            status_code=404, 
            detail=f"Documentation file error: {str(e)}"
        ) from e

@app.get("/packages/{package_id}/distilled")
async def get_distilled_documentation(
    package_id: int,
    session: AsyncSession = Depends(get_async_session)
):
    """Get the distilled documentation for a package."""
    result = await session.execute(select(Package).where(Package.id == package_id))
    package = result.scalar_one_or_none()
    
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
        
    if not package.distilled_doc_path:
        raise HTTPException(
            status_code=404,
            detail=f"No distilled documentation available for package '{package.name}'"
        )
    
    # Sanitize the path to prevent directory traversal attacks
    try:
        # Normalize path and resolve symlinks
        doc_path = Path(package.distilled_doc_path).resolve(strict=True)
        distilled_docs_dir = Path("distilled_docs").resolve()
        current_dir = Path.cwd().resolve()
        
        # Verify file exists
        if not doc_path.exists() or not doc_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Distilled documentation file not found for package '{package.name}'"
            )
            
        # Check if the file is within allowed directories
        # Use is_relative_to if Python 3.9+, otherwise use string-based check
        try:
            if hasattr(doc_path, 'is_relative_to'):  # Python 3.9+
                is_in_distilled_dir = doc_path.is_relative_to(distilled_docs_dir)
                is_in_current_dir = doc_path.is_relative_to(current_dir)
            else:
                # Fallback for earlier Python versions
                distilled_dir_str = str(distilled_docs_dir)
                current_dir_str = str(current_dir)
                doc_path_str = str(doc_path)
                is_in_distilled_dir = doc_path_str.startswith(distilled_dir_str)
                is_in_current_dir = doc_path_str.startswith(current_dir_str)
                
            if not (is_in_distilled_dir or is_in_current_dir):
                raise HTTPException(
                    status_code=403,
                    detail="Access to this file is not allowed"
                )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=403,
                detail="Invalid file path"
            ) from None
        
        return FileResponse(
            str(doc_path),
            media_type="text/markdown",
            filename=f"{package.name}_distilled_docs.md"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Distilled documentation file not found for package '{package.name}'"
        ) from None
    except (ValueError, OSError) as e:
        # Handle path resolution errors
        raise HTTPException(
            status_code=404, 
            detail=f"Documentation file error: {str(e)}"
        ) from e


@app.get("/jobs", response_model=List[DistillationJobResponse])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_async_session)
):
    """List distillation jobs with optional filtering by status."""
    query = select(DistillationJob)
    
    if status:
        try:
            enum_status = DistillationJobStatus(status)
            query = query.where(DistillationJob.status == enum_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}") from None
            
    query = query.order_by(DistillationJob.created_at.desc()).offset(offset).limit(limit)
    result = await session.execute(query)
    jobs = result.scalars().all()
    
    result_list = []
    for job in jobs:
        # Get package name
        pkg_result = await session.execute(select(Package).where(Package.id == job.package_id))
        package = pkg_result.scalar_one_or_none()
        package_name = package.name if package else "Unknown"
        
        result_list.append(
            DistillationJobResponse(
                id=job.id,
                package_id=job.package_id,
                package_name=package_name,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                num_chunks=job.num_chunks,
                chunks_processed=job.chunks_processed,
                input_file_path=job.input_file_path,
                output_file_path=job.output_file_path,
                error_message=job.error_message
            )
        )
        
    return result_list


@app.post("/discover")
async def discover_packages(
    background_tasks: BackgroundTasks,
    limit: int = Query(100, ge=1, le=1000),
    process_top: int = Query(0, ge=0, description="Number of top packages to process immediately"),
    session: AsyncSession = Depends(get_async_session)
):
    """Discover new packages from PyPI."""
    # Initialize the package discovery
    # We need to create a PackageDiscovery instance that can work with AsyncSession
    # This requires modifying the PackageDiscovery class to accept AsyncSession
    # For now, we'll assume this has been updated
    discovery = PackageDiscovery(session)
    
    # Run discovery
    packages = await discovery.discover_and_store_packages(limit)
    
    # Process top packages if requested
    if process_top > 0:
        top_packages = packages[:min(process_top, len(packages))]
        for package in top_packages:
            background_tasks.add_task(process_package, package.id)
            
    await discovery.close()
    
    return {
        "message": f"Discovered {len(packages)} packages",
        "processing": f"Started processing {min(process_top, len(packages))} packages" if process_top > 0 else None
    }