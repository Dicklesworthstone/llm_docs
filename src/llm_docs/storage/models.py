"""
SQLModel models for the llm_docs database.
"""

from typing import Optional, List
from datetime import datetime
from enum import Enum, auto
from sqlmodel import Field, SQLModel, Relationship


class PackageStatus(str, Enum):
    """Status of a package in the processing pipeline."""
    DISCOVERED = "discovered"              # Package discovered, not yet processed
    DOCS_EXTRACTED = "docs_extracted"      # Documentation has been extracted
    DISTILLATION_PENDING = "distillation_pending"  # Ready for distillation
    DISTILLATION_IN_PROGRESS = "distillation_in_progress"  # Currently being distilled
    DISTILLATION_COMPLETED = "distillation_completed"  # Distillation complete
    DISTILLATION_FAILED = "distillation_failed"  # Distillation failed


class Package(SQLModel, table=True):
    """Model representing a Python package."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None
    discovery_date: datetime = Field(default_factory=datetime.now)
    status: PackageStatus = Field(default=PackageStatus.DISCOVERED)
    priority: int = Field(default=0)  # Higher number = higher priority
    
    # URLs and paths
    docs_url: Optional[str] = None
    original_doc_path: Optional[str] = None
    distilled_doc_path: Optional[str] = None
    
    # Processing timestamps
    docs_extraction_date: Optional[datetime] = None
    distillation_start_date: Optional[datetime] = None
    distillation_end_date: Optional[datetime] = None
    
    # Relationships
    stats: List["PackageStats"] = Relationship(back_populates="package")
    pages: List["DocumentationPage"] = Relationship(back_populates="package")
    jobs: List["DistillationJob"] = Relationship(back_populates="package")


class PackageStats(SQLModel, table=True):
    """Model for tracking package download statistics over time."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    package_id: int = Field(foreign_key="package.id", index=True)
    monthly_downloads: int = Field(default=0)
    recorded_at: datetime = Field(default_factory=datetime.now)
    
    # Relationship
    package: Package = Relationship(back_populates="stats")


class DocumentationPage(SQLModel, table=True):
    """Model representing a documentation page for a package."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    package_id: int = Field(foreign_key="package.id", index=True)
    url: str
    title: Optional[str] = None
    extraction_date: datetime = Field(default_factory=datetime.now)
    content_length: Optional[int] = None
    
    # Relationship
    package: Package = Relationship(back_populates="pages")


class DistillationJobStatus(str, Enum):
    """Status of a distillation job."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DistillationJob(SQLModel, table=True):
    """Model for tracking distillation jobs."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    package_id: int = Field(foreign_key="package.id", index=True)
    status: DistillationJobStatus = Field(default=DistillationJobStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    num_chunks: int = Field(default=5)
    chunks_processed: int = Field(default=0)
    
    input_file_path: Optional[str] = None
    output_file_path: Optional[str] = None
    
    error_message: Optional[str] = None
    
    # Relationship
    package: Package = Relationship(back_populates="jobs")


class ProcessingMetrics(SQLModel, table=True):
    """Model for tracking processing metrics."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    date: datetime = Field(default_factory=datetime.now)
    
    packages_discovered: int = Field(default=0)
    packages_docs_extracted: int = Field(default=0)
    packages_distilled: int = Field(default=0)
    
    total_tokens_processed: int = Field(default=0)
    total_api_cost: float = Field(default=0.0)
    
    avg_extraction_time_seconds: Optional[float] = None
    avg_distillation_time_seconds: Optional[float] = None


def create_db_and_tables(engine):
    """Create database tables."""
    SQLModel.metadata.create_all(engine)
