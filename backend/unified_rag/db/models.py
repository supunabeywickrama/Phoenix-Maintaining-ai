"""
Phoenix Industries schema.

Six tables — the knowledge side of the platform only. The sensor/ML tables from
the Zynaptrix product (AnomalyRecord, ChatMessage, SensorConfiguration,
AnomalyThreshold, MachineAsset, MachineEvaluation) are deliberately absent:
Phoenix has no telemetry in scope, so nothing writes or reads them.

Machine is kept even though Phoenix has no sensors, because it is what scopes a
question to a manual and gives InteractionMemory something to hang past fixes on.
"""
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from pgvector.sqlalchemy import Vector
from unified_rag.db.database import Base


class ManualChunk(Base):
    """One retrievable unit of a manual: a passage, a table summary, or a figure caption."""
    __tablename__ = "manual_chunks"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False)   # 'text' | 'image' | 'table'
    content = Column(Text, nullable=True)   # passage, table summary, or vision caption
    embedding = Column(Vector, nullable=False)  # dimension follows EMBEDDING_DIM
    page = Column(Integer, nullable=True)
    path = Column(String, nullable=True)    # Cloudinary URL of an extracted figure


class Manual(Base):
    """Metadata for a source PDF stored in Cloudinary."""
    __tablename__ = "manuals"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String, unique=True, index=True, nullable=False)
    filename = Column(String, nullable=False)
    url = Column(String, nullable=False)
    created_at = Column(String, nullable=True)


class Machine(Base):
    """A piece of equipment on the floor, mapped to the manual that documents it."""
    __tablename__ = "machines"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    location = Column(String, nullable=True)
    manual_id = Column(String, nullable=False)  # -> ManualChunk.manual_id


class InteractionMemory(Base):
    """
    A fix that actually worked, written when a technician resolves a session.

    RetrievalEngine.retrieve() queries this alongside the manual, so a repair
    that succeeded once is surfaced the next time the same machine is asked about.
    """
    __tablename__ = "interaction_memory"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    manual_id = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    operator_fix = Column(Text, nullable=True)
    embedding = Column(Vector, nullable=False)
    timestamp = Column(String, nullable=False)


class AssistantSession(Base):
    """One conversation thread, optionally pinned to a machine."""
    __tablename__ = "assistant_sessions"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=True)
    title = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)
    resolved_at = Column(String, nullable=True)  # set when a fix is archived


class AssistantMessage(Base):
    """A single turn within a session."""
    __tablename__ = "assistant_messages"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("assistant_sessions.id"), nullable=False)
    role = Column(String, nullable=False)   # 'agent' | 'user'
    content = Column(Text, nullable=False)
    type = Column(String, default='text')   # 'text' | 'wizard_step'
    step_data = Column(Text, nullable=True)  # JSON
    images = Column(Text, nullable=True)     # JSON list of URLs
    timestamp = Column(String, nullable=False)
