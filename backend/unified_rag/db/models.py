"""
Phoenix Industries schema.

Relational tables only — vector search moved to Qdrant (see
unified_rag/db/qdrant_store.py), since Postgres has no reason to hold the
sessions/messages/machines data ManualChunk and InteractionMemory used to sit
next to. Those two classes are kept below, unused by application code, purely
as a read-only migration source: scripts/migrate_to_qdrant.py reads their
already-computed embeddings straight out of pgvector rather than re-paying to
re-embed everything, and their data is left in place as a rollback path until
Qdrant is confirmed working. Safe to drop both tables once that's verified.

The sensor/ML tables from the Zynaptrix product (AnomalyRecord, ChatMessage,
SensorConfiguration, AnomalyThreshold, MachineAsset, MachineEvaluation) are
deliberately absent: Phoenix has no telemetry in scope, so nothing writes or
reads them.

Machine is kept even though Phoenix has no sensors, because it is what scopes a
question to a manual and gives interaction memory something to hang past fixes on.
"""
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from pgvector.sqlalchemy import Vector
from unified_rag.db.database import Base


class ManualChunk(Base):
    """LEGACY / READ-ONLY — superseded by Qdrant's manual_chunks collection.
    Nothing in the app writes here anymore; kept only so the migration script
    can read out already-computed embeddings. See module docstring."""
    __tablename__ = "manual_chunks"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False)   # 'text' | 'image' | 'table'
    content = Column(Text, nullable=True)   # passage, table summary, or vision caption
    embedding = Column(Vector, nullable=False)  # dimension follows EMBEDDING_DIM
    page = Column(Integer, nullable=True)
    path = Column(String, nullable=True)    # Cloudinary URL of an extracted figure

    # Figure structure. A composite drawing is stored as the whole figure plus one
    # chunk per component SAM isolated from it, so an answer can show the full
    # diagram for orientation before zooming into the part being discussed.
    # NULL means "ingested before this existed" and is treated as 'full'.
    figure_role = Column(String, nullable=True)    # 'full' | 'part'
    parent_path = Column(String, nullable=True)    # for a part: path of its full figure
    width = Column(Integer, nullable=True)         # pixels, used to drop unusably small crops
    height = Column(Integer, nullable=True)

    # What this asset actually IS, beyond text/image/table: 'diagram', 'schematic',
    # 'chart', 'flowchart', 'photo', 'table', 'exploded_view'. A technician reading
    # "Figure 3" learns nothing; "Wiring schematic (page 12)" tells them whether it
    # is worth opening. Also drives presentation order in an answer.
    kind = Column(String, nullable=True)

    # Markdown for assets that should be rendered as structure rather than shown as
    # a picture - currently tables. camelot already returns a DataFrame at ingestion;
    # previously only an LLM prose summary was kept and the grid was discarded, so a
    # table could never be displayed as a table.
    render_markdown = Column(Text, nullable=True)


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
    """LEGACY / READ-ONLY — superseded by Qdrant's interaction_memory
    collection. Nothing in the app writes here anymore; kept only so the
    migration script can read out already-computed embeddings."""
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
    # 'troubleshoot' | 'learn' — chosen when the chat starts. A fault report and a
    # "how does this work" question want different answers, so the whole thread is
    # steered by it rather than re-guessing the intent on every turn.
    intent = Column(String, nullable=True)
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
    # JSON list of typed assets shown with this reply: figures, schematics, charts
    # and rendered tables. `images` is kept alongside it for older messages and for
    # the report view, which only deals in pictures.
    attachments = Column(Text, nullable=True)
    timestamp = Column(String, nullable=False)
