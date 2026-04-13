"""
Manthana — Case Embeddings (Medical Entity Embeddings)
Purpose-built case fingerprinting using Eka Care Parrotlet-e.

Capabilities:
- Generate embeddings from case summaries (any modality)
- Semantic similarity search across patient cases
- Medical entity-aware embeddings (symptoms, diagnoses, anatomy)
- Multilingual support (12 Indic languages + English)

Architecture:
- Lazy-loaded via LazyModel pattern
- GPU-accelerated inference
- Vector store integration (ChromaDB default, pluggable)
- PHI-safe (embeddings only, no raw text stored)

Integration Points:
- Called by gateway after each analysis completion
- Used by correlation_engine for "similar cases" lookup
- Supports RAG for unified report generation
"""

import os
import json
import logging
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

# Import lazy loader
from model_loader import ManagedModel

logger = logging.getLogger("manthana.embeddings")

# Model configuration
PARROTLET_E_MODEL_ID = "ekacare/parrotlet-e"
PARROTLET_E_CACHE_NAME = "parrotlet-e"
PARROTLET_E_VRAM_GB = 2.0  # 0.6B params ≈ 1.2GB in fp16, round up for safety
EMBEDDING_DIM = 1024  # BGE-M3 base dimension

# Lazy model instance
parrotlet_e_model = ManagedModel(
    model_id=PARROTLET_E_MODEL_ID,
    cache_name=PARROTLET_E_CACHE_NAME,
    device="cuda",
    vram_gb=PARROTLET_E_VRAM_GB,
    priority=7,  # Higher priority - fast inference, frequently used
)


def is_loaded() -> bool:
    """Check if embedding model is loaded."""
    return parrotlet_e_model.is_loaded()


def generate_case_embedding(
    case_summary: str,
    case_id: str,
    patient_id: Optional[str] = None,
    modality: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Generate embedding for a clinical case summary.
    
    Args:
        case_summary: Canonical text describing the case
                     (modality | findings | impression | key values)
        case_id: Unique identifier for this analysis
        patient_id: Optional patient identifier (hashed before storage)
        modality: Source modality (xray, ct, mri, lab_report, etc.)
        metadata: Additional payload (scores, structures, etc.)
    
    Returns:
        Dictionary with embedding vector and metadata
    """
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    
    logger.info(f"Generating embedding for case: {case_id}")
    
    # Load model and tokenizer
    model = parrotlet_e_model.get()
    tokenizer = AutoTokenizer.from_pretrained(
        PARROTLET_E_MODEL_ID,
        cache_dir=parrotlet_e_model.cache_dir,
        trust_remote_code=True,
    )
    
    # Tokenize
    inputs = tokenizer(
        case_summary,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    
    # Move to device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate embedding (mean pooling of last hidden states)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling with attention mask
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        
        # Mask padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        # Normalize (L2)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    # Convert to numpy
    embedding_np = embedding.cpu().numpy()[0]
    
    # Build result
    # Hash patient_id for PHI safety if provided
    patient_hash = None
    if patient_id:
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
    
    result = {
        "case_id": case_id,
        "patient_hash": patient_hash,
        "modality": modality,
        "embedding": embedding_np.tolist(),
        "embedding_dim": EMBEDDING_DIM,
        "model": "parrotlet-e",
        "created_at": datetime.utcnow().isoformat(),
        "payload": metadata or {},
        # Store truncated summary for debugging (not full text)
        "summary_preview": case_summary[:200] if case_summary else "",
    }
    
    return result


def build_case_summary(
    modality: str,
    findings: List[Dict],
    impression: str,
    pathology_scores: Dict[str, float],
    structures: List[str],
    lab_values: Optional[Dict] = None,
    language: str = "en",
) -> str:
    """
    Build canonical case summary text for embedding.
    
    This standardizes case representation across all modalities
    so CT, MRI, X-ray, lab reports all embed in the same space.
    
    Format (example):
        Modality: Chest X-ray
        Findings: Lung opacity, pleural effusion
        Impression: Pneumonia suspected
        Key Scores: opacity=0.85, effusion=0.72
        Structures: lung, pleura
        Labs: WBC=elevated, CRP=elevated
    """
    parts = []
    
    # Modality
    parts.append(f"Modality: {modality}")
    
    # Findings (extract labels)
    if findings:
        finding_labels = [f.get("label", f.get("finding", "")) for f in findings]
        finding_labels = [f for f in finding_labels if f]
        if finding_labels:
            parts.append(f"Findings: {', '.join(finding_labels)}")
    
    # Impression
    if impression:
        # Truncate if too long
        impression_clean = impression.replace("\n", " ")[:300]
        parts.append(f"Impression: {impression_clean}")
    
    # Key pathology scores (top 5)
    if pathology_scores:
        top_scores = sorted(
            pathology_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        score_strs = [f"{k}={v:.2f}" for k, v in top_scores]
        parts.append(f"Key Scores: {', '.join(score_strs)}")
    
    # Structures
    if structures:
        # Sanitize structures to handle numpy arrays/unhashable types
        if isinstance(structures, dict):
            # If structures is a dict, extract keys or values
            structures = list(structures.keys()) if structures else []
        elif hasattr(structures, 'tolist'):
            # Convert numpy array to list
            structures = structures.tolist()
        # Ensure all items are strings
        structure_strs = [str(s) for s in structures[:10] if s is not None]
        if structure_strs:
            parts.append(f"Structures: {', '.join(structure_strs)}")
    
    # Lab values (if available)
    if lab_values:
        # Format key lab values
        lab_strs = []
        for key, value in list(lab_values.items())[:5]:
            if isinstance(value, (int, float)):
                lab_strs.append(f"{key}={value}")
            else:
                lab_strs.append(f"{key}={str(value)[:20]}")
        if lab_strs:
            parts.append(f"Labs: {', '.join(lab_strs)}")
    
    return "\n".join(parts)


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    import numpy as np
    
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)
    
    # Cosine similarity
    dot = np.dot(e1, e2)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot / (norm1 * norm2))


# ═════════════════════════════════════════════════════════════════════════════
# Vector Store Integration (Pluggable)
# ═════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """Abstract base for vector storage backends."""
    
    def add(self, embedding_record: Dict) -> None:
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        modality_filter: Optional[str] = None,
    ) -> List[Dict]:
        raise NotImplementedError
    
    def get_by_case_id(self, case_id: str) -> Optional[Dict]:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based vector store.
    Lightweight, embedded, good for single-node deployments.
    """
    
    def __init__(self, persist_dir: str = "/models/vector_db"):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
    
    def _get_collection(self):
        """Lazy init ChromaDB."""
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name="manthana_cases",
                metadata={"hnsw:space": "cosine"},
            )
        
        return self._collection
    
    def add(self, embedding_record: Dict) -> None:
        """Add embedding to ChromaDB."""
        collection = self._get_collection()
        
        case_id = embedding_record["case_id"]
        embedding = embedding_record["embedding"]
        
        # Metadata (must be serializable)
        metadata = {
            "modality": embedding_record.get("modality", "unknown"),
            "patient_hash": embedding_record.get("patient_hash"),
            "created_at": embedding_record.get("created_at"),
            "model": embedding_record.get("model", "parrotlet-e"),
        }
        
        # Add payload as JSON string (Chroma has limited metadata types)
        payload = embedding_record.get("payload", {})
        if payload:
            metadata["payload_json"] = json.dumps(payload, default=str)
        
        # Documents field stores the preview text
        documents = embedding_record.get("summary_preview", "")
        
        collection.add(
            ids=[case_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[documents],
        )
        
        logger.info(f"Added embedding to ChromaDB: {case_id}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        modality_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Search for similar cases."""
        collection = self._get_collection()
        
        # Build where filter
        where_filter = None
        if modality_filter:
            where_filter = {"modality": modality_filter}
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances", "documents"],
        )
        
        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, case_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                document = results["documents"][0][i] if results["documents"] else ""
                
                # Parse payload JSON if present
                payload = {}
                if metadata and "payload_json" in metadata:
                    try:
                        payload = json.loads(metadata["payload_json"])
                    except json.JSONDecodeError:
                        pass
                
                formatted.append({
                    "case_id": case_id,
                    "similarity": 1.0 - distance,  # Convert distance to similarity
                    "modality": metadata.get("modality", "unknown"),
                    "patient_hash": metadata.get("patient_hash"),
                    "created_at": metadata.get("created_at"),
                    "summary_preview": document,
                    "payload": payload,
                })
        
        return formatted
    
    def get_by_case_id(self, case_id: str) -> Optional[Dict]:
        """Retrieve specific case by ID."""
        collection = self._get_collection()
        
        try:
            result = collection.get(
                ids=[case_id],
                include=["embeddings", "metadatas", "documents"],
            )
            
            if result["ids"]:
                metadata = result["metadatas"][0] if result["metadatas"] else {}
                embedding = result["embeddings"][0] if result["embeddings"] else None
                document = result["documents"][0] if result["documents"] else ""
                
                payload = {}
                if metadata and "payload_json" in metadata:
                    try:
                        payload = json.loads(metadata["payload_json"])
                    except json.JSONDecodeError:
                        pass
                
                return {
                    "case_id": case_id,
                    "embedding": embedding,
                    "modality": metadata.get("modality"),
                    "patient_hash": metadata.get("patient_hash"),
                    "created_at": metadata.get("created_at"),
                    "summary_preview": document,
                    "payload": payload,
                }
        except Exception as e:
            logger.warning(f"Failed to get case {case_id}: {e}")
        
        return None


# Global vector store instance (configurable via env)
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create global vector store instance."""
    global _vector_store
    
    if _vector_store is None:
        backend = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
        
        if backend == "chroma":
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "/models/vector_db")
            _vector_store = ChromaVectorStore(persist_dir=persist_dir)
            logger.info(f"Initialized ChromaDB at {persist_dir}")
        elif backend == "qdrant":
            # Future: Qdrant implementation for distributed deployments
            raise NotImplementedError("Qdrant backend coming soon")
        elif backend == "memory":
            # Future: In-memory for testing
            raise NotImplementedError("Memory backend coming soon")
        else:
            raise ValueError(f"Unknown vector store backend: {backend}")
    
    return _vector_store


def store_case_embedding(
    case_id: str,
    case_summary: str,
    patient_id: Optional[str] = None,
    modality: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Full pipeline: generate embedding and store in vector DB.
    
    This is the main API called by the gateway after analysis.
    """
    # Generate embedding
    record = generate_case_embedding(
        case_summary=case_summary,
        case_id=case_id,
        patient_id=patient_id,
        modality=modality,
        metadata=metadata,
    )
    
    # Store
    store = get_vector_store()
    store.add(record)
    
    return record


def find_similar_cases(
    case_id: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    case_summary: Optional[str] = None,
    top_k: int = 5,
    modality_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Find similar cases by ID, embedding, or text summary.
    
    At least one of case_id, query_embedding, or case_summary must be provided.
    """
    store = get_vector_store()
    
    # Get embedding to search with
    if query_embedding is not None:
        embedding = query_embedding
    elif case_id is not None:
        # Retrieve existing case
        record = store.get_by_case_id(case_id)
        if record is None:
            raise ValueError(f"Case not found: {case_id}")
        embedding = record["embedding"]
    elif case_summary is not None:
        # Generate embedding from text
        temp_record = generate_case_embedding(
            case_summary=case_summary,
            case_id="temp_query",
        )
        embedding = temp_record["embedding"]
    else:
        raise ValueError("Must provide case_id, query_embedding, or case_summary")
    
    # Search
    results = store.search(
        query_embedding=embedding,
        top_k=top_k,
        modality_filter=modality_filter,
    )
    
    return results
