"""
surveillance_rag_engine.py
Upgraded RAG Engine for XCAM-Detect
- Semantic search via ChromaDB + SentenceTransformers
- Conversational memory for the session
- Confidence scoring + citations on every answer
- Ollama LLM with rule-based fallback
- Drop-in compatible with app.py (respond() interface)
"""

import re
import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("SurveillanceRAG")


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = [
    {
        "id": "kb_001", "section": "Confidence Scoring", "category": "confidence_scoring",
        "text": (
            "Very High Confidence (90%+): The system is highly certain this is a person. "
            "Both YOLO and ResNet agree strongly. GradCAM++ shows clear activation on human "
            "features such as head, shoulders, and body outline. Trust this detection."
        )
    },
    {
        "id": "kb_002", "section": "Confidence Scoring", "category": "confidence_scoring",
        "text": (
            "Medium Confidence (70-89%): The system detects a person but with some uncertainty. "
            "This can occur due to partial occlusion, camouflage effectiveness, or background "
            "interference. GradCAM++ may show scattered activation. Manual review suggested."
        )
    },
    {
        "id": "kb_003", "section": "Confidence Scoring", "category": "confidence_scoring",
        "text": (
            "Low Confidence (below 70%): Weak detection signals. Significant background noise "
            "or heavy camouflage. GradCAM++ shows weak or inconsistent activation patterns. "
            "High chance of false positive. Verification required."
        )
    },
    {
        "id": "kb_004", "section": "Pipeline Architecture", "category": "pipeline",
        "text": (
            "Detection Pipeline Stage 1 - YOLO: Fast object detection identifies potential persons "
            "using YOLOv8 trained on camouflaged person datasets. Confidence threshold 0.50. "
            "Inference ~50ms but may produce false positives at a rate of 15-20%."
        )
    },
    {
        "id": "kb_005", "section": "Pipeline Architecture", "category": "pipeline",
        "text": (
            "Detection Pipeline Stage 2 - ResNet Verification: Each YOLO crop is passed to ResNet50 "
            "for binary verification (person vs non-person). Confidence threshold 0.70. "
            "Reduces false positive rate to ~5%. Adds ~30ms latency. Combined accuracy ~88%."
        )
    },
    {
        "id": "kb_006", "section": "XAI Methods", "category": "xai_methods",
        "text": (
            "GradCAM++ (Gradient-weighted Class Activation Mapping++): Uses second-order gradients "
            "for better spatial localization than standard GradCAM. Red and yellow regions mean high "
            "importance for the classification. Green means moderate. Blue and purple mean low or "
            "background. Activation on head, shoulders, and torso means strong human features detected."
        )
    },
    {
        "id": "kb_007", "section": "XAI Methods", "category": "xai_methods",
        "text": (
            "LIME (Local Interpretable Model-agnostic Explanations): Explains predictions by "
            "perturbing superpixels and observing model output changes. Green regions support "
            "the person classification. Red regions contradict it. When GradCAM++ and LIME "
            "agree on the same regions, the model is focusing on genuine human body parts."
        )
    },
    {
        "id": "kb_008", "section": "Confidence Factors", "category": "confidence_factors",
        "text": (
            "Factors that increase detection confidence: clear human silhouette, distinct edge "
            "patterns, no occlusion, good lighting, texture matching human clothing, strong "
            "GradCAM++ activation on head and shoulders and torso."
        )
    },
    {
        "id": "kb_009", "section": "Confidence Factors", "category": "confidence_factors",
        "text": (
            "Factors that decrease detection confidence: partial occlusion by trees or bushes, "
            "effective camouflage blending with background, poor lighting such as rain or fog or "
            "night, small or distant target, weak GradCAM++ activation, conflicting signals "
            "between YOLO and ResNet scores."
        )
    },
    {
        "id": "kb_010", "section": "False Positive Filtering", "category": "filtering",
        "text": (
            "False positive filtering: YOLO alone has 15-20% false positive rate. ResNet "
            "verification reduces this to ~5%. Common false positives are tree stumps, rock "
            "formations, shadows, dense vegetation, and mannequins. ResNet distinguishes these "
            "using texture, shape, and contextual features."
        )
    },
    {
        "id": "kb_011", "section": "Threat Assessment", "category": "threat_assessment",
        "text": (
            "Threat level classification: "
            "CRITICAL means confidence above 90%, clear detection, urgent response recommended. "
            "HIGH means confidence 80-90%, strong detection, investigate promptly. "
            "MEDIUM means confidence 70-80%, verified human presence, monitor. "
            "LOW means confidence 50-70%, likely false alarm, flagged for review."
        )
    },
    {
        "id": "kb_012", "section": "Decision Making", "category": "decision_making",
        "text": (
            "When to trust detections: "
            "TRUST when both YOLO and ResNet above 85% with clear GradCAM++ on human features. "
            "REVIEW when scores 70-85% with scattered activation or partial occlusion. "
            "LIKELY FALSE when ResNet below 70% with unfocused GradCAM++ on background."
        )
    },
    {
        "id": "kb_013", "section": "Detection Scenarios", "category": "scenarios",
        "text": (
            "Clear detection scenario: Person in open area, good lighting, minimal camouflage. "
            "YOLO 85%+, ResNet 90%+. High trust. "
            "Partial occlusion scenario: Person behind tree or bush. YOLO 65-75%, ResNet 70-85%. "
            "Manual verification required."
        )
    },
    {
        "id": "kb_014", "section": "Detection Scenarios", "category": "scenarios",
        "text": (
            "Heavy camouflage scenario: Military-grade camo in matching environment. "
            "YOLO 50-65%, ResNet 60-75%. Difficult detection, human expert required. "
            "False positive scenario: Tree stump or shadow. YOLO 50-70%, ResNet below 60%. "
            "System should filter this out automatically."
        )
    },
    {
        "id": "kb_015", "section": "Performance Metrics", "category": "performance",
        "text": (
            "System performance: YOLO detection rate ~85% of real persons. "
            "ResNet verification accuracy ~94%. Combined system accuracy ~88%. "
            "False positive rate ~5%. Processing time under 1 second per image on CPU. "
            "12% reduction in false positives compared to YOLO-only."
        )
    },
    {
        "id": "kb_016", "section": "Combined Score", "category": "confidence_scoring",
        "text": (
            "Combined score is the average of YOLO detection confidence and ResNet verification "
            "score. Formula: Combined = (YOLO + ResNet) / 2. "
            "Above 85% is considered highly reliable. Above 90% is critical threat level."
        )
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL MEMORY
# ═══════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """Stores chat history for the current session instance."""

    def __init__(self, max_turns: int = 6):
        self.turns: List[Dict] = []
        self.max_turns = max_turns

    def add(self, role: str, text: str):
        self.turns.append({"role": role, "text": text.strip()})
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]

    def build_context(self) -> str:
        if not self.turns:
            return ""
        lines = ["CONVERSATION HISTORY:"]
        for t in self.turns:
            prefix = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"  {prefix}: {t['text']}")
        return "\n".join(lines)

    def clear(self):
        self.turns.clear()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class SurveillanceRAGEngine:
    """
    Surveillance RAG Engine.
    Uses ChromaDB + SentenceTransformers for semantic retrieval,
    Ollama for generation, and ConversationMemory for session context.

    Compatible with app.py via respond(user_message, detection_context).
    """

    def __init__(self,
                 ollama_url="http://localhost:11434",
                 model_name="llama3.2",
                 embedding_model="all-MiniLM-L6-v2"):

        self.ollama_url = ollama_url
        self.model_name = model_name
        self.knowledge_loaded = False

        # Embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="surveillance_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self.chroma_client.reset()
            self.collection = self.chroma_client.create_collection(
                name="surveillance_knowledge",
                metadata={"hnsw:space": "cosine"}
            )

        # Conversational memory (lives for the duration of this instance)
        self.memory = ConversationMemory(max_turns=6)

        # Load knowledge base into ChromaDB
        self._load_knowledge()

    # ───────────────────────────────────────────────────────────────────────
    # KNOWLEDGE LOADING
    # ───────────────────────────────────────────────────────────────────────

    def _load_knowledge(self):
        print("Loading surveillance domain knowledge...")
        existing_ids = set(self.collection.get()["ids"])
        to_add = [c for c in KNOWLEDGE_BASE if c["id"] not in existing_ids]

        if to_add:
            texts      = [c["text"] for c in to_add]
            ids        = [c["id"]   for c in to_add]
            metadatas  = [{"category": c["category"], "section": c["section"]} for c in to_add]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
                metadatas=metadatas,
            )

        self.knowledge_loaded = True
        print(f"Loaded {len(KNOWLEDGE_BASE)} knowledge chunks into ChromaDB")

    def add_detection_to_knowledge(self, detection_data: Dict):
        """Add a live detection result into the knowledge base."""
        det_id = "det_" + hashlib.md5(
            json.dumps(detection_data, sort_keys=True).encode()
        ).hexdigest()[:8]

        score = detection_data.get("resnet_score", 0)
        level = "HIGH" if score >= 0.9 else "MEDIUM" if score >= 0.7 else "LOW"

        text = (
            f"Detection record - Threat Level: {level}. "
            f"YOLO Score: {detection_data.get('yolo_score', 0)*100:.1f}%. "
            f"ResNet Score: {score*100:.1f}%. "
            f"Location: {detection_data.get('bbox', 'unknown')}. "
            f"Timestamp: {datetime.now().isoformat()}."
        )

        embedding = self.embedding_model.encode(text).tolist()
        try:
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[det_id],
                metadatas=[{"category": "detection_history", "section": "Detection History"}],
            )
        except Exception:
            pass  # already exists, skip

    # ───────────────────────────────────────────────────────────────────────
    # SEMANTIC RETRIEVAL
    # ───────────────────────────────────────────────────────────────────────

    def retrieve_context(self, query: str, top_k: int = 4) -> List[Dict]:
        """
        Retrieve the most semantically relevant knowledge chunks for a query.
        Returns list of dicts: text, section, category, score.
        """
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "distances", "metadatas"],
        )

        chunks = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            # cosine distance → similarity (0 = identical, 2 = opposite)
            similarity = round(max(0.0, 1.0 - dist), 3)
            chunks.append({
                "text":     doc,
                "section":  meta.get("section",  "General"),
                "category": meta.get("category", "general"),
                "score":    similarity,
            })

        log.info("Retrieved %d chunks for: '%s'", len(chunks), query[:60])
        return chunks

    # ───────────────────────────────────────────────────────────────────────
    # CONFIDENCE SCORING
    # ───────────────────────────────────────────────────────────────────────

    def _compute_confidence(self, query: str, answer: str, chunks: List[Dict]) -> float:
        """
        Estimate answer confidence (0-1) from three signals:
        - Average retrieval similarity of top chunks       (40%)
        - Semantic similarity between answer and top chunk (40%)
        - Answer length heuristic                          (20%)
        """
        if not chunks:
            return 0.0

        avg_retrieval = float(np.mean([c["score"] for c in chunks]))

        try:
            ans_emb   = self.embedding_model.encode(answer)
            chunk_emb = self.embedding_model.encode(chunks[0]["text"])
            dot       = float(np.dot(ans_emb, chunk_emb))
            norm      = float(np.linalg.norm(ans_emb) * np.linalg.norm(chunk_emb) + 1e-8)
            sim       = dot / norm
        except Exception:
            sim = 0.5

        length_score = min(1.0, len(answer.split()) / 40)

        confidence = (0.4 * avg_retrieval) + (0.4 * sim) + (0.2 * length_score)
        return round(min(1.0, max(0.0, confidence)), 3)

    # ───────────────────────────────────────────────────────────────────────
    # PROMPT BUILDING
    # ───────────────────────────────────────────────────────────────────────

    SYSTEM_PROMPT = """You are XCAM-Bot, an AI assistant built into the XCAM-Detect surveillance system.
You explain how the system works and help operators understand scan results.

STRICT RULES — follow these exactly:
1. Write in plain, clear paragraphs. No bullet points. No markdown headers. No bold text.
2. Do NOT invent URLs, links, or citations like "[Section]" or "(source)". Never do this.
3. Do NOT repeat the question back to the user.
4. Do NOT say "Great question" or use filler phrases.
5. Keep answers focused and under 5 sentences unless the question genuinely needs more.
6. If scan data is provided, mention the actual numbers naturally in your answer.
7. If you don't know something, say so plainly. Do not make things up."""

    def _build_prompt(self, query: str, chunks: List[Dict],
                      detection_context: Optional[Dict],
                      history_context: str) -> str:
        lines = [self.SYSTEM_PROMPT, ""]

        # Conversation history
        if history_context:
            lines += [history_context, ""]

        # Retrieved knowledge — shown as plain facts, no section labels
        lines.append("BACKGROUND KNOWLEDGE (use this to inform your answer, do not cite it directly):")
        for chunk in chunks:
            lines.append(f"- {chunk['text']}")

        # Live detection data — only inject if the question is actually about the scan
        detection_keywords = [
            "person", "detect", "scan", "confidence", "score", "threat", "yolo",
            "resnet", "result", "found", "high", "low", "medium", "how many",
            "gradcam", "lime", "heatmap", "trust", "false", "verified", "filter"
        ]
        query_lower = query.lower()
        is_detection_question = any(kw in query_lower for kw in detection_keywords)

        if detection_context and is_detection_question:
            dets = detection_context.get("detections", [])
            stats = detection_context.get("statistics", {})
            if dets:
                lines.append("\nCURRENT SCAN RESULTS:")
                for idx, det in enumerate(dets, 1):
                    yolo   = det.get("yolo_score",    det.get("yolo_confidence", 0)) * 100
                    resnet = det.get("resnet_score",   0) * 100
                    combo  = det.get("combined_score", (yolo + resnet) / 200) * 100
                    level  = "HIGH" if resnet >= 90 else "MEDIUM" if resnet >= 70 else "LOW"
                    lines.append(
                        f"  Person {idx}: YOLO={yolo:.1f}%, ResNet={resnet:.1f}%, "
                        f"Combined={combo:.1f}%, Threat={level}"
                    )
                if stats:
                    lines.append(
                        f"  Total: {stats.get('yolo_detections', 0)} detected by YOLO, "
                        f"{stats.get('verified_detections', 0)} verified, "
                        f"{stats.get('filtered_out', 0)} filtered out."
                    )

        lines += [
            f"\nUSER QUESTION: {query}",
            "",
            "Write a clear, direct answer in plain sentences. Do not use bullet points, bold text, "
            "headers, or any citations. If scan data is shown above, refer to the actual numbers naturally. "
            "Keep it concise.",
            "",
            "ANSWER:",
        ]
        return "\n".join(lines)

    # ───────────────────────────────────────────────────────────────────────
    # OLLAMA
    # ───────────────────────────────────────────────────────────────────────

    def query_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama. Returns None if unavailable (triggers fallback)."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=45,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return None
        except requests.exceptions.Timeout:
            return "⚠️ Ollama timed out. Try a simpler question or a smaller model."
        except Exception as e:
            log.warning("Ollama error: %s", e)
            return None

    def _rule_based_fallback(self, chunks: List[Dict]) -> str:
        """Return a factual summary from retrieved chunks when Ollama is unavailable."""
        if not chunks:
            return (
                "I couldn't find specific information about that. "
                "Try asking about confidence scores, GradCAM++, LIME, "
                "threat levels, or the detection pipeline."
            )
        best      = chunks[0]
        sentences = re.split(r"(?<=[.!?])\s+", best["text"])
        summary   = " ".join(sentences[:3])
        return (
            f"According to [{best['section']}]: {summary}\n\n"
            "*(Ollama is not running — answering from knowledge base directly. "
            "Start Ollama with `ollama serve` for richer answers.)*"
        )

    # ───────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE  — drop-in for app.py
    # ───────────────────────────────────────────────────────────────────────

    def respond(self,
                user_message: str,
                detection_context: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Main entry point called by app.py chatbot.

        Parameters
        ----------
        user_message      : raw text from the user
        detection_context : dict with 'detections' and 'statistics' from last scan

        Returns
        -------
        (answer, source)  where source is "rag" or "rule-based"
        """

        # Short-circuit for greetings and small talk — no need to call Ollama
        msg = user_message.strip().lower()
        small_talk = {
            ("hi", "hey", "hello", "hiya"):
                "Hello! I'm XCAM-Bot. Run a scan and ask me anything about the detections.",
            ("thanks", "thank you", "thx", "ty"):
                "You're welcome. Let me know if you have more questions.",
            ("ok", "okay", "k", "alright", "cool", "got it", "noted"):
                "Got it. Feel free to ask anything about the scan results.",
            ("who are you", "what are you"):
                "I'm XCAM-Bot, the AI assistant for the XCAM-Detect surveillance system. I can explain detection results, confidence scores, GradCAM++ heatmaps, LIME explanations, and threat levels.",
            ("bye", "goodbye", "see you"):
                "Goodbye. Stay vigilant.",
        }
        for triggers, reply in small_talk.items():
            if msg in triggers or any(msg == t for t in triggers):
                self.memory.add("user", user_message)
                self.memory.add("assistant", reply)
                return reply, "rule-based"

        # 1. Semantic retrieval
        chunks = self.retrieve_context(user_message, top_k=4)

        # 2. Conversation history
        history_ctx = self.memory.build_context()

        # 3. Build prompt
        prompt = self._build_prompt(
            query=user_message,
            chunks=chunks,
            detection_context=detection_context,
            history_context=history_ctx,
        )

        # 4. Call Ollama (fallback if unavailable)
        raw = self.query_ollama(prompt)
        if raw is None:
            answer = self._rule_based_fallback(chunks)
            source = "rule-based"
        else:
            answer = raw
            source = "rag"

        # 5. Confidence score
        confidence = self._compute_confidence(user_message, answer, chunks)

        # 6. Update conversational memory
        self.memory.add("user",      user_message)
        self.memory.add("assistant", answer)

        return answer, source

    # ───────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def clear_memory(self):
        """Reset conversation memory (e.g. on new session)."""
        self.memory.clear()

    def test_connection(self) -> str:
        if not self.is_available():
            return (
                "Ollama is NOT running.\n"
                "Fix:\n"
                "  1) Download from https://ollama.com/download\n"
                "  2) Run: ollama serve\n"
                "  3) Run: ollama pull llama3.2"
            )
        return "Surveillance RAG Engine ready."


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  XCAM Surveillance RAG Engine — Test")
    print("=" * 60)

    engine = SurveillanceRAGEngine()
    print("\n" + engine.test_connection())

    sample_context = {
        "detections": [
            {"yolo_score": 0.82, "resnet_score": 0.94, "combined_score": 0.88},
            {"yolo_score": 0.61, "resnet_score": 0.71, "combined_score": 0.66},
        ],
        "statistics": {
            "yolo_detections":     3,
            "verified_detections": 2,
            "filtered_out":        1,
        }
    }

    test_turns = [
        "Why is person 1 high confidence?",
        "What does the GradCAM heatmap show?",
        "Should I trust this detection?",
        "What about person 2, they have a lower score?",
        "Is there a risk of false positives?",
    ]

    print("\n[CONVERSATIONAL TEST]\n")
    for turn in test_turns:
        print(f"User: {turn}")
        answer, source = engine.respond(turn, sample_context)
        preview = answer[:350] + ("..." if len(answer) > 350 else "")
        print(f"Bot ({source}): {preview}\n")