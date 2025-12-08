"""RAG agent that enforces knowledge-base only answers."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

# Optional import to avoid hard dependency when Groq isn't used
try:
    from groq import Groq  # type: ignore
except ImportError:  # pragma: no cover
    Groq = None

from openai import OpenAI
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
try:
    from google import genai as genai_v2
    from google.genai import types as genai_types
except ImportError:
    genai_v2 = None

from .config import Settings
from .data_models import AgentAnswer, RetrievalResult
from .vector_store import EmbeddingClient, VectorStore

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are Yantra Live's support AI. Answer strictly from the provided knowledge base context.
If the answer is not contained in the context, respond with: "I don't have information about that in my knowledge base".

### Style & Formatting
- **Beautify the Output**: Use clear Markdown structure.
  - Use `###` for section headers.
  - **Highlight Key Info**: ALWAYS use **bold** for:
    - Model names (e.g., **JCB 3DX**)
    - Key specifications (e.g., **76 hp**, **0.30 cu.m**)
    - Critical warnings or notes.
  - Use bullet points for lists.
  - Use tables for structured data.
- **Tone**: Professional, structured, and helpful.

### Instructions
- Do NOT guess. Do NOT use outside knowledge.
- Respect the dataset tags [END_CUSTOMER], [SPARE_PARTS], [DEALERS] when reasoning.

### Special Handling

**1. SPECIFICATIONS & TECHNICAL DETAILS**
If the user asks for "specifications", "technical specifications", or general machine details (e.g. "Specs of 3DX", "Tell me about 3DXL Plus"):
- **MANDATORY**: You must extract and present the following details if available in the context:
  - **Engine**
  - **Transmission**
  - **Tyres**
  - **Shipping/Operating Weight**
  - **Brakes**
  - **Steering**
  - **Electrical**
- Present these clearly (e.g., in a list or table).

**2. SPECIFIC ATTRIBUTE QUESTIONS**
If the user asks for a *single* specific attribute (e.g., "What is the engine power?", "What is the bucket capacity?"), answer **only** that attribute directly. Do not provide the full specification list.

**3. COMPATIBILITY QUESTIONS** (e.g., "which breaker is compatible with SANY SY20")
- Scan ALL lines in the context.
- Return a **bullet list** of compatible models/SKUs.

**3. COMPARISON QUESTIONS** (e.g., "Compare JCB and CAT")
- Start with: "Here is a comparison between [A] and [B]:"
- Build a clean Markdown table:
    | Feature | [Model A] | [Model B] |
    |---------|-----------|-----------|
    | ...     | ...       | ...       |
- Choose meaningful features from the CONTEXT.
- Only include facts that are clearly present in the context.
"""


class YantraRAGAgent:
    """High-level facade that wraps retrieval + generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = EmbeddingClient(settings.embedding_model)
        self.vector_store = VectorStore(
            settings.faiss_index_path,
            settings.chunk_store_path,
            self.embedder,
        )
        self.vector_store.load()
        self.provider = self.settings.llm_provider.lower()
        self._gemini_model_name: str | None = None
        self.client = self._init_llm_client()

    def _init_llm_client(self):
        if self.provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when llm_provider=openai")
            return OpenAI(api_key=self.settings.openai_api_key)
        if self.provider == "gemini":
            self._gemini_model_name = self._normalize_gemini_model(self.settings.gemini_model)
            if not self.settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when llm_provider=gemini")
            
            # If using File Search (Store ID present), use the new v2 client
            if self.settings.gemini_store_id and genai_v2:
                return genai_v2.Client(api_key=self.settings.gemini_api_key)
            
            # Fallback to old client for standard generation
            genai.configure(api_key=self.settings.gemini_api_key)
            return genai.GenerativeModel(model_name=self._gemini_model_name)
        if self.provider == "groq":
            if Groq is None:
                raise ImportError("groq package is not installed. Please reinstall requirements.")
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when llm_provider=groq")
            return Groq(api_key=self.settings.groq_api_key)
        raise ValueError(f"Unsupported llm_provider '{self.settings.llm_provider}'")

    @staticmethod
    def _normalize_gemini_model(model_name: str) -> str:
        name = (model_name or "gemini-1.5-flash").strip().strip("'")
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        
        # If it's a specific version like 1.5, 2.0, 2.5, don't append latest automatically
        # unless it's a known alias that needs it.
        # Actually, the new SDK prefers exact names.
        return name


    def _filter_hits(self, hits: List[RetrievalResult]) -> List[RetrievalResult]:
        return [hit for hit in hits if hit.score >= self.settings.min_similarity]

    @staticmethod
    def _extract_section(content: str, start_marker: str, end_markers: List[str]) -> str | None:
        """Return substring between start_marker and the earliest end_marker."""
        upper_content = content.upper()
        start_token = start_marker.upper()
        start_idx = upper_content.find(start_token)
        if start_idx == -1:
            return None

        end_idx = len(content)
        for marker in end_markers:
            token = marker.upper()
            marker_idx = upper_content.find(token, start_idx + len(start_token))
            if marker_idx != -1 and marker_idx < end_idx:
                end_idx = marker_idx

        snippet = content[start_idx:end_idx].strip()
        if len(snippet) < 40:
            return None
        return snippet

    @staticmethod
    def _slice_content_for_question(question: str, content: str) -> str:
        lower_question = question.lower()
        section: str | None = None

        if "excavator" in lower_question:
            section = YantraRAGAgent._extract_section(
                content,
                "EXCAVATOR PERFORMANCE",
                [
                    "LOADER PERFORMANCE",
                    "STATIC DIMENSIONS",
                    "TRANSMISSION",
                ],
            )
        elif "loader" in lower_question:
            section = YantraRAGAgent._extract_section(
                content,
                "LOADER PERFORMANCE",
                [
                    "STATIC DIMENSIONS",
                    "TRANSMISSION",
                    "EXCAVATOR PERFORMANCE",
                ],
            )
        elif "static" in lower_question and "dimension" in lower_question:
            section = YantraRAGAgent._extract_section(
                content,
                "STATIC DIMENSIONS",
                [
                    "TRANSMISSION",
                    "EXCAVATOR PERFORMANCE",
                    "LOADER PERFORMANCE",
                ],
            )

        if section:
            return section
        return content

    def _format_context(self, question: str, hits: List[RetrievalResult]) -> str:
        formatted = []
        for idx, hit in enumerate(hits, start=1):
            citation = f"{hit.chunk.source_file}"
            if hit.chunk.page_number:
                citation += f", page {hit.chunk.page_number}"
            snippet = self._slice_content_for_question(question, hit.chunk.content)
            formatted.append(
                f"[Source {idx} | {citation}]\n{snippet.strip()}"
            )
        return "\n\n".join(formatted)

    @staticmethod
    def _gather_images(hits: List[RetrievalResult]) -> List[Path]:
        images: List[Path] = []
        for hit in hits:
            for image in hit.chunk.images:
                if image.path not in images:
                    images.append(image.path)
        return images

    def _select_images(self, hits: List[RetrievalResult], question: str) -> List[Path]:
        if not hits:
            return []

        tokens = [tok for tok in re.findall(r"[a-z0-9]+", question.lower()) if len(tok) >= 3]
        if not tokens:
            tokens = [question.lower()]

        source_scores: dict[str, int] = {}
        source_order: List[str] = []
        for hit in hits:
            source = hit.chunk.source_file
            if source not in source_scores:
                source_scores[source] = 0
                source_order.append(source)
            cleaned_name = re.sub(r"[^a-z0-9]+", " ", source.lower())
            cleaned_content = hit.chunk.content.lower()
            for token in tokens:
                if token in cleaned_name:
                    source_scores[source] += 3
                elif token in cleaned_content:
                    source_scores[source] += 1

        ranked_sources = sorted(
            source_scores.keys(),
            key=lambda s: (-source_scores[s], source_order.index(s)),
        )
        for source in ranked_sources:
            primary_hits = [hit for hit in hits if hit.chunk.source_file == source]
            images = self._gather_images(primary_hits)
            if images:
                return images
        return self._gather_images(hits)

    def answer(self, question: str, history: List[dict] = None) -> AgentAnswer:
        # 1. Gemini File Search Path
        if self.provider == "gemini" and self.settings.gemini_store_id and genai_v2:
            gemini_answer = self._answer_with_gemini_file_search(question, history)
            
            # Check if the answer is a refusal or error
            is_refusal = "I don't have information about that" in gemini_answer.answer
            is_error = "Error querying Gemini Knowledge Base" in gemini_answer.answer
            
            if not is_refusal and not is_error:
                # If successful, try to enrich with images from local vector store
                try:
                    hits = self.vector_store.search(question, self.settings.top_k)
                    images = self._select_images(hits, question)
                    gemini_answer.images = images
                except Exception as e:
                    LOGGER.warning(f"Failed to retrieve local images for Gemini answer: {e}")
                return gemini_answer
            
            LOGGER.info("Gemini File Search refused or failed. Falling back to local RAG.")

        # 2. Standard RAG Path (Groq / OpenAI / Gemini-Legacy)
        hits = self.vector_store.search(question, self.settings.top_k)
        hits = self._filter_hits(hits)
        if not hits:
            LOGGER.info("No grounded evidence found for question: %s", question)
            return AgentAnswer(
                answer="I don't have information about that in my knowledge base",
                grounded=False,
            )

        context_block = self._format_context(question, hits)
        user_prompt = self._build_user_prompt(question, context_block, history)

        answer_text = self._generate_model_answer(user_prompt)
        citations = [
            f"{hit.chunk.source_file}:{hit.chunk.page_number or '?'}"
            for hit in hits
        ]
        images = self._select_images(hits, question)
        return AgentAnswer(answer=answer_text, citations=citations, images=images)

    def _answer_with_gemini_file_search(self, question: str, history: List[dict] = None) -> AgentAnswer:
        """Use Google's managed File Search tool."""
        try:
            # Build conversation history for context
            contents = []
            if history:
                for msg in history:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append(genai_types.Content(
                        role=role,
                        parts=[genai_types.Part(text=msg["content"])]
                    ))
            
            # Add current question
            contents.append(genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=question)]
            ))

            # Enhanced System Instruction for Tables and Formatting
            system_instruction = (
                "You are Yantra Live's support AI. Answer strictly from the provided knowledge base context.\n"
                "If the answer is not contained in the context, respond with: \"I don't have information about that in my knowledge base\".\n\n"
                "### CRITICAL INSTRUCTIONS:\n"
                "1. **DIRECT ANSWERING**: Start the answer IMMEDIATELY. Do NOT use phrases like \"The provided information mentions\", \"However, I can tell you\", \"I don't have enough specific information\", or \"Based on available information\". Just state the facts directly.\n"
                "2. **PARTIAL DATA**: If you have partial information (e.g. specs for one machine but not the other), just present what you have in the requested format (Table/List). Do not explain what is missing or apologize.\n"
                "3. **CONTEXT RESOLUTION**: Always use the conversation history to understand what 'this', 'it', or 'the machine' refers to. Formulate your search queries using the specific entity names (e.g., 'Stage 5 Engine') instead of pronouns.\n"
                "4. **COMPARISONS**: If the user asks to 'compare', 'difference', or 'vs', you MUST provide the answer as a **Markdown Table**.\n"
                "   - Columns: Feature, [Model A], [Model B], ...\n"
                "   - Rows: Engine, Power, Weight, etc.\n"
                "   - Do NOT just list paragraphs. USE A TABLE.\n"
                "5. **SPECIFICATIONS**: Use bullet points or tables for specs. If a section like 'Service Refill Capacities' is requested, extract ALL values (Fuel tank, Hydraulic system, etc.) and present them clearly.\n"
                "6. **PRICE/COST**: If the user asks for 'price', 'cost', or 'range', look for the 'Range in USD' column in the dataset. Answer with the value from that column (e.g., '80-88K USD').\n"
                "7. **BOLDING**: Bold key numbers and model names.\n"
                "8. **IMAGES & PAGE NUMBERS**: \n"
                "   - If the user asks for images, photos, or diagrams, you MUST provide the page number where the information is found.\n"
                "   - **STRICT FORMAT**: You MUST use the format `[PAGE: <number>]` (with brackets and colon). Example: `[PAGE: 297]`.\n"
                "   - **ACCURACY**: Only cite page numbers that are EXPLICITLY mentioned in the retrieved context. Do not guess or hallucinate page numbers. If you don't see a page number in the text, do not invent one.\n"
                "9. **FOLLOW-UP QUESTIONS**: At the very end of your response, generate exactly 4 relevant follow-up questions that the user might want to ask next. Format them exactly like this:\n"
                "---FOLLOWUP---\n"
                "Question 1\n"
                "Question 2\n"
                "Question 3\n"
                "Question 4\n"
            )

            response = self.client.models.generate_content(
                model=self._gemini_model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[
                        genai_types.Tool(
                            file_search=genai_types.FileSearch(
                                file_search_store_names=[f"fileSearchStores/{self.settings.gemini_store_id}"]
                            )
                        )
                    ]
                )
            )
            
            full_text = response.text
            answer_text = full_text
            suggested_questions = []

            if "---FOLLOWUP---" in full_text:
                parts = full_text.split("---FOLLOWUP---")
                answer_text = parts[0].strip()
                questions_text = parts[1].strip()
                suggested_questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                # Limit to 4 just in case
                suggested_questions = suggested_questions[:4]

            citations = []
            
            # Extract citations from grounding metadata if available
            if (response.candidates and 
                response.candidates[0].grounding_metadata and 
                response.candidates[0].grounding_metadata.grounding_chunks):
                
                for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                    if chunk.retrieved_context:
                        citations.append(f"{chunk.retrieved_context.title} ({chunk.retrieved_context.uri})")

            return AgentAnswer(
                answer=answer_text,
                grounded=bool(citations),
                citations=citations,
                images=[], # File Search doesn't return local image paths easily yet
                suggested_questions=suggested_questions
            )
        except Exception as e:
            LOGGER.error("Gemini File Search failed: %s", e)
            return AgentAnswer(
                answer=f"Error querying Gemini Knowledge Base: {e}",
                grounded=False
            )

    def _generate_model_answer(self, user_prompt: str) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        # Gemini provider
        try:
            response = self.client.generate_content(
                f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                generation_config={"temperature": 0.0},
            )
        except google_exceptions.NotFound as exc:
            if self._gemini_model_name and not self._gemini_model_name.endswith("-latest"):
                self._gemini_model_name = f"{self._gemini_model_name}-latest"
                self.client = genai.GenerativeModel(model_name=self._gemini_model_name)
                response = self.client.generate_content(
                    f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                    generation_config={"temperature": 0.0},
                )
            else:
                raise exc
        text = getattr(response, "text", "").strip()
        if text:
            return text
        return "I don't have information about that in my knowledge base"

    def _build_user_prompt(self, question: str, context_block: str, history: List[dict] = None) -> str:
        lower_question = question.lower()
        comparison_keywords = (
            "compare",
            "comparison",
            "difference",
            "versus",
            " vs ",
            "side by side",
        )
        requires_table = any(keyword in lower_question for keyword in comparison_keywords)
        instructions = (
            "Instructions: Answer using only the context provided."
            " Do not mention sources or citations in your response."
            " Use hyphens for lists."
        )
        if requires_table:
            instructions += (
                " Present the response as a Markdown table with the first column named 'Attribute',"
                " and a separate column for each machine/model explicitly mentioned in the user question."
                " Include a final column titled 'Difference / Notes' that clearly states why each row is included."
                " Only include attributes whose values differ across the compared models."
                " If the context shows no differences, respond with a single sentence explaining that no differences were found and skip the table."
                " If data for a requested model is missing, include a note such as 'Data unavailable in knowledge base'."
                " After the table (when present), include a concise bullet list (max 3 bullets) summarizing the most important distinctions."
            )

        history_text = ""
        if history:
            history_text = "Conversation History:\n"
            for msg in history[-5:]: # Include last 5 messages for context
                history_text += f"{msg['role'].upper()}: {msg['content']}\n"
            history_text += "\n"

        return (
            "Knowledge Base Context:\n"
            f"{context_block}\n\n"
            f"{history_text}"
            f"User Question: {question}\n\n"
            f"{instructions}"
        )
