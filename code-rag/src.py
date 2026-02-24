import os
from xml.parsers.expat import model

from dotenv import load_dotenv
load_dotenv()  # Loads .env file

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from urllib import response
import uuid
import base64
from xmlrpc import client
import networkx as nx
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io
import openai

class LearningRAGPipeline:
    def __init__(self, qdrant_url=":memory:", collection_name="curriculum_graph"):
        self.client = QdrantClient(qdrant_url)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )

        # Knowledge graph to store topic relationships
        self.graph = nx.DiGraph()
        self._setup_collection()

    def _setup_collection(self):
        """Create Qdrant collection with multimodal vectors"""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                # Store text chunks, images (base64), graph nodes, topics
                sparse_vectors_config=None
            )

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF"""
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            texts.append(page.get_text())
        doc.close()
        return texts

    def extract_text_from_image(self, image_path: str) -> str:
        """Simple OCR placeholder - replace with Tesseract or multimodal LLM"""
        # For demo, we'll use image embedding directly
        return f"Image content from {image_path}"

    def process_uploaded_file(self, file_path: str) -> List[Document]:
        """Process PDF, DOC, or Image into text chunks"""
        docs = []

        if file_path.lower().endswith('.pdf'):
            texts = self.extract_text_from_pdf(file_path)
            for text in texts:
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    docs.append(Document(page_content=chunk, metadata={"source": file_path}))

        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = self.extract_text_from_image(file_path)
            docs.append(Document(page_content=text, metadata={"source": file_path, "type": "image"}))

        return docs

    def build_knowledge_graph(self, docs: List[Document]):
        """Build simple topic graph from documents"""
        topics = []

        for doc in docs:
            # Extract potential topics (simple keyword extraction)
            content = doc.page_content.lower()
            topic_candidates = [
                word for word in content.split()
                if len(word) > 4 and content.count(word) > 1
            ][:5]  # Top 5 repeated terms

            main_topic = topic_candidates[0] if topic_candidates else "general"

            # Add to graph: source -> main_topic -> subtopics
            self.graph.add_node(main_topic, type="topic", content=doc.page_content[:200])
            topics.append(main_topic)

        # Connect related topics
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                if topic1 != topic2:
                    self.graph.add_edge(topic1, topic2, weight=0.5)

    def embed_and_store(self, docs: List[Document]):
        """Embed documents and store in Qdrant"""
        points = []

        for i, doc in enumerate(docs):
            # Create embedding
            embedding = self.embedding_model.encode(doc.page_content).tolist()

            # Create point with metadata
            point = PointStruct(
                id=uuid.uuid4().hex,
                vector=embedding,
                payload={
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "type": doc.metadata.get("type", "text"),
                    "topic": list(self.graph.nodes)[i % len(self.graph.nodes)] if self.graph.nodes else "general"
                }
            )
            points.append(point)

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def ingest_curriculum(self, file_paths: List[str]):
        """Main ingestion pipeline"""
        print("🚀 Ingesting curriculum...")
        all_docs = []

        # Process all files
        for file_path in file_paths:
            print(f"📄 Processing {file_path}")
            docs = self.process_uploaded_file(file_path)
            all_docs.extend(docs)

        # Build knowledge graph
        self.build_knowledge_graph(all_docs)

        # Embed and store
        self.embed_and_store(all_docs)

        print(f"✅ Stored {len(all_docs)} chunks with {len(self.graph.nodes)} topics!")

    def hybrid_search(self, query: str, topic_filter: str = None) -> List[Dict]:
        """Hybrid search: semantic + graph-based retrieval"""
        query_embedding = self.embedding_model.encode(query).tolist()

        # Semantic search filter
        filters = []
        if topic_filter:
            filters.append(FieldCondition(
                key="topic",
                match=MatchValue(value=topic_filter)
            ))

        # Search top 5 similar chunks
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=Filter(must=filters) if filters else None,
            limit=5
        ).points

        # Get graph context for better relevance
        graph_context = self._get_graph_context(query)

        results = []
        for hit in search_result:
            results.append({
                "content": hit.payload["content"],
                "score": hit.score,
                "source": hit.payload["source"],
                "topic": hit.payload["topic"]
            })

        # Add graph context
        if graph_context:
            results.append({
                "content": graph_context,
                "score": 0.8,
                "source": "knowledge_graph",
                "topic": "related_topics"
            })

        return results[:5]

    def _get_graph_context(self, query: str) -> str:
        """Get related topics from knowledge graph"""
        query_words = query.lower().split()

        # Find matching topics
        matching_topics = []
        for node, data in self.graph.nodes(data=True):
            if any(word in str(node).lower() for word in query_words):
                matching_topics.append(node)

        if matching_topics:
            context = f"Related topics: {', '.join(matching_topics[:3])}"
            return context

        return None

    def generate_lesson_plan(self, student_query: str, llm_prompt_template: str = None) -> str:
        """Retrieve context and generate lesson plan"""
        print(f"🔍 Searching for: {student_query}")

        # Hybrid retrieval
        contexts = self.hybrid_search(student_query)

        # Format context for LLM
        context_text = "\n\n".join([f"[{i+1}] {ctx['content']}" for i, ctx in enumerate(contexts)])

        # Simple lesson plan template (replace with your LLM call)
        lesson_prompt = f"""
        STUDENT QUERY: {student_query}

        CURRICULUM CONTEXT:
        {context_text}

        Create a 3-part lesson plan for a 3D avatar teacher:
        1. Learning Objectives (2-3 bullet points)
        2. Main Explanation (300 words max, use simple language)
        3. Visual Aids (suggest 3D models/images to show)

        LESSON PLAN:
        """

        # Here you would call your LLM (OpenAI, Grok, etc.)
        lesson_plan = self._mock_llm_response(lesson_prompt)

        return lesson_plan, contexts
    
        
    def _mock_llm_response(self, prompt: str) -> str:
        """✅ WORKING Gemini via OpenAI API compatibility"""
        try:
            from openai import OpenAI
        
            # Gemini OpenAI-compatible endpoint
            client = OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        
            response = client.chat.completions.create(
                model="gemini-2.5-flash",  # ✅ Working model name
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error: {str(e)}"




'''
    def _mock_llm_response(self, prompt: str) -> str:
        """Mock LLM - replace with actual LLM call"""
        # Extract query outside f-string to avoid backslash issue
        query = prompt.split('STUDENT QUERY: ')[1].split('\n')[0]
        return f"""
## 📚 Lesson Plan for "{query}"

### 🎯 Learning Objectives
- Understand core concepts from uploaded materials
- Apply knowledge through examples
- Visualize concepts using 3D models

### 📖 Main Explanation
[Detailed explanation based on retrieved context would go here]

### 🖼️ Visual Aids
- 3D Model: [Relevant model suggestion]
- Diagram: [Topic diagram]
- Animation: [Process visualization]
        """

'''