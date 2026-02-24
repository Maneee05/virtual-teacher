# Initialize pipeline
from src import LearningRAGPipeline
rag = LearningRAGPipeline()

# Ingest student materials
file_paths = [
    "C://Users//manee//Downloads//CSIT_III-II_CRYPTOGRAPHY AND NETWORK SECURITY DIGITAL NOTES (1).pdf",
    "C://Users//manee//Pictures//Screenshots//Screenshot 2026-02-10 142104.png"
]

rag.ingest_curriculum(file_paths)

# Student asks about a topic
query = "Explain Cipher"
lesson_plan, contexts = rag.generate_lesson_plan(query)

print("📝 GENERATED LESSON PLAN:")
print(lesson_plan)
print("\n📚 RETRIEVED CONTEXTS:")
for i, ctx in enumerate(contexts):
    print(f"{i+1}. {ctx['topic']}: {ctx['content'][:100]}...")
