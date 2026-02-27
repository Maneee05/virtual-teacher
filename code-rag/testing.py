# Initialize pipeline
from src import LearningRAGPipeline
rag = LearningRAGPipeline()

# Ingest student materials
file_paths = [
    "somefile.pdf",
    "someimage.png"
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
