"""
Migration script to convert from LlamaIndex/Ollama to OpenAI + Pinecone.
"""
import asyncio
import logging
import os
from datetime import datetime

# Import current system components
import chromadb
from llama_index.core.schema import Document as LlamaDocument

# Import new components
from knowledge_base.pinecone_manager import PineconeManager
from knowledge_base.document_processor import DocumentProcessor
from knowledge_base.openai_assistant_manager import OpenAIAssistantManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseMigration:
    """Migrate knowledge base from LlamaIndex to OpenAI + Pinecone."""
    
    def __init__(self):
        self.pinecone_manager = PineconeManager()
        self.doc_processor = DocumentProcessor()
        self.openai_manager = OpenAIAssistantManager()
    
    async def migrate_from_chromadb(self, chroma_collection_name: str = "company_knowledge"):
        """Migrate documents from ChromaDB to Pinecone."""
        logger.info("Starting migration from ChromaDB to Pinecone...")
        
        # Connect to ChromaDB
        chroma_client = chromadb.Client()
        collection = chroma_client.get_collection(chroma_collection_name)
        
        # Get all documents
        results = collection.get(
            include=["metadatas", "documents", "embeddings"]
        )
        
        logger.info(f"Found {len(results['ids'])} documents in ChromaDB")
        
        # Convert to new format
        documents = []
        for i, (doc_id, metadata, document) in enumerate(zip(
            results['ids'], results['metadatas'], results['documents']
        )):
            # Extract text content
            text = document if isinstance(document, str) else metadata.get('text', '')
            
            # Create document object
            doc = {
                "id": doc_id,
                "text": text,
                "source": metadata.get("source", "unknown"),
                "document_id": metadata.get("document_id", doc_id),
                "metadata": metadata
            }
            documents.append(doc)
        
        # Initialize Pinecone and upload
        await self.pinecone_manager.init()
        upserted_count = await self.pinecone_manager.upsert_documents(documents)
        
        logger.info(f"Migrated {upserted_count} documents to Pinecone")
    
    async def migrate_documents_directory(self, directory_path: str):
        """Migrate documents from a directory to the new system."""
        logger.info(f"Migrating documents from {directory_path}...")
        
        await self.pinecone_manager.init()
        
        # Process each file in directory
        import os
        processed_count = 0
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                try:
                    # Process file
                    documents = self.doc_processor.process_file(file_path)
                    
                    # Upload to Pinecone
                    await self.pinecone_manager.upsert_documents(documents)
                    
                    processed_count += len(documents)
                    logger.info(f"Processed {filename}: {len(documents)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        logger.info(f"Migration complete. Processed {processed_count} document chunks")
    
    async def create_assistant(self):
        """Create the OpenAI assistant."""
        logger.info("Creating OpenAI Assistant...")
        
        assistant_id = await self.openai_manager.create_assistant()
        logger.info(f"Created assistant: {assistant_id}")
        
        return assistant_id
    
    async def run_migration(self, 
                          migrate_chromadb: bool = True,
                          chroma_collection: str = "company_knowledge",
                          documents_directory: Optional[str] = None):
        """Run the complete migration process."""
        logger.info("Starting complete migration process...")
        
        # Step 1: Migrate from ChromaDB if requested
        if migrate_chromadb:
            try:
                await self.migrate_from_chromadb(chroma_collection)
            except Exception as e:
                logger.error(f"Error migrating from ChromaDB: {e}")
        
        # Step 2: Migrate documents from directory if provided
        if documents_directory and os.path.exists(documents_directory):
            try:
                await self.migrate_documents_directory(documents_directory)
            except Exception as e:
                logger.error(f"Error migrating documents: {e}")
        
        # Step 3: Create OpenAI Assistant
        try:
            assistant_id = await self.create_assistant()
            logger.info(f"Migration complete! Assistant ID: {assistant_id}")
        except Exception as e:
            logger.error(f"Error creating assistant: {e}")
        
        # Step 4: Verify migration
        await self.verify_migration()
    
    async def verify_migration(self):
        """Verify the migration was successful."""
        logger.info("Verifying migration...")
        
        try:
            # Check Pinecone stats
            stats = await self.pinecone_manager.get_stats()
            logger.info(f"Pinecone stats: {stats}")
            
            # Test a query
            results = await self.pinecone_manager.query("test query", top_k=1)
            logger.info(f"Test query returned {len(results)} results")
            
            logger.info("Migration verification complete")
            
        except Exception as e:
            logger.error(f"Error in verification: {e}")

async def main():
    """Main migration function."""
    print("Knowledge Base Migration - ChromaDB/LlamaIndex to OpenAI + Pinecone")
    print("=" * 60)
    
    # Create migration instance
    migration = KnowledgeBaseMigration()
    
    # Run migration
    await migration.run_migration(
        migrate_chromadb=True,
        chroma_collection="company_knowledge",
        documents_directory="./knowledge_base/knowledge_docs"  # Adjust path as needed
    )
    
    print("\nMigration complete!")
    print("Please update your application to use the new knowledge base components.")

if __name__ == "__main__":
    asyncio.run(main())