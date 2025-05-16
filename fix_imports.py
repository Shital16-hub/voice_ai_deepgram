#!/usr/bin/env python3
"""
Manual fix script to resolve import issues
"""
import os
import shutil

def fix_imports():
    """Fix import issues by removing/backing up old files"""
    
    print("🔧 Fixing import issues...")
    
    # Check if llama_index directory exists and back it up
    llama_index_path = "knowledge_base/llama_index"
    if os.path.exists(llama_index_path):
        backup_path = "knowledge_base/llama_index_backup"
        print(f"📁 Backing up {llama_index_path} to {backup_path}")
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.move(llama_index_path, backup_path)
        print("✅ Backed up old llama_index directory")
    else:
        print("ℹ️ No llama_index directory found")
    
    # Remove __pycache__ directories
    print("🧹 Cleaning __pycache__ directories...")
    for root, dirs, files in os.walk("knowledge_base"):
        if "__pycache__" in dirs:
            cache_path = os.path.join(root, "__pycache__")
            shutil.rmtree(cache_path)
            print(f"   Removed: {cache_path}")
    
    # Remove .pyc files
    print("🧹 Cleaning .pyc files...")
    for root, dirs, files in os.walk("knowledge_base"):
        for file in files:
            if file.endswith(".pyc"):
                pyc_path = os.path.join(root, file)
                os.remove(pyc_path)
                print(f"   Removed: {pyc_path}")
    
    print("✅ Import issues fixed!")
    print("\nNow you can run: python index_knowledge_docs.py")

if __name__ == "__main__":
    fix_imports()