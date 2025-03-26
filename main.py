from memory.session_memory import HybridSessionMemory

def main():
    """Initialize System"""
    memory = HybridSessionMemory()
    memory.cleanup_old_memory()

if __name__ == "__main__":
    main()
