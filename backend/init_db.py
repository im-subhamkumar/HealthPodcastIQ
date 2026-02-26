from backend.database import init_db

if __name__ == "__main__":
    print("Initializing HealthPodcastIQ Database...")
    init_db()
    print("✓ Database initialized successfully.")
