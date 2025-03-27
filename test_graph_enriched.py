from config import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def enrich_with_openai(user_query):
    print("🔍 Enriching query with OpenAI...")
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an intelligent memory enricher. Extract emotion, intent, topic, priority, and lifespan from this query."},
                {"role": "user", "content": user_query}
            ]
        )
        enriched = response.choices[0].message.content
        print("✅ Enrichment success:\n", enriched)
        return enriched
    except Exception as e:
        print("❌ OpenAI enrichment failed:\n", e)
        return None

if __name__ == "__main__":
    query = "I’m feeling overwhelmed preparing for our April product launch."
    enriched_output = enrich_with_openai(query)
