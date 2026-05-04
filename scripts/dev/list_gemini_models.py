# scripts/list_gemini_models.py
import os

from google import genai


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY ")

    client = genai.Client(api_key=api_key)

    print("Listing models that support generateContent:\n")
    found = 0
    for m in client.models.list():
        # m.supported_actions is typically like ['generateContent', ...]
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print(f"- {m.name} | actions={actions}")
            found += 1

    print(f"\nTotal generateContent-capable models: {found}")


if __name__ == "__main__":
    main()
