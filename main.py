import os
import sys

# Add the project root to the module search path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DATASET_PATH, KNOWLEDGE_BASE_PATH
from src.data_loader import load_dataset, save_knowledge_base, load_knowledge_base
from src.pipline import ResponsePipeline
from src.vectorizer import Vectorizer


def build_knowledge_base() -> dict:
    df = load_dataset(DATASET_PATH)
    col_to_embed = "complaint_anon" if "complaint_anon" in df.columns else "complaint"

    print(f"Search column: '{col_to_embed}'")

    vectorizer = Vectorizer()
    embeddings = vectorizer.encode_texts(df[col_to_embed].tolist())

    data = {"df": df, "embeddings": embeddings}
    save_knowledge_base(data, KNOWLEDGE_BASE_PATH)
    return data


def main():
    print("Starting the prosecutor response RAG system...")

    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("\nKnowledge base not found. Creating a new one...")
        try:
            data = build_knowledge_base()
        except Exception as e:
            print(f"Failed to create the knowledge base: {e}")
            return
    else:
        print("Knowledge base found. Loading...")
        data = load_knowledge_base(KNOWLEDGE_BASE_PATH)
        if data is None:
            print("Knowledge base cache is corrupted or incompatible. Rebuilding...")
            try:
                data = build_knowledge_base()
            except Exception as e:
                print(f"Failed to rebuild the knowledge base: {e}")
                return

    df = data["df"]
    embeddings = data["embeddings"]

    pipeline = ResponsePipeline(df, embeddings)

    print("\nSystem is ready.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        user_input = input("Your request: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye.")
            break

        if not user_input:
            continue

        try:
            response = pipeline.process(user_input)

            print("\n" + "-" * 50)
            print("SYSTEM RESPONSE:")
            print(response)
            print("-" * 50 + "\n")

        except Exception as e:
            print(f"\nProcessing error: {e}")


if __name__ == "__main__":
    main()
