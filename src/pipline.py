from src.anonymizer import SimpleAnonymizer
from src.vectorizer import Vectorizer
from src.search_engine import SearchEngine
from src.generator import AnswerGenerator


class ResponsePipeline:

    def __init__(self, df, embeddings):
        self.anonymizer = SimpleAnonymizer()
        self.vectorizer = Vectorizer()
        self.search_engine = SearchEngine(df, embeddings)
        self.generator = AnswerGenerator()

    def process(self, user_query: str) -> str:
        print("\n" + "=" * 50)
        print(f"RECEIVED REQUEST: {user_query[:50]}...")
        print("=" * 50)

        print("Step 1: Data anonymization...")
        anon_query, mapping = self.anonymizer.anonymize(user_query)

        print("Step 2: Query vectorization...")
        query_vector = self.vectorizer.encode_single(anon_query)

        print("Step 3: Searching for similar precedents...")
        examples = self.search_engine.find_similar(query_vector)
        print(f"   Examples found: {len(examples)}")

        print("Step 4: Response generation...")
        prompt = self.generator.create_prompt(anon_query, examples)
        anon_response = self.generator.generate(prompt)

        print("Step 5: Data restoration...")
        final_response = self.anonymizer.deanonymize(anon_response, mapping)

        return final_response
