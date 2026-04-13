
import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests




def prepare_knowledge_base(csv_file='data.csv'):
    print("Загружаем датасет...")
    df = pd.read_csv(csv_file)
    print("Загружаем модель для векторизации...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    print("Создаем векторы для поиска...")
    embeddings = model.encode(df['complaint'].tolist())
    with open('data/knowledge_base.pkl', 'wb') as f:
        pickle.dump({'df': df, 'embeddings': embeddings}, f)
    print("База знаний готова!")
    return model, df, embeddings




class SimpleAnonymizer:
    def __init__(self):
        self.mapping = {}
        self.person_id = 0
        self.addr_id = 0

    def anonymize(self, text):
        result = text
        mapping = {}
        keywords_person = ['Иванов', 'Петров', 'Сидоров', 'Иван', 'Мария']
        keywords_addr = ['ул.', 'улица', 'дом', 'д.', 'квартира', 'кв.']

        # Замена ФИО
        for word in keywords_person:
            if word in text and word not in mapping.values():
                self.person_id += 1
                tag = f"<PERSON_{self.person_id}>"
                mapping[tag] = word
                result = result.replace(word, tag)

        # Замена адресов
        words = result.split()
        for i, word in enumerate(words):
            if any(kw in word.lower() for kw in keywords_addr):
                addr_snippet = ' '.join(words[max(0, i - 1):min(len(words), i + 3)])
                if addr_snippet not in mapping.values():
                    self.addr_id += 1
                    tag = f"<ADDR_{self.addr_id}>"
                    mapping[tag] = addr_snippet
                    result = result.replace(addr_snippet, tag)

        self.mapping.update(mapping)
        return result, mapping

    def deanonymize(self, text, mapping):
        for tag, real_value in mapping.items():
            text = text.replace(tag, real_value)
        return text




def find_similar(new_complaint, df_base, embeddings, top_k=3):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    new_emb = model.encode([new_complaint])
    similarities = cosine_similarity(new_emb, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    examples = [{'complaint': df_base.iloc[idx]['complaint'], 'response': df_base.iloc[idx]['response']} for idx in
                top_indices]
    return examples




def generate_answer(new_complaint_anon, examples):
    """Генерирует ответ через локальную модель Ollama"""

    context = "\n\n".join([f"[Пример {i + 1}] Жалоба: {ex['complaint']} | Ответ: {ex['response']}"
                           for i, ex in enumerate(examples)])


    prompt = f"""Приветствие: Уважаемый заявитель!

На ваше обращение по вопросу: "{new_complaint_anon[:150]}..." сообщаем следующее.

Прокуратурой проведена проверка изложенных фактов. По результатам проверки выявлено...

Опирайтесь на следующие примеры правильных ответов:

{context}

Напишите полный официальный ответ от имени прокуратуры. Используйте официально-деловой стиль. Не используйте первые лица 
типа 'Я', вместо этого пишите 'Прокуратурой установлено...'."""

    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.2",  # Или ваша модель, если она другая
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.6}
    }

    try:
        print("Подключение к локальному серверу...")
        response = requests.post(url, json=data, timeout=120)
        result = response.json()
        return result.get("response", "(Ошибка получения ответа)")
    except Exception as e:
        return (f"Ошибка подключения к Ollama: {e}\n\n(Шаблонный ответ)\nУважаемый заявитель! Ваше обращение поступило"
                f" в прокуратуру...")# ========== ГЛАВНАЯ ФУНКЦИЯ ==========

def main():
    print("=" * 50)
    print("СИСТЕМА АДАПТИВНОЙ ГЕНЕРАЦИИ ОТВЕТОВ (PROCURATOR_RAG)")
    print("=" * 50)

    # Инициализация базы
    if not os.path.exists('data/knowledge_base.pkl'):
        print("Первая инициализация...")
        prepare_knowledge_base()

    with open('data/knowledge_base.pkl', 'rb') as f:
        data = pickle.load(f)

    anon = SimpleAnonymizer()

    while True:
        print("\n" + "-" * 30)
        print("Введите жалобу (или 'выход'):")
        user_input = input("> ").strip()

        if user_input.lower() in ['выход', 'exit']:
            print("👋 Система закрыта.")
            break

        # Анонимизация
        complaint_anon, mapping = anon.anonymize(user_input)
        print(f"[Обезличено]: {complaint_anon[:80]}...")

        # Поиск по базе
        examples = find_similar(complaint_anon, data['df'], data['embeddings'])
        print(f"Найдено похожих кейсов: {len(examples)}")

        # Генерация
        answer_anon = generate_answer(complaint_anon, examples)


        final_answer = anon.deanonymize(answer_anon, mapping)

        # Вывод
        print("\n" + "=" * 50)
        print("ПРОЕКТ ОТВЕТА:")
        print("=" * 50)
        print(final_answer)
        print("=" * 50)


if __name__ == "__main__":
    main()