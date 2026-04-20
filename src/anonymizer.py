import re


class SimpleAnonymizer:
    def __init__(self):
        self.global_mapping = {}
        self.person_counter = 0
        self.addr_counter = 0
        self.common_names = [
            "Иванов", "Петров", "Сидоров", "Кузнецов", "Попов",
            "Васильев", "Смирнов", "Иван", "Петр", "Сергей",
            "Мария", "Елена", "Ольга", "Анна"
        ]
        self.address_keywords = [
            "ул.", "улица", "пр.", "проспект", "пер.", "переулок",
            "дом", "д.", "квартира", "кв.", "офис", "оф."
        ]

    def _get_person_tag(self):
        self.person_counter += 1
        return f"<PERSON_{self.person_counter}>"

    def _get_addr_tag(self):
        self.addr_counter += 1
        return f"<ADDR_{self.addr_counter}>"

    def anonymize(self, text: str) -> tuple[str, dict]:
        """
        Anonymize text by replacing names and addresses with tags.

        Args:
            text: Original complaint text.

        Returns:
            Tuple with anonymized text and a replacement mapping.
        """
        if not isinstance(text, str):
            return text, {}

        anon_text = text
        local_mapping = {}

        for name in self.common_names:
            pattern = r'\b' + re.escape(name) + r'\b'

            if re.search(pattern, anon_text, re.IGNORECASE):
                tag = self._get_person_tag()
                local_mapping[tag] = name

                anon_text = re.sub(pattern, tag, anon_text, flags=re.IGNORECASE)

        words = anon_text.split()
        i = 0

        while i < len(words):
            word = words[i]
            is_addr_part = any(kw.lower() in word.lower() for kw in self.address_keywords)

            if is_addr_part:
                start_idx = max(0, i - 1)
                end_idx = min(len(words), i + 3)
                addr_snippet = " ".join(words[start_idx:end_idx])

                if addr_snippet not in local_mapping.values():
                    tag = self._get_addr_tag()
                    local_mapping[tag] = addr_snippet
                    anon_text = anon_text.replace(addr_snippet, tag)
                    i = end_idx
                else:
                    i += 1
            else:
                i += 1

        self.global_mapping.update(local_mapping)

        return anon_text, local_mapping

    def deanonymize(self, text: str, mapping: dict) -> str:
        """
        Restore original values from anonymization tags.

        Args:
            text: Text with tags.
            mapping: Replacement mapping returned by anonymize.

        Returns:
            Text with restored names and addresses.
        """
        result = text
        for tag, original_value in mapping.items():
            result = result.replace(tag, original_value)
        return result
