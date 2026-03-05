import pandas as pd
from typing import List, Dict

class Vocabulary:
    def __init__(self, token_map: Dict[str, int]):
        if not token_map:
            raise ValueError("O mapa de tokens não pode estar vazio.")
        
        self._vocab_df = pd.DataFrame(list(token_map.items()), columns=['token', 'id'])
        self._token_to_id = token_map

    def encode(self, text: str) -> List[int]:
        tokens = text.lower().split()
        encoded = []
        
        for token in tokens:
            if token not in self._token_to_id:
                raise KeyError(f"Token desconhecido no vocabulário: '{token}'")
            encoded.append(self._token_to_id[token])
            
        return encoded

    @property
    def size(self) -> int:
        return len(self._vocab_df)