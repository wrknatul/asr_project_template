from typing import List, NamedTuple
import numpy as np
import heapq
import torch

from .char_text_encoder import CharTextEncoder
from hw_asr.base.base_text_encoder import BaseTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab =  list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_token = 0
        result = ""
        for ind in inds:
            if ind == last_token:
                continue

            if ind != 0:
                result += self.ind2char[ind]
            
            last_token = ind
        
        return result

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        probs = probs[:probs_length]
        char_length, voc_size = probs.shape
        assert char_length == probs_length
        probs = torch.exp(probs)

        assert voc_size == len(self.ind2char)
        state: dict[tuple[str, str], float] = {('', self.EMPTY_TOK): 1.0}
        best_prefixes: dict[str, float] = {'': 1.0}


        # for probs_for_time_t in probs:
        #     # Remove unlikely prefixes
        #     state = self._truncate_state_to_best_prefixes(state, best_prefixes)
             
        #     # Do 1 dynamical programming step
        #     state = self._extend_and_merge(probs_for_time_t, state)
        #     # Calculate the prefixes with highest probabilities
        #     best_prefixes = self._get_best_prefixes(state, beam_size)


        for probs_for_time_t in probs:
            state_new = {}
            for (pref, last_char), pref_prob in state.items():
                if pref in best_prefixes:
                    state_new.update({(pref, last_char): pref_prob})
            state = state_new
            next_values = {}
            for char_ind, proba_ in enumerate(probs_for_time_t.tolist()):
                for (pref, last_char), pref_prob in state.items():
                    new_pref = pref
                    end_char = self.ind2char[char_ind]
                    if end_char != last_char and end_char != self.EMPTY_TOK:
                        new_pref += end_char
                    next_values[(new_pref, end_char)] = next_values.get((new_pref, end_char), 0) + pref_prob * proba_
            state = next_values
            print(probs_for_time_t)
            assert len(next_values) != 0
            
            best_prefixes = {val for val in heapq.nlargest(beam_size, state.items(), key=lambda i: i[1])}
        
        hypos = [Hypothesis(self._correct_sentence(prefix), prob) for prefix, prob in best_prefixes.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    
