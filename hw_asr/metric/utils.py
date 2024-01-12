# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1.

    distance = editdistance.distance(target_text, predicted_text)
    return distance / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    tgt_words = target_text.split()
    if len(tgt_words) == 0:
        return 1.

    distance = editdistance.distance(tgt_words, predicted_text.split())
    return distance / len(tgt_words)