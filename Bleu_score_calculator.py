import nltk
from nltk.translate.bleu_score import *
import sys

def nltk_bleu(hypotheses, references):
    refs = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()

        score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
        total_score += score
        count += 1
    print(total_score)
    avg_score = total_score / count
    print ('avg_score: %.4f' % avg_score)
    return corpus_bleu, avg_score


def evaluate():

    hypotheses = []
    print('start evaluation')
    with open("predictions", 'r') as file:
        for line in file:
            hypotheses.append(line.strip())

    references = []
    with open("reference", 'r') as file:
        for line in file:
            references.append(line.strip())

    nltk_bleu(hypotheses, references)

evaluate()
