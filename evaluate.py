from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def simple_eval(predicted_captions, true_captions):
    """
    Simple evaluation with BLEU, METEOR, and ROUGE
    """
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    
    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    for pred, true in zip(predicted_captions, true_captions):
        # Clean text
        pred = pred.strip().lower()
        true = true.strip().lower()
        
        # Tokenize
        pred_tokens = pred.split()
        true_tokens = true.split()
        
        # BLEU score (4-gram)
        if len(pred_tokens) > 0:
            bleu = sentence_bleu([true_tokens], pred_tokens)
            bleu_scores.append(bleu)
        
        # METEOR score
        if len(pred_tokens) > 0 and len(true_tokens) > 0:
            meteor = meteor_score([true_tokens], pred_tokens)
            meteor_scores.append(meteor)
        
        # ROUGE score
        rouge_scores_dict = rouge_scorer_obj.score(true, pred)
        rouge_scores.append(rouge_scores_dict['rouge1'].fmeasure)
    
    # Calculate averages
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    
    return {
        'BLEU': avg_bleu,
        'METEOR': avg_meteor,
        'ROUGE-1': avg_rouge
    }

def print_eval(predicted_captions, true_captions):
    """Print evaluation metrics in a simple format"""
    metrics = simple_eval(predicted_captions, true_captions)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"BLEU:     {metrics['BLEU']:.3f}")
    print(f"METEOR:   {metrics['METEOR']:.3f}")
    print(f"ROUGE-1:  {metrics['ROUGE-1']:.3f}")
    print("="*50)
    
    return metrics 