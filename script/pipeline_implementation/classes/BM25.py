from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Download tokenizer data if needed
nltk.download('punkt')

class BM25_Model:
    def __init__(self, qa_pairs):
        """
        Initialize the BM25 model with a list of question-answer pairs.

        Each pair is expected to be a dict with 'question' and 'answer' keys.
        """
        self.qa_pairs = qa_pairs

        # Combine question and answer into a single string and tokenize
        self.corpus = [word_tokenize(f"{pair['question']} {pair['answer']}") for pair in qa_pairs]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.corpus)

    def rank_qa_pairs(self, query, top_k=5):
        """
        Rank question-answer pairs based on similarity to the query.

        Parameters:
            query (str): Input query string.
            top_k (int): Number of top results to return.

        Returns:
            List[Tuple[Dict, float]]: List of (qa_pair, bm25_score) tuples sorted by score.
        """
        tokenized_query = word_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        results = sorted(
            zip(self.qa_pairs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return results[:top_k]
