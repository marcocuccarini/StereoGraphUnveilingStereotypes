from sentence_transformers import SentenceTransformer, util
from sentence_transformers import util

class BERT_Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def text_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=True)

    def rank_qa_pairs(self, query, qa_pairs_dict, top_k=None):
        """
        Rank question-answer pairs based on their similarity to a given query.

        Parameters:
            query (str): User input or query.
            qa_pairs_dict (Dict[str, Dict]): Dictionary where keys are 'id' and values are dicts with 'question' and 'answer'.
            top_k (int, optional): Number of top results to return.

        Returns:
            List[Dict]: Each dict contains 'id', 'answer', 'score'.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        for id_, pair in qa_pairs_dict.items():
            if not all(k in pair for k in ['question', 'answer']):
                raise ValueError("Each value in qa_pairs_dict must have 'question' and 'answer'.")

        query_embedding = self.text_embedding(query)

        # Concatenate question + answer for each item
        qa_texts = [f"{pair['question']} {pair['answer']}" for pair in qa_pairs_dict.values()]
        qa_embeddings = self.model.encode(qa_texts, convert_to_tensor=True)

        scores = util.cos_sim(query_embedding, qa_embeddings)[0]

        # Create results list
        results = sorted(
            [
                {'id': id_, 'answer': pair['answer'], 'score': float(score)}
                for (id_, pair), score in zip(qa_pairs_dict.items(), scores)
            ],
            key=lambda x: x['score'],
            reverse=True
        )

        if top_k:
            results = results[:top_k]

        return results

    def get_results(self, qa_pairs_dict, top_k=None):
        results = {}

        # Estrai gli ID, embeddings e risposte
        ids = list(qa_pairs_dict.keys())
        question_embeddings = [pair['question_embedding'] for pair in qa_pairs_dict.values()]
        answer_embeddings = [pair['answer_embedding'] for pair in qa_pairs_dict.values()]
        answers = [pair['answer'] for pair in qa_pairs_dict.values()]

        # Calcola la matrice di similarità
        scores = util.cos_sim(question_embeddings, answer_embeddings)

        # Per ogni domanda (riga della matrice)
        for i, question_id in enumerate(ids):
            # Estrai i punteggi della riga i
            row_scores = scores[i]

            # Costruisci lista di risposte con punteggi
            ranked_answers = sorted(
                [
                    {
                        'id': ids[j],
                        'answer': answers[j],
                        'score': float(row_scores[j])
                    }
                    for j in range(len(ids))
                ],
                key=lambda x: x['score'],
                reverse=True
            )

            # Applica top_k se richiesto
            if top_k:
                ranked_answers = ranked_answers[:top_k]

            # Salva i risultati per la domanda corrente
            results[question_id] = ranked_answers

        return results



