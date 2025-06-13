import json
import numpy as np
import pandas as pd
from ast import literal_eval
from sentence_transformers import SentenceTransformer, util
from Script.Structure.Document import Document





from sklearn.metrics.pairwise import cosine_similarity
from Script.Varius import util1
import os
from collections import defaultdict, deque
from datetime import datetime



class Pipeline:

    def __init__(self, model, dataset="dataset_llama.json", graph="pseudo-graph_BERT.json"):

        with open("/Users/marco/Documents/GitHub/ECAI2025/dataset/"+dataset, "r", encoding="utf-8") as f:
            
            data = json.load(f)

        self.dataset = data

        with open("/Users/marco/Documents/GitHub/ECAI2025/KG/"+graph, "r", encoding="utf-8") as f:

            graph = json.load(f)

        self.graph = graph
        '''                                     
                                                {
              graph is a list of                "from_id": 
              dictionary                        "from_text"
                                                "to_id":
                                                "to_text":
                                                }
        '''
        self.model=model



    def baseline(self):

        self.results=self.model.get_results(self.dataset)

        '''the results dictionaty contain all the candidate of all the possible question:
        dict[id_question] -----> {id_question0: [{'id': '261', 'answer': ' The ground plan ...', 'score': 0.3}, ....], 
                                  id_question1: [{'id': '11', 'answer': ' The ...', 'score': 0.5}, ...., ...] ...}'''

        return self.results


    def extract_candidates(self, n_levels, top_k):
        results_level = {}
        seen_global = {}

        # Build an adjacency list for fast neighbor lookups
        adjacency = {}
        for edge in self.graph:
            from_id = str(edge['from_id'])
            to_id = str(edge['to_id'])
            adjacency.setdefault(from_id, set()).add(to_id)
            adjacency.setdefault(to_id, set()).add(from_id)

        for level in range(n_levels):
            level_dict = {}

            for question_key in self.results:
                if question_key not in seen_global:
                    seen_global[question_key] = set()

                # Initialize candidates at level 0
                if level == 0:
                    top_k_candidates = list(self.results[question_key])[:top_k]
                    id_candidates = [str(item['id']) for item in top_k_candidates]
                    seen_global[question_key].update(id_candidates)
                else:
                    # Use previously seen candidates to expand
                    id_candidates = results_level[level - 1].get(question_key, [])
                    id_candidates = [str(item['id']) for item in id_candidates]

                new_candidates = set()

                # Expand each candidate through the graph
                for candidate_id in id_candidates:
                    neighbors = adjacency.get(candidate_id, set())
                    for neighbor_id in neighbors:
                        if neighbor_id not in seen_global[question_key]:
                            new_candidates.add(neighbor_id)

                # Update the seen set with new candidates
                seen_global[question_key].update(new_candidates)

                # Filter available items from original results using updated seen set
                all_seen_ids = seen_global[question_key]
                filtered_data = [
                    item for item in self.results[question_key]
                    if str(item['id']) in all_seen_ids
                ]

                level_dict[question_key] = filtered_data

            results_level[level] = level_dict

        return results_level


    def get_correct_answer_ranks(self, data):
        """
        Handles format: {question_id: [list of answers]}
        Assumes correct answer has ID == str(question_id)
        """
        rankings = {}

        for question_id, answers in data.items():
            # Sort by score descending
            sorted_answers = sorted(answers, key=lambda x: x['score'], reverse=True)
            correct_id = str(question_id)

            for rank, answer in enumerate(sorted_answers, start=1):
                if answer['id'] == correct_id:
                    rankings[question_id] = rank
                    break
            else:
                rankings[question_id] = None

        return rankings


    def compute_ir_metrics(self, rankings, k_values=[1, 5, 10, 20, 50]):
        """
        Computes standard IR metrics from a dictionary of ranks.

        Parameters:
            rankings (dict): {question_id: rank_of_correct_answer}
            k_values (list): List of cutoff values for Hits@K

        Returns:
            dict: Contains MRR, Hits@K for each K in k_values
        """
        num_samples = len(rankings)
        reciprocal_ranks = []
        hits_at_k = {k: 0 for k in k_values}

        for rank in rankings.values():
            if rank is None:
                reciprocal_ranks.append(0.0)
            else:
                reciprocal_ranks.append(1.0 / rank)
                for k in k_values:
                    if rank <= k:
                        hits_at_k[k] += 1

        # Final metrics
        mrr = sum(reciprocal_ranks) / num_samples
        hits_at_k = {f"Hits@{k}": hits / num_samples for k, hits in hits_at_k.items()}

        metrics = {"MRR": mrr}
        metrics.update(hits_at_k)

        return metrics





