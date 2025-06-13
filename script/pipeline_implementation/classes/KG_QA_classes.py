import json
import numpy as np
import pandas as pd
from ast import literal_eval
from sentence_transformers import SentenceTransformer, util
from Script.Structure.Document import Document
from Script.Model.BERT import BERT_Model
from Script.Model.NER import NamedEntityExtractor
from Script.Model.EU import EntityUnifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Script.Varius import util1
import os
from collections import defaultdict, deque
from datetime import datetime


# Scarica/carica il modello una sola volta
model = SentenceTransformer('all-MiniLM-L6-v2')
vectorizer = TfidfVectorizer()

def are_same_concept_fast(entity1, entity2, threshold=0.4):

    entity1 = str(entity1)
    entity2 = str(entity2)
    threshold = float(threshold)

    if entity1 == entity2:  # early stop
        return True

    vectors = vectorizer.fit_transform([entity1, entity2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return similarity >= threshold

def are_same_concept(self, entity1: str, entity2: str, threshold: float = 0.8) -> bool:
    """
    Ritorna True se le due entità rappresentano lo stesso concetto.
    
    Args:
        entity1 (str): Prima entità
        entity2 (str): Seconda entità
        threshold (float): Soglia di similarità per considerare "uguali"

    Returns:
        bool: True se simili, False altrimenti
    """
    embeddings = model.encode([entity1, entity2], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    #print(f"Similarità: {similarity:.4f}")
    return similarity >= threshold




class EntityExtarction:

    def __init__(self, KG_pp=False, DB_small=False, KG_small=False, sim_threshold=0.7):

        self.NER = NamedEntityExtractor()

        if DB_small:

            self.DB_name="dataset/dataset_llama_small.json"


        else:
            self.DB_name="dataset/dataset_llama.json"


        if KG_pp:

            if KG_small:

                self.KG_name='/Users/marco/Documents/GitHub/ECAI2025/KG/unified_triples_small.json'

            else:

                self.KG_name='/Users/marco/Documents/GitHub/ECAI2025/KG/unified_triples.json'

        else:

            self.KG_name='/Users/marco/Documents/GitHub/ECAI2025/KG/triples.json'

        with open(self.DB_name, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.answer_real = [entry["answer"].strip() for entry in data.values()]


        self.KB_flat = self.load_kb_from_json(self.KG_name)


        self.answer_real_entities = [self.NER.get_entities(answer) for answer in self.answer_real]


    def load_kb_from_json(self, filepath):
    
        #Carica un file JSON contenente una lista di triple e le mette in self.KB_flat.
        
        #param filepath: Percorso del file JSON


        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(triple, list) and len(triple) == 3 for triple in data):
                    
                    print("✅ KB_flat caricato con successo.")

                    return data

                else:

                    raise ValueError("Il file JSON non è nel formato previsto: lista di triple [s, p, o].")

        except Exception as e:

            print(f"❌ Errore durante il caricamento del file JSON: {e}")





#Vecchia classe non più utilizzata. Pezzi di codice che possono servire. 

class KG_QA_Pipeline:

    def __init__(self, model_name, limit_ret=5, dataset_len=10, KG_pp=False, DB_small=False, KG_small=False):

        if DB_small:

            self.DB_name="dataset/dataset_llama_small.json"


        else:
            self.DB_name="dataset/dataset_llama.json"


        if KG_pp:

            if KG_small:

                self.KG_name='/Users/marco/Documents/GitHub/ECAI2025/KG/unified_triples_small.json'

            else:

                self.KG_name='/Users/marco/Documents/GitHub/ECAI2025/KG/unified_triples.json'

        else:

            self.KG_name='/Users/marco/Documents/GitHub/ECAI2025/KG/triples.json'


        self.model_name = model_name
        self.limit_ret = limit_ret
        self.dataset_len = dataset_len
        self.model = BERT_Model(model_name, SentenceTransformer)

        self.df_tot = None
        self.df_text = None
        self.answer_enc_dict = {}
        self.answer_enc = []
        self.question_enc = []
        self.question = []
        self.answer_real = []
        self.answer_real_entities=[]
        self.KB_flat = []

        self.answer_pred = []


        self.question_entities=[]

        self.NER = NamedEntityExtractor()
        

    def load_data(self):

        with open(self.DB_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.df_tot = pd.DataFrame(data)
        self.df_tot.rename(columns={"Paragrafo": "answer", "Risposta": "question"}, inplace=True)
        self.df_tot.reset_index(drop=True, inplace=True)

        print(self.df_tot.columns)

        #self.df_text = pd.read_csv("dataset/dataset_wikipedia_firenze.csv", sep=";")
        #self.df_text.reset_index(drop=True, inplace=True)
        self.KB_flat = self.load_kb_from_json(self.KG_name)

    def build_graph(self, KB):

        graph = defaultdict(set)
        for head, rel, tail in KB:
            graph[head].add(tail)
            graph[tail].add(head)  # bidirectional
        return graph

    def bfs_entities(self, graph, seeds, max_depth):

        visited = set(seeds)
        queue = deque([(ent, 0) for ent in seeds])
        result = set(seeds)

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def extract_entities(self, level=0):
        
        graph = self.build_graph(self.KB_flat)
        questions_entities = []
        answers_entities = []

        for i in range(len(self.answer_pred)):


            q_seeds = set()
            a_seeds = set()

            # Seed entities based on exact match in question and answer
            for head, rel, tail in self.KB_flat:
                if self.iterative_check_concept(head,self.question_entities[i]): q_seeds.add(head)
                if self.iterative_check_concept(tail,self.question_entities[i]): q_seeds.add(tail)

                if self.iterative_check_concept(head,self.answer_real_entities[i]): a_seeds.add(head)
                if self.iterative_check_concept(tail,self.answer_real_entities[i]): a_seeds.add(tail)

            # Traverse the graph from seeds
            q_ent = self.bfs_entities(graph, q_seeds, level)
            a_ent = self.bfs_entities(graph, a_seeds, level)

            questions_entities.append(list(q_ent))
            answers_entities.append(list(a_ent))

        return questions_entities, answers_entities

    def iterative_check_concept(self, head,question_entities):

        for question_entity in question_entities:

            if are_same_concept(head, question_entity):

                return True

        return False

  

    #Function that updatete the KB on json format

    def load_kb_from_json(self, filepath):
        """
        Carica un file JSON contenente una lista di triple e le mette in self.KB_flat.
        
        :param filepath: Percorso del file JSON
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(triple, list) and len(triple) == 3 for triple in data):
                    
                    print("✅ KB_flat caricato con successo.")

                    return data
                else:
                    raise ValueError("Il file JSON non è nel formato previsto: lista di triple [s, p, o].")
        except Exception as e:
            print(f"❌ Errore durante il caricamento del file JSON: {e}")



    def encode_texts(self):

        print(self.df_tot.columns)

        self.question_enc = self.model.text_emebedding(self.df_tot['question'].tolist())
        self.answer_enc = self.model.text_emebedding(self.df_tot['answer'].tolist())

        self.question = self.df_tot['question'].tolist()
        self.answer_real = self.df_tot['answer'].tolist()
        self.answer_pred = [a[0] for a in util1.order_candidate(
            util1.create_similarity_matrix(self.question_enc, self.answer_enc),
            self.df_tot['answer']
        )[1]]

        for i in range(len(self.answer_real)):
            self.answer_enc_dict[self.answer_real[i]] = self.answer_enc[i]


        self.answer_real_entities = [self.NER.get_entities(answer) for answer in self.answer_real]
        self.question_entities = [self.NER.get_entities(question) for question in self.question]





    def generate_candidates(self, questions_entities, answers_entities):
        candidates_KG = []
        candidates_KG_enc = []

        candidates_KG_pure = []

        for i in range(len(questions_entities)):
            candidate_KG = [self.answer_pred[i]]
            candidate_KG_enc = [self.answer_enc_dict[self.answer_pred[i]]]

            for ent in questions_entities[i]:
                for k in range(len(answers_entities)):
                    for a_ent in answers_entities[k]:
                        if ent == a_ent and self.answer_real[k] not in candidate_KG:
                            candidate_KG.append(self.answer_real[k])
                            candidate_KG_enc.append(self.answer_enc_dict[self.answer_real[k]])

            candidates_KG_pure.append(candidate_KG)

            if len(candidate_KG) < self.limit_ret:

                candidate_KG = self.df_tot['answer'].tolist()
                candidate_KG_enc = self.answer_enc

            candidates_KG.append(candidate_KG)
            candidates_KG_enc.append(candidate_KG_enc)

        return candidates_KG, candidates_KG_enc, candidates_KG_pure



    def rank_candidates(self, candidates_KG, candidates_KG_enc):
        scores_similarity = util1.create_similarity_matrix_KB(self.question_enc, candidates_KG_enc)

        possible_candidates_ordered = []
        for i in range(len(scores_similarity)):
            list_score, list_candidate = zip(*sorted(zip(scores_similarity[i], candidates_KG[i]), reverse=True))
            possible_candidates_ordered.append(list(dict.fromkeys(list_candidate)))

        return possible_candidates_ordered

    def run(self):
        print(f"Running for model: {self.model_name}, limit_ret: {self.limit_ret}, dataset_len: {self.dataset_len}")

        data_ora = datetime.now().strftime("%d/%m/%Y %H:%M")

        results = []

        #print("Baseline Evaluation:")
        scores = util1.create_similarity_matrix(self.question_enc, self.answer_enc)
        _, candidates_baseline_order = util1.order_candidate(scores, self.df_tot['answer'])
        metrics = util1.evaluation_function(candidates_baseline_order, self.df_tot['answer'])
        results.append({
            "model": self.model_name,
            "limit_ret": self.limit_ret,
            "dataset_len": self.dataset_len,
            "method": "baseline",
            "dataset_name": self.DB_name,
            "KG_type": self.KG_name,
            "distance_level": -1,
            "@10": "-1",
            "Threshold": "None",
            "data_ora": data_ora,
            "avg_candidate_len": len(self.df_tot['answer']),
            **metrics
        })

        for level in [3, 4, 5]:


            print("⚙️ Level "+str(level)+ " starting caculation")
            #print(f"\nKnowledge Graph Evaluation (Distance Level {level}):")
            q_ents, a_ents = self.extract_entities(level)
            candidates_KG, candidates_KG_enc, candidates_KG_pure = self.generate_candidates(q_ents, a_ents)
            ranked = self.rank_candidates(candidates_KG, candidates_KG_enc)
            metrics = util1.evaluation_function(ranked, self.df_tot['answer'])
            print("🎯 Level "+str(level)+ " calculated")

            avg_len = sum(len(c) for c in candidates_KG_pure) / len(candidates_KG_pure)


            vet=[len(i) for i in candidates_KG_pure]
            vet1=[i for i in vet if i < 5]
            results.append({
                "model": self.model_name,
                "limit_ret": self.limit_ret,
                "dataset_len": self.dataset_len,
                "method": "KG",
                "dataset_name": self.DB_name,
                "KG_type": self.KG_name,
                "distance_level": level,
                "@10": len(vet1),
                "Threshold": self.threshold,
                "data_ora": data_ora,
                "avg_candidate_len": avg_len,
                **metrics
            })




        df_new_result = pd.DataFrame(results)

        # Percorso del file risultati
        results_file = "risultati_kgqa.csv"

        # Se esiste già, lo carica e concatena i nuovi risultati
        if os.path.exists(results_file):
            df_existing = pd.read_csv(results_file)
            df_results = pd.concat([df_existing, df_new_result], ignore_index=True)
        else:
            df_results = df_new_result

        # Rimuove eventuali duplicati (opzionale)
        df_results.drop_duplicates(inplace=True)

        # Salva il risultato aggiornato
        df_results.to_csv(results_file, index=False)

        print("\nFinal Evaluation Results:")
        print(df_new_result.to_string(index=False))

        return df_new_result




