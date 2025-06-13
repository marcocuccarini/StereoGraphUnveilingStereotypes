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
# Scarica/carica il modello una sola volta
model = SentenceTransformer('all-MiniLM-L6-v2')
vectorizer = TfidfVectorizer()
from pyvis.network import Network


def are_same_concept_fast(entity1, entity2, threshold=0.4):

    entity1 = str(entity1)
    entity2 = str(entity2)
    threshold = float(threshold)

    if entity1 == entity2:  # early stop
        return True

    vectors = vectorizer.fit_transform([entity1, entity2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return similarity >= threshold

def are_same_concept(entity1: str, entity2: str, threshold: float = 0.7) -> bool:
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


import json
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self):
        self.nodes = dict()  # key: node_id, value: text
        self.edges = dict()  # key: (node_id1, node_id2), value: 1 or -1

    def add_node(self, node_id, text):
        #if node_id in self.nodes:
        #    raise ValueError(f"Node ID '{node_id}' already exists.")
        self.nodes[node_id] = text

    def add_edge(self, node_id1, node_id2, value):
        if value not in (1, 0):
            raise ValueError("Edge value must be 1 or -1")
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            raise ValueError("Both node IDs must be present in the graph")
        self.edges[(node_id1, node_id2)] = value

    def get_relation(self, node_id1, node_id2):
        return self.edges.get((node_id1, node_id2))

    def get_text(self, node_id):
        return self.nodes.get(node_id)

    def get_relations_as_list(self):
        return [
            {
                "from_id": node1,
                "from_text": self.get_text(node1),
                "to_id": node2,
                "to_text": self.get_text(node2),
                "value": value
            }
            for (node1, node2), value in self.edges.items()
        ]

    def print_relations_as_json(self, indent=2):
        relations = self.get_relations_as_list()
        print(json.dumps(relations, indent=indent))
        return relations

    def save_relations_to_json(self, filename):
        relations = self.get_relations_as_list()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
        print(f"Relations saved to '{filename}' ✅")

    def draw_graph_html(self, filename='graph.html'):
        net = Network(directed=True, notebook=False)

        # Aggiunge i nodi
        for node_id, text in self.nodes.items():
            label = text if len(text) < 50 else f"{node_id}: {text[:47]}..."
            net.add_node(node_id, label=label, title=text, shape='box')

        # Aggiunge gli archi
        for (node1, node2), value in self.edges.items():
            color = 'green' if value == 1 else 'red'
            label = f"+1" if value == 1 else "-1"
            net.add_edge(node1, node2, color=color, label=label, arrows='to')

        net.set_options("""
        var options = {
          "edges": {
            "color": {
              "inherit": false
            },
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "horizontal",
              "roundness": 0.4
            }
          },
          "physics": {
            "stabilization": true
          }
        }
        """)
        net.show(filename)
        print(f"Grafo interattivo salvato in '{filename}' 🌐")

    def draw_graph(self, filename='graph.png'):
        G = nx.DiGraph()  # Directed graph

        # Add nodes with labels
        for node_id, text in self.nodes.items():
            G.add_node(node_id, label=text)

        # Add edges with relation values
        for (node1, node2), value in self.edges.items():
            color = 'green' if value == 1 else 'red'
            G.add_edge(node1, node2, color=color, label=str(value))

        # Node labels: use text if short, else just ID
        labels = {node_id: text if len(text) < 30 else node_id for node_id, text in self.nodes.items()}

        pos = nx.spring_layout(G)  # nice layout
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightblue',
                edge_color=edge_colors, node_size=1500, font_size=8, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
        plt.title("Graph Relations")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Graph image saved to '{filename}' 📸")

class Pseudo_KG_creation:

    def __init__(self, passages_entities, passages, KG, sim_threshold=0.8):
        self.passages_entities = {i: val for i, val in enumerate(passages_entities)}
        self.passages = {i: val for i, val in enumerate(passages)}
        self.KG = KG
        self.pseudo_graph = Graph()
        self.dict_equal={}
        self.sim_threshold=sim_threshold

    def compare_two_entity_vec(self, set_A, set_B):
        for x in set_A:
            for y in set_B:
                if are_same_concept(x, y):
                    return True
        return False

    def html_graph(self,filename):

        self.pseudo_graph.draw_graph_html(filename)

    def print_graph(self, filename):

        self.pseudo_graph.draw_graph(filename)

    def print_graph_as_json(self, filename):

        self.pseudo_graph.save_relations_to_json(filename)

    def compare_two_entity_vec_deep(self, set_A, set_B, KB):
        for ent1 in set_A:
            for ent2 in set_B:
                for row in KB:
                
                    if (ent1, row[0]) not in self.dict_equal:

                        self.dict_equal[(ent1, row[0])] = are_same_concept(ent1, row[0], self.sim_threshold)
                        self.dict_equal[(row[0], ent1)] = self.dict_equal[(ent1, row[0])]

                    if (ent2, row[2]) not in self.dict_equal:

                        self.dict_equal[(ent2, row[2])] = are_same_concept(ent2, row[2], self.sim_threshold)
                        self.dict_equal[(row[2], ent2)] = self.dict_equal[(ent2, row[2])]

                    if (ent2, row[0]) not in self.dict_equal:

                        self.dict_equal[(ent2, row[0])] = are_same_concept(ent2, row[0], self.sim_threshold)
                        self.dict_equal[(row[0], ent2)] = self.dict_equal[(ent2, row[0])]

                    if (ent1, row[2]) not in self.dict_equal:

                        self.dict_equal[(ent1 , row[2])] = are_same_concept(ent1, row[2], self.sim_threshold)
                        self.dict_equal[(row[2] , ent1)] = self.dict_equal[(ent1, row[2])]

                     
                    if  self.dict_equal[(ent1, row[0])] and self.dict_equal[(ent2, row[2])]:

                        return True

                    if self.dict_equal[(ent2, row[0])] and self.dict_equal[(ent1, row[2])]:

                        return True
        return False

    def _safe_add_nodes_and_edge(self, i, j, edge_value):
        if i not in self.pseudo_graph.nodes:
            self.pseudo_graph.add_node(i, self.passages[i])
        if j not in self.pseudo_graph.nodes:
            self.pseudo_graph.add_node(j, self.passages[j])
        self.pseudo_graph.add_edge(i, j, edge_value)

    def create_graph(self):
        processed_pairs = set()

        for i in self.passages_entities.keys():
            print(f"Index {i}")
            entities_i = self.passages_entities[i]

            for j in self.passages_entities.keys():
                if j <= i:
                    continue  # evita confronti duplicati e self-comparison

                if (i, j) in processed_pairs or (j, i) in processed_pairs:
                    continue
                processed_pairs.add((i, j))

                entities_j = self.passages_entities[j]

                # Verifica similarità superficiale


                # Verifica connessione profonda tramite KG
                if self.compare_two_entity_vec_deep(entities_i, entities_j, self.KG):
                    self._safe_add_nodes_and_edge(i, j, 1)

                if self.compare_two_entity_vec(entities_i, entities_j):
                    self._safe_add_nodes_and_edge(i, j, 0)
