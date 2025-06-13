
import pandas as pd
from string import Template
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import uuid
import datetime

class RDFMappable(ABC):
    """Abstract base class for objects that can be mapped to RDF."""
    @abstractmethod
    def to_rdf(self, graph: Graph) -> URIRef:
        pass

    def get_uri(self) -> URIRef:
        """Generate a unique URI for this instance."""
        return CI[f"{self.__class__.__name__}_{str(uuid.uuid4())}"]

# Response Type Enum
class ResponseType(Enum):
    REFUSAL = "refusal"
    GENERATED = "generated"
    MIXED = "mixed"

    
@dataclass(kw_only=True)
class LLMResponse(RDFMappable):
    """Base class for all LLM responses."""
    prompt_id: str
    raw_text: str = field(default="")
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    response_type: ResponseType = field(default=ResponseType.GENERATED)
    
    def to_rdf(self, graph: Graph) -> URIRef:
        uri = self.get_uri()
        graph.add((uri, RDF.type, CI.LLMResponse))
        graph.add((uri, CI.promptId, Literal(self.prompt_id)))
        graph.add((uri, CI.timestamp, Literal(self.timestamp)))
        graph.add((uri, CI.responseType, Literal(self.response_type.value)))
        graph.add((uri, CI.rawText, Literal(self.raw_text)))
        return uri

class PromptCreation:


    def __init__(self, few_shot=True, path_prompt="/Users/marco/Documents/GitHub/FrameworkArgomentativo/LLM_argument_mining/dataset_input/test_prompts.csv", path_samples="/Users/marco/Documents/GitHub/FrameworkArgomentativo/LLM_argument_mining/dataset_input/samples_fewshot.csv"):

        self.config=pd.read_csv(path_prompt, sep=";")
        self.samples=pd.read_csv(path_samples, sep=",")

        self.few_shot=few_shot



    def prompt_creation(self):

        #this function create the prompt according the parameters provide:

        #prompt_method=["Baseline", "CoT", "CARP"]
        #samples= list of sample in the form ["label","text"]
        #language=["Eng","Ita","Spa"]
        #task=["text_postion_classification"]

        row = self.config

        prompt=row['Context']+" "+row['Question']+" "+row['Output_format']+ "\n\n"

        if (self.few_shot):
            
            for i, row1 in self.samples.iterrows():

                prompt+="Input post: "+row1["post"]+"\n"
                prompt+="Expected output: "+row1["implied_statement"]+"\n\n"
              

        
        prompt+="Input post: {}"+"\n"
        prompt+="Expected output: "+"\n"

        self.prompt_output=prompt
        
        return prompt











