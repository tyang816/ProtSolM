import torch
import gc
import torch.nn as nn
from torch_geometric.data import Batch, Dataset
from transformers import AutoTokenizer, EsmModel
from typing import *
from src.module.egnn.network import EGNN

class PLM_model(nn.Module):
    possible_amino_acids = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
        'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]
    one_letter = {
        'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
        'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',
        'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
        'GLY':'G', 'PRO':'P', 'CYS':'C'
        }
    
    def __init__(self, args):
        super().__init__()
        # load global config
        self.args = args
        
        # esm on the first cuda
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.plm)
        self.model = EsmModel.from_pretrained(self.args.plm).cuda()
        
        
    def forward(self, batch):
        with torch.no_grad():
            if not isinstance(batch, List):
                batch = [batch]
            # get the target sequence
            one_hot_truth_seqs = [elem.y for elem in batch]
            aa_seqs = ["".join([self.one_letter[self.possible_amino_acids[idx]] for idx in seq_idx]) for seq_idx in one_hot_truth_seqs]

            batch_graph = self._nlp_inference(aa_seqs, batch)
        return batch_graph

    
    
    @torch.no_grad()
    def _nlp_inference(self, input_seqs, batch):    
        inputs = self.tokenizer(input_seqs, return_tensors="pt", padding=True).to("cuda:0")
        batch_lens = (inputs["attention_mask"] == 1).sum(1) - 2
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        for idx, (hidden_state, seq_len) in enumerate(zip(last_hidden_states, batch_lens)):
            batch[idx].esm_rep = hidden_state[1: 1+seq_len]
            del batch[idx].seq
                
        # move to the GNN devices
        batch = [elem.cuda() for elem in batch]
        batch_graph = Batch.from_data_list(batch)
        gc.collect()
        torch.cuda.empty_cache()
        return batch_graph



class GNN_model(nn.Module):    
    def __init__(self, args):
        super().__init__()
        # load graph network config which usually not change
        self.gnn_config = args.gnn_config
        # load global config
        self.args = args
        
        # calculate input dim according to the input feature
        self.out_dim = 20
        self.input_dim = self.args.plm_hidden_size
        
        # gnn on the rest cudas
        self.GNN_model = EGNN(self.gnn_config, self.args, self.input_dim, self.out_dim).cuda()

    def forward(self, batch_graph):
        gnn_out = self.GNN_model(batch_graph)
        return gnn_out
    
