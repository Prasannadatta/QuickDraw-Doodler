import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional

class RecurDropLayerNormLSTMCell(nn.Module):
    '''
    Custom LSTM Cell similar to Pytorch LSTM,
    but with added layer norm and recurrent dropout added.
    '''
    def __init__(self, input_size, hidden_size, use_layer_norm, recurrent_dropout=0.1):
        super(RecurDropLayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout
        self.use_layer_norm = use_layer_norm
        
        # initialize lstm gates
        #self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        #self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        #self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        #self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # use one fc layer for all gates
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=True)
        
        # layer normalization layer
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
            self,
            x: torch.Tensor,
            hidden: Tuple[torch.Tensor, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Using type enforcement for torch.jit.script to work'''

        # hidden and cell states from previous step
        cur_hidden, cur_cell = hidden # (batch_size, hidden_size)
        combined = torch.cat((x, cur_hidden), dim=1) # (batch_size, input_size + hidden_size)
        
        # gate activations, all output size (batch_size, hidden_size)
        #input_t = torch.sigmoid(self.input_gate(combined)) # controls influence of input on cell state
        #forget_t = torch.sigmoid(self.forget_gate(combined)) # how much previous cell state should be retained
        #output_t = torch.sigmoid(self.output_gate(combined)) # controls influence of current cell state on hidden state
        #gen_candidate_t = torch.tanh(self.cell_gate(combined)) # how much current cell state influences hidden state
        
        # single layer forr all gates for efficiency
        gate_values = self.gates(combined)
        input_t, forget_t, output_t, gen_candidate_t = gate_values.chunk(4, dim=1)

        input_t = torch.sigmoid(input_t)
        forget_t = torch.sigmoid(forget_t)
        output_t = torch.sigmoid(output_t)
        gen_candidate_t = torch.tanh(gen_candidate_t)

        # compute new cell state using gate outputs of combined input and hidden values
        cell_update_t = input_t * gen_candidate_t + forget_t * cur_cell 

        # compute new hidden state 
        if self.use_layer_norm:
            norm_cell_t = self.layer_norm(cell_update_t) # apply layer norm to cell state only for hidden update
            hidden_update_t = output_t * torch.tanh(norm_cell_t)
        else:
            hidden_update_t = output_t * torch.tanh(cell_update_t)
        
        # apply recurrent dropout only during training
        if self.training:
            hidden_update_t = F.dropout(hidden_update_t, p=self.recurrent_dropout, training=self.training)
        
        # return original cell update, not the layer normalized one
        return hidden_update_t, cell_update_t

class RecurDropLayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, recurrent_dropout=0.1, use_layer_norm=True):
        super(RecurDropLayerNormLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # assemble lstm cells
        # first cell in_size is data input size
        cell_list = [RecurDropLayerNormLSTMCell(input_size, hidden_size, recurrent_dropout, use_layer_norm)]
        for _ in range(1, num_layers): # subsequent layers have input size the hidden size from previous layer
            cell_list.append(RecurDropLayerNormLSTMCell(hidden_size, hidden_size, recurrent_dropout, use_layer_norm))
        
        # assemble list of cells into a module (connect them)
        self.cells = nn.ModuleList(cell_list)
        
    def forward(
            self,
            x: torch.Tensor,
            hidden: List[Tuple[torch.Tensor, torch.Tensor]]
        ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        '''Using type enforcement for torch.jit.script to work'''

        if hidden is None:
            # init hidden and cell states for hidden layer with 0s if not given
            # will be properly initialized in vae class
            hidden = []
            for _ in range(self.num_layers): # subsequent layers have input size the hidden size from previous layer
                h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
                c = torch.zeros(x.size(0), self.hidden_size, device=x.device)
                hidden.append((h, c))

        # rnn sequences flow better and have less computational overhead when batch_size second
        # i.e. can loop through sequence time steps as the outer index and get all data across batch for each time step
        if self.batch_first: # if data is passed in with shape (batch_size, seq_len, input_size)
            x = x.transpose(0,1) # transpose to shape (seq_len, batch_size, input_size)
        
        seq_len, batch_size, _ = x.size()
        
        # process sequence where hidden is a list of tuples of hidden, cell states
        # [(h_0, c_0), (h_1, c_1), ...]
        outputs = []
        for t in range(seq_len): # iterate over each time step (sequence point)
            input_t = x[t] # (batch_size, input_size)
            hidden_update: List[Tuple[torch.Tensor, torch.Tensor]] = [] # updated hidden states from each cell
            for layer_idx, cell in enumerate(self.cells): # pass sequence point through all lstm cells
                # give input and hidden and cell states for time step t, layer i
                h_ti_update, c_ti_update = cell.forward(input_t, hidden[layer_idx])
                
                # next input to lstm cell is the new hidden state
                input_t = h_ti_update
                hidden_update.append((h_ti_update, c_ti_update))

            hidden = hidden_update
            outputs.append(input_t)
        
        outputs = torch.stack(outputs, dim=0)

        if self.batch_first: # if we had to tranpose for batch_first=True,
            # we need to transpose outputs back after processing
            outputs = outputs.transpose(0,1) # (batch_size, seq_len, input_size)

        return outputs, hidden
