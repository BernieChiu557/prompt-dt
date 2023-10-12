import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch, Data, HeteroData


def get_structural_mask(agent, max_length, prompt_length):
    num_node = agent['num_node']
    # single timestep structural_edge
    structural_edge = torch.LongTensor(agent['edge'])
    # spatial edge across full timestep
    full_structural_edge = structural_edge.repeat(max_length + prompt_length, 1)
    arange = torch.arange(0, num_node * (max_length + prompt_length), step=num_node).view(-1, 1).repeat(1, len(structural_edge)).view(-1, 1)
    full_structural_edge = full_structural_edge + arange
    structural_mask = torch.ones(num_node * (max_length + prompt_length), num_node * (max_length + prompt_length)).to(torch.bool)
    for edge in full_structural_edge:
        structural_mask[edge[0], edge[1]] = False
        structural_mask[edge[1], edge[0]] = False # add reverse direction edge

    return structural_mask

def get_temporal_mask_list(num_node, max_length, prompt_length):
    temporal_masks = []
    for num_zero in range(max_length):
        temporal_edge = torch.tril(torch.ones(max_length + prompt_length, max_length + prompt_length), diagonal=0)
        temporal_edge[:, prompt_length:prompt_length+num_zero] = 0
        temporal_edge[prompt_length:prompt_length+num_zero, :] = 0
        temporal_edge = temporal_edge.nonzero()*num_node
        full_temporal_edge = torch.cat([temporal_edge+i for i in range(num_node)], dim=0)
        temporal_mask = torch.ones(num_node * (max_length + prompt_length), num_node * (max_length + prompt_length)).to(torch.bool)
        for edge in full_temporal_edge:
            temporal_mask[edge[0], edge[1]] = False
        temporal_mask[:num_node * prompt_length, :num_node * prompt_length] = True
        
        temporal_masks.append(temporal_mask)

    return temporal_masks


class GPDT_V2_Torch(nn.Module):
    def __init__(self, agent, max_length, max_ep_length, prompt_length, hidden_size, n_layers, dropout=0.1, num_heads=1):
        super().__init__()

        self.agent = agent
        self.num_node = agent['num_node']
        self.node_list = list(agent['state_position'].keys())
        self.max_length = max_length
        self.max_ep_length = max_ep_length
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size

        mask_store = []
        structural_mask = get_structural_mask(agent, max_length, prompt_length)
        temporal_mask_list = get_temporal_mask_list(self.num_node, max_length, prompt_length)
        for temporal_mask in temporal_mask_list:
            mask = ~(~structural_mask + ~temporal_mask)
            # mask = torch.logical_and(structural_mask, temporal_mask)
            mask_store.append(mask)
        mask_store = torch.stack(mask_store, dim=0).to(torch.bool)
        self.register_buffer('mask_store', mask_store)

        # create embed and predict net
        self.embed_net = nn.ModuleDict()
        self.predict_net = nn.ModuleDict()
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_length, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # prompt net
        self.prompt_embed_timestep = nn.Embedding(max_ep_length, hidden_size)
        self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
        self.prompt_embed_net = nn.ModuleDict()

        node_types = agent['node_type_state_action_len'].keys()
        for node_type in node_types:
            state_action_size = agent['node_type_state_action_len'][node_type]
            self.embed_net.add_module(node_type, nn.Linear(state_action_size, hidden_size))
            self.prompt_embed_net.add_module(node_type, nn.Linear(state_action_size, hidden_size))
            if node_type != 'root':
                action_size = len(agent['action_position'][self.node_list[1]])
                self.predict_net.add_module(node_type, nn.Sequential(*([nn.Linear(hidden_size, action_size)] + ([nn.Tanh()]))))

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size*4, dropout=dropout, layer_norm_eps=1e-05, batch_first=True)
        self.torch_graph_transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        states = states[:, :, :self.agent['state_dim']]
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        # discard last timestep, equivalent to shift 1 timestep
        actions = actions[:, :-1, :]

        # construct 1 graph for 1 timestep of a agent
        node_embeddings = []
        for node in self.node_list:
            node_states = states[:, :, self.agent['state_position'][node]]
            node_actions = actions[:, :, self.agent['action_position'][node]]
            node_state_action = torch.cat([node_states, node_actions], dim=2)
            node_embeddings.append(self.embed_net[self.agent['node_type'][node]](node_state_action))
        # [batch_size, seq_length, num_node, hidden_size]
        node_embeddings = torch.stack(node_embeddings, dim=2)

        time_embeddings = self.embed_timestep(timesteps)
        time_embeddings = time_embeddings.repeat(1, 1, self.num_node).reshape(batch_size, seq_length, self.num_node, self.hidden_size)

        rtg_embeddings = self.embed_return(returns_to_go)
        rtg_embeddings = rtg_embeddings.repeat(1, 1, self.num_node).reshape(batch_size, seq_length, self.num_node, self.hidden_size)

        # [batch_size, seq_length, num_node, hidden_size]
        transformer_inputs = node_embeddings + rtg_embeddings + time_embeddings
        transformer_inputs = self.embed_ln(transformer_inputs)

        # process prompt the same
        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            
            # I wonder if there's a case where prompt mask has 0 inside?
            assert prompt_attention_mask.min() == 1

            prompt_states = prompt_states[:, :, :self.agent['state_dim']]
            prompt_seq_length = prompt_states.shape[1]
            prompt_actions = prompt_actions[:, :-1, :]

            # construct 1 graph for 1 timestep of a agent
            prompt_node_embeddings = []
            for node in self.node_list:
                node_states = prompt_states[:, :, self.agent['state_position'][node]]
                node_actions = prompt_actions[:, :, self.agent['action_position'][node]]
                node_state_action = torch.cat([node_states, node_actions], dim=2)
                prompt_node_embeddings.append(self.prompt_embed_net[self.agent['node_type'][node]](node_state_action))
            # [batch_size, prompt_seq_length, num_node, hidden_size]
            prompt_node_embeddings = torch.stack(prompt_node_embeddings, dim=2)

            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)
            prompt_time_embeddings = prompt_time_embeddings.repeat(1, 1, self.num_node).reshape(batch_size, prompt_seq_length, self.num_node, self.hidden_size)

            prompt_rtg_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_rtg_embeddings = prompt_rtg_embeddings.repeat(1, 1, self.num_node).reshape(batch_size, prompt_seq_length, self.num_node, self.hidden_size)

            # [batch_size, prompt_seq_length, num_node, hidden_size]
            prompt_transformer_inputs = prompt_node_embeddings + prompt_rtg_embeddings + prompt_time_embeddings
            prompt_transformer_inputs = self.embed_ln(prompt_transformer_inputs)

            transformer_inputs = torch.cat((prompt_transformer_inputs, transformer_inputs), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)
        
        mask_batch = []
        for mask_ in attention_mask:
            num_zeros = len(mask_) - int(mask_.sum())
            mask_batch.append(self.mask_store[num_zeros].clone())

        mask_batch = torch.stack(mask_batch, dim=0)
        if mask_batch.shape[0] == 1:
            mask_batch = mask_batch.reshape(mask_batch.shape[1], mask_batch.shape[2])

        transformer_inputs = transformer_inputs.reshape(batch_size, -1, self.hidden_size)

        breakpoint()
        transformer_outputs = self.torch_graph_transformer(transformer_inputs, mask=mask_batch)
        breakpoint()
        transformer_outputs = transformer_outputs.reshape([batch_size, -1, self.num_node, self.hidden_size])

        # predict action condition on previous state_rtg token
        action_preds = []
        for idx, node in enumerate(self.node_list):
            if node != 'root':
                action_pred = self.predict_net[self.agent['node_type'][node]](transformer_outputs[:, :, idx, :])
                action_preds.append(action_pred)

        action_preds = torch.cat(action_preds, dim=2)

        # we don't predict state and reward
        state_preds, reward_preds = 0.0, 0.0

        return state_preds, action_preds[:, -seq_length:, :], reward_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        states = states[:, :self.agent['state_dim']]
        states = states.reshape(1, -1, self.agent['state_dim'])
        actions = actions.reshape(1, -1, self.agent['act_dim'])
        # pad a action in front
        actions = torch.cat([torch.ones((1, 1, self.agent['act_dim']), device=actions.device) * -10., actions], dim=1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length-1:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.agent['state_dim']), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length + 1 - actions.shape[1], self.agent['act_dim']), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=True):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm

        self.graph_attention = pyg_nn.TransformerConv(in_dim, out_dim, heads=num_heads, bias=use_bias)

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)          
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)    
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        x_in1 = x # for first residual connection
        
        x = self.graph_attention(x, edge_index)
        x = x.view(-1, self.out_channels) # what is this?
        x = F.dropout(x, self.dropout, training=self.training)     
        x = self.O(x)
        
        if self.residual:
            x = x_in1 + x # residual connection
        
        if self.layer_norm:
            x = self.layer_norm1(x)    
        if self.batch_norm:
            x = self.batch_norm1(x)
        
        x_in2 = x # for second residual connection
        
        # FFN
        x = self.FFN_layer1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.FFN_layer2(x)

        if self.residual:
            x = x_in2 + x # residual connection
        
        if self.layer_norm:
            x = self.layer_norm2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)       

        return x

class GPDT_V2(nn.Module):
    def __init__(self, agent, max_length, max_ep_length, prompt_length, hidden_size, n_layers, dropout=0.1, num_heads=1, library='pyg'):
        super().__init__()

        self.agent = agent
        self.max_length = max_length
        self.max_ep_length = max_ep_length
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.num_node = self.agent['num_node']
        self.node_list = list(self.agent['state_position'].keys())
        self.library = library

        # create structural_edges
        # single timestep structural_edge
        structural_edge = torch.LongTensor(self.agent['edge'])
        structural_edge = torch.cat((structural_edge, torch.flip(structural_edge, [1])), dim=0) # add reverse direction edge
        # spatial edge across full timestep
        full_structural_edge = structural_edge.repeat(max_length + prompt_length, 1)
        arange = torch.arange(0, self.num_node * (max_length + prompt_length), step=self.num_node).view(-1, 1).repeat(1, len(structural_edge)).view(-1, 1)
        full_structural_edge = full_structural_edge + arange
        if library == 'pyg':
            self.register_buffer('structural_edge', full_structural_edge.T.contiguous()) # register to self, [2, num_node*single_space_edge_len]
        elif library == 'torch':
            structural_mask = torch.ones(self.num_node * (max_length + prompt_length), self.num_node * (max_length + prompt_length)).to(torch.bool)
            for edge in full_structural_edge:
                structural_mask[edge[0], edge[1]] = False
            self.register_buffer('structural_mask', structural_mask.contiguous())
        else:
            print(f'library {library} not implemented')
            raise NotImplementedError

        # create embed and predict net
        self.embed_net = nn.ModuleDict()
        self.predict_net = nn.ModuleDict()
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_timestep = nn.Embedding(self.max_ep_length, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # prompt net
        self.prompt_embed_timestep = nn.Embedding(self.max_ep_length, hidden_size)
        self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
        self.prompt_embed_net = nn.ModuleDict()

        # root node
        state_size = len(self.agent['state_position']['root'])
        self.embed_net.add_module('root', nn.Linear(state_size, self.hidden_size))
        self.prompt_embed_net.add_module('root', nn.Linear(state_size, self.hidden_size))

        # non root node
        state_size = len(self.agent['state_position'][self.node_list[1]])
        action_size = len(self.agent['action_position'][self.node_list[1]])
        self.embed_net.add_module('nonroot', nn.Linear(state_size + action_size, self.hidden_size))
        self.prompt_embed_net.add_module('nonroot', nn.Linear(state_size + action_size, self.hidden_size))
        self.predict_net.add_module('nonroot', nn.Sequential(*([nn.Linear(self.hidden_size, action_size)] + ([nn.Tanh()]))))

        if library == 'pyg':
            self.pyg_graph_transformer = nn.ModuleList([Block(self.hidden_size, self.hidden_size, num_heads=num_heads, dropout=dropout, layer_norm=True, batch_norm=False) for _ in range(n_layers)])
        elif library == 'torch':
            encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, num_heads, dim_feedforward=self.hidden_size*2, dropout=0, layer_norm_eps=1e-05, batch_first=True)
            self.torch_graph_transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        states = states[:, :, :self.agent['state_dim']]
        device = states.device
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        node_list = self.agent['state_position'].keys()
        
        # discard last timestep, equivalent to shift 1 timestep
        actions = actions[:, :-1, :]
        # construct 1 graph for 1 timestep of a agent
        node_embeddings = []
        for node in node_list:
            node_states = states[:, :, self.agent['state_position'][node]]
            node_actions = actions[:, :, self.agent['action_position'][node]]
            node_state_action = torch.cat([node_states, node_actions], dim=2)
            if node == 'root':
                node_embeddings.append(self.embed_net['root'](node_state_action))
            else:
                node_embeddings.append(self.embed_net['nonroot'](node_state_action))
        # [batch_size, seq_length, num_node, hidden_size]
        node_embeddings = torch.stack(node_embeddings, dim=2)

        time_embeddings = self.embed_timestep(timesteps)
        time_embeddings = time_embeddings.repeat(1, 1, self.num_node).reshape(batch_size, seq_length, self.num_node, self.hidden_size)

        rtg_embeddings = self.embed_return(returns_to_go)
        rtg_embeddings = rtg_embeddings.repeat(1, 1, self.num_node).reshape(batch_size, seq_length, self.num_node, self.hidden_size)

        # [batch_size, seq_length, num_node, hidden_size]
        transformer_inputs = node_embeddings + rtg_embeddings + time_embeddings
        transformer_inputs = self.embed_ln(transformer_inputs)

        # process prompt the same as d-t
        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]
            prompt_actions = prompt_actions[:, :-1, :]

            # construct 1 graph for 1 timestep of a agent
            prompt_node_embeddings = []
            for node in node_list:
                node_states = prompt_states[:, :, self.agent['state_position'][node]]
                node_actions = prompt_actions[:, :, self.agent['action_position'][node]]
                node_state_action = torch.cat([node_states, node_actions], dim=2)
                if node == 'root':
                    prompt_node_embeddings.append(self.prompt_embed_net['root'](node_state_action))
                else:
                    prompt_node_embeddings.append(self.prompt_embed_net['nonroot'](node_state_action))
            # [batch_size, prompt_seq_length, num_node, hidden_size]
            prompt_node_embeddings = torch.stack(prompt_node_embeddings, dim=2)

            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)
            prompt_time_embeddings = prompt_time_embeddings.repeat(1, 1, self.num_node).reshape(batch_size, prompt_seq_length, self.num_node, self.hidden_size)

            prompt_rtg_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_rtg_embeddings = prompt_rtg_embeddings.repeat(1, 1, self.num_node).reshape(batch_size, prompt_seq_length, self.num_node, self.hidden_size)

            # [batch_size, prompt_seq_length, num_node, hidden_size]
            prompt_transformer_inputs = prompt_node_embeddings + prompt_rtg_embeddings + prompt_time_embeddings
            # prompt_transformer_inputs = self.embed_ln(prompt_transformer_inputs)

            # stacked_inputs add prompted sequence
            # if prompt_transformer_inputs.shape[1] == 3 * seq_length: # if only smaple one prompt
            #     prompt_transformer_inputs = prompt_transformer_inputs.reshape(1, -1, self.num_node, self.hidden_size)
            #     prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
            #     stacked_inputs = torch.cat((prompt_transformer_inputs.repeat(batch_size, 1, 1), transformer_inputs), dim=1)
            #     stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            # else: # if sample one prompt for each traj in batch
            transformer_inputs = torch.cat((prompt_transformer_inputs, transformer_inputs), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)
        
        if self.library == 'pyg':
            # create edges for every batch
            edge_batch = []
            for mask_ in attention_mask:
                zero_idxs = torch.where(mask_ == 0)[0]
                '''
                TODO
                1. do i need to do deepcopy? or copy?
                2. upper right or bottom left???
                '''
                # temporal_edge = self.temporal_edge.clone()
                temporal_edge = torch.triu(torch.ones(self.max_length + self.prompt_length, self.max_length + self.prompt_length, device=device), diagonal=0)
                temporal_edge[:, zero_idxs] = 0
                temporal_edge[zero_idxs, :] = 0
                temporal_edge = temporal_edge.nonzero()*self.agent['num_node']
                full_temporal_edge = torch.cat([temporal_edge+i for i in range(self.agent['num_node'])], dim=0).T.contiguous()

                edge_batch.append(torch.cat([self.structural_edge, full_temporal_edge], dim=1))
        elif self.library == 'torch':
            mask_batch = []
            for mask_ in attention_mask:
                zero_idxs = torch.where(mask_ == 0)[0]
                temporal_edge = torch.triu(torch.ones(self.max_length + self.prompt_length, self.max_length + self.prompt_length, device=device), diagonal=0)
                temporal_edge[:, zero_idxs] = 0
                temporal_edge[zero_idxs, :] = 0
                temporal_edge = temporal_edge.nonzero()*self.agent['num_node']
                full_temporal_edge = torch.cat([temporal_edge+i for i in range(self.agent['num_node'])], dim=0)
                temporal_mask = torch.ones(self.num_node * (self.max_length + self.prompt_length), self.num_node * (self.max_length + self.prompt_length), device=device).to(torch.bool)
                # print(len(full_temporal_edge))
                for edge in full_temporal_edge:
                    temporal_mask[edge[0], edge[1]] = False
                mask_batch.append(self.structural_mask + temporal_mask)
            mask_batch = torch.stack(mask_batch, dim=0)
        else:
            raise NotImplementedError

        # since pyg_nn.transformerConv can't deal with additional dimension, sequeeze all dimension before hidden_size
        transformer_inputs = transformer_inputs.reshape(batch_size, -1, self.hidden_size)

        if self.library == 'pyg':
            transformer_inputs = [Data(x=x_, edge_index=edge_) for x_, edge_ in zip(transformer_inputs, edge_batch)]
            batch = Batch.from_data_list(transformer_inputs)
            for block in self.pyg_graph_transformer:
                batch.x = block(batch.x, edge_index=batch.edge_index)
            transformer_outputs = batch.x.reshape([batch_size, -1, self.num_node, self.hidden_size])
        elif self.library == 'torch':
            if mask_batch.shape[0] == 1:
                mask_batch = mask_batch.reshape(mask_batch.shape[1], mask_batch.shape[2])
            transformer_outputs = self.torch_graph_transformer(transformer_inputs, mask=mask_batch)
            transformer_outputs = transformer_outputs.reshape([batch_size, -1, self.num_node, self.hidden_size])
        else:
            raise NotImplementedError

        # predict action condition on previous state_rtg token
        action_preds = []
        for idx, node in enumerate(node_list):
            if node == 'root':
                continue
            action_pred = self.predict_net['nonroot'](transformer_outputs[:, :, idx, :])
            action_preds.append(action_pred)

        action_preds = torch.cat(action_preds, dim=2)

        # we don't predict state and reward
        state_preds, reward_preds = 0.0, 0.0

        return state_preds, action_preds[:, -seq_length:, :], reward_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        states = states[:, :self.agent['state_dim']]
        states = states.reshape(1, -1, self.agent['state_dim'])
        actions = actions.reshape(1, -1, self.agent['act_dim'])
        # pad a action in front
        actions = torch.cat([torch.ones((1, 1, self.agent['act_dim']), device=actions.device) * -10., actions], dim=1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # print('max_length', self.max_length)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length-1:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.agent['state_dim']), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length + 1 - actions.shape[1], self.agent['act_dim']), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        # print('action_pred shape: ', action_preds.shape)

        return action_preds[0,-1]

if __name__ == '__main__':
    pass