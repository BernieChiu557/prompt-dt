import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch, Data, HeteroData

def get_structural_mask(agent, max_length, prompt_length, max_num_node=None):
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
    
    for row in structural_mask:
        assert row.sum() != num_node * (max_length + prompt_length)

    return structural_mask

def get_temporal_mask_list(num_node, max_length, prompt_length, max_num_node=None):
    temporal_masks = []
    for num_zero in range(max_length):
        temporal_edge = torch.tril(torch.ones(max_length + prompt_length, max_length + prompt_length), diagonal=0)
        temporal_edge[:, prompt_length:prompt_length+num_zero] = 0
        temporal_edge[prompt_length:prompt_length+num_zero, :] = 0
        temporal_edge = temporal_edge.nonzero()*num_node
        full_temporal_edge = torch.cat([temporal_edge+i for i in range(num_node)], dim=0)

        temporal_mask = torch.eye(num_node * (max_length + prompt_length)) * -1 + 1
        temporal_mask = temporal_mask.to(torch.bool)
        for edge in full_temporal_edge:
            temporal_mask[edge[0], edge[1]] = False

        for row in temporal_mask:
            assert row.sum() != num_node * (max_length + prompt_length)

        temporal_masks.append(temporal_mask)

    return temporal_masks

class GPDT_Base(nn.Module):
    def __init__(self, agents, max_length, max_ep_length, prompt_length, hidden_size, n_layers, dropout=0.1, num_heads=1):
        super().__init__()

        self.agents = agents
        self.lengths = list(agents.keys())

        self.max_num_node = 0
        self.max_act_dim = 0
        for length in self.lengths:
            agent = self.agents[length]
            num_node = agent['num_node']
            act_dim = agent['act_dim']
            if num_node > self.max_num_node:
                self.max_num_node = num_node
            if act_dim > self.max_act_dim:
                self.max_act_dim = act_dim

        self.max_length = max_length
        self.max_ep_length = max_ep_length
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size

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

        agent_ = self.agents[self.lengths[0]]
        node_types = agent_['node_type_state_action_len'].keys()
        for node_type in node_types:
            state_action_size = agent_['node_type_state_action_len'][node_type]
            self.embed_net.add_module(node_type, nn.Linear(state_action_size, hidden_size))
            self.prompt_embed_net.add_module(node_type, nn.Linear(state_action_size, hidden_size))
            if node_type != 'root':
                action_size = 1
                self.predict_net.add_module(node_type, nn.Sequential(*([nn.Linear(hidden_size, action_size)] + ([nn.Tanh()]))))
        
    def create_mask_store(self, lengths, max_length, prompt_length):
        pass

    def create_transformer(self, hidden_size, num_heads, dropout, n_layers):
        pass

    def forward(self, per_env_b_size, lengths, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        device = actions.device
        actions = actions[:, :-1, :]
        batch_size, seq_length = states.shape[0], states.shape[1]
        num_token = seq_length*self.max_num_node

        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]
            prompt_actions = prompt_actions[:, :-1, :]
            assert prompt_attention_mask.min() == 1 # I wonder if there's a case where prompt mask has 0 inside?
            num_token = (prompt_seq_length+seq_length)*self.max_num_node

        # [batch_size, seq_length, hidden_size]
        time_embeddings = self.embed_timestep(timesteps)
        rtg_embeddings = self.embed_return(returns_to_go)
        if prompt is not None:
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)
            prompt_rtg_embeddings = self.prompt_embed_return(prompt_returns_to_go)
        
        transformer_outputs = []
        for idx, length in enumerate(lengths):
            start_idx = idx * per_env_b_size
            agent = self.agents[length]
            node_list = list(agent['state_position'].keys())
            num_node = agent['num_node']

            states_ = states[start_idx:start_idx+per_env_b_size, :, :agent['state_dim']]
            actions_ = actions[start_idx:start_idx+per_env_b_size, :, :agent['act_dim']]

            node_embeddings_ = []
            for node in node_list:
                node_states = states_[:, :, agent['state_position'][node]]
                node_actions = actions_[:, :, agent['action_position'][node]]
                node_state_action = torch.cat([node_states, node_actions], dim=2)
                # breakpoint()
                node_embeddings_.append(self.embed_net[agent['node_type'][node]](node_state_action))
            

            # [per_env_b_size, seq_length, num_node, hidden_size]
            node_embeddings_ = torch.stack(node_embeddings_, dim=2)
            time_embeddings_ = time_embeddings[start_idx:start_idx+per_env_b_size].repeat(1, 1, num_node).reshape(per_env_b_size, seq_length, num_node, self.hidden_size)
            rtg_embeddings_ = rtg_embeddings[start_idx:start_idx+per_env_b_size].repeat(1, 1, num_node).reshape(per_env_b_size, seq_length, num_node, self.hidden_size)
            # print(f'shape: node = {node_embeddings_.shape} | time = {time_embeddings_.shape} | rtg = {rtg_embeddings_.shape}')

            transformer_inputs_ = node_embeddings_ + rtg_embeddings_ + time_embeddings_
            transformer_inputs_ = self.embed_ln(transformer_inputs_)

            masks_ = attention_mask[start_idx:start_idx+per_env_b_size]

            if prompt is not None:
                prompt_states_ = prompt_states[start_idx:start_idx+per_env_b_size, :, :agent['state_dim']]
                prompt_actions_ = prompt_actions[start_idx:start_idx+per_env_b_size, :, :agent['act_dim']]

                prompt_node_embeddings_ = []
                for node in node_list:
                    node_states = prompt_states_[:, :, agent['state_position'][node]]
                    node_actions = prompt_actions_[:, :, agent['action_position'][node]]
                    node_state_action = torch.cat([node_states, node_actions], dim=2)
                    prompt_node_embeddings_.append(self.prompt_embed_net[agent['node_type'][node]](node_state_action))
                

                # [per_env_b_size, prompt_seq_length, num_node, hidden_size]
                prompt_node_embeddings_ = torch.stack(prompt_node_embeddings_, dim=2)
                prompt_time_embeddings_ = prompt_time_embeddings[start_idx:start_idx+per_env_b_size].repeat(1, 1, num_node).reshape(per_env_b_size, prompt_seq_length, num_node, self.hidden_size)
                prompt_rtg_embeddings_ = prompt_rtg_embeddings[start_idx:start_idx+per_env_b_size].repeat(1, 1, num_node).reshape(per_env_b_size, prompt_seq_length, num_node, self.hidden_size)
                
                prompt_transformer_inputs_ = prompt_node_embeddings_ + prompt_rtg_embeddings_ + prompt_time_embeddings_
                # prompt_transformer_inputs_ = self.embed_ln(prompt_transformer_inputs_)
                prompt_masks_ = prompt_attention_mask[start_idx:start_idx+per_env_b_size]

                # [per_env_b_size, prompt_seq_length+seq_length, num_node, hidden_size]
                transformer_inputs_ = torch.cat((prompt_transformer_inputs_, transformer_inputs_), dim=1)
                masks_ = torch.cat((prompt_masks_, masks_), dim=1)
            

            # [per_env_b_size, (prompt_seq_length+seq_length)*num_node, hidden_size]
            transformer_inputs_ = transformer_inputs_.reshape(per_env_b_size, -1, self.hidden_size)

            mask_batch = self.create_mask_batch(masks_, length)

            transformer_outputs_ = self.inference_transformer(transformer_inputs_, mask_batch, length)
            transformer_outputs.append(transformer_outputs_)

        action_preds = []
        for idx, (length, transformer_outputs_) in enumerate(zip(lengths, transformer_outputs)):
            start_idx = idx * per_env_b_size
            agent = self.agents[length]
            node_list = list(agent['state_position'].keys())
            num_node = agent['num_node']
            act_dim = agent['act_dim']

            transformer_outputs_ = transformer_outputs_.reshape([per_env_b_size, -1, num_node, self.hidden_size])

            action_preds_ = []
            for idx, node in enumerate(node_list):
                if node != 'root':
                    action_pred = self.predict_net[agent['node_type'][node]](transformer_outputs_[:, :, idx, :])
                    action_preds_.append(action_pred)

            # [per_env_b_size, prompt_seq_lengt+seq_length, act_dim]
            action_preds_ = torch.cat(action_preds_, dim=2)
            # padding to max_act_dim
            num_padding = self.max_act_dim - act_dim
            padding = torch.zeros(per_env_b_size, action_preds_.shape[1], num_padding, device=device)
            action_preds_ = torch.cat([action_preds_, padding], dim=2)

            action_preds.append(action_preds_)
        

        action_preds = torch.cat(action_preds, dim=0)
        if prompt is not None:
            assert action_preds.shape ==torch.Size([batch_size, prompt_seq_length+seq_length, self.max_act_dim])

        # we don't predict state and reward, these are just dummy variables
        state_preds, reward_preds = 0.0, 0.0

        return state_preds, action_preds[:, -seq_length:, :], reward_preds
    
    def create_mask_batch(self, masks_, length):
        pass
    
    def inference_transformer(self, transformer_inputs_, mask_batch):
        pass

    def get_action(self, length, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        agent = self.agents[length]
        states = states[:, :agent['state_dim']]
        states = states.reshape(1, -1, agent['state_dim'])
        actions = actions.reshape(1, -1, agent['act_dim'])
        # pad a action in front
        actions = torch.cat([torch.ones((1, 1, agent['act_dim']), device=actions.device) * -10., actions], dim=1)
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
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], agent['state_dim']), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length + 1 - actions.shape[1], agent['act_dim']), device=actions.device), actions],
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
            1, [length], states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
        
        action_preds = action_preds[:, :, :agent['act_dim']]

        return action_preds[0,-1]
    

class GPDT_V2(GPDT_Base):
    def __init__(self, agents, max_length, max_ep_length, prompt_length, hidden_size, n_layers, dropout=0.1, num_heads=1):
        super().__init__(agents, max_length, max_ep_length, prompt_length, hidden_size, n_layers, dropout=0.1, num_heads=1)

        self.create_mask_store(self.lengths, max_length, prompt_length)
        self.create_transformer(hidden_size, num_heads, dropout, n_layers)
        
    def create_mask_store(self, lengths, max_length, prompt_length):
        # mask store for every agent
        for length in self.lengths:
            agent = self.agents[length]
            num_node = agent['num_node']

            structural_mask = get_structural_mask(agent, max_length, prompt_length, self.max_num_node)
            temporal_mask_list = get_temporal_mask_list(num_node, max_length, prompt_length, self.max_num_node)
            masks = []
            for temporal_mask in temporal_mask_list:
                # breakpoint()
                # mask = structural_mask + temporal_mask
                mask = torch.logical_and(structural_mask, temporal_mask)
                
                masks.append(mask)
            masks = torch.stack(masks, dim=0)
            self.register_buffer(f'mask_store_{length}', masks)

    def create_transformer(self, hidden_size, num_heads, dropout, n_layers):
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True)
        self.torch_graph_transformer = nn.TransformerEncoder(encoder_layer, n_layers)

    def create_mask_batch(self, masks_, length):
        mask_batch = []
        for mask_ in masks_:
            num_zeros = len(mask_) - int(mask_.sum())
            mask_batch.append(getattr(self, f'mask_store_{length}')[num_zeros].clone())

        mask_batch = torch.stack(mask_batch, dim=0)
        if mask_batch.shape[0] == 1:
            mask_batch = mask_batch.reshape(mask_batch.shape[1], mask_batch.shape[2])
        
        return mask_batch
    
    def inference_transformer(self, transformer_inputs_, mask_batch, length):
        transformer_outputs_ = self.torch_graph_transformer(transformer_inputs_, mask=mask_batch)

        return transformer_outputs_

class GPDT_V3(GPDT_Base):
    def __init__(self, agents, max_length, max_ep_length, prompt_length, hidden_size, n_layers, dropout=0.1, num_heads=1):
        super().__init__(agents, max_length, max_ep_length, prompt_length, hidden_size, n_layers, dropout=0.1, num_heads=1)

        self.create_mask_store(self.lengths, max_length, prompt_length)
        self.create_transformer(hidden_size, num_heads, dropout, n_layers)
        
    def create_mask_store(self, lengths, max_length, prompt_length):
        # mask store for every agent
        for length in self.lengths:
            agent = self.agents[length]
            num_node = agent['num_node']

            structural_mask = get_structural_mask(agent, max_length, prompt_length, self.max_num_node)
            self.register_buffer(f'mask_store_{length}_structural',  structural_mask)

            temporal_mask_list = get_temporal_mask_list(num_node, max_length, prompt_length, self.max_num_node)
            temporal_mask_list = torch.stack(temporal_mask_list, dim=0)
            self.register_buffer(f'mask_store_{length}_temporal', temporal_mask_list)

    def create_transformer(self, hidden_size, num_heads, dropout, n_layers):
        self.torch_graph_transformer = nn.ModuleList([nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True) for _ in range(n_layers)])
    
    def create_mask_batch(self, masks_, length):
        temporal_mask_batch = []
        for mask_ in masks_:
            num_zeros = len(mask_) - int(mask_.sum())
            temporal_mask_batch.append(getattr(self, f'mask_store_{length}_temporal')[num_zeros].clone())

        temporal_mask_batch = torch.stack(temporal_mask_batch, dim=0)
        if temporal_mask_batch.shape[0] == 1:
            temporal_mask_batch = temporal_mask_batch.reshape(temporal_mask_batch.shape[1], temporal_mask_batch.shape[2])
        
        return temporal_mask_batch
    
    def inference_transformer(self, transformer_inputs_, temporal_mask_batch, length):
        # breakpoint()
        transformer_outputs_ = transformer_inputs_
        structural_mask = getattr(self, f'mask_store_{length}_structural')
        for idx, layer in enumerate(self.torch_graph_transformer):
            if idx < 2:
                transformer_outputs_ = layer(transformer_outputs_, src_mask=temporal_mask_batch)
            else:
                transformer_outputs_ = layer(transformer_outputs_, src_mask=structural_mask)
        return transformer_outputs_