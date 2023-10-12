# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn

import transformers

from .trajectory_gpt2 import GPT2Model

class Prompt_DT_MMML(nn.Module):

    def __init__(
            self,
            agents,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__()

        self.agents = agents
        self.lengths = list(agents.keys())

        self.max_act_dim = 0
        self.max_state_dim = 0
        for length in self.lengths:
            agent = self.agents[length]
            act_dim = agent['act_dim']
            if act_dim > self.max_act_dim:
                self.max_act_dim = act_dim
            state_dim = agent['state_dim']
            if state_dim > self.max_state_dim:
                self.max_state_dim = state_dim

        self.max_length = max_length
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        for length in self.lengths:
            agent = self.agents[length]
            state_dim = agent['state_dim']
            act_dim = agent['act_dim']

            setattr(self, f'embed_state_{length}', torch.nn.Linear(state_dim, hidden_size))
            setattr(self, f'embed_action_{length}', torch.nn.Linear(act_dim, hidden_size))
            setattr(self, f'prompt_embed_state_{length}', torch.nn.Linear(state_dim, hidden_size))
            setattr(self, f'prompt_embed_action_{length}', torch.nn.Linear(act_dim, hidden_size))
            setattr(self, f'predict_state_{length}', torch.nn.Linear(hidden_size, state_dim))
            setattr(self, f'predict_action_{length}', nn.Sequential(
                *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
            ))

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)

        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return = torch.nn.Linear(1, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, per_env_b_size, lengths, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        device = actions.device
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = []
        action_embeddings = []
        for idx, length in enumerate(lengths):
            start_idx = idx * per_env_b_size
            agent = self.agents[length]

            states_ = states[start_idx:start_idx+per_env_b_size, :, :agent['state_dim']]
            actions_ = actions[start_idx:start_idx+per_env_b_size, :, :agent['act_dim']]

            state_embeddings_ = getattr(self, f'embed_state_{length}')(states_)
            action_embeddings_ = getattr(self, f'embed_action_{length}')(actions_)

            state_embeddings.append(state_embeddings_)
            action_embeddings.append(action_embeddings_)
        state_embeddings = torch.cat(state_embeddings, dim=0)
        action_embeddings = torch.cat(action_embeddings, dim=0)
        assert state_embeddings.shape == torch.Size([batch_size, seq_length, self.hidden_size])
        assert action_embeddings.shape == torch.Size([batch_size, seq_length, self.hidden_size])

        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # process prompt the same as d-t
        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]

            prompt_state_embeddings = []
            prompt_action_embeddings = []
            for idx, length in enumerate(lengths):
                start_idx = idx * per_env_b_size
                agent = self.agents[length]

                prompt_states_ = prompt_states[start_idx:start_idx+per_env_b_size, :, :agent['state_dim']]
                prompt_actions_ = prompt_actions[start_idx:start_idx+per_env_b_size, :, :agent['act_dim']]

                prompt_state_embeddings_ = getattr(self, f'prompt_embed_state_{length}')(prompt_states_)
                prompt_action_embeddings_ = getattr(self, f'prompt_embed_action_{length}')(prompt_actions_)

                prompt_state_embeddings.append(prompt_state_embeddings_)
                prompt_action_embeddings.append(prompt_action_embeddings_)
            prompt_state_embeddings = torch.cat(prompt_state_embeddings, dim=0)
            prompt_action_embeddings = torch.cat(prompt_action_embeddings, dim=0)
            assert prompt_state_embeddings.shape == torch.Size([batch_size, prompt_seq_length, self.hidden_size])
            assert prompt_action_embeddings.shape == torch.Size([batch_size, prompt_seq_length, self.hidden_size])

            if prompt_returns_to_go.shape[1] % 10 == 1:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
            else:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

            prompt_stacked_inputs = torch.stack(
                (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            prompt_stacked_attention_mask = torch.stack(
                (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
            ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)

            # stacked_inputs add prompted sequence
            # breakpoint()
            if prompt_stacked_inputs.shape[1] == 3 * seq_length: # if only smaple one prompt
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            else: # if sample one prompt for each traj in batch
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
            # breakpoint()
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        if prompt is None:
            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action

         # predict next action given state
        action_preds = []
        state_preds = []
        for idx, length in enumerate(lengths):
            start_idx = idx * per_env_b_size
            agent = self.agents[length]
            act_dim = agent['act_dim']
            state_dim = agent['state_dim']

            x_ = x[start_idx:start_idx+per_env_b_size,1]

            action_preds_ = getattr(self, f'predict_action_{length}')(x_)
            num_padding = self.max_act_dim - act_dim
            padding = torch.zeros(per_env_b_size, action_preds_.shape[1], num_padding, device=device)
            action_preds_ = torch.cat([action_preds_, padding], dim=2)
            action_preds.append(action_preds_)

            x_ = x[start_idx:start_idx+per_env_b_size,2]

            state_preds_ = getattr(self, f'predict_state_{length}')(x_)
            num_padding = self.max_state_dim - state_dim
            padding = torch.zeros(per_env_b_size, state_preds_.shape[1], num_padding, device=device)
            state_preds_ = torch.cat([state_preds_, padding], dim=2)
            state_preds.append(state_preds_)

        action_preds = torch.cat(action_preds, dim=0)
        state_preds = torch.cat(state_preds, dim=0)
        if prompt is not None:
            assert action_preds.shape == torch.Size([batch_size, prompt_seq_length + seq_length, self.max_act_dim])
            assert state_preds.shape == torch.Size([batch_size, prompt_seq_length + seq_length, self.max_state_dim])

        action_preds = action_preds[:, -seq_length:, :]
        state_preds = state_preds[:, -seq_length:, :] 

        return state_preds, action_preds, return_preds

    def get_action(self, length, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        agent = self.agents[length]
        state_dim = agent['state_dim']
        act_dim = agent['act_dim']

        states = states[:, :state_dim]
        states = states.reshape(1, -1, state_dim)
        actions = actions.reshape(1, -1, act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], act_dim),
                             device=actions.device), actions],
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

        # Note: prompt within kwargs
        _, action_preds, return_preds = self.forward(
            1, [length], states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        action_preds = action_preds[:, :, :agent['act_dim']]

        return action_preds[0,-1]
