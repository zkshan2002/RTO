import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device)


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory()


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.values = to(self.values, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.values = pin_memory(self.values)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
        dpo_model: Optional[Actor] = None,
        dpo_ref: Optional[Actor] = None,
        rl_config: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

        self._use_dpo = dpo_model is not None
        self._use_dpo_ref = dpo_ref is not None
        self._dpo_model = dpo_model
        self._dpo_ref = dpo_ref
        self._rl_config = rl_config

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        value = self.critic(sequences, num_actions, attention_mask)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        if self._use_dpo:
            dpo_rewards = self._compute_dpo_rewards(
                sequences, num_actions, attention_mask, base_action_log_probs, action_mask,
            )
        else:
            dpo_rewards = None

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            token_reward=dpo_rewards,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.inference_mode()
    def _compute_dpo_rewards(self, sequences, num_actions, attention_mask, base_action_log_probs, action_mask):
        dpo_action_log_probs = self._dpo_model(sequences, num_actions, attention_mask)
        dpo_rewards = (dpo_action_log_probs - base_action_log_probs) * action_mask

        scale = self._rl_config["dpo_reward_scale"]
        if scale != 1.0 and scale > 0:
            dpo_rewards = dpo_rewards * scale
        
        clip = self._rl_config["dpo_reward_clip"]
        if clip > 0:
            dpo_rewards = torch.clamp(dpo_rewards, min=-clip, max=clip)
        return dpo_rewards

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        start = time.time()
        if not self.packing_samples:
            sequences, attention_mask, action_mask = (
                self._generate_local(prompts, **generate_kwargs)
                if self.vllm_engines is None
                else self._generate_vllm(prompts, **generate_kwargs)
            )
            num_actions = action_mask.size(1)
            packed_seq_lens = None
            response_length = action_mask.float().sum(dim=-1)
            total_length = attention_mask.float().sum(dim=-1)
        else:
            assert self.vllm_engines is not None, "vllm_engines must be provided for packed samples"
            sequences, attention_mask, packed_seq_lens, num_actions = self._generate_vllm(prompts, **generate_kwargs)
            action_mask = None
            response_length = torch.tensor(num_actions, device=device, dtype=torch.float)
            total_length = torch.tensor(packed_seq_lens, device=device, dtype=torch.float)
        generate_time = time.time() - start

        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        value_ref = self.critic.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward:
            ray.get([value_ref])
            ray.get([self.critic.empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        else:
            # remote RM
            for rm in self.remote_rm_url:
                queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                r = remote_rm_fn_ray.remote(rm, queries=queries)
                r_refs.append(r)

        # log probs
        start = time.time()
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            num_actions=num_actions,
        )

        if not self.packing_samples:
            kl = masked_mean(kl, action_mask, dim=-1)
            return_sums = reward.sum(dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            value = unpacking_samples(value, num_actions)
            reward = unpacking_samples(reward, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl = torch.tensor([each_kl.mean() for each_kl in kl], device=device)
            return_sums = torch.tensor([each_reward.sum() for each_reward in reward], device=device)

        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": kl,
            "reward": r,
            "return": return_sums,
            "response_length": response_length,
            "total_length": total_length,
        }

        if self.strategy.args.perf:
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        prompt_token_ids = self.tokenize_fn(prompts, self.prompt_max_len, padding=False)["input_ids"]
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        if not self.packing_samples:
            # NOTE: concat all outputs to following format:
            #
            # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
            # | token token token token token | token token [EOS] [PAD] |
            # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
            # |<---------- prompt ----------->|<-------- answer ------->|
            max_input_len, max_output_len = 0, 0
            for output in outputs:
                max_input_len = max(max_input_len, len(output.prompt_token_ids))
                max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            sequences = []
            for output in outputs:
                # left padding input
                input_len = len(output.prompt_token_ids)
                input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                # right padding output
                output_len = len(output.outputs[0].token_ids)
                output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                if output_ids[output_len - 1] != eos_token_id:
                    output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

                # concat input and output
                sequences.append(input_ids + output_ids)

            sequences = torch.tensor(sequences)
            sequences, attention_mask, action_mask = self.actor.process_sequences(
                sequences, max_input_len, eos_token_id, pad_token_id
            )
            return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")
        else:
            # NOTE: concat all outputs to following format:
            #
            # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
            # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            sequences = []
            packed_seq_lens = []
            attention_mask = []
            num_actions = []
            for i, output in enumerate(outputs):
                input_len = len(output.prompt_token_ids)
                output_len = len(output.outputs[0].token_ids)
                packed_seq_lens.append(input_len + output_len)
                sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                attention_mask.extend([i + 1] * (input_len + output_len))

                # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                # num_actions.append(max(1, sum(current_action_mask)))
                num_actions.append(max(1, output_len))

            sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
            attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
            return sequences, attention_mask, packed_seq_lens, num_actions

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None
