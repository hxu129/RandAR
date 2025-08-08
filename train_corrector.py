# This script trains a Corrector model, which is a transformer-based model that
# learns to predict which tokens in a sequence are "incorrect" or "perturbed".
#
# It leverages a pre-trained RandAR model to generate rich embeddings for the input sequence.
# The core idea is to:
# 1. Take a sequence of image tokens.
# 2. Randomly perturb a certain ratio of these tokens.
# 3. Feed the perturbed sequence through the frozen RandAR model to get embeddings.
# 4. Train the Corrector model to take these embeddings and output a probability for each token's position,
#    indicating whether it was one of the perturbed tokens.
#
# This script is heavily inspired by the structure of train_c2i.py and uses accelerate for
# distributed training and experiment management.

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import os
import time
import argparse
import sys
from typing import Tuple
sys.path.append("./")
import shutil
import wandb

from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

from RandAR.util import instantiate_from_config, load_safetensors
from RandAR.dataset.builder import build_dataset
from RandAR.utils.logger import create_logger
from RandAR.utils.lr_scheduler import get_scheduler
from RandAR.model.utils import interleave_tokens
from RandAR.model.randar_gpt import RandARTransformer

import debugpy
import os

# Only start debugpy in the main process to avoid port conflicts
# if int(os.environ.get('LOCAL_RANK', 0)) == 0:
#     debugpy.listen(65432)
#     print("Waiting for debugger to attach...")
#     debugpy.wait_for_client()
#     print("Debugger attached")

def cycle(dl: torch.utils.data.DataLoader):
    while True:
        for data in dl:
            yield data

def generate_and_label_draft_online(
    gpt_model: RandARTransformer,
    cond_tokens: torch.Tensor,
    seq_len: int,
    block_size: int,
    bad_token_rank_threshold_ratio: float,
    device: torch.device,
    cfg_scales: Tuple[float, float] = (1.0, 1.0),
    num_inference_steps: int = -1, # Use full autoregressive by default
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """
    在线生成草稿并基于其自身上下文进行标注，为学生模型提供训练数据。
    该方法基于您的建议，分为两个阶段：

    1.  **生成草稿 (Draft Generation)**:
        - 直接调用教师模型(gpt_model)的 `.generate()` 方法，使用随机顺序(random order)
          一次性生成一个批次的“初稿”序列。

    2.  **事后审查与标注 (Review and Label)**:
        - 对生成的“初稿”进行一次独立的、高效的单次前向传播评估。
        - 对于初稿中的每一个token，计算它在其他所有token作为上下文的条件下的概率排名。
        - 如果一个token的概率排名不在最优的top N% (由bad_token_rank_threshold_ratio决定)，
          则将其标记为“坏”(bad) token。

    Args:
        gpt_model: 预训练的、冻结的教师模型 (RandARTransformer)。
        cond_tokens: 条件token (例如类别标签)。
        seq_len: 生成序列的长度。
        bad_token_rank_threshold_ratio: 用于判定“坏”token的排名百分比阈值 (例如, 0.001 for top 0.1%)。
        device: 计算设备。
        ... (generation-related parameters) ...

    Returns:
        draft_tokens (torch.Tensor): 生成的“初稿”token序列, shape [bs, seq_len]。
        labels (torch.Tensor): 每个token对应的“坏”标签 (True表示坏, False表示好), shape [bs, seq_len]。
    """
    bs = cond_tokens.shape[0]
    gpt_model.eval()

    # --- 步骤 1: 使用 .generate() 直接生成草稿 ---
    # Note: This call activates the KV cache inside the model.
    with torch.no_grad():
        # 为生成过程创建随机排列
        gen_token_order = torch.stack([torch.randperm(seq_len, device=device) for _ in range(bs)])

        draft_tokens = gpt_model.generate(
            cond=cond_tokens,
            token_order=gen_token_order,
            cfg_scales=cfg_scales,
            num_inference_steps=num_inference_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # CRITICAL FIX: Clear the KV cache after generation.
    # The .generate() call activates the KV cache, which is not needed for the
    # full forward pass below and would cause a crash because `input_pos` is not provided.
    gpt_model.remove_caches()
    
    # --- 步骤 2: 事后审查与标注 (单次前向传播) ---
    with torch.no_grad():
        # 使用全新的随机排列来对“初稿”进行独立的、无偏的评估
        eval_token_order = torch.stack([torch.randperm(seq_len, device=device) for _ in range(bs)])
        permuted_draft_tokens = torch.gather(draft_tokens, 1, eval_token_order)[:, :block_size]

        # make bidirectional attention mask
        no_self_attention_mask = torch.ones(block_size*2+1, block_size*2+1, device=device, dtype=torch.bool)
        for i in range(block_size*2):
            no_self_attention_mask[i, i+1] = False
        
        # 获取评估的logits
        eval_logits, _ = get_last_n_hidden_states(
            gpt_model,
            permuted_draft_tokens,
            cond_tokens,
            eval_token_order,
            output_last_n=0,
            device=device,
            block_size=block_size,
            mask=no_self_attention_mask
        )
        # 提取与图像token对应的logits
        image_eval_logits = eval_logits[:, gpt_model.cls_token_num::2, :]

        # 计算每个位置上真实token的排名
        eval_probs = torch.softmax(image_eval_logits, dim=-1)
        
        # 获取“初稿”中每个token在被打乱顺序后对应的概率
        gt_probs_permuted = torch.gather(eval_probs, 2, permuted_draft_tokens.unsqueeze(-1))

        # 排名 = 概率比它大的token的数量。排名0为最好。
        ranks_permuted = (eval_probs > gt_probs_permuted).sum(dim=-1).float()

        # 计算排名阈值
        rank_threshold = gpt_model.vocab_size * bad_token_rank_threshold_ratio

        # 如果排名大于阈值，则认为是"坏"token
        labels_permuted = ranks_permuted > rank_threshold

        if block_size == seq_len:
            # --- Calculate average rank per generation step for logging ---
            # 1. Map ranks from evaluation order back to raster order.
            inverse_eval_order = torch.argsort(eval_token_order, dim=1)
            ranks_raster = torch.gather(ranks_permuted, 1, inverse_eval_order)

            # 2. Map ranks from raster order to the original generation order.
            ranks_generation_order = torch.gather(ranks_raster, 1, gen_token_order)

            # 3. Average across the batch to get the mean rank for each generation step.
            avg_rank_per_gen_step = ranks_generation_order.mean(dim=0)
        else:
            avg_rank_per_gen_step = None

    return permuted_draft_tokens, labels_permuted, eval_token_order, avg_rank_per_gen_step


def perturb_image_tokens_adversarial(
    image_tokens: torch.Tensor,
    vocab_size: int,
    perturb_mode: str,
    perturb_ratio: float,
    perturb_num: int,
    gpt_model: RandARTransformer,
    token_order: torch.Tensor,
    cond_tokens: torch.Tensor,
):
    """
    Efficiently generates perturbed tokens using a pre-trained RandAR model for adversarial training.
    
    Args:
        gpt_model: Pre-trained, frozen RandAR model.
        image_tokens: [bs, seq_len] image tokens, already permuted by token_order.
        cond_tokens: [bs] Class condition tokens.
        token_order: [bs, seq_len] The permutation order used for image_tokens.
        perturb_ratio: float, ratio of tokens to perturb (used in "ratio" mode).
        perturb_num: int, number of tokens to perturb (used in "num" mode).
        perturb_mode: str, perturbation mode ("ratio" or "num").
    """
    bs, seq_len = image_tokens.shape
    device = image_tokens.device

    # 1. Determine positions to perturb and create a boolean mask.
    if perturb_mode == "ratio":
        if perturb_ratio is None: perturb_ratio = 0.2
        perturbed_indices = torch.rand(bs, seq_len, device=device) < perturb_ratio
    elif perturb_mode == "num":
        if perturb_num is None: perturb_num = 1
        perturb_num = min(perturb_num, seq_len)
        perturbed_indices = torch.zeros_like(image_tokens, dtype=torch.bool)
        if perturb_num > 0:
            # Efficiently select `perturb_num` random indices per sample.
            row_indices = torch.arange(bs, device=device).unsqueeze(1)
            col_indices = torch.rand(bs, seq_len, device=device).topk(perturb_num, dim=1).indices
            perturbed_indices[row_indices, col_indices] = True
    else:
        # Fallback to default ratio mode.
        perturbed_indices = torch.rand(bs, seq_len, device=device) < 0.2

    # Clone original tokens to start with.
    perturbed_image_tokens = image_tokens.clone()

    # If no tokens are selected for perturbation, return early.
    if not torch.any(perturbed_indices):
        # Create dummy prediction quality for consistency
        dummy_quality = {
            'perturbed_gt_prob_mean': torch.tensor(0.0, device=device),
            'perturbed_gt_prob_std': torch.tensor(0.0, device=device),
            'perturbed_high_confidence_ratio': torch.tensor(0.0, device=device),
            'perturbed_count': torch.tensor(0, device=device),
            'perturbed_rank': torch.tensor(0.0, device=device),
        }
        # Return original token_order since no shuffling happened
        return perturbed_image_tokens, perturbed_indices, dummy_quality, token_order

    # 2. Perform a single forward pass with the RandAR model to get all logits.
    with torch.no_grad():
        gpt_model.eval()

        cond_embeddings = gpt_model.cls_embedding(cond_tokens, train=False)[:, :gpt_model.cls_token_num]
        token_embeddings = gpt_model.tok_embeddings(image_tokens)
        pos_instruct_tokens = gpt_model.get_position_instruction_tokens(token_order)

        h = torch.cat(
            (cond_embeddings, interleave_tokens(pos_instruct_tokens, token_embeddings)),
            dim=1
        )

        gpt_model.freqs_cis = gpt_model.freqs_cis.to(device)
        token_freqs_cis_ordered = gpt_model.freqs_cis[gpt_model.cls_token_num:].clone()[token_order]
        freqs_cis = torch.cat(
            (gpt_model.freqs_cis[:gpt_model.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
             interleave_tokens(token_freqs_cis_ordered, token_freqs_cis_ordered)),
            dim=1
        )

        for layer in gpt_model.layers:
            h = layer(h, freqs_cis, start_pos=None, mask=None)

        h = gpt_model.norm(h)
        logits = gpt_model.output(h).float()
        # --- End of forward pass ---

        # 3. Sample and replace tokens efficiently.

        # Select logits for image token positions from the interleaved sequence.
        # Shape: [bs, seq_len, vocab_size]
        image_token_logits = logits[:, gpt_model.cls_token_num::2, :]

        # Check RandAR prediction quality BEFORE perturbation
        prediction_quality = check_randar_prediction_quality(
            logits=image_token_logits,
            ground_truth_tokens=image_tokens,
            perturbed_indices=perturbed_indices,
            threshold=0.1,
            log_details=False  # Set to True for debugging
        )

        # 3. Sample and replace tokens efficiently.

        # Use the boolean mask to gather logits for all positions to be perturbed.
        # Shape: [num_total_perturbations, vocab_size]
        logits_to_sample = image_token_logits[perturbed_indices]

        # Sample new tokens for all positions in a single operation.
        temperature = 1.0
        probs = torch.softmax(logits_to_sample / temperature, dim=-1)
        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Place the newly generated tokens back into their positions in one go.
        perturbed_image_tokens[perturbed_indices] = new_tokens

        # 4. Handle sampling collisions (new_token == old_token).
        # This is rare but handled for correctness.
        collision_mask = (perturbed_image_tokens == image_tokens) & perturbed_indices
        max_attempts = 10
        attempts = 0
        while torch.any(collision_mask) and attempts < max_attempts:
            # Gather logits for collided positions.
            collided_logits = image_token_logits[collision_mask]
            
            # Resample.
            new_probs = torch.softmax(collided_logits / temperature, dim=-1)
            replacements = torch.multinomial(new_probs, num_samples=1).squeeze(1)

            # Place replacements back into collided positions.
            perturbed_image_tokens[collision_mask] = replacements
            
            # Update collision mask for the next check.
            collision_mask = (perturbed_image_tokens == image_tokens) & perturbed_indices
            attempts += 1

        # 5. Shuffle the perturbed sequence and recompute labels.
        bs, seq_len = perturbed_image_tokens.shape
        new_token_order = token_order.clone()
        for i in range(bs):
            shuffle_order = torch.randperm(seq_len, device=perturbed_image_tokens.device)
            original_shuffled = image_tokens[i][shuffle_order]
            perturbed_image_tokens[i] = perturbed_image_tokens[i][shuffle_order]
            # Also shuffle the token_order to keep it consistent
            new_token_order[i] = token_order[i][shuffle_order]
            # Recompute which positions are perturbed after shuffling
            perturbed_indices[i] = perturbed_indices[i][shuffle_order]

    # Return the new token order as well
    return perturbed_image_tokens, perturbed_indices, prediction_quality, new_token_order

def perturb_image_tokens_replacement(image_tokens, vocab_size, perturb_mode="ratio", perturb_ratio=None, perturb_num=None):
    '''Train the corrector to tell the replacement tokens and the original tokens apart.
    Args:
        image_tokens: [bsz, seq_len]
        vocab_size: int, the size of the vocabulary to sample replacements from
        perturb_mode: str, "ratio" or "num"
        perturb_ratio: float, the ratio of perturbed tokens
        perturb_num: int, the number of perturbed tokens
    '''
    # Validate parameters - prioritize perturb_num if both are provided
    if perturb_mode not in ["ratio", "num"]:
        raise ValueError(f"Invalid perturb_mode: {perturb_mode}, must be 'ratio' or 'num'")
    
    if perturb_mode == "ratio" and perturb_ratio is None:
        raise ValueError("perturb_ratio must be provided when perturb_mode is 'ratio'")
    
    if perturb_mode == "num" and perturb_num is None:
        raise ValueError("perturb_num must be provided when perturb_mode is 'num'")

    bs, seq_len = image_tokens.shape
    device = image_tokens.device
    
    # 1. Determine which tokens to perturb
    if perturb_mode == "ratio":
        # Note: this is stochastic, and the number of perturbed tokens is not guaranteed to be the same as perturb_ratio
        # Note: this is not real indicies, but the order of the tokens in the permuted sequence
        perturbed_indices = torch.rand(bs, seq_len, device=device) < perturb_ratio
   
        # 2. Clone original tokens and generate the exact number of replacements
        perturbed_image_tokens = image_tokens.clone()
        num_to_perturb = torch.sum(perturbed_indices)
    
        if num_to_perturb > 0:
            replacements = torch.randint(0, vocab_size, (num_to_perturb,), device=device)
            perturbed_image_tokens[perturbed_indices] = replacements
        
            # 3. Fix collisions: re-sample for any position that was selected for perturbation 
            #    but randomly got assigned its original value.
            collision_mask = (perturbed_image_tokens == image_tokens) & perturbed_indices
            while torch.any(collision_mask):
                num_collisions = torch.sum(collision_mask)
                new_replacements = torch.randint(0, vocab_size, (num_collisions,), device=device)
                perturbed_image_tokens[collision_mask] = new_replacements
                # Update the mask to check for new collisions
                collision_mask = (perturbed_image_tokens == image_tokens) & perturbed_indices
            
    elif perturb_mode == "num":
        # Deterministic number of perturbed tokens per sample
        assert perturb_num <= seq_len, f"perturb_num ({perturb_num}) cannot be larger than seq_len ({seq_len})"
        
        perturbed_image_tokens = image_tokens.clone()
        perturbed_indices = torch.zeros_like(image_tokens, dtype=torch.bool)
        
        if perturb_num > 0:
            # Batch-optimized implementation for perturb_num
            # Generate random permutations for all samples at once
            batch_perms = torch.stack([torch.randperm(seq_len, device=device) for _ in range(bs)])
            selected_positions = batch_perms[:, :perturb_num]  # [bs, perturb_num]
            
            # Create batch indices for advanced indexing
            batch_indices = torch.arange(bs, device=device).unsqueeze(1).expand(-1, perturb_num)
            
            # Set perturbed indices
            perturbed_indices[batch_indices, selected_positions] = True
            
            # Generate replacement tokens for all positions at once
            replacements = torch.randint(0, vocab_size, (bs, perturb_num), device=device)
            perturbed_image_tokens[batch_indices, selected_positions] = replacements
            
            # Fix collisions in batch
            original_tokens = image_tokens[batch_indices, selected_positions]
            collision_mask = (perturbed_image_tokens[batch_indices, selected_positions] == original_tokens)
            
            while torch.any(collision_mask):
                # Find positions that need re-sampling
                collision_batch_idx, collision_pos_idx = torch.where(collision_mask)
                num_collisions = len(collision_batch_idx)
                
                if num_collisions > 0:
                    new_replacements = torch.randint(0, vocab_size, (num_collisions,), device=device)
                    collision_positions = selected_positions[collision_batch_idx, collision_pos_idx]
                    perturbed_image_tokens[collision_batch_idx, collision_positions] = new_replacements
                    
                    # Update collision mask
                    collision_mask = (perturbed_image_tokens[batch_indices, selected_positions] == original_tokens)
                else:
                    break
            
    return perturbed_image_tokens, perturbed_indices # [bsz, seq_len], [bsz, seq_len]

def perturb_image_tokens(image_tokens, vocab_size, supervision_mode="adversarial", perturb_mode="ratio", perturb_ratio=None, perturb_num=None, ar_model=None, token_order=None, class_tokens=None):
    '''
    Args:
        image_tokens: [bsz, seq_len]
        vocab_size: int, the size of the vocabulary to sample replacements from
        perturb_mode: str, "ratio" or "num"
        perturb_ratio: float, the ratio of perturbed tokens
        perturb_num: int, the number of perturbed tokens
        ar_model: RandARTransformer model for generation (required for adversarial mode)
        token_order: [bsz, seq_len] the permutation order (required for adversarial mode)
        class_tokens: [bsz] class tokens (required for adversarial mode)
    '''
    if supervision_mode == "adversarial":
        # This function will now consistently return 4 items
        return perturb_image_tokens_adversarial(image_tokens, vocab_size, perturb_mode, perturb_ratio, perturb_num, ar_model, token_order, class_tokens)
    elif supervision_mode == "replacement":
        perturbed_tokens, perturbed_indices = perturb_image_tokens_replacement(image_tokens, vocab_size, perturb_mode, perturb_ratio, perturb_num)
        # Return a tuple of 4 with None as placeholders for consistency
        return perturbed_tokens, perturbed_indices, None, token_order
    else:
        raise ValueError(f"Invalid supervision_mode: {supervision_mode}, must be 'adversarial' or 'replacement'")

def get_last_n_hidden_states(
    model: RandARTransformer,
    image_tokens: torch.Tensor,
    cond_tokens: torch.Tensor,
    token_order: torch.Tensor,
    output_last_n: int,
    device: torch.device,
    block_size: int,
    mask: torch.Tensor = None,
):
    """
    Feeds a sequence through the RandAR model and returns the hidden states
    from the last n layers.

    Args:
        model: The RandARTransformer model.
        image_tokens: Tensor of shape [bs, seq_len] with **already permuted** image token IDs.
        cond_tokens: Tensor of shape [bs] or [bs, 1] with class token IDs.
        token_order: The permutation order used for the image_tokens. (full length)
        output_last_n: The number of final layers to return hidden states from.
        device: The torch device to run on.

    Returns:
        A tuple of (logits, hidden_states).
        - logits: Tensor of shape [bs, seq_len_out, vocab_size]
        - hidden_states: Tensor of shape [bs, seq_len_out, output_last_n * dim]
    """
    with torch.no_grad():
        model.eval()  # Ensure the model is in evaluation mode

        bs, seq_len = image_tokens.shape
        assert seq_len == block_size, f"Input sequence length ({seq_len}) must match the model's block_size ({model.block_size})."

        cond_embeddings = model.cls_embedding(cond_tokens, train=False)[:, :model.cls_token_num]
        token_embeddings = model.tok_embeddings(image_tokens)
        pos_instruct_tokens = model.get_position_instruction_tokens(token_order)[:, :block_size]

        h = torch.cat(
            (cond_embeddings, interleave_tokens(pos_instruct_tokens, token_embeddings)),
            dim=1
        )

        # Correctly prepare RoPE embeddings for the sequence
        model.freqs_cis = model.freqs_cis.to(device)
        # 1. Select freqs for image tokens (excluding class tokens)
        # 2. Permute them according to the token_order
        token_freqs_cis_ordered = model.freqs_cis[model.cls_token_num:].clone()[token_order][:, :block_size]
        # 3. Build the full freqs_cis sequence for interleaved input
        freqs_cis = torch.cat(
            (model.freqs_cis[:model.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
             interleave_tokens(token_freqs_cis_ordered, token_freqs_cis_ordered)),
            dim=1
        )

        h_for_inference = h
        hidden_states = []
        for layer_idx, layer in enumerate(model.layers):
            # The layer forward in TransformerBlock doesn't use `start_pos` when KV cache is off
            # We pass the custom mask to control attention behavior (e.g., bidirectional without self-attention).
            h_for_inference = layer(h_for_inference, freqs_cis, start_pos=None, mask=mask)
            if output_last_n > 0 and layer_idx >= model.n_layer - output_last_n:
                hidden_states.append(h_for_inference)
        
        h_final = model.norm(h_for_inference)
        logits = model.output(h_final).float()

        if output_last_n > 0:
            return logits, torch.cat(hidden_states, dim=-1)
        else:
            return logits, None


def compute_loss(logits, perturbed_indices):
    # Logits from corrector have shape [bs, 1 + 2 * block_size, 1]
    # We need to select the logits for the image token positions.
    # The interleaved sequence is: [cls_token, pos_instruct_0, image_token_0, pos_instruct_1, image_token_1, ...]
    # So, we need to select the logits at indices 2, 4, 6, ...
    image_token_logits = logits[:, 2::2, :] # Shape: [bs, block_size, 1]
    
    # Squeeze the last dimension to match the target shape
    image_token_logits = image_token_logits.squeeze(-1) # Shape: [bs, block_size]

    # compute accuracy
    pred_pos = image_token_logits.sigmoid() > 0.5
    acc = ((image_token_logits.sigmoid() > 0.5) == perturbed_indices).float().mean() # (tp + tn) / (tp + tn + fp + fn)

    # compute f1 score
    f1 = f1_score(perturbed_indices.float().cpu().numpy(), 
                  (image_token_logits.sigmoid() > 0.5).cpu().numpy(), 
                  average='macro')

    # compute top k accuracy: 1, 5, 10, 15
    k_range = [15, 10, 5, 3, 2, 1]
    top_k_prob, top_k_pos = image_token_logits.sigmoid().topk(min(max(k_range), image_token_logits.shape[1]), dim=-1, largest=True, sorted=True)

    top_k_acc = {}
    top_k_mean_prob = {}
    top_k_pos_idx = {}


    bs = perturbed_indices.shape[0]
    batch_indices = torch.arange(bs, device=perturbed_indices.device).unsqueeze(1)
    for k in k_range:
        current_top_k_pos = top_k_pos[:, :k]
        top_k_pos_idx[k] = current_top_k_pos # [bs, k]

        # Use advanced indexing instead of torch.gather for better readability.
        ground_truth_at_top_k = perturbed_indices[batch_indices, current_top_k_pos]
        
        top_k_acc[k] = ground_truth_at_top_k.float().mean() # (tp) / (tp + tn + fp + fn)
        top_k_mean_prob[k] = top_k_prob[:, :k].mean()

    # compute more meaningful metrics
    tp_rate = (perturbed_indices[pred_pos] == 1).float().sum() / (pred_pos).float().sum() # True positive rate, real positive / total predicted positive
    fp_rate = (perturbed_indices[pred_pos] == 0).float().sum() / (pred_pos).float().sum() # False positive rate, real negative / total predicted positive
    fn_rate = (perturbed_indices[~pred_pos] == 1).float().sum() / (~pred_pos).float().sum() # False negative rate, real positive / total predicted negative
    tn_rate = (perturbed_indices[~pred_pos] == 0).float().sum() / (~pred_pos).float().sum() # True negative rate, real negative / total predicted negative

    # get the position information about fp, fn, tp, tn
    position_init = torch.arange(image_token_logits.shape[1], device=image_token_logits.device).repeat(bs, 1)
    fp_pos = position_init[(perturbed_indices == 0) & (pred_pos == 1)].bincount(minlength=image_token_logits.shape[1]) / bs # position distribution of false positive samples
    fn_pos = position_init[(perturbed_indices == 1) & (pred_pos == 0)].bincount(minlength=image_token_logits.shape[1]) / bs # position distribution of false negative samples
    tp_pos = position_init[(perturbed_indices == 1) & (pred_pos == 1)].bincount(minlength=image_token_logits.shape[1]) / bs # position distribution of true positive samples
    tn_pos = position_init[(perturbed_indices == 0) & (pred_pos == 0)].bincount(minlength=image_token_logits.shape[1]) / bs # position distribution of true negative samples

    # compute pos_weight for the loss function
    pos_ratio = perturbed_indices.float().mean()
    neg_ratio = 1 - pos_ratio
    pos_weight = neg_ratio / pos_ratio
    
    # Target (perturbed_indices) also has shape [bs, block_size]
    # It needs to be converted to float for the loss function
    loss = F.binary_cross_entropy_with_logits(image_token_logits,
                                              perturbed_indices.float(),
                                              pos_weight=pos_weight)

    return_value = {
        'loss': loss,
        'acc': acc,
        'f1': f1,
        'top_k_acc': top_k_acc,
        'top_k_mean_prob': top_k_mean_prob,
        'top_k_pos_idx': top_k_pos_idx,
        'tp_pos': tp_pos,
        'fn_pos': fn_pos,
        'fp_pos': fp_pos,
        'tn_pos': tn_pos,
        'tp_rate': tp_rate,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'tn_rate': tn_rate,
    }
    
    return return_value

def check_randar_prediction_quality(
    logits: torch.Tensor,
    ground_truth_tokens: torch.Tensor,
    perturbed_indices: torch.Tensor,
    threshold: float = 0.1,
    log_details: bool = False,
):
    """
    检查RandAR模型对扰动位置真实token的预测质量
    
    Args:
        logits: [bs, seq_len, vocab_size] RandAR输出的logits
        ground_truth_tokens: [bs, seq_len] 真实的token
        perturbed_indices: [bs, seq_len] 扰动位置的mask
        threshold: float, 认为预测概率"高"的阈值
        log_details: bool, 是否记录详细信息
    
    Returns:
        dict: 包含扰动位置预测质量指标的字典
    """
    with torch.no_grad():
        bs, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # 1. 计算真实token的预测概率
        probs = torch.softmax(logits, dim=-1) # [bs, seq_len, vocab_size]
        
        # 使用高级索引获取真实token的概率
        batch_indices = torch.arange(bs, device=device).unsqueeze(1)  # [bs, 1]
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        gt_probs = probs[batch_indices, pos_indices, ground_truth_tokens]  # [bs, seq_len]
        
        # 2. 只分析扰动位置
        perturbed_mask = perturbed_indices.bool()
        
        if not perturbed_mask.any():
            # 如果没有扰动位置，返回默认值
            metrics = {
                'perturbed_gt_prob_mean': torch.tensor(0.0, device=device),
                'perturbed_gt_prob_std': torch.tensor(0.0, device=device),
                'perturbed_high_confidence_ratio': torch.tensor(0.0, device=device),
                'perturbed_count': torch.tensor(0, device=device),
                'rank_per_position': torch.zeros(seq_len, device=device),
                'position_axis': torch.arange(seq_len, device=device),
            }
        else:
            # 提取扰动位置的预测概率
            perturbed_gt_probs = gt_probs[perturbed_mask]

            # Calculate the rank of the ground truth token at each position.
            # The rank is the number of other tokens with a higher probability.
            ranks = (probs > gt_probs.unsqueeze(-1)).sum(dim=-1).float()  # Shape: [bs, seq_len]

            rank_per_position = ranks.float().mean(0)

            # Create x-axis for plotting (positions 0 to seq_len-1)
            position_axis = torch.arange(seq_len, device=device)

            metrics = {
                'perturbed_gt_prob_mean': perturbed_gt_probs.mean(),
                'perturbed_gt_prob_std': perturbed_gt_probs.std(),
                'perturbed_high_confidence_ratio': (perturbed_gt_probs > threshold).float().mean(),
                'perturbed_count': perturbed_mask.sum(),
                'rank_per_position': rank_per_position,
                'position_axis': position_axis,
            }
        
        # 3. 记录详细信息
        if log_details and perturbed_mask.any():
            print(f"RandAR Perturbed Positions Quality:")
            print(f"  Count: {metrics['perturbed_count']}")
            print(f"  GT Probability: {metrics['perturbed_gt_prob_mean']:.4f} ± {metrics['perturbed_gt_prob_std']:.4f}")
            print(f"  High Confidence Ratio (>{threshold}): {metrics['perturbed_high_confidence_ratio']:.4f}")
        
        return metrics

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    config = OmegaConf.load(args.config)
    ar_config = OmegaConf.load(args.ar_model_config_path)

    #################### Accelerator ####################
    args.exp_name = args.exp_name + f'_bs_{config.training_params.global_batch_size}_lr_{config.optimizer.lr}'
    experiment_dir = os.path.join(args.results_dir, args.exp_name)
    
    accelerator_config = ProjectConfiguration(project_dir=experiment_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_config=accelerator_config,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=config.accelerator.mixed_precision,
        log_with=config.accelerator.log_with,
        gradient_accumulation_steps=config.accelerator.gradient_accumulation_steps,
    )
    set_seed(config.global_seed + accelerator.process_index)

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Corrector Config: {config}")
        logger.info(f"Loaded AR Config: {ar_config}")
    else:
        logger = create_logger(None)

    #################### Data, Model, Optimization ####################
    dataset = build_dataset(is_train=True, args=args, transform=transforms.ToTensor())
    per_gpu_batch_size = int(
        config.training_params.global_batch_size
        // accelerator.num_processes
        // config.accelerator.gradient_accumulation_steps
    )
    data_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    logger.info("Datasets contains {} samples".format(len(dataset)))

    # --- Load Pre-trained RandAR Model (and freeze it) ---
    logger.info(f"Loading pre-trained RandAR model from: {args.gpt_ckpt}")
    gpt = instantiate_from_config(ar_config.ar_model).to(accelerator.device)
    gpt_weight = load_safetensors(args.gpt_ckpt)
    gpt.load_state_dict(gpt_weight, strict=True)
    gpt.eval()
    # It is crucial to freeze the pre-trained model to save memory and compute
    for param in gpt.parameters():
        param.requires_grad = False

    # --- Create Corrector Model ---
    corrector = instantiate_from_config(config.corrector_model).to(accelerator.device)
    logger.info(f"Corrector Parameters: {sum(p.numel() for p in corrector.parameters()):,}")
    
    ################## Continue from Pre-trained Weights ##################
    if args.continue_from_weight:
        if not os.path.exists(args.continue_from_weight):
            raise FileNotFoundError(f"Continue checkpoint file not found: {args.continue_from_weight}")
        
        if accelerator.is_main_process:
            logger.info(f"Loading pre-trained corrector weights from: {args.continue_from_weight}")
        
        # Load the checkpoint state dict
        if args.continue_from_weight.endswith('.safetensors'):
            from safetensors.torch import load_file 
            checkpoint = load_file(args.continue_from_weight)
        else:
            # Temporarily patch torch.load to use weights_only=False for trusted checkpoint loading
            original_load = torch.load
            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            try:
                checkpoint = torch.load(args.continue_from_weight, map_location=accelerator.device)
            finally:
                torch.load = original_load
        
        # Extract model state dict if it's wrapped in accelerator checkpoint format
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                # If checkpoint contains model key (from accelerator.save_state)
                model_state_dict = checkpoint['model']
            elif any(key.startswith('module.') for key in checkpoint.keys()):
                # If it's a DataParallel/DistributedDataParallel wrapped model
                model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            else:
                # Assume it's already a clean state dict
                model_state_dict = checkpoint
        else:
            model_state_dict = checkpoint
        
        # Load the state dict into the corrector model
        # Use unwrap to get the actual model if it's wrapped by accelerator
        unwrapped_model = accelerator.unwrap_model(corrector)
        unwrapped_model.load_state_dict(model_state_dict, strict=True)

    # --- Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=config.optimizer.betas)
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warm_up_iters * config.accelerator.gradient_accumulation_steps,
        num_training_steps=config.training_params.max_iters * config.accelerator.gradient_accumulation_steps,
    )
    
    corrector, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        corrector, optimizer, data_loader, lr_scheduler
    )
    data_loader = cycle(data_loader)
    
    ################## Resume Training ##################
    # Ensure only one resume method is used
    if args.resume_from and args.continue_from_weight:
        raise ValueError("Cannot use both --resume-from and --continue-from-weight simultaneously. "
                        "Use --resume-from for full state resumption or --continue-from-weight for weight-only loading.")
    
    train_steps = 0
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Resume checkpoint directory not found: {args.resume_from}")
        
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
        
        # Temporarily patch torch.load to use weights_only=False for trusted checkpoint loading
        # This is safe since we trust our own checkpoint files
        original_load = torch.load
        def patched_load(*args, **kwargs):
            # If weights_only is not explicitly set, set it to False
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        try:
            accelerator.load_state(args.resume_from)
        finally:
            # Restore original torch.load
            torch.load = original_load
        
        # Extract train_steps from checkpoint directory name
        # Expected format: "iters_00000500" or "iters_00000500_final"
        ckpt_dir_name = os.path.basename(args.resume_from)
        if ckpt_dir_name.startswith("iters_"):
            # Split by '_' and find the numeric part
            parts = ckpt_dir_name.split("_")
            if len(parts) >= 2:
                try:
                    train_steps = int(parts[1])
                    if accelerator.is_main_process:
                        logger.info(f"Resuming training from step: {train_steps}")
                except ValueError:
                    if accelerator.is_main_process:
                        logger.warning(f"Could not parse train_steps from {ckpt_dir_name}, starting from 0")
                    train_steps = 0
            else:
                if accelerator.is_main_process:
                    logger.warning(f"Unexpected checkpoint directory name format: {ckpt_dir_name}, starting from 0")
                train_steps = 0
        else:
            if accelerator.is_main_process:
                logger.warning(f"Checkpoint directory name does not start with 'iters_': {ckpt_dir_name}, starting from 0")
            train_steps = 0

    #################### Wandb Setup ####################
    if accelerator.is_main_process:
        # Setup wandb configuration based on whether we're resuming or starting fresh
        wandb_config = {
            "entity": args.wandb_entity,
            "config": OmegaConf.to_container(config, resolve=True),
            "name": args.exp_name,
            "dir": experiment_dir,
        }
        
        # If resuming from checkpoint and user wants to resume wandb, try to resume the wandb run
        if args.resume_from and args.resume_wandb:
            wandb_config["resume"] = "allow"  # Allow resuming if run exists, otherwise create new
            # Use experiment name as run ID for consistent resuming
            # This ensures we always resume the same wandb run for the same experiment
            wandb_config["id"] = args.exp_name.replace('/', '_').replace(' ', '_')  # Clean run ID
            if accelerator.is_main_process:
                logger.info(f"Attempting to resume wandb run with ID: {wandb_config['id']}")
        else:
            # Fresh training, continuing from weights, or user chose not to resume wandb - create new run
            wandb_config["resume"] = False
            if args.resume_from and not args.resume_wandb:
                if accelerator.is_main_process:
                    logger.info("Resuming training but creating new wandb run (--resume-wandb not specified)")
            elif args.continue_from_weight:
                if accelerator.is_main_process:
                    logger.info("Starting fresh training from pre-trained weights - creating new wandb run")
            
        accelerator.init_trackers(
            project_name=args.wandb_project,
            init_kwargs={"wandb": wandb_config},
        )
        
        # Define wandb metrics after initialization
        wandb.define_metric("perturbed_rank_x_axis")
        wandb.define_metric("perturbed_rank", step_metric="perturbed_rank_x_axis")

    #################### Training Loop ####################
    corrector.train()
    total_iters = config.training_params.max_iters
    log_iters, running_loss, start_time = 0, 0, time.time()
    full_block_size = config.corrector_model.params.block_size
    cls_token_num = config.corrector_model.params.cls_token_num

    # Variables for running average of rank per position
    running_rank_sum = torch.zeros(full_block_size, device=accelerator.device)
    running_rank_count = 0

    running_gen_step_rank_sum = torch.zeros(full_block_size, device=accelerator.device)
    running_gen_step_rank_count = 0

    # Variables for running average of top-k position indices
    running_top_k_pos_idx_sum = {k: torch.zeros(full_block_size, device=torch.device('cpu')) for k in [1, 2, 3, 5, 10, 15]}
    running_top_k_pos_idx_mean = {k: torch.zeros(full_block_size, device=torch.device('cpu')) for k in [1, 2, 3, 5, 10, 15]}
    running_pos_count = {pos_type: torch.zeros(full_block_size, device=torch.device('cpu')) for pos_type in ['tp', 'fn', 'fp', 'tn']}
    running_pos_count_mean = {pos_type: torch.zeros(full_block_size, device=torch.device('cpu')) for pos_type in ['tp', 'fn', 'fp', 'tn']}

    logger.info(f"Starting training from iteration {train_steps} to {total_iters}")
    while train_steps < total_iters:
        block_size = full_block_size if config.training_params.full_length_mode else torch.randint(128, 201, (1,)).item()
        if train_steps % 250 == 0:
            block_size = full_block_size
        x, y, _ = next(data_loader)
        x = x.to(accelerator.device, non_blocking=True)
        y = y.to(accelerator.device, non_blocking=True)
        image_tokens = x.reshape(x.shape[0], -1)
        cond = y.reshape(-1)
        bs = image_tokens.shape[0]

        with accelerator.accumulate(corrector):
            # 1. Prepare token sequence and labels based on the supervision mode
            if config.training_params.supervision_mode == "confidence_scoring":
                # Mode 1: Teacher generates a draft and labels it.
                corr_perturbed_tokens, corr_perturbed_indices, token_order, avg_rank_per_gen_step = generate_and_label_draft_online(
                    gpt_model=gpt,
                    cond_tokens=cond,
                    seq_len=full_block_size,
                    bad_token_rank_threshold_ratio=config.training_params.bad_token_rank_threshold_ratio,
                    device=accelerator.device,
                    # Note: You might want to expose these generation params in the config as well
                    cfg_scales=(1.0, 1.0), 
                    temperature=1.0,
                    top_k=0,
                    block_size=block_size,
                )

                prediction_quality = None # Not applicable in this mode
            
            else:
                # Mode 2 & 3: Perturb existing real data.
                token_order = torch.arange(full_block_size, device=accelerator.device).unsqueeze(0).repeat(bs, 1)
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(full_block_size, device=accelerator.device)]
                
                permuted_image_tokens = torch.gather(image_tokens, 1, token_order)

                perturbed_tokens, perturbed_indices, prediction_quality, new_token_order = perturb_image_tokens(
                    permuted_image_tokens,
                    config.corrector_model.params.vocab_size,
                    supervision_mode=config.training_params.supervision_mode,
                    perturb_mode=config.training_params.perturb_mode,
                    perturb_ratio=config.training_params.perturb_ratio,
                    perturb_num=config.training_params.perturb_num,
                    ar_model=gpt,
                    token_order=token_order,
                    class_tokens=y
                )
                token_order = new_token_order

                # prepare tokens and token_order to pass in under partial sequence length mode
                corr_perturbed_tokens = perturbed_tokens[:, :block_size]
                corr_perturbed_indices = perturbed_indices[:, :block_size]
            
            # 2. Get hidden states from the frozen gpt model
            logits, hidden_states = get_last_n_hidden_states(
                gpt,
                corr_perturbed_tokens,
                cond,
                token_order,
                output_last_n=config.corrector_model.params.num_ar_layers_for_input,
                device=accelerator.device,
                block_size=block_size
            )

            # 3. Forward pass through corrector
            logits = corrector(hidden_states)
            return_value = compute_loss(logits, corr_perturbed_indices)
            
            # 5. Backward pass
            accelerator.backward(return_value['loss'])
            if config.optimizer.max_grad_norm > 0:
                accelerator.clip_grad_norm_(corrector.parameters(), config.optimizer.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            running_loss += accelerator.gather(return_value['loss'].repeat(per_gpu_batch_size)).mean().item() / config.accelerator.gradient_accumulation_steps

        if accelerator.sync_gradients:
            train_steps += 1
            
            if train_steps % args.log_every == 0 and accelerator.is_main_process:
                average_loss = running_loss / args.log_every
                end_time = time.time()
                average_time = (end_time - start_time) / args.log_every

                logger.info(f"Step {train_steps:08d} | Loss {average_loss:.4f} | Acc {return_value['acc']:.4f} | F1 {return_value['f1']:.4f} | Time {average_time:.4f}s | LR {lr_scheduler.get_last_lr()[0]:.6f}")
                
                # Prepare logging metrics
                log_metrics = {
                    "loss": average_loss, 
                    "acc": return_value['acc'], 
                    "f1": return_value['f1'], 
                    "top_1_acc": return_value['top_k_acc'][1],
                    "top_2_acc": return_value['top_k_acc'][2],
                    "top_3_acc": return_value['top_k_acc'][3],
                    "top_5_acc": return_value['top_k_acc'][5],
                    "top_10_acc": return_value['top_k_acc'][10],
                    "top_15_acc": return_value['top_k_acc'][15],
                    "top_1_mean_prob": return_value['top_k_mean_prob'][1],
                    "top_2_mean_prob": return_value['top_k_mean_prob'][2],
                    "top_3_mean_prob": return_value['top_k_mean_prob'][3],
                    "top_5_mean_prob": return_value['top_k_mean_prob'][5],
                    "top_10_mean_prob": return_value['top_k_mean_prob'][10],
                    "top_15_mean_prob": return_value['top_k_mean_prob'][15],
                    "tp_rate": return_value['tp_rate'].item(),
                    "fp_rate": return_value['fp_rate'].item(),
                    "fn_rate": return_value['fn_rate'].item(),
                    "tn_rate": return_value['tn_rate'].item(),
                    "lr": lr_scheduler.get_last_lr()[0], 
                    "time": average_time,
                    "block_size": block_size
                }

                if block_size == full_block_size:
                    # Log top-k position indices
                    for k in return_value['top_k_pos_idx'].keys():
                        # top_k_pos_idx[k] has shape [bs, k], we flatten it to count occurrences
                        indices = return_value['top_k_pos_idx'][k].flatten()
                        counts = torch.bincount(indices, minlength=full_block_size) / bs # normalize by batch size
                        running_top_k_pos_idx_sum[k] += counts.cpu()
                    
                        running_top_k_pos_idx_mean[k] = (running_top_k_pos_idx_sum[k] / (train_steps % 100)).numpy()
                    
                        x_axis = range(full_block_size)
                        table_data = [[pos, count] for pos, count in zip(x_axis, running_top_k_pos_idx_mean[k])]
                        table = wandb.Table(data=table_data, columns=["Position", "Average Count"])
                        log_metrics[f"randar_perturbed/top_k_pos_idx_{k}"] = table
                
                    # log tp, fn, fp, tn position distribution
                    pos_types = ['tp', 'fn', 'fp', 'tn']
                    for pos_type in pos_types:
                        running_pos_count[pos_type] += return_value[f'{pos_type}_pos'].cpu()
                        running_pos_count_mean[pos_type] = (running_pos_count[pos_type] / train_steps % 100).cpu().numpy()

                        table_data = [[pos, count] for pos, count in zip(range(full_block_size), running_pos_count_mean[pos_type])]
                        table = wandb.Table(data=table_data, columns=["Position", "Average Count"])
                        log_metrics[f"randar_perturbed/{pos_type}_pos_distribution"] = table

                
                # Add RandAR prediction quality metrics if available
                if prediction_quality is not None:
                    # Log standard metrics
                    for key, value in prediction_quality.items():
                        if key not in ['rank_per_position', 'position_axis']:
                            log_metrics[f"randar_perturbed/{key}"] = value.item() if torch.is_tensor(value) and value.dim() == 0 else value

                    # Update and log the running average of rank per position
                    if 'rank_per_position' in prediction_quality:
                        # Update running totals
                        running_rank_sum += prediction_quality['rank_per_position']
                        running_rank_count += 1
                        
                        # Calculate the current running average
                        cumulative_avg_rank = running_rank_sum / running_rank_count
                        
                        # Prepare data for wandb plot
                        position_data = torch.arange(full_block_size, device=accelerator.device).cpu().numpy()
                        rank_data = cumulative_avg_rank.cpu().numpy()
                        
                        table_data = [[pos, rank] for pos, rank in zip(position_data, rank_data)]
                        table = wandb.Table(data=table_data, columns=["Position", "Average Rank"])
                        log_metrics["randar_perturbed/cumulative_rank_per_position"] = wandb.plot.bar(
                            table, "Position", "Average Rank", title="Cumulative Average Rank per Position"
                        )
                
                # Log rank per generation step if in confidence_scoring mode
                if config.training_params.supervision_mode == "confidence_scoring" and avg_rank_per_gen_step is not None:
                    running_gen_step_rank_sum += avg_rank_per_gen_step
                    running_gen_step_rank_count += 1
                    cumulative_avg_gen_step_rank = running_gen_step_rank_sum / running_gen_step_rank_count
                    
                    gen_step_data = torch.arange(full_block_size, device=accelerator.device).cpu().numpy()
                    gen_rank_data = cumulative_avg_gen_step_rank.cpu().numpy()
                    
                    gen_step_table_data = [[step, rank] for step, rank in zip(gen_step_data, gen_rank_data)]
                    gen_step_table = wandb.Table(data=gen_step_table_data, columns=["Generation Step", "Average Rank"])
                    log_metrics["confidence_scoring/cumulative_rank_per_gen_step"] = wandb.plot.bar(
                        gen_step_table, "Generation Step", "Average Rank", title="Cumulative Average Rank per Generation Step"
                    )

                accelerator.log(log_metrics, step=train_steps)
                
                running_loss, start_time = 0, time.time()

            if train_steps % args.ckpt_every == 0 and accelerator.is_main_process:
                ckpt_path = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}")
                accelerator.save_state(ckpt_path)
                logger.info(f"Saved Iter {train_steps} checkpoint to {ckpt_path}")

                # Clean up old checkpoints
                saved_ckpt_dirs = sorted(
                    [d for d in os.listdir(checkpoint_dir) if d.startswith("iters")],
                    key=lambda x: int(x.split('_')[-1])
                )
                if len(saved_ckpt_dirs) > args.keep_last_k:
                    for old_ckpt_dir in saved_ckpt_dirs[:-args.keep_last_k]:
                        shutil.rmtree(os.path.join(checkpoint_dir, old_ckpt_dir))
                        logger.info(f"Removed old checkpoint: {old_ckpt_dir}")

                # copy the checkpoint to the disk location
                if args.disk_location:
                    disk_location = os.path.join(args.disk_location, args.exp_name)
                    # using try-catch to bypass random disk error or quota issues
                    try:
                        if os.path.exists(disk_location):
                            shutil.rmtree(disk_location)
                        shutil.copytree(checkpoint_dir, disk_location)
                        logger.info(f"Copied checkpoint to {disk_location}")
                    except Exception as e:
                        logger.error(f"Error copying checkpoint to {disk_location}: {e}")

    # Save the final checkpoint
    if accelerator.is_main_process:
        final_ckpt_dir = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}_final")
        accelerator.save_state(final_ckpt_dir)
        logger.info(f"Saved Final Iter {train_steps} checkpoint to {final_ckpt_dir}")

    logger.info("Training Done.")
    accelerator.end_training()

    # using shutil to copy the final checkpoint to the disk location
    if args.disk_location:
        disk_location = os.path.join(args.disk_location, args.exp_name)
        if os.path.exists(disk_location):
            shutil.rmtree(disk_location)
        shutil.copytree(checkpoint_dir, disk_location)
        logger.info(f"Copied final checkpoint to {disk_location}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/corrector/corrector_base.yaml")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_corrector")
    parser.add_argument("--ar-model-config-path", type=str, default="configs/randar/randar_xl_0.7b.yaml")
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="Path to the pre-trained RandAR model checkpoint (.safetensors)")
    
    # Data related
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="latent", help="Dataset type, matches builder.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)

    # Resume training
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint directory to resume from (full state: model, optimizer, scheduler)")
    parser.add_argument("--continue-from-weight", type=str, default=None, help="Path to checkpoint file to load model weights from (fresh optimizer/scheduler states)")
    parser.add_argument("--resume-wandb", action="store_true", help="Resume wandb logging when resuming from checkpoint")

    # Logging and Checkpointing
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--keep-last-k", type=int, default=3)

    # W&B
    parser.add_argument("--wandb-entity", type=str, default="hxu129-hkust")
    parser.add_argument("--wandb-project", type=str, default="image-corrector-cvpr-26")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--disk-location", type=str, default='')
    args = parser.parse_args()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    main(args)
