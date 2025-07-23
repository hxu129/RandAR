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

import os
import time
import argparse
import sys
sys.path.append("./")
import shutil

from omegaconf import OmegaConf
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

from RandAR.util import instantiate_from_config, load_safetensors
from RandAR.dataset.builder import build_dataset
from RandAR.utils.logger import create_logger
from RandAR.utils.lr_scheduler import get_scheduler
from RandAR.corrector.utils import interleave_tokens

def cycle(dl: torch.utils.data.DataLoader):
    while True:
        for data in dl:
            yield data

def perturb_image_tokens(image_tokens, perturb_ratio, vocab_size):
    '''
    Args:
        image_tokens: [bsz, seq_len]
        perturb_ratio: float, the ratio of perturbed tokens
        vocab_size: int, the size of the vocabulary to sample replacements from
    '''
    bs, seq_len = image_tokens.shape
    device = image_tokens.device
    
    # 1. Determine which tokens to perturb
    # Note: this is stochastic, and the number of perturbed tokens is not guaranteed to be the same as perturb_ratio
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
            
    return perturbed_image_tokens, perturbed_indices # [bsz, seq_len], [bsz, seq_len]

def compute_loss(logits, perturbed_indices):
    # Logits from corrector have shape [bs, 1 + 2 * block_size, 1]
    # We need to select the logits for the image token positions.
    image_token_logits = logits[:, 1::2, :] # Shape: [bs, block_size, 1]
    
    # Squeeze the last dimension to match the target shape
    image_token_logits = image_token_logits.squeeze(-1) # Shape: [bs, block_size]
    
    # Target (perturbed_indices) also has shape [bs, block_size]
    # It needs to be converted to float for the loss function
    return F.binary_cross_entropy_with_logits(image_token_logits,
                                              perturbed_indices.float())

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    config = OmegaConf.load(args.config)
    ar_config = OmegaConf.load(config.ar_model_config_path)

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
    for param in gpt.parameters():
        param.requires_grad = False
    del gpt_weight

    # --- Extract necessary modules from the frozen GPT ---
    tok_embeddings = gpt.tok_embeddings
    cls_embedding = gpt.cls_embedding
    get_position_instruction_tokens = gpt.get_position_instruction_tokens
    freqs_cis = gpt.freqs_cis

    # --- Create Corrector Model ---
    corrector = instantiate_from_config(config.corrector_model).to(accelerator.device)
    logger.info(f"Corrector Parameters: {sum(p.numel() for p in corrector.parameters()):,}")

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
    
    #################### Wandb Setup ####################
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            init_kwargs={
                "wandb": {
                    "entity": args.wandb_entity,
                    "config": OmegaConf.to_container(config, resolve=True),
                    "name": args.exp_name,
                    "dir": experiment_dir,
                }
            },
        )
        
    ################## Resume Training ##################
    train_steps = 0
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        saved_ckpt_dirs = [_ for _ in os.listdir(checkpoint_dir) if _.startswith("iters")]
        if len(saved_ckpt_dirs) > 0:
            saved_ckpt_dirs = sorted(saved_ckpt_dirs, key=lambda x: int(x.split('_')[-1]))
            ckpt_dir = os.path.join(checkpoint_dir, saved_ckpt_dirs[-1])
            if accelerator.is_main_process:
                logger.info(f"Resuming from checkpoint: {ckpt_dir}")
            accelerator.load_state(ckpt_dir)
            train_steps = int(saved_ckpt_dirs[-1].split("_")[-1])

    #################### Training Loop ####################
    corrector.train()
    total_iters = config.training_params.max_iters
    log_iters, running_loss, start_time = 0, 0, time.time()
    block_size = config.corrector_model.params.block_size
    cls_token_num = config.corrector_model.params.cls_token_num

    logger.info(f"Starting training from iteration {train_steps} to {total_iters}")
    while train_steps < total_iters:
        x, y, _ = next(data_loader)
        image_tokens = x.reshape(x.shape[0], -1)
        cond = y.reshape(-1)
        bs = image_tokens.shape[0]

        with accelerator.accumulate(corrector):
            # 1. Prepare token sequence
            token_order = torch.arange(block_size, device=accelerator.device).unsqueeze(0).repeat(bs, 1)
            for i in range(bs):
                token_order[i] = token_order[i][torch.randperm(block_size)]
            
            permuted_image_tokens = torch.gather(image_tokens, 1, token_order)

            perturbed_tokens, perturbed_indices = perturb_image_tokens(
                permuted_image_tokens,
                config.training_params.perturb_ratio,
                config.corrector_model.params.vocab_size
            )

            # 2. Prepare embeddings using the frozen gpt model
            cond_embeddings = cls_embedding(cond, train=False)[:, :cls_token_num]
            token_embeddings = tok_embeddings(perturbed_tokens)
            pos_instruct_tokens = get_position_instruction_tokens(token_order)
            
            h = torch.cat(
                (cond_embeddings, interleave_tokens(pos_instruct_tokens, token_embeddings)),
                dim=1
            )
            
            # 3. Prepare RoPE embeddings
            token_freqs_cis = freqs_cis[cls_token_num:].clone()[token_order]
            token_freqs_cis = torch.cat(
                (freqs_cis[:cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), interleave_tokens(token_freqs_cis, token_freqs_cis)),
                dim=1
            )

            # 4. Forward pass through corrector
            logits = corrector(h, token_freqs_cis)
            loss = compute_loss(logits, perturbed_indices, block_size)
            
            # 5. Backward pass
            accelerator.backward(loss)
            if config.optimizer.max_grad_norm > 0:
                accelerator.clip_grad_norm_(corrector.parameters(), config.optimizer.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            running_loss += accelerator.gather(loss.repeat(per_gpu_batch_size)).mean().item() / config.accelerator.gradient_accumulation_steps

        if accelerator.sync_gradients:
            train_steps += 1
            
            if train_steps % args.log_every == 0 and accelerator.is_main_process:
                average_loss = running_loss / args.log_every
                end_time = time.time()
                average_time = (end_time - start_time) / args.log_every

                logger.info(f"Step {train_steps:08d} | Loss {average_loss:.4f} | Time {average_time:.4f}s | LR {lr_scheduler.get_last_lr()[0]:.6f}")
                accelerator.log({"loss": average_loss, "lr": lr_scheduler.get_last_lr()[0], "time": average_time}, step=train_steps)
                
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

    # Save the final checkpoint
    if accelerator.is_main_process:
        final_ckpt_dir = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}_final")
        accelerator.save_state(final_ckpt_dir)
        logger.info(f"Saved Final Iter {train_steps} checkpoint to {final_ckpt_dir}")

    logger.info("Training Done.")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/corrector/corrector_base.yaml")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_corrector")
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="Path to the pre-trained RandAR model checkpoint (.safetensors)")
    
    # Data related
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="latent", help="Dataset type, matches builder.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)

    # Logging and Checkpointing
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--keep-last-k", type=int, default=3)

    # W&B
    parser.add_argument("--wandb-entity", type=str, default="hxu129-hkust")
    parser.add_argument("--wandb-project", type=str, default="image-corrector-cvpr-26")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--disk-location", type=str, default='')
    args = parser.parse_args()
    main(args)
