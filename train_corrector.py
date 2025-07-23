import torch
from RandAR.corrector.corrector import LinearCorrector, MLPCorrector, TransformerCorrector
from RandAR.dataset.builder import build_dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from RandAR.corrector.utils import interleave_tokens
from RandAR.util import instantiate_from_config, load_safetensors
from argparse import ArgumentParser
from omegaconf import OmegaConf


# Note: the hard coded parameters need to be replaced with hyper-param
# later when the logic is ok
# The params below are from the RandAR corrector's config
cls_token_num = 1
block_size = 256
dim = 1024
n_head = 16
rope_base = 10000
vocab_size = 16384

# unique params for corrector
max_iters = 10000
num_workers = 8
per_gpu_batch_size = 16

def perturb_image_tokens(image_tokens, perturb_ratio=0.1):
    '''
    Args:
        image_tokens: [bsz, seq_len]
        perturb_ratio: float, the ratio of perturbed tokens
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
    # logits from corrector have shape [bs, 1 + 2 * block_size, 1]
    # we need to select the logits for the image token positions
    image_token_logits = logits[:, 1::2, :] # Shape: [bs, block_size, 1]
    
    # Squeeze the last dimension to match the target shape
    image_token_logits = image_token_logits.squeeze(-1) # Shape: [bs, block_size]
    
    # Target (perturbed_indices) also has shape [bs, block_size]
    # It needs to be converted to float for the loss function
    return F.binary_cross_entropy_with_logits(image_token_logits,
                                              perturbed_indices.float())

def main(args):
    # prepare data
    dataset = build_dataset(is_train=True, args=args, transform=transforms.ToTensor())
    data_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # prepare RandAR corrector instance
    gpt = instantiate_from_config(config.ar_model).to(device=device, dtype=precision)
    corrector_weight = load_safetensors(args.gpt_ckpt)
    gpt.load_state_dict(corrector_weight, strict=True)
    gpt.eval()

    # read pretrained modules from RandAR corrector
    tok_embeddings = gpt.tok_embeddings
    cls_embedding = gpt.cls_embedding
    get_position_instruction_tokens = gpt.get_position_instruction_tokens
    freqs_cis = gpt.freqs_cis

    # freeze imported modules
    for p in tok_embeddings.parameters():
        p.requires_grad = False
    for p in cls_embedding.parameters():
        p.requires_grad = False
    gpt.pos_instruct_embeddings.requires_grad = False

    # prepare corrector instance, optimizer, scheduler 
    corrector = TransformerCorrector()

    optimizer = torch.optim.AdamW(corrector.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    # start training
    corrector.train()
    for i in range(max_iters):
        x, y, inat_index = next(data_loader)
        image_tokens = x.reshape(x.shape[0], -1)
        cond = y.reshape(-1)

        # 1. prepare token sequence in the format of [cls_token, query_token_0, ..., query_token_n]
        # 1.1. prepare the token order
        bs = image_tokens.shape[0]
        token_order = torch.arange(block_size, device=image_tokens.device)
        token_order = token_order.unsqueeze(0).repeat(bs, 1)
        for i in range(bs):
            token_order[i] = token_order[i][torch.randperm(block_size)]
        token_order = token_order.contiguous()

        # 1.2. permute the image tokens according to the random order
        image_tokens = torch.gather(image_tokens.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()

        # 1.3. perturb the image tokens
        perturbed_image_tokens, perturbed_indices = perturb_image_tokens(image_tokens)

        # 1.4. prepare embeddings and freqs_cis
        freqs_cis = freqs_cis.to(cond.device)
        cond_embeddings = cls_embedding(cond, train=corrector.training)[
            :, : corrector.cls_token_num
        ] # [bsz, cls_token_num, dim]

        token_embeddings = tok_embeddings(perturbed_image_tokens)
        position_instruction_tokens = get_position_instruction_tokens(token_order) # [bsz, seq_len, dim]
        h = torch.cat(
            (cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)),
            dim=1
        )

        # 1.5 prepare 2d rope embd for the corrector
        token_freqs_cis = freqs_cis[cls_token_num:].clone().to(token_order.device)[token_order]
        token_freqs_cis = torch.cat(
            (freqs_cis[:cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), interleave_tokens(token_freqs_cis, token_freqs_cis)),
            dim=1
        )

        # 2. forward
        logits = corrector(h, token_freqs_cis)

        # 3. compute loss
        loss = compute_loss(logits, perturbed_indices)

        # 4. backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/randar/randar_l_0.3b_llamagen.yaml")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--ema", action="store_true", help="whether using ema training")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    # with 512 bs. 2.5k iters is 1 epoch
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=5000)  # save every 5k iters
    # keep last k checkpoints; 1 means only keep the last checkpoint
    parser.add_argument("--keep-last-k", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    # data
    parser.add_argument("--dataset", type=str, default="latent")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--visualize-every", type=int, default=2000)
    parser.add_argument("--visualize-num", type=int, default=32)
    # wandb
    parser.add_argument("--wandb-entity", type=str, default="hxu129-hkust")
    parser.add_argument("--wandb-project", type=str, default="image-corrector-cvpr-26")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--disk-location", type=str, default='')
    args = parser.parse_args()
    main(args)
