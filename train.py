import os
import json
import argparse
import torch
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader

from config import Config
from utils.dataset import ControlNetDataset
from utils.image_processors import get_processor_for_condition

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ControlNet model on custom data")
    parser.add_argument("--config", type=str, default="config.py", help="Path to config file")
    parser.add_argument("--condition_type", type=str, choices=["canny", "depth", "pose", "seg"],
                       help="Type of condition to use")
    parser.add_argument("--use_text_condition", default=True, help="Use text prompts along with image conditions")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()
    
    # Update config from args
    if args.condition_type:
        config.condition_type = args.condition_type
    if args.use_text_condition :
        config.use_text_condition = args.use_text_condition
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize wandb
    if config.log_wandb:
        if config.wandb_api_key:
            os.environ["WANDB_API_KEY"] = config.wandb_api_key
        wandb.init(entity="danny_paper_project" ,project=config.project_name, config=vars(config))
    
    # Set up accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    # Load the ControlNet model
    controlnet = ControlNetModel.from_pretrained(
        config.controlnet_model,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    )
    
    # Load the StableDiffusion pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        config.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    )
    pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
    print("accelerator divice : ",accelerator.device)
    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)
    pipeline.unet.to(accelerator.device)
    
    # Load prompts if using text conditions
    prompts = {}
    if config.use_text_condition and os.path.exists(config.prompt_file):
        with open(config.prompt_file, 'r', encoding="utf-8") as f:
            prompts = json.load(f)
    # Set up the training dataset
    # processor = get_processor_for_condition(config.condition_type)
    
    # Create dataset and dataloader
    train_dataset = ControlNetDataset(
        image_dir=config.image_dir,
        condition_dir=config.condition_dir,
        prompts=prompts,
        # processor=processor,
        resolution=config.resolution,
        use_text_condition=config.use_text_condition
    )
    
    sample = train_dataset[0]
    image = sample["images"]
    condition_image = sample["condition_images"]
    prompt = sample["prompts"]

    # 如果是 torch tensor，轉回 PIL 或 numpy
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)
    if isinstance(condition_image, torch.Tensor):
        condition_image = TF.to_pil_image(condition_image)

    print("Prompt",prompt)
    # 顯示圖像與條件圖像
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(condition_image)
    axes[1].set_title("Condition Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=config.learning_rate
    )
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.max_train_steps,
    )
    
    # Prepare everything with accelerator
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training loop
    global_step = 0
    best_loss = float("inf")
    progress_bar = tqdm(total=config.max_train_steps, disable=not accelerator.is_local_main_process)
    
    for epoch in range(config.max_train_steps // len(train_dataloader) + 1):
        controlnet.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Get batch data
                images = batch["images"]
                images = images.to(dtype=torch.float32) / 255.0  # <- 新增這行
                condition_images = batch["condition_images"]
                condition_images = condition_images.to(dtype=torch.float32) / 255.0  # <- 新增這行
                
                
                # If using text conditions
                
                if config.use_text_condition:
                    prompts = batch["prompts"]
                else:
                    prompts = [""] * len(images)  # Empty prompts if not using text
                
                # Forward pass through the model
                with torch.no_grad():
                    latents = pipeline.vae.encode(images).latent_dist.sample() * 0.18215
                
                # Get noise and noisy latents
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                
                text_inputs = pipeline.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(accelerator.device)

                encoder_hidden_states = pipeline.text_encoder(**text_inputs).last_hidden_state
                
                controlnet_outputs = pipeline.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=condition_images,
                    return_dict=True,
                )
                
                # Predict noise with ControlNet
                noise_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=controlnet_outputs.down_block_res_samples,
                    mid_block_additional_residual=controlnet_outputs.mid_block_res_sample,
                ).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            
            # Log to wandb
            if config.log_wandb and global_step % 10 == 0:
                wandb.log({
                    "train_loss": loss.detach().item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                })
            
            # Save checkpoint if loss is lower than the best loss
            current_loss = loss.detach().item()
            if global_step % config.save_steps == 0:
                if current_loss < best_loss:
                    # Save model
                    pipeline.save_pretrained(f"{config.output_dir}/checkpoint-{global_step}")
                    best_loss = current_loss
                    print(f"New best loss: {best_loss:.6f} - Saved checkpoint at step {global_step}")
            
            # Validation
            if global_step % config.validation_steps == 0:
                controlnet.eval()
                # Generate and log validation images
                if config.log_wandb:
                    validation_images = []
                    for i in range(min(2, len(train_dataset))):
                        sample = train_dataset[i]
                        prompt = sample["prompts"] if config.use_text_condition else ""
                        condition_image = sample["condition_images"].float().div(255).unsqueeze(0).to(accelerator.device)
                        
                        pipeline.to(accelerator.device)
                        
                        with torch.no_grad():
                            image = pipeline(
                                prompt,
                                condition_image,
                                num_inference_steps=config.num_inference_steps,
                                guidance_scale=config.guidance_scale
                            ).images[0]
                        
                        validation_images.append(wandb.Image(
                            image, 
                            caption=f"Step {global_step}: {prompt}"
                        ))
                    
                    wandb.log({"validation_images": validation_images})
                controlnet.train()
            
            if global_step >= config.max_train_steps:
                break
    
    # Save final model
    pipeline.save_pretrained(f"{config.output_dir}/final-model")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
