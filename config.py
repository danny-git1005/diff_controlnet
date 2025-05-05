class Config:
    # Project settings
    project_name = "diff_phase_3070ti"
    output_dir = "outputs"
    
    # Data settings
    data_dir = "E:/aerial_img/phase2_diffusion_mdoel/data/japna_train_data"
    image_dir = f"{data_dir}/japan_label"
    condition_dir = f"{data_dir}/japan_mask"
    prompt_file = f"{data_dir}/image_analysis_results.json"
    
    # Training settings
    base_model = "runwayml/stable-diffusion-v1-5"
    resolution = 512
    train_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    max_train_steps = 15000
    save_steps = 1000
    validation_steps = 500
    
    # ControlNet settings
    controlnet_model = "lllyasviel/sd-controlnet-seg"  # Default model
    condition_type = None  # Options: canny, depth, pose, seg, None
    use_text_condition = False  # Use both image and text conditions
    
    # Hardware settings
    mixed_precision = "no"  # Options: "no", "fp16", "bf16"
    
    # Wandb settings
    wandb_api_key = None  # Set your API key here or through environment variable
    log_wandb = True
    
    # Inference settings
    num_inference_steps = 100
    guidance_scale = 7.5