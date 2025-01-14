# HuggingFace Cache to save T5 text encoder, video tokenizer, prompt upsampler, and guardrails weights.
# export PYTHONPATH=/maindata/data/user/chunli.peng/envs/anaconda3/envs/cosmos/lib/python3.10/site-packages/
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_TOKEN="<your/HF/access/token>"
export HF_HOME="checkpoints/"

# Number of GPU devices available for inference. Supports up to 8 GPUs for accelerated inference.
export NUM_DEVICES=1

# Prompt describing world scene and actions taken by subject (if provided).
# export PROMPT="The teal robot is cooking food in a kitchen. Steam rises from a simmering pot as the robot chops vegetables on a worn wooden cutting board. Copper pans hang from an overhead rack, catching glints of afternoon light, while a well-loved cast iron skillet sits on the stovetop next to scattered measuring spoons and a half-empty bottle of olive oil."
export PROMPT=eval/prompt.jsonl


python cosmos1/models/diffusion/inference/text2world.py --num_steps 35 \
    --height 704 --width 1280 --num_video_frames 121 --fps 16 \
    --video_save_folder eval/output_14B_promptrefiner \
    --tokenizer_dir Cosmos-1.0-Tokenizer-CV8x8x8 \
    --text_enc_dir models--google-t5--t5-11b/snapshots/90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3 \
    --batch_input_path ${PROMPT} \
    --checkpoint_dir /maindata/data/user/chunli.peng/genie/Cosmos/checkpoints \
    --diffusion_transformer_dir "Cosmos-1.0-Diffusion-14B-Text2World" \
    --offload_diffusion_transformer \
    --offload_tokenizer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \

python cosmos1/models/diffusion/inference/text2world.py --num_steps 35 \
    --height 704 --width 1280 --num_video_frames 121 --fps 16 \
    --video_save_folder eval/output_14B_disable_promptrefiner \
    --tokenizer_dir Cosmos-1.0-Tokenizer-CV8x8x8 \
    --text_enc_dir models--google-t5--t5-11b/snapshots/90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3 \
    --batch_input_path ${PROMPT} \
    --checkpoint_dir /maindata/data/user/chunli.peng/genie/Cosmos/checkpoints \
    --diffusion_transformer_dir "Cosmos-1.0-Diffusion-14B-Text2World" \
    --offload_diffusion_transformer \
    --offload_tokenizer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --disable_prompt_upsampler
    
python cosmos1/models/diffusion/inference/text2world.py --num_steps 35 \
    --height 704 --width 1280 --num_video_frames 121 --fps 16 \
    --video_save_folder eval/output_7B_promptrefiner \
    --tokenizer_dir Cosmos-1.0-Tokenizer-CV8x8x8 \
    --text_enc_dir models--google-t5--t5-11b/snapshots/90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3 \
    --batch_input_path ${PROMPT} \
    --checkpoint_dir /maindata/data/user/chunli.peng/genie/Cosmos/checkpoints \
    --diffusion_transformer_dir "Cosmos-1.0-Diffusion-7B-Text2World" \
    --offload_diffusion_transformer \
    --offload_tokenizer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \