import subprocess
import os, sys
from glob import glob
from datetime import datetime
import math
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
from OmniAvatar.utils.args_config import parse_args

from OmniAvatar.utils.io_utils import load_state_dict 
from peft import LoraConfig, inject_adapter_in_model
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
import torch.distributed as dist
import torchvision.transforms as TT
from transformers import Wav2Vec2FeatureExtractor
import torchvision.transforms as transforms
import torch.nn.functional as F
from OmniAvatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg
from OmniAvatar.distributed.fsdp import shard_model
from OmniAvatar.utils.args_config import get_args

args = get_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 设置当前GPU
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU

def read_from_file(p):
    with open(p, "r") as fin:
        for l in fin:
            yield l.strip()

def match_size(image_size, h, w):
    ratio_ = 9999
    size_ = 9999
    select_size = None
    for image_s in image_size:
        ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
        size_tmp = abs(max(image_s) - max(w, h))
        if ratio_tmp < ratio_:
            ratio_ = ratio_tmp
            size_ = size_tmp
            select_size = image_s
        if ratio_ == ratio_tmp:
            if size_ == size_tmp:
                select_size = image_s
    return select_size

def resize_pad(image, ori_size, tgt_size):
    h, w = ori_size
    scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
    scale_h = int(h * scale_ratio)
    scale_w = int(w * scale_ratio)

    image = transforms.Resize(size=[scale_h, scale_w])(image)

    padding_h = tgt_size[0] - scale_h
    padding_w = tgt_size[1] - scale_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return image


class WanInferencePipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.rank}")
        if args.dtype=='bf16':
            self.dtype = torch.bfloat16
        elif args.dtype=='fp16':
            self.dtype = torch.float16
        else:   
            self.dtype = torch.float32
        self.pipe = self.load_model()
        if args.i2v:
            chained_trainsforms = []
            chained_trainsforms.append(TT.ToTensor())
            self.transform = TT.Compose(chained_trainsforms)
        if args.use_audio:
            from OmniAvatar.models.wav2vec import Wav2VecModel
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    args.wav2vec_path
                )
            self.audio_encoder = Wav2VecModel.from_pretrained(args.wav2vec_path, local_files_only=True).to(device=self.device)
            self.audio_encoder.feature_extractor._freeze_parameters()

    def load_model(self):
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
            from xfuser.core.distributed import (initialize_model_parallel,
                                                init_distributed_environment)

            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=self.args.sp_size,
                ring_degree=1,
                ulysses_degree=self.args.sp_size,
            )

            torch.cuda.set_device(dist.get_rank())
            rank = dist.get_rank()
        else:
            print("[Info] Running without distributed initialization (single-GPU inference)")
            rank = self.args.local_rank  # typically 0
        ckpt_path = f'{args.exp_path}/pytorch_model.pt'
        assert os.path.exists(ckpt_path), f"pytorch_model.pt not found in {args.exp_path}"
        if args.train_architecture == 'lora':
            args.pretrained_lora_path = pretrained_lora_path = ckpt_path
        else:
            resume_path = ckpt_path
        
        self.step = 0

        # Load models
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models(
            [
                args.dit_path.split(","),
                args.text_encoder_path,
                args.vae_path
            ],
            torch_dtype=self.dtype, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
            device='cpu',
        )

        pipe = WanVideoPipeline.from_model_manager(model_manager, 
                                                torch_dtype=self.dtype, 
                                                device=f"cuda:{rank}", 
                                                use_usp=True if args.sp_size > 1 else False,
                                                infer=True)
        if args.train_architecture == "lora":
            print(f'Use LoRA: lora rank: {args.lora_rank}, lora alpha: {args.lora_alpha}')
            self.add_lora_to_model(
                    pipe.denoising_model(),
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_target_modules=args.lora_target_modules,
                    init_lora_weights=args.init_lora_weights,
                    pretrained_lora_path=pretrained_lora_path,
                )
        else:
            missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(load_state_dict(resume_path), strict=True)
            print(f"load from {resume_path}, {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
        pipe.requires_grad_(False)
        pipe.eval()
        pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required. 
        if args.use_fsdp:
            shard_fn = partial(shard_model, device_id=self.device)
            pipe.dit = shard_fn(pipe.dit)
        return pipe
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    
    def forward(self, prompt, 
                image_path=None, 
                audio_path=None, 
                seq_len=101, # not used while audio_path is not None
                height=720, 
                width=720,
                overlap_frame=None,
                num_steps=None,
                negative_prompt=None,
                guidance_scale=None,
                audio_scale=None):
        overlap_frame = overlap_frame if overlap_frame is not None else self.args.overlap_frame
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale

        if image_path is not None:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            _, _, h, w = image.shape
            select_size = match_size(getattr(self.args, f'image_sizes_{self.args.max_hw}'), h, w)
            image = resize_pad(image, (h, w), select_size)
            image = image * 2.0 - 1.0
            image = image[:, :, None]
        else:
            image = None
            select_size = [height, width]
        L = int(args.max_tokens * 16 * 16 * 4 / select_size[0] / select_size[1])
        L = L // 4 * 4 + 1 if L % 4 != 0 else L - 3  # video frames
        T = (L + 3) // 4  # latent frames

        if self.args.i2v:
            if self.args.random_prefix_frames:
                fixed_frame = overlap_frame
                assert fixed_frame % 4 == 1
            else:
                fixed_frame = 1
            prefix_lat_frame = (3 + fixed_frame) // 4
            first_fixed_frame = 1
        else:
            fixed_frame = 0
            prefix_lat_frame = 0
            first_fixed_frame = 0


        if audio_path is not None and args.use_audio:
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
            input_values = np.squeeze(
                    self.wav_feature_extractor(audio, sampling_rate=16000).input_values
                )
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            ori_audio_len = audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
            input_values = input_values.unsqueeze(0)
            # padding audio
            if audio_len < L - first_fixed_frame:
                audio_len = audio_len + ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
            elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
                audio_len = audio_len + ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
            input_values = F.pad(input_values, (0, audio_len * int(self.args.sample_rate / self.args.fps) - input_values.shape[1]), mode='constant', value=0)
            with torch.no_grad():
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            seq_len = audio_len
            audio_embeddings = audio_embeddings.squeeze(0)
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
        else:
            audio_embeddings = None

        # loop
        times = (seq_len - L + first_fixed_frame) // (L-fixed_frame) + 1
        if times * (L-fixed_frame) + fixed_frame < seq_len:
            times += 1
        video = []
        image_emb = {}
        img_lat = None
        if args.i2v:
            self.pipe.load_models_to_device(['vae'])
            img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)

            msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1])
            image_cat = img_lat.repeat(1, 1, T, 1, 1)
            msk[:, :, 1:] = 1
            image_emb["y"] = torch.cat([image_cat, msk], dim=1)
        for t in range(times):
            print(f"[{t+1}/{times}]")
            audio_emb = {}
            if t == 0:
                overlap = first_fixed_frame
            else:
                overlap = fixed_frame
                image_emb["y"][:, -1:, :prefix_lat_frame] = 0 # 第一次推理是mask只有1，往后都是mask overlap
            prefix_overlap = (3 + overlap) // 4
            if audio_embeddings is not None:
                if t == 0:
                    audio_tensor = audio_embeddings[
                            :min(L - overlap, audio_embeddings.shape[0])
                        ]
                else:
                    audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                    audio_tensor = audio_embeddings[
                        audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])
                    ]
                    
                audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                audio_prefix = audio_tensor[-fixed_frame:]
                audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                audio_emb["audio_emb"] = audio_tensor
            else:
                audio_prefix = None
            if image is not None and img_lat is None:
                self.pipe.load_models_to_device(['vae'])
                img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
                assert img_lat.shape[2] == prefix_overlap
            img_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))], dim=2)
            frames, _, latents = self.pipe.log_video(img_lat, prompt, prefix_overlap, image_emb, audio_emb,
                                                 negative_prompt, num_inference_steps=num_steps, 
                                                 cfg_scale=guidance_scale, audio_cfg_scale=audio_scale if audio_scale is not None else guidance_scale,
                                                 return_latent=True,
                                                 tea_cache_l1_thresh=args.tea_cache_l1_thresh,tea_cache_model_id="Wan2.1-T2V-14B")
            img_lat = None
            image = (frames[:, -fixed_frame:].clip(0, 1) * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()
            if t == 0:
                video.append(frames)
            else:
                video.append(frames[:, overlap:])
        video = torch.cat(video, dim=1)
        video = video[:, :ori_audio_len + 1]
        return video

