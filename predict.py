from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download
import subprocess
import uuid
from pathlib import Path as SysPath
def download_if_not_exists(repo_id, local_dir):
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"✅ 模型已存在，跳过下载: {local_dir}")
        return
    print(f"🔽 正在下载模型: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

# 用法：
# class Predictor(BasePredictor):
#     def setup(self):
        # print("🔽 正在下载模型...")
        # snapshot_download(
        #     repo_id="OmniAvatar/OmniAvatar-1.3B",
        #     local_dir="pretrained_models/OmniAvatar-1.3B",
        #     local_dir_use_symlinks=False
        # )
        # snapshot_download(
        #     repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        #     local_dir="pretrained_models/Wan2.1-T2V-1.3B",
        #     local_dir_use_symlinks=False
        # )
        # snapshot_download(
        #     repo_id="facebook/wav2vec2-base-960h",
        #     local_dir="pretrained_models/wav2vec2-base-960h",
        #     local_dir_use_symlinks=False
        # )

#     def predict(self, prompt: str = Input(description="Text + image + audio triple prompt")) -> Path:
#         input_dir = SysPath("tmp_inputs")
#         input_dir.mkdir(exist_ok=True)
#         input_file = input_dir / f"{uuid.uuid4().hex}.txt"
#         input_file.write_text(prompt.strip())

#         cmd = [
#             "torchrun",
#             "--standalone",
#             "--nproc_per_node=1",
#             "scripts/inference.py",
#             "--config", "configs/inference_1.3B.yaml",
#             "--input_file", str(input_file)
#         ]

#         result = subprocess.run(cmd, capture_output=True, text=True)
#         print("STDOUT:", result.stdout)
#         print("STDERR:", result.stderr)

#         output_dir = SysPath("output")
#         if output_dir.exists():
#             files = sorted(output_dir.glob("*.png"))
#             if files:
#                 return Path(str(files[0]))

#         raise RuntimeError("No output file generated.")

from cog import BasePredictor, Input, Path
from OmniAvatar.utils.pipeline import WanInferencePipeline, set_seed
from OmniAvatar.utils.args_config import parse_args
from OmniAvatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
from OmniAvatar.utils.args_config import get_args
import os
import uuid
import torch
import shutil

class Predictor(BasePredictor):
    def setup(self):
        print("🚀 初始化模型...")
        download_if_not_exists("OmniAvatar/OmniAvatar-1.3B", "pretrained_models/OmniAvatar-1.3B")
        download_if_not_exists("Wan-AI/Wan2.1-T2V-1.3B", "pretrained_models/Wan2.1-T2V-1.3B")
        download_if_not_exists("facebook/wav2vec2-base-960h", "pretrained_models/wav2vec2-base-960h")
        self.args = get_args()
        self.pipe = WanInferencePipeline(self.args)
        set_seed(self.args.seed)

    def predict(
        self,
        prompt: str = Input(description="文本提示"),
        image_path: str = Input(default=None, description="图像路径"),
        audio_path: str = Input(default=None, description="音频路径"),
    ) -> Path:
        print("📝 接收输入:", prompt)
        text = prompt
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError(f"图像路径无效: {image_path}")
        if audio_path and not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频路径无效: {audio_path}")

        output_dir = f"outputs/{uuid.uuid4().hex}"
        os.makedirs(output_dir, exist_ok=True)

        # 添加静音段
        if self.args.silence_duration_s > 0 and audio_path:
            input_audio_path = os.path.join(output_dir, "audio_input.wav")
            add_silence_to_audio_ffmpeg(audio_path, input_audio_path, self.args.silence_duration_s)
        else:
            input_audio_path = audio_path

        print("🎬 开始生成视频...")
        video = self.pipe(
            prompt=text,
            image_path=image_path,
            audio_path=input_audio_path,
            seq_len=self.args.seq_len
        )

        # 添加一帧延迟音频
        if audio_path and self.args.use_audio:
            audio_out_path = os.path.join(output_dir, "audio_out.wav")
            add_silence_to_audio_ffmpeg(audio_path, audio_out_path, 1.0 / self.args.fps + self.args.silence_duration_s)
        else:
            audio_out_path = None

        prompt_path = os.path.join(output_dir, "prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(text)

        print("💾 保存视频结果...")
        result_path = save_video_as_grid_and_mp4(
            video, output_dir, self.args.fps, prompt=text,
            prompt_path=prompt_path,
            audio_path=audio_out_path,
            prefix="result"
        )

        # 返回 mp4 文件
        return Path(result_path)
