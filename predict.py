import subprocess
import uuid
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

def predict(prompt: str):
    model_dir = snapshot_download(
        repo_id="OmniAvatar/OmniAvatar-1.3B",
        local_dir="pretrained_models/OmniAvatar-1.3B",
        local_dir_use_symlinks=False
    )

    print(f"✅ 模型已下载至: {model_dir}")

    model_dir = snapshot_download(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        local_dir="pretrained_models/Wan2.1-T2V-1.3B",
        local_dir_use_symlinks=False
    )

    print(f"✅ 模型已下载至: {model_dir}")

    model_dir = snapshot_download(
        repo_id="facebook/wav2vec2-base-960h",
        local_dir="pretrained_models/wav2vec2-base-960h",
        local_dir_use_symlinks=False
    )

    print(f"✅ 模型已下载至: {model_dir}")
    
    # 2. 写入复合格式输入
    input_dir = Path("tmp_inputs")
    input_dir.mkdir(exist_ok=True)
    input_file = input_dir / f"{uuid.uuid4().hex}.txt"
    input_file.write_text(prompt.strip())  # 🧠 prompt 是完整的一行含文本+图像+音频路径

    # 2. 构建命令
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "scripts/inference.py",
        "--config", "configs/inference_1.3B.yaml",
        "--input_file", str(input_file)
    ]

    # 3. 运行推理
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 4. 你需要根据 inference.py 的逻辑来提取输出路径
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    # 5. 假设输出图像保存在 output/ 目录
    output_path = Path("output")  # ← 按照实际输出位置修改
    if output_path.exists():
        files = list(output_path.glob("*.png"))  # 假设生成的是 PNG
        if files:
            return str(files[0])
    
    return "No output generated"
