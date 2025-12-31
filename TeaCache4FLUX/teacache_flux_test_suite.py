# -*- coding: utf-8 -*-
"""
TeaCache for FLUX 集成测试脚本（简化版）
- 固定参数在脚本内设置，避免命令行传参，直接运行即可
- 中文注释，便于理解与修改
- 支持：关闭/开启 offload、开启快速注意力、ABBA交替顺序验证、阈值多档、重复次数与 warmup
- 输出：结果汇总 CSV（summary.csv），图像输出到 images 目录

使用（示例）：
    CUDA_VISIBLE_DEVICES=1 python teacache_flux_test_suite.py
"""

import os
import time
import csv
import re
import torch
import numpy as np
from PIL import Image

from diffusers import DiffusionPipeline
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_metric
import lpips

from diffusers.models import FluxTransformer2DModel
from teacache_flux import teacache_forward


# ===========================
# 固定参数配置（可在此处修改）
# ===========================
CONFIG = {
    # 提示词文件路径（至少20条）
    "prompts_path": "prompts.txt",
    # 输出主目录
    "outdir": "flux_teacache_test_suite",
    # 生成图片的输出子目录（自动创建）
    "images_subdir": "images",
    # 推理步数（FLUX 默认28）
    "steps": 28,
    # 基础随机种子
    "seed": 42,
    # TeaCache 阈值档位（可增删）
    "thresholds": [0.25, 0.4, 0.6, 0.8],
    # 每个 prompt 的重复次数（建议 3 或 5）
    "repeats": 3,
    # 每个设置的 warmup 次数（不计入统计，稳定内核与显存）
    "warmup": 1,
    # 是否关闭 offload（True：整管线驻留 GPU，速度测试推荐；False：省显存模式）
    "no_offload": True,
    # 是否尝试启用快速注意力（xFormers/SDPA/FA2），需要环境支持
    "enable_fast_attn": False,
    # 是否运行 ABBA 交替顺序验证（baseline→TeaCache→baseline→TeaCache）
    "run_abba": False,
    # 是否开启严格 CUDA 确定性（需要预先设置 CUBLAS_WORKSPACE_CONFIG）
    "strict_cuda_determinism": False,
}


# ===========================
# 工具函数
# ===========================
def set_seed(seed: int, deterministic: bool = True, strict_cuda_determinism: bool = False):
    """
    设置随机种子与确定性选项。
    - deterministic=True：开启 cudnn 的确定性；不启用 torch.use_deterministic_algorithms(True)，避免 CuBLAS 报错
    - strict_cuda_determinism=True：尝试启用 torch.use_deterministic_algorithms(True)，需要环境变量 CUBLAS_WORKSPACE_CONFIG
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False

    if strict_cuda_determinism:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # 若环境未设置 CUBLAS_WORKSPACE_CONFIG，会失败；此处忽略异常以保证可运行
            pass


def load_prompts(path: str):
    """
    从文件读取提示词，每行一个提示，去除空行与两端空白。
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def tensor_from_pil(img: Image.Image):
    """
    将 PIL 图像转为张量 [C,H,W]，范围 [0,1]，用于 SSIM/PSNR。
    """
    return transforms.ToTensor()(img)


def safe_slug(text: str, maxlen: int = 60):
    """
    将提示词文本转为安全的文件名片段（只保留字母数字、短横线、下划线），并限制长度。
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-_]+", "_", text)
    return text[:maxlen].strip("_")


def make_generator(seed: int, device: torch.device):
    """
    根据种子与设备创建 torch.Generator。
    - GPU 下用 device=cuda，可减少设备不一致带来的非复现性。
    """
    g = torch.Generator(device=device if device.type == "cuda" else "cpu")
    g.manual_seed(seed)
    return g


def synchronize_if_cuda():
    """
    若使用 GPU，则在计时前后同步，确保时间测量准确。
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def measure_latency_generate(pipeline: DiffusionPipeline, prompt: str, steps: int, generator: torch.Generator):
    """
    端到端生成计时：关闭进度条、在调用前后同步 GPU、返回耗时与生成的首张图像。
    """
    try:
        pipeline.set_progress_bar_config(disable=True)
    except Exception:
        pass

    synchronize_if_cuda()
    start = time.perf_counter()
    out = pipeline(prompt, num_inference_steps=steps, generator=generator)
    synchronize_if_cuda()
    elapsed = time.perf_counter() - start
    image = out.images[0]
    return elapsed, image


def compute_psnr_from_arrays(ref_np: np.ndarray, cmp_np: np.ndarray, data_range: float = 1.0, eps: float = 1e-10):
    """
    计算 PSNR（避免 MSE=0 导致 inf），当完全相同时，最终由快速路径覆盖，PSNR=100dB。
    """
    mse = np.mean((ref_np - cmp_np) ** 2, dtype=np.float64)
    mse = max(mse, eps)
    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(psnr)


@torch.inference_mode()
def compute_image_metrics(img_ref: Image.Image, img_cmp: Image.Image, lpips_net):
    """
    计算 LPIPS/SSIM/PSNR：
    - 完全相同图像走快速路径：LPIPS=0, SSIM=1, PSNR=100dB（有限大值）
    - LPIPS 使用 AlexNet backbone，输入需归一化到 [-1,1]
    - SSIM/PSNR 使用 [0,1] 的 numpy 数组
    """
    ref_u8 = np.asarray(img_ref)
    cmp_u8 = np.asarray(img_cmp)
    if ref_u8.shape == cmp_u8.shape and np.array_equal(ref_u8, cmp_u8):
        return 0.0, 1.0, 100.0

    # LPIPS
    t_ref = tensor_from_pil(img_ref)
    t_cmp = tensor_from_pil(img_cmp)
    t_ref_lpips = (t_ref.unsqueeze(0) * 2 - 1).float()
    t_cmp_lpips = (t_cmp.unsqueeze(0) * 2 - 1).float()
    lpips_score = float(lpips_net(t_ref_lpips, t_cmp_lpips).item())

    # SSIM/PSNR
    ref_np = ref_u8.astype(np.float32) / 255.0
    cmp_np = cmp_u8.astype(np.float32) / 255.0
    ssim_score = float(ssim_metric(ref_np, cmp_np, data_range=1.0, channel_axis=-1))
    psnr_score = compute_psnr_from_arrays(ref_np, cmp_np, data_range=1.0)
    return lpips_score, ssim_score, psnr_score


def enable_teacache_on_pipeline(pipeline: DiffusionPipeline, steps: int, rel_l1_thresh: float):
    """
    启用 TeaCache：猴子补丁替换 forward，并设置类级别的 TeaCache 控制参数。
    """
    FluxTransformer2DModel.forward = teacache_forward
    transformer_cls = pipeline.transformer.__class__
    transformer_cls.enable_teacache = True
    transformer_cls.cnt = 0
    transformer_cls.num_steps = steps
    transformer_cls.rel_l1_thresh = rel_l1_thresh
    transformer_cls.accumulated_rel_l1_distance = 0
    transformer_cls.previous_modulated_input = None
    transformer_cls.previous_residual = None


def disable_teacache_restore_forward(original_forward):
    """
    禁用 TeaCache：恢复原始 forward，并重置类级别标记，避免副作用。
    """
    FluxTransformer2DModel.forward = original_forward
    try:
        transformer_cls = FluxTransformer2DModel
        for attr in [
            "enable_teacache", "cnt", "num_steps", "rel_l1_thresh",
            "accumulated_rel_l1_distance", "previous_modulated_input", "previous_residual"
        ]:
            if hasattr(transformer_cls, attr):
                if attr == "enable_teacache":
                    setattr(transformer_cls, attr, False)
                elif attr in ["cnt", "accumulated_rel_l1_distance"]:
                    setattr(transformer_cls, attr, 0)
                else:
                    setattr(transformer_cls, attr, None)
    except Exception:
        pass


def maybe_enable_fast_attention(pipeline: DiffusionPipeline):
    """
    尝试启用快速注意力（xFormers/SDPA/FA2），需要环境支持。
    """
    if not torch.cuda.is_available():
        return
    try:
        from diffusers.utils.torch_utils import is_xformers_available
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
        if hasattr(pipeline, "transformer") and hasattr(pipeline.transformer, "set_attn_processor"):
            pipeline.transformer.set_attn_processor(AttnProcessor2_0())
    except Exception:
        pass


def prepare_pipeline(no_offload: bool, enable_fast_attn: bool):
    """
    加载与准备 FLUX 管线：
    - no_offload=True：整管线驻留 GPU（推荐速度测试）
    - no_offload=False：开启 offload，常驻 CPU，子模块按需迁移（省显存但更慢）
    - enable_fast_attn=True：尝试启用快速注意力（如环境支持）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preferred_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=preferred_dtype)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    if no_offload:
        pipe.to(device)
        if device.type == "cpu":
            # 若在 CPU 上运行，确保关键模块使用 float32，避免 LayerNorm FP16 报错
            for name in ["text_encoder", "text_encoder_2", "transformer", "vae"]:
                if hasattr(pipe, name) and getattr(pipe, name) is not None:
                    getattr(pipe, name).to(dtype=torch.float32)
    else:
        # 开启 offload，避免手动迁移到 GPU
        pipe.enable_model_cpu_offload()
        try:
            pipe.to("cpu")
        except Exception:
            pass

    if enable_fast_attn and torch.cuda.is_available():
        maybe_enable_fast_attention(pipe)

    return pipe, device


# ===========================
# 具体测试流程
# ===========================
def run_baseline(pipeline, device, prompt, steps, repeats, warmup, out_dir, p_idx, slug, base_seed):
    """
    运行 baseline：按 repeats 次数生成，并保存图像与耗时。
    """
    for _ in range(max(warmup, 0)):
        g_warm = make_generator(seed=base_seed, device=device)
        _ = measure_latency_generate(pipeline, prompt, steps, g_warm)
    latencies = []
    images = []
    for r in range(repeats):
        g = make_generator(seed=base_seed + r, device=device)
        lat, img = measure_latency_generate(pipeline, prompt, steps, g)
        img.save(os.path.join(out_dir, f"{p_idx:03d}_baseline_{slug}_{r}.png"))
        latencies.append(lat)
        images.append(img)
    return latencies, images


def run_teacache(pipeline, device, prompt, steps, repeats, warmup, out_dir, p_idx, slug, thr, original_forward, base_seed):
    """
    运行 TeaCache：启用指定阈值，按 repeats 次数生成，并保存图像与耗时；结束后恢复 forward。
    """
    enable_teacache_on_pipeline(pipeline, steps, thr)
    for _ in range(max(warmup, 0)):
        g_warm = make_generator(seed=base_seed, device=device)
        _ = measure_latency_generate(pipeline, prompt, steps, g_warm)
    latencies = []
    images = []
    for r in range(repeats):
        g = make_generator(seed=base_seed + r, device=device)
        lat, img = measure_latency_generate(pipeline, prompt, steps, g)
        img.save(os.path.join(out_dir, f"{p_idx:03d}_teacache_thr{thr}_{slug}_{r}.png"))
        latencies.append(lat)
        images.append(img)
    disable_teacache_restore_forward(original_forward)
    return latencies, images


def run_abba_sequence(pipeline, device, prompt, steps, repeats, warmup, out_dir, p_idx, slug, thr, original_forward, base_seed):
    """
    ABBA 验证序列：baseline → teacache → baseline → teacache
    用于检测顺序或缓存对速度测量的影响。
    """
    b1_lats, b1_imgs = run_baseline(pipeline, device, prompt, steps, repeats, warmup, out_dir, p_idx, slug, base_seed)
    t1_lats, t1_imgs = run_teacache(pipeline, device, prompt, steps, repeats, warmup, out_dir, p_idx, slug, thr, original_forward, base_seed)
    b2_lats, b2_imgs = run_baseline(pipeline, device, prompt, steps, repeats, warmup, out_dir, p_idx, slug, base_seed)
    t2_lats, t2_imgs = run_teacache(pipeline, device, prompt, steps, repeats, warmup, out_dir, p_idx, slug, thr, original_forward, base_seed)
    return (b1_lats, t1_lats, b2_lats, t2_lats), (b1_imgs, t1_imgs, b2_imgs, t2_imgs)


def main():
    # 读取配置
    cfg = CONFIG

    # 创建输出目录
    os.makedirs(cfg["outdir"], exist_ok=True)
    img_dir = os.path.join(cfg["outdir"], cfg["images_subdir"])
    os.makedirs(img_dir, exist_ok=True)

    # 设置随机性
    set_seed(cfg["seed"], deterministic=True, strict_cuda_determinism=cfg["strict_cuda_determinism"])

    # 读取提示词
    prompts = load_prompts(cfg["prompts_path"])

    # 准备管线与设备
    pipeline, device = prepare_pipeline(no_offload=cfg["no_offload"], enable_fast_attn=cfg["enable_fast_attn"])

    # 保存原始 forward，用于恢复
    original_forward = FluxTransformer2DModel.forward
    # LPIPS 只加载一次，并放到 CPU，避免与生成争用 GPU
    lpips_net = lpips.LPIPS(net='alex').eval().cpu()

    # 结果汇总 CSV
    summary_csv = os.path.join(cfg["outdir"], "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fsum:
        ws = csv.writer(fsum)
        if cfg["run_abba"]:
            ws.writerow(["prompt", "thr", "B1_mean", "B1_std", "T1_mean", "T1_std", "B2_mean", "B2_std", "T2_mean", "T2_std"])
        else:
            ws.writerow(["prompt", "setting", "thr", "latency_mean", "latency_std", "speedup_vs_baseline", "LPIPS_mean", "SSIM_mean", "PSNR_mean"])

        for p_idx, prompt in enumerate(prompts):
            slug = safe_slug(prompt)
            base_seed = cfg["seed"]

            if not cfg["run_abba"]:
                # Baseline
                b_lats, b_imgs = run_baseline(
                    pipeline, device, prompt, cfg["steps"], cfg["repeats"], cfg["warmup"], img_dir, p_idx, slug, base_seed
                )
                b_mean = float(np.mean(b_lats))
                b_std = float(np.std(b_lats)) if len(b_lats) > 1 else 0.0
                ws.writerow([prompt, "baseline", "", f"{b_mean:.4f}", f"{b_std:.4f}", 1.0, "", "", ""])
                print(f"[Baseline] {prompt}  {b_mean:.4f}s ± {b_std:.4f}s")

                # TeaCache thresholds
                for thr in cfg["thresholds"]:
                    t_lats, t_imgs = run_teacache(
                        pipeline, device, prompt, cfg["steps"], cfg["repeats"], cfg["warmup"], img_dir, p_idx, slug, thr, original_forward, base_seed
                    )
                    t_mean = float(np.mean(t_lats))
                    t_std = float(np.std(t_lats)) if len(t_lats) > 1 else 0.0
                    speedup = b_mean / t_mean if t_mean > 0 else float("nan")

                    # 配对比较：第 r 次 baseline vs 第 r 次 teacache
                    lp_list, ss_list, ps_list = [], [], []
                    for r in range(cfg["repeats"]):
                        lp, ss, ps = compute_image_metrics(b_imgs[r], t_imgs[r], lpips_net)
                        lp_list.append(lp); ss_list.append(ss); ps_list.append(ps)

                    ws.writerow([
                        prompt, "teacache", thr,
                        f"{t_mean:.4f}", f"{t_std:.4f}", f"{speedup:.4f}",
                        f"{float(np.mean(lp_list)):.6f}",
                        f"{float(np.mean(ss_list)):.6f}",
                        f"{float(np.mean(ps_list)):.6f}",
                    ])
                    print(f"[TeaCache thr={thr}] {prompt}  {t_mean:.4f}s ± {t_std:.4f}s  Speedup {speedup:.2f}x  "
                          f"LPIPS {np.mean(lp_list):.4f}  SSIM {np.mean(ss_list):.4f}  PSNR {np.mean(ps_list):.2f}dB")
            else:
                # ABBA 交替验证
                for thr in cfg["thresholds"]:
                    (b1_l, t1_l, b2_l, t2_l), _imgs = run_abba_sequence(
                        pipeline, device, prompt, cfg["steps"], cfg["repeats"], cfg["warmup"], img_dir, p_idx, slug, thr, original_forward, base_seed
                    )
                    ws.writerow([
                        prompt, thr,
                        f"{float(np.mean(b1_l)):.4f}", f"{float(np.std(b1_l)) if len(b1_l)>1 else 0.0:.4f}",
                        f"{float(np.mean(t1_l)):.4f}", f"{float(np.std(t1_l)) if len(t1_l)>1 else 0.0:.4f}",
                        f"{float(np.mean(b2_l)):.4f}", f"{float(np.std(b2_l)) if len(b2_l)>1 else 0.0:.4f}",
                        f"{float(np.mean(t2_l)):.4f}", f"{float(np.std(t2_l)) if len(t2_l)>1 else 0.0:.4f}",
                    ])
                    print(f"[ABBA thr={thr}] {prompt}  "
                          f"B1 {np.mean(b1_l):.4f}s  T1 {np.mean(t1_l):.4f}s  "
                          f"B2 {np.mean(b2_l):.4f}s  T2 {np.mean(t2_l):.4f}s")

    print(f"\nDone. Summary saved to: {summary_csv}\nImages saved to: {img_dir}")
    print("提示：")
    print("- 修改脚本顶部 CONFIG，即可快速调整开关与参数；")
    print("- 速度基准建议 no_offload=True（整管线GPU驻留、fp16），省显存则设为 False；")
    print("- 如需验证缓存/顺序影响，将 run_abba 设为 True；")
    print("- 若需严格确定性，strict_cuda_determinism 设为 True，并在环境中设置 CUBLAS_WORKSPACE_CONFIG。")


if __name__ == "__main__":
    main()