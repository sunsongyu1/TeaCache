import argparse
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


def set_seed(seed: int, deterministic: bool = True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 避免 CuBLAS 非确定性报错，不启用 use_deterministic_algorithms
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def tensor_from_pil(img: Image.Image):
    return transforms.ToTensor()(img)  # [C,H,W] in [0,1]


def safe_slug(text: str, maxlen: int = 80):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-_]+", "_", text)
    return text[:maxlen].strip("_")


def make_generator(seed: int, device: torch.device):
    g = torch.Generator(device=device if device.type == "cuda" else "cpu")
    g.manual_seed(seed)
    return g


def synchronize_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def measure_latency_generate(pipeline: DiffusionPipeline, prompt: str, steps: int, generator: torch.Generator):
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
    mse = np.mean((ref_np - cmp_np) ** 2, dtype=np.float64)
    mse = max(mse, eps)
    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(psnr)


@torch.inference_mode()
def compute_image_metrics(img_ref: Image.Image, img_cmp: Image.Image, lpips_net):
    t_ref = tensor_from_pil(img_ref)
    t_cmp = tensor_from_pil(img_cmp)

    ref_np_u8 = np.asarray(img_ref)
    cmp_np_u8 = np.asarray(img_cmp)
    if ref_np_u8.shape == cmp_np_u8.shape and np.array_equal(ref_np_u8, cmp_np_u8):
        return 0.0, 1.0, 100.0

    t_ref_lpips = (t_ref.unsqueeze(0) * 2 - 1).float()
    t_cmp_lpips = (t_cmp.unsqueeze(0) * 2 - 1).float()
    lpips_score = float(lpips_net(t_ref_lpips, t_cmp_lpips).item())

    ref_np = ref_np_u8.astype(np.float32) / 255.0
    cmp_np = cmp_np_u8.astype(np.float32) / 255.0
    ssim_score = float(ssim_metric(ref_np, cmp_np, data_range=1.0, channel_axis=-1))
    psnr_score = compute_psnr_from_arrays(ref_np, cmp_np, data_range=1.0)
    return lpips_score, ssim_score, psnr_score


def enable_teacache_on_pipeline(pipeline: DiffusionPipeline, steps: int, rel_l1_thresh: float):
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
    FluxTransformer2DModel.forward = original_forward
    try:
        transformer_cls = FluxTransformer2DModel
        if hasattr(transformer_cls, "enable_teacache"):
            transformer_cls.enable_teacache = False
        if hasattr(transformer_cls, "cnt"):
            transformer_cls.cnt = 0
        if hasattr(transformer_cls, "num_steps"):
            transformer_cls.num_steps = 0
        if hasattr(transformer_cls, "rel_l1_thresh"):
            transformer_cls.rel_l1_thresh = None
        if hasattr(transformer_cls, "accumulated_rel_l1_distance"):
            transformer_cls.accumulated_rel_l1_distance = 0
        if hasattr(transformer_cls, "previous_modulated_input"):
            transformer_cls.previous_modulated_input = None
        if hasattr(transformer_cls, "previous_residual"):
            transformer_cls.previous_residual = None
    except Exception:
        pass


def prepare_pipeline(no_offload: bool):
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
            for name in ["text_encoder", "text_encoder_2", "transformer", "vae"]:
                if hasattr(pipe, name) and getattr(pipe, name) is not None:
                    getattr(pipe, name).to(dtype=torch.float32)
    else:
        pipe.enable_model_cpu_offload()
        try:
            pipe.to("cpu")
        except Exception:
            pass

    return pipe, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.4, 0.6, 0.8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="flux_teacache_benchmark")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no_offload", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img_dir = os.path.join(args.outdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    set_seed(args.seed)

    prompts = load_prompts(args.prompts)

    pipeline, device = prepare_pipeline(args.no_offload)

    if not args.no_progress:
        try:
            pipeline.set_progress_bar_config(disable=True)
        except Exception:
            pass

    original_forward = FluxTransformer2DModel.forward

    lpips_net = lpips.LPIPS(net='alex').eval().cpu()

    csv_path = os.path.join(args.outdir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "prompt",
            "setting",
            "rel_l1_thresh",
            "latency_s_mean",
            "latency_s_std",
            "speedup_vs_baseline",
            "LPIPS_mean",
            "SSIM_mean",
            "PSNR_mean",
        ])

        for p_idx, prompt in enumerate(prompts):
            slug = safe_slug(prompt)

            baseline_latencies = []
            baseline_images = []

            for _ in range(max(args.warmup, 0)):
                g_warm = make_generator(seed=args.seed, device=device)
                _ = measure_latency_generate(pipeline, prompt, args.steps, g_warm)

            for r in range(args.repeats):
                g = make_generator(seed=args.seed + r, device=device)
                lat, img = measure_latency_generate(pipeline, prompt, args.steps, g)
                out_path = os.path.join(img_dir, f"{p_idx:02d}_baseline_{slug}_{r}.png")
                img.save(out_path)
                baseline_latencies.append(lat)
                baseline_images.append(img)

            baseline_latency_mean = float(np.mean(baseline_latencies))
            baseline_latency_std = float(np.std(baseline_latencies)) if len(baseline_latencies) > 1 else 0.0

            writer.writerow([prompt, "baseline", "", f"{baseline_latency_mean:.4f}", f"{baseline_latency_std:.4f}", 1.0, "", "", ""])
            print(f"[Baseline] Prompt: {prompt}  Latency: {baseline_latency_mean:.4f}s ± {baseline_latency_std:.4f}s over {args.repeats} runs")

            for thr in args.thresholds:
                teacache_latencies = []
                lpips_list, ssim_list, psnr_list = [], [], []

                enable_teacache_on_pipeline(pipeline, args.steps, thr)
                for _ in range(max(args.warmup, 0)):
                    g_warm = make_generator(seed=args.seed, device=device)
                    _ = measure_latency_generate(pipeline, prompt, args.steps, g_warm)

                for r in range(args.repeats):
                    g = make_generator(seed=args.seed + r, device=device)
                    lat, img = measure_latency_generate(pipeline, prompt, args.steps, g)
                    out_path = os.path.join(img_dir, f"{p_idx:02d}_teacache_thr{thr}_{slug}_{r}.png")
                    img.save(out_path)
                    teacache_latencies.append(lat)

                    lp, ss, pn = compute_image_metrics(baseline_images[r], img, lpips_net)
                    lpips_list.append(lp)
                    ssim_list.append(ss)
                    psnr_list.append(pn)

                teacache_latency_mean = float(np.mean(teacache_latencies))
                teacache_latency_std = float(np.std(teacache_latencies)) if len(teacache_latencies) > 1 else 0.0
                speedup = baseline_latency_mean / teacache_latency_mean if teacache_latency_mean > 0 else float("nan")
                lpips_mean = float(np.mean(lpips_list))
                ssim_mean = float(np.mean(ssim_list))
                psnr_mean = float(np.mean(psnr_list))

                writer.writerow([
                    prompt,
                    "teacache",
                    thr,
                    f"{teacache_latency_mean:.4f}",
                    f"{teacache_latency_std:.4f}",
                    f"{speedup:.4f}",
                    f"{lpips_mean:.6f}",
                    f"{ssim_mean:.6f}",
                    f"{psnr_mean:.6f}"
                ])
                print(f"[TeaCache thr={thr}] Prompt: {prompt}  Latency: {teacache_latency_mean:.4f}s ± {teacache_latency_std:.4f}s  "
                      f"Speedup: {speedup:.2f}x  LPIPS: {lpips_mean:.4f}  SSIM: {ssim_mean:.4f}  PSNR: {psnr_mean:.2f}dB")

                disable_teacache_restore_forward(original_forward)

    print(f"\nDone. Results saved to: {csv_path}\nImages saved to: {os.path.join(args.outdir, 'images')}")
    print("Interpretation:")
    print("- speedup_vs_baseline > 1 表示加速；")
    print("- LPIPS 越低越好，SSIM/PSNR 越高越好；随阈值增大，质量可能略降但延时显著降低。")


if __name__ == "__main__":
    main()