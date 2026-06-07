"""
深度学习方法版 - 主入口

使用方法:
  # 视频输入（默认）
  python deep_method/run_deep.py --video data/custom/videos/scene1.mp4
  python deep_method/run_deep.py --video data/custom/videos/scene1.mp4 --frames 100 --device cpu

  # 使用LoRA微调后的模型
  python deep_method/run_deep.py --video data/custom/videos/scene1.mp4 \
      --model yolo11m.pt --lora outputs/deep/lora_weights/lora_best.pt

  # 图像序列输入（兼容旧版）
  python deep_method/run_deep.py --data-dir data/kitti/images --seq 0017 0019
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

_current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_current_dir))

import argparse

from tracking_pipeline import DeepTrackingPipeline, DeepTrackingConfig


def main():
    parser = argparse.ArgumentParser(
        description="深度学习方法多目标跟踪 (YOLO11 + AdvancedTracker + ByteTrack + ReID)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 视频输入（默认）
  python run_deep.py --video data/custom/videos/scene1.mp4
  python run_deep.py --video data/custom/videos/scene1.mp4 --frames 100 --device cpu

  # 使用LoRA微调模型
  python run_deep.py --video data/custom/videos/scene1.mp4 \\
      --lora outputs/deep/lora_weights/lora_best.pt

  # 图像序列输入（兼容旧版）
  python run_deep.py --data-dir data/kitti/images --seq 0017 0019
        """
    )

    # 输入方式（二选一，默认视频）
    parser.add_argument("--video", type=str, default=None,
                       help="视频文件路径（默认输入方式）")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="图像数据根目录（兼容旧版，与 --video 二选一）")
    parser.add_argument("--seq", type=str, nargs="+", default=None,
                       help="要处理的序列名称（仅 --data-dir 模式使用）")
    parser.add_argument("--frames", type=int, default=None,
                       help="最大处理帧数")

    # 模型
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                       help="YOLO模型 (yolo11n/s/m/l, yolo11m.pt)")
    parser.add_argument("--lora", type=str, default=None,
                       help="LoRA权重路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="推理设备")

    # 检测
    parser.add_argument("--conf", type=float, default=0.25,
                       help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="NMS IoU阈值")

    # 跟踪
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--track-iou", type=float, default=0.3)

    # ByteTrack
    parser.add_argument("--no-bytetrack", action="store_true",
                       help="禁用 ByteTrack 低分框策略")

    # 输出
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认自动生成）")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-trajectory", action="store_true")

    args = parser.parse_args()

    # 判断输入模式
    use_video_mode = args.video is not None

    if not use_video_mode and args.data_dir is None:
        parser.error("请指定输入方式: --video <视频路径> 或 --data-dir <图像目录> --seq <序列名>")

    # 创建配置
    config = DeepTrackingConfig(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        use_bytetrack=not args.no_bytetrack,
        max_age=args.max_age,
        min_hits=args.min_hits,
        track_iou_threshold=args.track_iou,
        lora_path=args.lora,
        save_video=not args.no_video,
        draw_trajectory=not args.no_trajectory,
    )

    # 初始化管道
    pipeline = DeepTrackingPipeline(config)

    print("=" * 70)
    print("  动态场景多目标跟踪 - 深度学习方法版")
    print("=" * 70)
    print(f"  检测器: YOLO11 ({Path(args.model).stem})")
    print(f"  跟踪器: AdvancedTracker + ByteTrack + ReID")
    print(f"  ByteTrack: {'启用' if config.use_bytetrack else '禁用'}")
    print(f"  LoRA微调: {'已加载' if args.lora else '未使用'}")
    print(f"  设备: {args.device}")
    print(f"  输入: {'视频文件' if use_video_mode else '图像序列'}")
    print("=" * 70)

    if args.lora:
        print(f"\n[已加载LoRA权重] {args.lora}")

    total_stats = []

    if use_video_mode:
        # 视频输入模式
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"错误: 视频文件不存在 {video_path}")
            return

        output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/deep") / video_path.stem

        stats = pipeline.process_video(
            video_path=video_path,
            output_dir=output_dir,
            max_frames=args.frames,
            verbose=True,
        )
        total_stats.append((video_path.stem, stats))

    else:
        # 图像序列输入模式（兼容旧版）
        data_root = Path(args.data_dir)
        output_root = Path(args.output_dir) if args.output_dir else Path("outputs/deep")

        if not data_root.exists():
            print(f"错误: 数据目录不存在: {data_root}")
            return

        seq_names = args.seq or ["0017"]

        for seq_name in seq_names:
            seq_dir = data_root / seq_name

            if not seq_dir.exists() or not seq_dir.is_dir():
                print(f"\n跳过: 序列目录不存在: {seq_dir}")
                continue

            output_dir = output_root / f"seq_{seq_name}"

            stats = pipeline.process_sequence(
                image_dir=seq_dir,
                output_dir=output_dir,
                max_frames=args.frames,
                verbose=True,
            )
            total_stats.append((seq_name, stats))
            pipeline.reset()

    # 总结
    print("\n" + "=" * 70)
    print("  所有处理完成!")
    print("=" * 70)

    for name, stats in total_stats:
        if isinstance(stats, dict) and "error" not in stats:
            avg_fps = stats["total_frames"] / max(stats["processing_time"], 0.001)
            print(f"  {name}: {stats['total_frames']}帧, "
                  f"检测 {stats['total_detections']}次, "
                  f"最大 {stats['max_tracks']}个目标, "
                  f"平均 {avg_fps:.1f} FPS")

    print("=" * 70)


if __name__ == "__main__":
    main()
