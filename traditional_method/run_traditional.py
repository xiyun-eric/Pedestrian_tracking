"""
传统CV方法版 - 主入口

使用方法:
  # 视频输入（默认）
  python traditional_method/run_traditional.py --video data/custom/videos/scene1.mp4
  python traditional_method/run_traditional.py --video data/custom/videos/scene1.mp4 --frames 100

  # 图像序列输入（兼容旧版）
  python traditional_method/run_traditional.py --data-dir data/kitti/images --seq 0017 0019
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

import argparse
from pathlib import Path

# 允许直接运行此文件时导入同目录模块
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from tracking_pipeline import TraditionalTrackingPipeline, TraditionalTrackingConfig


def main():
    parser = argparse.ArgumentParser(
        description="传统CV方法多目标跟踪 (HOG+SVM + 光流 + 卡尔曼滤波 + 匈牙利算法)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 视频输入（默认）
  python run_traditional.py --video data/custom/videos/scene1.mp4
  python run_traditional.py --video data/custom/videos/scene1.mp4 --frames 100

  # 图像序列输入（兼容旧版）
  python run_traditional.py --data-dir data/kitti/images --seq 0017 0019
  python run_traditional.py --data-dir data/custom/images --seq campus_scene
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

    # 检测参数
    parser.add_argument("--hog-conf", type=float, default=0.3,
                       help="HOG检测置信度阈值")
    parser.add_argument("--hog-scale", type=float, default=1.05,
                       help="HOG图像金字塔缩放系数(<1.05更精细)")
    parser.add_argument("--use-hog-api", action="store_true", default=True,
                       help="HOG特征提取使用OpenCV API（默认开启，速度快）")
    parser.add_argument("--no-hog-api", action="store_true",
                       help="HOG特征提取使用手动实现（用于验证算法原理，非常慢）")
    parser.add_argument("--use-svm-api", action="store_true", default=True,
                       help="SVM使用OpenCV预训练权重（默认开启）")
    parser.add_argument("--no-svm-api", action="store_true",
                       help="SVM不使用OpenCV预训练权重（仅用于标识，决策函数始终手动实现）")

    # 跟踪参数
    parser.add_argument("--max-age", type=int, default=30,
                       help="轨迹最大丢失帧数")
    parser.add_argument("--min-hits", type=int, default=3,
                       help="确认轨迹所需最小检测数")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                       help="IoU匹配阈值")

    # 功能开关
    parser.add_argument("--no-optical-flow", action="store_true",
                       help="禁用光流运动估计")
    parser.add_argument("--no-reid", action="store_true",
                       help="禁用ReID重识别")
    parser.add_argument("--no-video", action="store_true",
                       help="不生成视频")

    # 评估参数
    parser.add_argument("--eval", action="store_true",
                       help="启用GT评估")
    parser.add_argument("--labels", type=str, default=None,
                       help="YOLO格式GT标注目录")
    parser.add_argument("--scene", type=str, default=None,
                       help="场景名称过滤（如scene1, scene2）")

    # 输出参数
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认自动生成）")

    args = parser.parse_args()

    # 判断是否使用API
    use_hog_api = not args.no_hog_api
    use_svm_api = not args.no_svm_api

    # 判断输入模式
    use_video_mode = args.video is not None

    if not use_video_mode and args.data_dir is None:
        parser.error("请指定输入方式: --video <视频路径> 或 --data-dir <图像目录> --seq <序列名>")

    # 创建配置
    config = TraditionalTrackingConfig(
        hog_conf_threshold=args.hog_conf,
        hog_scale=args.hog_scale,
        use_hog_api=use_hog_api,
        use_svm_api=use_svm_api,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        use_optical_flow=not args.no_optical_flow,
        use_reid=not args.no_reid,
        save_video=not args.no_video,
    )

    # 初始化管道
    pipeline = TraditionalTrackingPipeline(config)

    print("=" * 70)
    print("  动态场景多目标跟踪 - 传统CV方法版")
    print("=" * 70)
    print(f"  检测:  HOG ({'OpenCV API' if use_hog_api else '手动实现'}) + SVM ({'预训练权重' if use_svm_api else '手动实现'})")
    print(f"  运动:  {'Farneback 稠密光流' if config.use_optical_flow else '无'}")
    print(f"  跟踪:  卡尔曼滤波 + 级联匹配 + 匈牙利算法")
    print(f"  ReID:  {'颜色直方图 + HOG 特征' if config.use_reid else '无'}")
    print(f"  输入:  {'视频文件' if use_video_mode else '图像序列'}")
    print("=" * 70)
    
    if not use_hog_api:
        print("\n[警告] HOG手动实现模式非常慢，仅用于验证算法原理！")
        print("       对于1280x720图像，每帧约需处理3600+个窗口。")
        print("       建议减少处理帧数：--frames 10\n")

    total_stats = []

    if use_video_mode:
        # 视频输入模式
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"错误: 视频文件不存在 {video_path}")
            return

        output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/traditional") / video_path.stem

        stats = pipeline.process_video(
            video_path=video_path,
            output_dir=output_dir,
            max_frames=args.frames,
            verbose=True,
            eval_gt=args.eval,
            labels_dir=Path(args.labels) if args.eval and args.labels else None,
            scene_name=args.scene,
        )
        total_stats.append((video_path.stem, stats))

    else:
        # 图像序列输入模式（兼容旧版）
        data_root = Path(args.data_dir)
        output_root = Path(args.output_dir) if args.output_dir else Path("outputs/traditional")

        if not data_root.exists():
            print(f"错误: 数据目录不存在: {data_root}")
            return

        seq_names = args.seq or ["0017"]

        for seq_name in seq_names:
            seq_dir = data_root / seq_name

            if not seq_dir.exists() or not seq_dir.is_dir():
                print(f"\n跳过: 序列目录不存在: {seq_dir}")
                continue

            # 支持MOT17格式：图像在 img1/ 子目录下
            img_dir = seq_dir
            if (seq_dir / "img1").is_dir():
                img_dir = seq_dir / "img1"

            output_dir = output_root / f"seq_{seq_name}"

            stats = pipeline.process_sequence(
                image_dir=img_dir,
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
