"""
Phase 5: 消融实验

实现计划文档中定义的 7 组消融实验:
  A1: 无 LoRA (仅评估原始模型, 验证 LoRA 的必要性)
  A2: LoRA r=16 vs r=32 vs r=64 (LoRA 秩的影响)
  A3: 无 L_format (格式约束损失的作用)
  A4: 无 L_iou (IoU 约束损失的作用)
  A5: 无 L_track (跟踪一致性损失的作用)
  A6: 单阶段 vs 三阶段训练 (渐进训练的必要性)
  A7: seq_len=2 vs 4 (输入帧数的影响)

每个消融实验:
  1. 按配置训练 LoRA (可选, 可跳过使用已有权重)
  2. 在 MOT17 上推理 + 评估
  3. 收集 MOTA/MOTP/IDF1 等指标

使用方式:
  # 列出所有消融配置
  python model_method/ablation.py --list

  # 运行所有消融实验 (快速模式, 仅 20 帧评估)
  python model_method/ablation.py --quick

  # 运行特定消融
  python model_method/ablation.py --ablations A1,A2

  # 跳过训练, 只评估已有权重
  python model_method/ablation.py --ablations A2 --eval-only

  # 只生成对比报告 (使用已有评估结果)
  python model_method/ablation.py --report-only
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import OrderedDict

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "Qwen"
MOT17_PATH = PROJECT_ROOT / "data" / "MOT17"
OUTPUT_DIR = PROJECT_ROOT / "runs"
ABLATION_DIR = OUTPUT_DIR / "ablation"

# 修复 Windows GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# 消融实验配置
# ============================================================

# 每个消融实验的配置:
#   name: 实验名
#   desc: 描述
#   lora_path: 评估时使用的 LoRA 权重路径 (None=原始模型)
#   train_cmd: 训练命令 (None=不训练, 使用已有权重或原始模型)
#   train_needed: 是否需要训练

ABLATION_CONFIGS = OrderedDict([
    # A1: 无 LoRA (原始模型) vs 有 LoRA
    # 验证 LoRA 微调的必要性
    ('A1', {
        'name': 'A1_no_lora',
        'desc': '无 LoRA (原始模型) - 验证 LoRA 的必要性',
        'lora_path': None,  # 原始模型
        'baseline': 'runs/stage2/final',  # 对比基线
        'train_needed': False,
    }),

    # A2: LoRA 秩的影响 (r=16, 32, 64)
    # 需要训练 3 个不同 r 的 LoRA
    ('A2_r16', {
        'name': 'A2_lora_r16',
        'desc': 'LoRA r=16 (对比 r=32, r=64)',
        'lora_path': str(ABLATION_DIR / 'A2_r16' / 'final'),
        'train_needed': True,
        'train_cmd': [
            'python', 'model_method/train.py',
            '--stage', '1',
            '--quick',  # 默认快速训练, 可通过 --full 覆盖
        ],
        'train_env': {'LORA_R': '16', 'LORA_ALPHA': '32'},
    }),
    ('A2_r32', {
        'name': 'A2_lora_r32',
        'desc': 'LoRA r=32',
        'lora_path': str(ABLATION_DIR / 'A2_r32' / 'final'),
        'train_needed': True,
        'train_cmd': [
            'python', 'model_method/train.py',
            '--stage', '1',
            '--quick',
        ],
        'train_env': {'LORA_R': '32', 'LORA_ALPHA': '64'},
    }),
    # A2_r64 使用默认 stage1 权重作为对比
    ('A2_r64', {
        'name': 'A2_lora_r64',
        'desc': 'LoRA r=64 (默认配置)',
        'lora_path': str(OUTPUT_DIR / 'stage1' / 'final'),
        'train_needed': False,  # 使用已有 stage1 权重
    }),

    # A3: 无 L_format (坐标加权损失)
    # 基线: 使用 CoordinateWeightedLoss (默认)
    # 消融: 不使用辅助损失 (仅 L_lm)
    ('A3', {
        'name': 'A3_no_format_loss',
        'desc': '无 L_format (仅 L_lm) - 验证格式约束损失的作用',
        'lora_path': str(ABLATION_DIR / 'A3_no_format' / 'final'),
        'train_needed': True,
        'train_cmd': [
            'python', 'model_method/train.py',
            '--stage', '2',
            '--quick',
        ],
        'train_env': {'AUX_LOSS': 'none'},  # 自定义环境变量控制
        'baseline': str(OUTPUT_DIR / 'stage2' / 'final'),
    }),

    # A4: 无 L_iou
    # 当前默认实现 iou_every_n_steps=0, 即不计算 IoU 损失
    # 所以 A4 = 默认配置, 基线 = 启用 IoU 损失的版本
    ('A4', {
        'name': 'A4_no_iou_loss',
        'desc': '无 L_iou (默认配置, iou_every_n_steps=0)',
        'lora_path': str(OUTPUT_DIR / 'stage2' / 'final'),
        'train_needed': False,
        'baseline': str(ABLATION_DIR / 'A4_with_iou' / 'final'),
        'baseline_desc': '启用 IoU 损失 (iou_every_n_steps=10)',
    }),

    # A5: 无 L_track (跟踪一致性损失)
    # 当前 CoordinateWeightedLoss 不使用 L_track, 所以 A5 = 默认配置
    ('A5', {
        'name': 'A5_no_track_loss',
        'desc': '无 L_track (默认配置, 仅坐标加权)',
        'lora_path': str(OUTPUT_DIR / 'stage2' / 'final'),
        'train_needed': False,
        'baseline': str(ABLATION_DIR / 'A5_with_track' / 'final'),
        'baseline_desc': '启用跟踪一致性损失 (CombinedTrackingLoss)',
    }),

    # A6: 单阶段 vs 三阶段训练
    ('A6_stage1', {
        'name': 'A6_stage1_only',
        'desc': '仅 Stage 1 (单帧检测, 无跟踪训练)',
        'lora_path': str(OUTPUT_DIR / 'stage1' / 'final'),
        'train_needed': False,
    }),
    ('A6_stage2', {
        'name': 'A6_stage1_2',
        'desc': 'Stage 1+2 (双帧跟踪)',
        'lora_path': str(OUTPUT_DIR / 'stage2' / 'final'),
        'train_needed': False,
    }),
    ('A6_stage3', {
        'name': 'A6_stage1_2_3',
        'desc': 'Stage 1+2+3 (完整三阶段, 含 RL 精调)',
        'lora_path': str(OUTPUT_DIR / 'stage3' / 'final'),
        'train_needed': False,
    }),

    # A7: seq_len=2 vs 4
    # A7_seq2 使用 stage2 权重 (训练时 seq_len=2)
    ('A7_seq2', {
        'name': 'A7_seq_len_2',
        'desc': '推理时 window_size=2 (与训练一致)',
        'lora_path': str(OUTPUT_DIR / 'stage2' / 'final'),
        'train_needed': False,
        'eval_window_size': 2,
    }),
    ('A7_seq4', {
        'name': 'A7_seq_len_4',
        'desc': '推理时 window_size=4 (大于训练时 seq_len=2)',
        'lora_path': str(OUTPUT_DIR / 'stage2' / 'final'),
        'train_needed': False,
        'eval_window_size': 4,
    }),
])


# ============================================================
# 训练 + 评估
# ============================================================

def run_training(config, full=False, dry_run=False):
    """
    运行训练 (调用 train.py)

    Args:
        config: 消融配置
        full: True=完整训练, False=quick 模式
        dry_run: True=只打印命令不执行

    Returns:
        True 如果训练成功或权重已存在
    """
    lora_path = Path(config['lora_path'])

    # 如果权重已存在, 跳过训练
    if lora_path.exists() and (lora_path / 'adapter_config.json').exists():
        print(f"  ✅ LoRA 权重已存在, 跳过训练: {lora_path}")
        return True

    if not config.get('train_needed', False):
        if not lora_path.exists():
            print(f"  ⚠️ 权重不存在且无需训练: {lora_path}")
            print(f"     请先运行基线训练")
        return lora_path.exists()

    train_cmd = list(config.get('train_cmd', []))
    if not train_cmd:
        print(f"  ⚠️ 无训练命令")
        return False

    # 调整 quick/full 模式
    if full and '--quick' in train_cmd:
        train_cmd.remove('--quick')

    # 设置输出目录
    output_dir = lora_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建训练命令 (使用环境变量传递消融参数)
    env = os.environ.copy()
    env.update(config.get('train_env', {}))

    # 添加输出目录参数 (train.py 不直接支持, 通过环境变量)
    env['ABLATION_OUTPUT_DIR'] = str(output_dir)

    print(f"  训练命令: {' '.join(train_cmd)}")
    print(f"  环境变量: {config.get('train_env', {})}")
    print(f"  输出目录: {output_dir}")

    if dry_run:
        print(f"  [Dry Run] 跳过实际训练")
        return True

    try:
        result = subprocess.run(
            train_cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if result.returncode != 0:
            print(f"  ❌ 训练失败 (exit code: {result.returncode})")
            return False

        # 训练完成后, 复制权重到目标路径 (如果不在目标路径)
        # train.py 默认保存到 runs/stage{N}/final
        # 消融实验需要保存到 ablation/{name}/final
        default_save = OUTPUT_DIR / 'stage1' / 'final'
        if default_save.exists() and not lora_path.exists():
            import shutil
            lora_path.mkdir(parents=True, exist_ok=True)
            for item in default_save.iterdir():
                if item.is_file():
                    shutil.copy2(str(item), str(lora_path / item.name))
            print(f"  ✅ 复制权重: {default_save} -> {lora_path}")

        return lora_path.exists()

    except Exception as e:
        print(f"  ❌ 训练异常: {e}")
        return False


def run_evaluation(config, quick=True, sequences=None, max_frames=None,
                   window_size=None, stride=1):
    """
    运行评估 (调用 evaluate.py)

    Args:
        config: 消融配置
        quick: 快速模式
        sequences: 评估序列列表
        max_frames: 最大帧数
        window_size: 推理窗口大小 (None=使用配置默认值)
        stride: 滑动步长

    Returns:
        评估结果字典, 或 None
    """
    lora_path = config['lora_path']
    name = config['name']

    # 结果目录
    result_dir = ABLATION_DIR / name / 'mot_results'
    report_path = ABLATION_DIR / name / 'report.json'

    # 如果已有评估结果, 直接加载
    if report_path.exists():
        print(f"  ✅ 评估结果已存在: {report_path}")
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 构建评估命令
    eval_cmd = [
        'python', 'model_method/evaluate.py',
        '--result-dir', str(result_dir),
        '--report-path', str(ABLATION_DIR / name / 'report.txt'),
    ]

    if lora_path:
        eval_cmd.extend(['--lora-path', lora_path])
    else:
        eval_cmd.append('--no-lora')

    if window_size is not None:
        eval_cmd.extend(['--window-size', str(window_size)])
    elif 'eval_window_size' in config:
        eval_cmd.extend(['--window-size', str(config['eval_window_size'])])
    else:
        eval_cmd.extend(['--window-size', '2'])

    eval_cmd.extend(['--stride', str(stride)])

    if quick:
        eval_cmd.append('--quick')

    if sequences:
        eval_cmd.extend(['--sequences', ','.join(sequences)])

    if max_frames is not None:
        eval_cmd.extend(['--max-frames', str(max_frames)])

    print(f"  评估命令: {' '.join(eval_cmd)}")

    try:
        result = subprocess.run(
            eval_cmd,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if result.returncode != 0:
            print(f"  ❌ 评估失败 (exit code: {result.returncode})")
            return None
    except Exception as e:
        print(f"  ❌ 评估异常: {e}")
        return None

    # 加载评估结果
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


# ============================================================
# 报告生成
# ============================================================

def generate_comparison_report(results, output_path):
    """
    生成消融实验对比报告

    Args:
        results: Dict[exp_name, eval_result]
        output_path: 报告输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("Phase 5: 消融实验对比报告")
    lines.append("=" * 80)
    lines.append("")

    # 收集所有指标
    if not results:
        lines.append("无可用结果")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return

    # 表格头部
    headers = ['实验', 'MOTA', 'MOTP', 'IDF1', 'IDSW', 'FP', 'FN', 'TP', 'MT', 'ML', 'Prec', 'Reca']
    lines.append('| ' + ' | '.join(f'{h:>10}' for h in headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')

    # 每个实验一行
    for exp_name, result in results.items():
        if result is None:
            lines.append(f'| {exp_name:>10} |' + ' N/A |' * (len(headers) - 1))
            continue

        # 提取 overall 指标
        metrics = result.get('metrics', {})
        overall = metrics.get('OVERALL', metrics.get('overall', {}))

        if not overall:
            # 如果没有 overall, 取第一个序列
            for k, v in metrics.items():
                if k not in ['OVERALL', 'overall']:
                    overall = v
                    break

        row = [exp_name]
        for key in ['mota', 'motp', 'idf1', 'num_switches',
                    'num_false_positives', 'num_misses', 'num_matches',
                    'mostly_tracked', 'mostly_lost', 'precision', 'recall']:
            val = overall.get(key, None)
            if val is None:
                row.append('N/A')
            elif key in ['mota', 'motp', 'idf1', 'precision', 'recall']:
                # 百分比
                row.append(f'{val * 100:.1f}%' if val is not None else 'N/A')
            else:
                row.append(str(int(val)) if val is not None else 'N/A')

        lines.append('| ' + ' | '.join(f'{c:>10}' for c in row) + ' |')

    lines.append("")
    lines.append("说明:")
    lines.append("  MOTA: 多目标跟踪准确度 (越高越好)")
    lines.append("  MOTP: 多目标跟踪精度 (越低越好, 表示 IoU 距离)")
    lines.append("  IDF1: 身份维持 F1 (越高越好)")
    lines.append("  IDSW: 身份切换次数 (越低越好)")
    lines.append("  FP/FN: 误检/漏检数 (越低越好)")
    lines.append("  TP: 正确匹配数 (越高越好)")
    lines.append("  MT/ML: 主要跟踪/丢失轨迹数")

    # 各实验详细描述
    lines.append("")
    lines.append("=" * 80)
    lines.append("各实验详细说明")
    lines.append("=" * 80)
    for exp_name, config in ABLATION_CONFIGS.items():
        if exp_name not in results:
            continue
        lines.append("")
        lines.append(f"  {exp_name} ({config['name']}):")
        lines.append(f"    描述: {config['desc']}")
        lines.append(f"    LoRA: {config['lora_path'] or 'None (原始模型)'}")
        result = results[exp_name]
        if result:
            lines.append(f"    序列: {result.get('sequences', [])}")
            lines.append(f"    窗口: {result.get('window_size', '?')}, "
                        f"步长: {result.get('stride', '?')}")

    report = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n  报告已保存: {output_path}")


def generate_json_report(results, output_path):
    """生成 JSON 格式的对比报告"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        'experiments': {},
        'config_descriptions': {},
    }

    for exp_name, result in results.items():
        config = ABLATION_CONFIGS.get(exp_name, {})
        report['experiments'][exp_name] = result
        report['config_descriptions'][exp_name] = {
            'name': config.get('name', ''),
            'desc': config.get('desc', ''),
            'lora_path': config.get('lora_path', ''),
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"  JSON 报告已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================

def list_ablations():
    """列出所有消融实验配置"""
    print("=" * 80)
    print("消融实验配置列表")
    print("=" * 80)
    for key, config in ABLATION_CONFIGS.items():
        print(f"\n  {key} ({config['name']})")
        print(f"    描述: {config['desc']}")
        print(f"    LoRA: {config['lora_path'] or 'None (原始模型)'}")
        print(f"    需要训练: {config.get('train_needed', False)}")
        if 'baseline' in config:
            print(f"    基线对比: {config['baseline']}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: 消融实验")
    parser.add_argument(
        '--list', action='store_true',
        help='列出所有消融实验配置'
    )
    parser.add_argument(
        '--ablations', type=str, default=None,
        help='要运行的消融实验 (逗号分隔, 如 A1,A2_r16; 默认全部)'
    )
    parser.add_argument(
        '--quick', action='store_true', default=True,
        help='快速模式: 少量帧评估 (默认开启)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='完整模式: 完整训练 + 完整评估 (覆盖 --quick)'
    )
    parser.add_argument(
        '--no-quick', action='store_true',
        help='关闭快速模式 (与 --full 类似但只影响评估)'
    )
    parser.add_argument(
        '--sequences', type=str, default=None,
        help='评估序列 (逗号分隔, 如 MOT17-02-FRCNN)'
    )
    parser.add_argument(
        '--max-frames', type=int, default=None,
        help='每个序列最大评估帧数'
    )
    parser.add_argument(
        '--window-size', type=int, default=None,
        help='推理窗口大小 (覆盖配置默认值)'
    )
    parser.add_argument(
        '--stride', type=int, default=1,
        help='滑动步长'
    )
    parser.add_argument(
        '--train-only', action='store_true',
        help='只训练, 不评估'
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='只评估, 不训练 (使用已有权重)'
    )
    parser.add_argument(
        '--report-only', action='store_true',
        help='只生成对比报告 (使用已有评估结果)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='只打印命令不执行'
    )
    parser.add_argument(
        '--report-path', type=str,
        default=str(ABLATION_DIR / 'ablation_report.md'),
        help='对比报告输出路径'
    )
    args = parser.parse_args()

    # --list: 列出配置
    if args.list:
        list_ablations()
        return

    # 确定快速模式
    quick = not args.full and not args.no_quick

    print("=" * 80)
    print("Phase 5: 消融实验")
    print("=" * 80)
    print(f"  快速模式: {quick}")
    print(f"  输出目录: {ABLATION_DIR}")
    print(f"  报告路径: {args.report_path}")
    print()

    # 确定要运行的消融实验
    if args.ablations:
        exp_keys = [k.strip() for k in args.ablations.split(',')]
        # 验证
        invalid = [k for k in exp_keys if k not in ABLATION_CONFIGS]
        if invalid:
            print(f"  ❌ 未知的消融实验: {invalid}")
            print(f"     可用: {list(ABLATION_CONFIGS.keys())}")
            return
    else:
        exp_keys = list(ABLATION_CONFIGS.keys())

    print(f"  消融实验: {exp_keys}")
    print()

    # 序列
    sequences = None
    if args.sequences:
        sequences = [s.strip() for s in args.sequences.split(',')]

    # ============================================================
    # --report-only: 只生成报告
    # ============================================================
    if args.report_only:
        results = {}
        for exp_key in exp_keys:
            report_path = ABLATION_DIR / ABLATION_CONFIGS[exp_key]['name'] / 'report.json'
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    results[exp_key] = json.load(f)
                print(f"  ✅ 加载 {exp_key}: {report_path}")
            else:
                print(f"  ⚠️ 跳过 {exp_key}: 无评估结果 ({report_path})")
                results[exp_key] = None

        generate_comparison_report(results, args.report_path)
        generate_json_report(results, str(Path(args.report_path).with_suffix('.json')))
        return

    # ============================================================
    # 运行消融实验
    # ============================================================
    results = {}

    for exp_key in exp_keys:
        config = ABLATION_CONFIGS[exp_key]
        print(f"\n{'=' * 80}")
        print(f"消融实验: {exp_key} ({config['name']})")
        print(f"  描述: {config['desc']}")
        print(f"{'=' * 80}")

        # 步骤 1: 训练 (如果需要)
        if not args.eval_only and config.get('train_needed', False):
            print(f"\n  [训练]")
            success = run_training(
                config,
                full=args.full,
                dry_run=args.dry_run,
            )
            if not success:
                print(f"  ⚠️ 训练失败, 跳过评估")
                results[exp_key] = None
                continue

        if args.train_only:
            continue

        # 步骤 2: 评估
        print(f"\n  [评估]")
        result = run_evaluation(
            config,
            quick=quick,
            sequences=sequences,
            max_frames=args.max_frames,
            window_size=args.window_size,
            stride=args.stride,
        )
        results[exp_key] = result

        if result is not None:
            print(f"  ✅ 评估完成: {exp_key}")
        else:
            print(f"  ❌ 评估失败: {exp_key}")

    # ============================================================
    # 生成对比报告
    # ============================================================
    print(f"\n{'=' * 80}")
    print("生成对比报告")
    print(f"{'=' * 80}")

    if results:
        generate_comparison_report(results, args.report_path)
        generate_json_report(results, str(Path(args.report_path).with_suffix('.json')))
    else:
        print("  ⚠️ 无可用结果, 跳过报告生成")


if __name__ == '__main__':
    main()
