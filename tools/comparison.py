"""
对比实验分析模块

收集三种方法在多个场景上的评估指标，生成对比表格和报告。
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import csv


class ComparisonReporter:
    """
    对比实验报告生成器
    
    收集多种方法在多个场景上的评估指标，生成对比表格（Markdown/CSV/JSON）。
    """
    
    def __init__(self, output_dir: Path):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []  # List[Dict]: {"method": str, "scene": str, **metrics}
        
    def add_result(self, method_name: str, scene_name: str, metrics: Dict):
        """
        添加一个实验结果
        
        Args:
            method_name: 方法名称（如 "传统方法", "深度方法(原始)", "深度方法(微调)"）
            scene_name: 场景名称（如 "scene1", "scene2"）
            metrics: 评估指标字典（包含 MOTA, MOTP, FP, FN, IDSW, FPS 等）
        """
        result = {
            "method": method_name,
            "scene": scene_name,
        }
        result.update(metrics)
        self.results.append(result)
        
    def generate_table_markdown(self) -> str:
        """
        生成 Markdown 格式的对比表格
        
        Returns:
            Markdown 表格字符串
        """
        if not self.results:
            return "无结果数据"
            
        # 收集所有指标键（排除 method 和 scene）
        metric_keys = set()
        for r in self.results:
            metric_keys.update(r.keys())
        metric_keys.discard("method")
        metric_keys.discard("scene")
        
        # 按优先级排序指标
        priority_keys = ["MOTA", "MOTP", "FP", "FN", "IDSW", "FPS"]
        sorted_keys = [k for k in priority_keys if k in metric_keys] + \
                     sorted([k for k in metric_keys if k not in priority_keys])
        
        # 生成表头
        header = "| 方法 | 场景 | " + " | ".join(sorted_keys) + " |"
        separator = "| --- | --- | " + " | ".join(["---"] * len(sorted_keys)) + " |"
        
        # 生成表格行
        rows = []
        for r in self.results:
            row = f"| {r['method']} | {r['scene']} | "
            values = []
            for k in sorted_keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    values.append(f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}")
                else:
                    values.append(str(v))
            row += " | ".join(values) + " |"
            rows.append(row)
            
        # 计算平均值（可选）
        # 这里暂不实现，保持简单
        
        return "\n".join([header, separator] + rows)
        
    def generate_table_csv(self, output_path: Optional[Path] = None) -> str:
        """
        生成 CSV 格式的对比表格
        
        Args:
            output_path: 输出文件路径（可选，默认保存到 output_dir/comparison_table.csv）
            
        Returns:
            CSV 表格字符串
        """
        if not self.results:
            return ""
            
        # 收集所有指标键
        metric_keys = set()
        for r in self.results:
            metric_keys.update(r.keys())
        metric_keys.discard("method")
        metric_keys.discard("scene")
        
        priority_keys = ["MOTA", "MOTP", "FP", "FN", "IDSW", "FPS"]
        sorted_keys = [k for k in priority_keys if k in metric_keys] + \
                     sorted([k for k in metric_keys if k not in priority_keys])
        
        # 生成 CSV
        output = []
        header = ["method", "scene"] + sorted_keys
        output.append(",".join(header))
        
        for r in self.results:
            row = [r['method'], r['scene']]
            for k in sorted_keys:
                v = r.get(k, "")
                row.append(str(v))
            output.append(",".join(row))
            
        csv_str = "\n".join(output)
        
        # 保存文件
        if output_path is None:
            output_path = self.output_dir / "comparison_table.csv"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(csv_str)
            
        return csv_str
        
    def save_json_report(self, output_path: Optional[Path] = None):
        """
        保存 JSON 格式的详细报告
        
        Args:
            output_path: 输出文件路径（可选，默认保存到 output_dir/comparison_report.json）
        """
        if output_path is None:
            output_path = self.output_dir / "comparison_report.json"
            
        report = {
            "results": self.results,
            "summary": self._compute_summary(),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
    def _compute_summary(self) -> Dict:
        """
        计算汇总统计（各方法的平均值）
        
        Returns:
            汇总字典
        """
        summary = {}
        methods = set(r['method'] for r in self.results)
        
        for method in methods:
            method_results = [r for r in self.results if r['method'] == method]
            if not method_results:
                continue
                
            # 计算数值指标的平均值
            numeric_keys = []
            for k, v in method_results[0].items():
                if isinstance(v, (int, float)) and k not in ('method', 'scene'):
                    numeric_keys.append(k)
                    
            method_summary = {"method": method, "num_scenes": len(method_results)}
            for k in numeric_keys:
                values = [r.get(k, 0) for r in method_results if isinstance(r.get(k), (int, float))]
                if values:
                    method_summary[f"avg_{k}"] = sum(values) / len(values)
                    
            summary[method] = method_summary
            
        return summary
        
    def print_summary(self):
        """打印汇总统计"""
        summary = self._compute_summary()
        
        print("\n" + "="*80)
        print("  对比实验汇总")
        print("="*80)
        
        for method, method_summary in summary.items():
            print(f"\n方法: {method}")
            print(f"  场景数: {method_summary['num_scenes']}")
            for k, v in method_summary.items():
                if k.startswith("avg_"):
                    metric_name = k[4:]
                    print(f"  平均 {metric_name}: {v:.4f}")
                    
        print("\n" + "="*80)
