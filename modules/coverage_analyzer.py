#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
覆盖率分析模块

分析LLM分割后的覆盖率丢失原因
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CoverageReport:
    """覆盖率报告"""
    chapter_title: str
    original_len: int
    split_len: int
    coverage: float
    lost_chars: int
    lost_ratio: float
    segments_count: int
    reconstructed_len: int
    is_fully_covered: bool
    lost_parts: List[Dict[str, str]]


def analyze_coverage(
    original_text: str,
    lines: List,
    chapter_title: str = ""
) -> CoverageReport:
    """
    分析单个章节的覆盖率

    Args:
        original_text: 原始文本
        lines: 分割后的 DialogueLine 列表
        chapter_title: 章节标题

    Returns:
        CoverageReport 覆盖率报告
    """
    # 计算基本统计
    original_len = len(original_text)
    split_len = sum(len(line.text) for line in lines)
    coverage = split_len / original_len if original_len > 0 else 0.0
    lost_chars = original_len - split_len
    lost_ratio = 1 - coverage

    # 拼接所有段落文本
    reconstructed = "".join([line.text for line in lines])
    reconstructed_len = len(reconstructed)

    # 检查是否完全覆盖
    is_fully_covered = (original_text in reconstructed)

    # 分析丢失的部分
    lost_parts = []
    if not is_fully_covered:
        lost_parts = _find_lost_parts(original_text, reconstructed)

    return CoverageReport(
        chapter_title=chapter_title,
        original_len=original_len,
        split_len=split_len,
        coverage=coverage,
        lost_chars=lost_chars,
        lost_ratio=lost_ratio,
        segments_count=len(lines),
        reconstructed_len=reconstructed_len,
        is_fully_covered=is_fully_covered,
        lost_parts=lost_parts
    )


def _find_lost_parts(original: str, reconstructed: str) -> List[Dict[str, str]]:
    """
    找出丢失的文本部分

    Args:
        original: 原始文本
        reconstructed: 拼接后的文本

    Returns:
        丢失部分列表，每个元素包含位置和内容
    """
    lost_parts = []

    # 简单方法：检查前100字和后100字
    if original[:100] != reconstructed[:100]:
        lost_parts.append({
            "type": "开头",
            "original": original[:100],
            "reconstructed": reconstructed[:100]
        })

    if original[-100:] != reconstructed[-100:]:
        lost_parts.append({
            "type": "结尾",
            "original": original[-100:],
            "reconstructed": reconstructed[-100:]
        })

    # 如果拼接后长度与分割后长度不同，说明有重复
    if len(reconstructed) != sum(len(line.text) for line in []):
        # 这里需要传入lines，但为了简化，先跳过
        pass

    return lost_parts


def print_coverage_report(report: CoverageReport, verbose: bool = False) -> None:
    """
    打印覆盖率报告

    Args:
        report: 覆盖率报告
        verbose: 是否显示详细信息
    """
    print(f"\n📊 覆盖率分析: {report.chapter_title}")
    print(f"{'─' * 50}")
    print(f"   原始文本长度: {report.original_len}字")
    print(f"   分割后总长度: {report.split_len}字")
    print(f"   段落数量: {report.segments_count}")
    print(f"   覆盖率: {report.coverage:.1%}")
    print(f"   丢失字数: {report.lost_chars}字")
    print(f"   丢失比例: {report.lost_ratio:.1%}")

    if report.is_fully_covered:
        print(f"   ✅ 原始文本完全包含在拼接结果中")
    else:
        print(f"   ❌ 原始文本未完全包含在拼接结果中")

    if verbose and report.lost_parts:
        print(f"\n   🔍 丢失详情:")
        for part in report.lost_parts:
            print(f"      [{part['type']}]")
            print(f"         原始: {part['original'][:50]}...")
            print(f"         拼接: {part['reconstructed'][:50]}...")


def print_summary(reports: List[CoverageReport]) -> None:
    """
    打印汇总报告

    Args:
        reports: 覆盖率报告列表
    """
    if not reports:
        print("\n📊 没有覆盖率数据")
        return

    print(f"\n{'='*60}")
    print(f"  覆盖率汇总")
    print(f"{'='*60}")

    total_original = sum(r.original_len for r in reports)
    total_split = sum(r.split_len for r in reports)
    total_lost = sum(r.lost_chars for r in reports)
    avg_coverage = total_split / total_original if total_original > 0 else 0

    print(f"  章节数: {len(reports)}")
    print(f"  总原始长度: {total_original}字")
    print(f"  总分割长度: {total_split}字")
    print(f"  总丢失字数: {total_lost}字")
    print(f"  平均覆盖率: {avg_coverage:.1%}")

    # 按覆盖率排序
    sorted_reports = sorted(reports, key=lambda r: r.coverage)

    print(f"\n  📈 覆盖率排名（从低到高）:")
    for i, report in enumerate(sorted_reports[:5], 1):
        emoji = "✅" if report.coverage >= 0.9 else "⚠️" if report.coverage >= 0.7 else "❌"
        print(f"     {i}. {emoji} {report.chapter_title}: {report.coverage:.1%} ({report.lost_chars}字丢失)")

    if len(sorted_reports) > 5:
        print(f"     ... 还有 {len(sorted_reports) - 5} 个章节")

    # 统计覆盖率分布
    high = sum(1 for r in reports if r.coverage >= 0.9)
    medium = sum(1 for r in reports if 0.7 <= r.coverage < 0.9)
    low = sum(1 for r in reports if r.coverage < 0.7)

    print(f"\n  📊 覆盖率分布:")
    print(f"     ✅ 优秀 (≥90%): {high}章")
    print(f"     ⚠️ 一般 (70-90%): {medium}章")
    print(f"     ❌ 较差 (<70%): {low}章")
