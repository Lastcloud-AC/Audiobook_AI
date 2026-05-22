#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析覆盖率丢失原因
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import get_config
from modules.novel_reader import read_novel, split_chapters
from modules.dialogue_splitter import split_dialogues_async, clear_global_character_map


async def analyze_coverage(file_path: str, max_chars: int = 3000):
    """分析覆盖率丢失原因"""
    print("=" * 60)
    print(f"  覆盖率分析")
    print(f"  文件: {file_path}")
    print(f"  限制: 前{max_chars}字")
    print("=" * 60)

    # 加载配置
    config = get_config()

    # 读取小说
    print("\n📖 读取小说...")
    try:
        book_name, content = read_novel(file_path)
        print(f"  书名: {book_name}")
        print(f"  原始长度: {len(content)}字")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 清空全局人物映射表
    clear_global_character_map()

    # 分割章节
    print("\n📑 分割章节...")
    chapters = split_chapters(content)
    print(f"  章节数: {len(chapters)}")

    # 限制字符数
    if max_chars > 0:
        total_chars = 0
        filtered_chapters = []
        for chapter in chapters:
            if total_chars + len(chapter.text) > max_chars:
                # 截断当前章节
                remaining = max_chars - total_chars
                if remaining > 0:
                    from modules.novel_reader import Chapter
                    filtered_chapter = Chapter(
                        number=chapter.number,
                        title=chapter.title,
                        text=chapter.text[:remaining]
                    )
                    filtered_chapters.append(filtered_chapter)
                    total_chars += remaining
                break
            filtered_chapters.append(chapter)
            total_chars += len(chapter.text)
        chapters = filtered_chapters
        print(f"  限制后: {len(chapters)}章, {total_chars}字")

    # 分析每个章节
    for i, chapter in enumerate(chapters, 1):
        print(f"\n{'─'*50}")
        print(f"📝 章节 {i}/{len(chapters)}: {chapter.title}")
        print(f"   原始文本长度: {len(chapter.text)}字")
        print(f"{'─'*50}")

        # 分割
        lines = await split_dialogues_async(chapter.text, chapter.title, book_name)

        if not lines:
            print(f"   ❌ 返回空结果")
            continue

        # 计算最终覆盖率
        split_len = sum(len(line.text) for line in lines)
        coverage = split_len / len(chapter.text) if len(chapter.text) > 0 else 0

        print(f"\n   📊 最终结果:")
        print(f"      段落数: {len(lines)}")
        print(f"      分割后总长度: {split_len}字")
        print(f"      原始文本长度: {len(chapter.text)}字")
        print(f"      覆盖率: {coverage:.1%}")

        # 分析丢失的文本
        print(f"\n   🔍 丢失分析:")
        print(f"      丢失字数: {len(chapter.text) - split_len}字")
        print(f"      丢失比例: {1 - coverage:.1%}")

        # 显示每个段落的长度
        print(f"\n   📄 段落长度分布:")
        for j, line in enumerate(lines[:10], 1):
            print(f"      {j:2d}. [{line.character:6s}] {len(line.text):3d}字: {line.text[:30]}...")
        if len(lines) > 10:
            print(f"      ... 还有 {len(lines) - 10} 段")

        # 检查是否有重复或遗漏
        print(f"\n   🔎 内容检查:")
        # 简单检查：拼接所有段落文本
        reconstructed = "".join([line.text for line in lines])
        print(f"      拼接后长度: {len(reconstructed)}字")

        # 检查原始文本是否被包含
        if chapter.text in reconstructed:
            print(f"      ✅ 原始文本完全包含在拼接结果中")
        else:
            print(f"      ❌ 原始文本未完全包含在拼接结果中")

            # 找出丢失的部分
            print(f"\n      🔍 尝试定位丢失内容...")
            # 简单方法：检查前100字和后100字
            print(f"         原始文本前100字: {chapter.text[:100]}")
            print(f"         拼接结果前100字: {reconstructed[:100]}")
            print(f"         原始文本后100字: {chapter.text[-100:]}")
            print(f"         拼接结果后100字: {reconstructed[-100:]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_coverage.py <小说文件> [最大字符数]")
        print("示例: python analyze_coverage.py input/嫦娥.docx 3000")
        sys.exit(1)

    file_path = sys.argv[1]
    max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    import asyncio
    asyncio.run(analyze_coverage(file_path, max_chars))
