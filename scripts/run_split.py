#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行LLM分割（不生成TTS）
支持选择块数或章节
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import get_config
from modules.novel_reader import read_novel, split_chapters
from modules.dialogue_splitter import split_dialogues_async, clear_global_character_map, get_global_character_map, _split_text_chunks


async def run_split(file_path: str):
    """
    运行LLM分割

    Args:
        file_path: 小说文件路径
    """
    print("=" * 60)
    print(f"  LLM分割工具")
    print(f"  文件: {file_path}")
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

    # 显示章节信息
    print("\n📋 章节列表:")
    for i, chapter in enumerate(chapters, 1):
        print(f"  {i}. {chapter.title} ({len(chapter.text)}字)")

    # 用户选择
    print("\n" + "=" * 60)
    if len(chapters) == 1:
        # 没有章节标题，让用户输入块数
        print("  检测到没有章节标题")
        print("  请输入要分割的块数（例如：2）:")
        try:
            num_chunks = int(input("  > ").strip())
            if num_chunks < 1:
                print("❌ 块数必须大于0")
                return
        except ValueError:
            print("❌ 请输入有效的数字")
            return

        # 按chunk_size分块，然后取前N块
        chunk_size = config.generation.chunk_size
        print(f"\n  按{chunk_size}字分块，取前{num_chunks}块")

        # 分块
        all_chunks = _split_text_chunks(chapters[0].text, chunk_size)
        print(f"  总分块数: {len(all_chunks)}")
        
        # 取前N块
        chunks = all_chunks[:num_chunks]
        print(f"  实际使用: {len(chunks)}块")
        for i, chunk in enumerate(chunks):
            print(f"    块{i+1}: {len(chunk)}字")

        # 开始分割
        print("\n" + "=" * 60)
        print(f"  开始LLM分割")
        print("=" * 60)

        start_time = time.time()
        all_lines = []

        for i, chunk in enumerate(chunks, 1):
            print(f"\n{'─'*50}")
            print(f"📝 块 {i}/{len(chunks)}: {len(chunk)}字")
            print(f"{'─'*50}")

            chunk_start = time.time()

            try:
                lines = await split_dialogues_async(chunk, f"块{i}", book_name)
                chunk_elapsed = time.time() - chunk_start

                if not lines:
                    print(f"   ❌ 返回空结果")
                    continue

                # 计算覆盖率
                split_len = sum(len(line.text) for line in lines)
                coverage = split_len / len(chunk) if len(chunk) > 0 else 0

                print(f"\n   ✅ 分割完成:")
                print(f"      段落数: {len(lines)}")
                print(f"      覆盖率: {coverage:.1%}")
                print(f"      耗时: {chunk_elapsed:.1f}秒")

                # 统计角色
                characters = {}
                for line in lines:
                    if line.character not in characters:
                        characters[line.character] = 0
                    characters[line.character] += 1

                print(f"\n   🎭 角色统计:")
                for char, count in sorted(characters.items(), key=lambda x: -x[1]):
                    print(f"      {char}: {count}段")

                all_lines.extend(lines)

            except Exception as e:
                print(f"   ❌ 分割失败: {e}")
                import traceback
                traceback.print_exc()

        # 汇总
        elapsed = time.time() - start_time
        total_chunk_len = sum(len(chunk) for chunk in chunks)
        total_coverage = sum(len(line.text) for line in all_lines) / total_chunk_len if total_chunk_len > 0 else 0

        print(f"\n" + "=" * 60)
        print(f"  测试汇总")
        print(f"=" * 60)
        print(f"  块数: {len(chunks)}")
        print(f"  总段落数: {len(all_lines)}")
        print(f"  平均覆盖率: {total_coverage:.1%}")
        print(f"  总耗时: {elapsed:.1f}秒")

    else:
        # 有章节，让用户选择章节
        print("  检测到章节标题")
        print("  请输入要分割的章节号（例如：2,3,4,6）:")
        try:
            chapter_input = input("  > ").strip()
            chapter_indices = [int(x.strip()) for x in chapter_input.split(",")]
            
            # 验证章节号
            for idx in chapter_indices:
                if idx < 1 or idx > len(chapters):
                    print(f"❌ 章节号 {idx} 无效，有效范围: 1-{len(chapters)}")
                    return
        except ValueError:
            print("❌ 请输入有效的章节号（用逗号分隔）")
            return

        # 获取选中的章节
        selected_chapters = [chapters[i-1] for i in chapter_indices]
        print(f"\n  选中的章节:")
        for i, chapter in enumerate(selected_chapters, 1):
            print(f"    {i}. {chapter.title} ({len(chapter.text)}字)")

        # 开始分割
        print("\n" + "=" * 60)
        print(f"  开始LLM分割")
        print("=" * 60)

        start_time = time.time()
        all_lines = []

        for i, chapter in enumerate(selected_chapters, 1):
            print(f"\n{'─'*50}")
            print(f"📝 章节 {i}/{len(selected_chapters)}: {chapter.title}")
            print(f"   文本长度: {len(chapter.text)}字")
            print(f"{'─'*50}")

            # 判断是否需要分块
            if len(chapter.text) <= config.generation.chunk_size:
                # 不超过chunk_size，直接作为一个块
                chunks = [chapter.text]
                print(f"   不超过{config.generation.chunk_size}字，直接处理")
            else:
                # 超过chunk_size，需要分块
                chunks = _split_text_chunks(chapter.text, config.generation.chunk_size)
                print(f"   超过{config.generation.chunk_size}字，分为{len(chunks)}块")
                for j, chunk in enumerate(chunks):
                    print(f"     块{j+1}: {len(chunk)}字")

            # 处理每个块
            for j, chunk in enumerate(chunks, 1):
                print(f"\n   {'─'*40}")
                print(f"   📝 块 {j}/{len(chunks)}: {len(chunk)}字")
                print(f"   {'─'*40}")

                chunk_start = time.time()

                try:
                    lines = await split_dialogues_async(chunk, f"{chapter.title}_块{j}", book_name)
                    chunk_elapsed = time.time() - chunk_start

                    if not lines:
                        print(f"      ❌ 返回空结果")
                        continue

                    # 计算覆盖率
                    split_len = sum(len(line.text) for line in lines)
                    coverage = split_len / len(chunk) if len(chunk) > 0 else 0

                    print(f"\n      ✅ 分割完成:")
                    print(f"         段落数: {len(lines)}")
                    print(f"         覆盖率: {coverage:.1%}")
                    print(f"         耗时: {chunk_elapsed:.1f}秒")

                    # 统计角色
                    characters = {}
                    for line in lines:
                        if line.character not in characters:
                            characters[line.character] = 0
                        characters[line.character] += 1

                    print(f"\n      🎭 角色统计:")
                    for char, count in sorted(characters.items(), key=lambda x: -x[1]):
                        print(f"         {char}: {count}段")

                    all_lines.extend(lines)

                except Exception as e:
                    print(f"      ❌ 分割失败: {e}")
                    import traceback
                    traceback.print_exc()

        # 汇总
        elapsed = time.time() - start_time
        total_chars = sum(len(chapter.text) for chapter in selected_chapters)
        total_coverage = sum(len(line.text) for line in all_lines) / total_chars if total_chars > 0 else 0

        print(f"\n" + "=" * 60)
        print(f"  测试汇总")
        print(f"=" * 60)
        print(f"  章节数: {len(selected_chapters)}")
        print(f"  总段落数: {len(all_lines)}")
        print(f"  平均覆盖率: {total_coverage:.1%}")
        print(f"  总耗时: {elapsed:.1f}秒")

    # 打印全局人物映射表
    character_map = get_global_character_map()
    if character_map:
        print(f"\n🎭 全局人物映射表:")
        for char, voice in character_map.items():
            print(f"   {char} -> {voice}")

    print(f"\n🎉 测试完成！")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python run_split.py <小说文件>")
        print("示例: python run_split.py input/嫦娥.docx")
        sys.exit(1)

    file_path = sys.argv[1]
    asyncio.run(run_split(file_path))
