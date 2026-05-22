#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
有声书生成主脚本

两阶段执行：
  阶段1: LLM分割所有章节 → 验证全部成功
  阶段2: TTS生成音频 → 合并输出
"""

import os
import sys
import json
import time
import asyncio
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import get_config, ConfigManager
from modules.novel_reader import read_novel, split_chapters, Chapter
from modules.dialogue_splitter import split_dialogues_async, DialogueLine, clear_global_character_map, get_global_character_map
from modules.voice_assigner import assign_voices
from modules.tts_generator import generate_tts_batch, save_wav
from modules.audio_processor import (
    get_wav_duration, merge_wav_files, split_audio_by_duration,
    create_silence, format_duration
)
from modules.coverage_analyzer import analyze_coverage, print_coverage_report, print_summary


@dataclass
class ChapterSplitResult:
    """章节分割结果"""
    chapter: Chapter
    lines: List[DialogueLine]
    coverage: float
    success: bool
    error: str = ""


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="有声书生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate.py novel.txt                    # 使用默认配置生成
  python generate.py novel.txt -c 3               # 只处理前3章
  python generate.py novel.txt --first-paragraphs 10  # 只处理每个章节的前10段
  python generate.py novel.txt --first-chars 5000     # 只处理每个章节的前5000字
  python generate.py novel.txt --no-skip           # 不跳过已生成的章节
  python generate.py novel.txt --chunk-size 2000   # 设置LLM分割的chunk大小
        """
    )

    parser.add_argument("novel", help="小说文件路径（支持 input/ 下相对路径）")
    parser.add_argument("-o", "--output", help="输出目录（默认 output/{书名}_有声书）")
    parser.add_argument("-c", "--chapters", type=int, default=0, help="处理章节数（0=全部）")
    parser.add_argument("--first-paragraphs", type=int, help="只处理每个章节的前N段")
    parser.add_argument("--first-chars", type=int, help="只处理每个章节的前N个字符")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已生成的章节")
    parser.add_argument("--chunk-size", type=int, help="LLM分割的chunk大小（字符数）")
    parser.add_argument("--max-duration", type=int, help="单个音频文件最大时长（秒）")
    parser.add_argument("--split-only", action="store_true", help="只执行LLM分割，不生成音频")
    parser.add_argument("--test", action="store_true", help="测试API连接")
    parser.add_argument("--min-split-length", type=int, help="最小分割长度（字符数），低于此长度的文本将被跳过")

    return parser.parse_args()


# ============================================================
# 阶段1: LLM分割
# ============================================================

async def split_chapter(
    chapter: Chapter,
    book_name: str,
    config: ConfigManager
) -> ChapterSplitResult:
    """
    分割单个章节（LLM调用）

    Args:
        chapter: 章节数据
        book_name: 书名
        config: 配置管理器

    Returns:
        章节分割结果
    """
    print(f"  📝 章节 {chapter.number:03d}: {chapter.title} ({len(chapter.text)}字)")

    try:
        lines = await split_dialogues_async(chapter.text, chapter.title, book_name)

        if not lines:
            return ChapterSplitResult(
                chapter=chapter,
                lines=[],
                coverage=0.0,
                success=False,
                error="LLM返回空结果"
            )

        # 计算覆盖率
        original_len = len(chapter.text)
        split_len = sum(len(line.text) for line in lines)
        coverage = split_len / original_len if original_len > 0 else 0

        print(f"    ✅ 分割完成: {len(lines)}段, 覆盖率{coverage:.1%}")

        return ChapterSplitResult(
            chapter=chapter,
            lines=lines,
            coverage=coverage,
            success=True
        )

    except Exception as e:
        print(f"    ❌ 分割失败: {e}")
        return ChapterSplitResult(
            chapter=chapter,
            lines=[],
            coverage=0.0,
            success=False,
            error=str(e)
        )


async def split_all_chapters(
    chapters: List[Chapter],
    book_name: str,
    config: ConfigManager
) -> Tuple[List[ChapterSplitResult], List[ChapterSplitResult]]:
    """
    阶段1: 分割所有章节

    Args:
        chapters: 章节列表
        book_name: 书名
        config: 配置管理器

    Returns:
        (成功结果列表, 失败结果列表)
    """
    print(f"\n{'='*60}")
    print(f"  阶段1: LLM分割所有章节")
    print(f"{'='*60}")

    start_time = time.time()
    results = []

    # 逐章分割（顺序执行，避免同时触发审核）
    for chapter in chapters:
        result = await split_chapter(chapter, book_name, config)
        results.append(result)

    elapsed = time.time() - start_time

    # 分类结果
    success_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]

    print(f"\n  📊 分割统计:")
    print(f"     成功: {len(success_results)}/{len(chapters)} 章")
    print(f"     失败: {len(failed_results)}/{len(chapters)} 章")
    print(f"     耗时: {elapsed:.1f}秒")

    if failed_results:
        print(f"\n  ⚠️ 失败章节:")
        for r in failed_results:
            print(f"     - {r.chapter.title}: {r.error}")

    return success_results, failed_results


# ============================================================
# 阶段2: TTS生成
# ============================================================

async def generate_chapter_audio(
    result: ChapterSplitResult,
    book_name: str,
    config: ConfigManager,
    skip_existing: bool = True
) -> Optional[List[str]]:
    """
    生成单个章节的音频

    Args:
        result: 章节分割结果
        book_name: 书名
        config: 配置管理器
        skip_existing: 是否跳过已生成的章节

    Returns:
        生成的音频文件路径列表，失败返回None
    """
    chapter = result.chapter
    lines = result.lines

    print(f"\n  🎤 章节 {chapter.number:03d}: {chapter.title}")

    # 检查是否已存在
    output_dir = config.get_output_dir(book_name)
    existing_files = list(output_dir.glob(f"{chapter.number:03d}_*.wav"))

    if skip_existing and existing_files:
        print(f"    ⏭ 已存在（{len(existing_files)}个文件），跳过")
        return [str(f) for f in existing_files]

    # 分配音色
    voice_map = assign_voices(lines)

    # 为每行设置音色
    for line in lines:
        line.voice_id = voice_map.get(line.character, config.config.get("voice_presets", {}).get("default", "冰糖"))

    # 准备TTS任务
    temp_dir = config.get_temp_dir(book_name, chapter.number)
    tasks = []
    for j, line in enumerate(lines):
        if not line.text.strip():
            continue

        segment_file = str(temp_dir / f"seg_{j:04d}.wav")
        tasks.append({
            "idx": j,
            "text": line.text,
            "voice": line.voice_id,
            "emotion": line.emotion,
            "output_file": segment_file
        })

    # 批量生成TTS
    tts_results = await generate_tts_batch(tasks)

    # 统计结果并报告失败片段
    tts_failed_indices = []
    for result in tts_results:
        if result["success"]:
            pass  # 成功的不打印，避免刷屏
        else:
            error_msg = result.get("error", "未知错误")
            tts_failed_indices.append(result["idx"] + 1)  # 转为1-based
            print(f"      ❌ 片段 {result['idx']+1} 生成失败: {error_msg}")

    success_count = sum(1 for r in tts_results if r["success"])
    print(f"    ✅ TTS生成: {success_count}/{len(tasks)}段成功")

    # 收集成功的音频文件
    segment_files = [r["file"] for r in tts_results if r["success"]]
    segment_files.sort()

    # 片段完整性校验：检查是否有缺失的片段
    expected_count = sum(1 for line in lines if line.text.strip())
    if len(segment_files) != expected_count:
        print(f"    ⚠️ 片段数量不匹配: 期望{expected_count}，实际{len(segment_files)}")

        # 找出缺失的片段索引
        generated_indices = set()
        for seg_file in segment_files:
            # 从文件名提取索引: seg_0001.wav -> 1
            basename = Path(seg_file).name
            if basename.startswith("seg_") and basename.endswith(".wav"):
                try:
                    idx = int(basename[4:8])
                    generated_indices.add(idx)
                except ValueError:
                    pass

        missing_indices = []
        missing_details = []
        for j, line in enumerate(lines):
            if line.text.strip() and j not in generated_indices:
                missing_indices.append(j + 1)  # 转为1-based
                missing_details.append(f"{j+1}[{line.character}]: {line.text[:30]}...")

        if missing_indices:
            print(f"    ❌ 缺失片段({len(missing_indices)}个): {missing_indices[:20]}{'...' if len(missing_indices) > 20 else ''}")
            for detail in missing_details[:10]:
                print(f"       {detail}")
    else:
        print(f"    ✅ 片段完整性校验通过: {len(segment_files)}/{expected_count}")

    # 报告TTS失败的片段汇总
    if tts_failed_indices:
        print(f"    ⚠️ TTS生成失败的片段({len(tts_failed_indices)}个): {tts_failed_indices[:20]}{'...' if len(tts_failed_indices) > 20 else ''}")

    if not segment_files:
        print(f"    ❌ 没有生成任何音频")
        return None

    # 合并音频
    max_duration = config.generation.max_duration_per_file
    silence_sec = config.generation.silence_between_segments
    batches = split_audio_by_duration(segment_files, max_duration)

    output_files = []
    for i, batch in enumerate(batches):
        if len(batches) == 1:
            output_filename = f"{chapter.number:03d}_{chapter.title}.wav"
        else:
            output_filename = f"{chapter.number:03d}_{chapter.title}_part{i+1:02d}.wav"

        output_path = str(output_dir / output_filename)

        if merge_wav_files(batch, output_path, silence_sec):
            duration = get_wav_duration(output_path)
            print(f"      ✅ {output_filename} ({format_duration(duration)})")
            output_files.append(output_path)
        else:
            print(f"      ❌ {output_filename} 合并失败")

    # 保存脚本
    script_file = output_dir / f"{chapter.number:03d}_{chapter.title}_script.json"
    script_data = {
        "chapter": chapter.title,
        "lines": [
            {
                "character": line.character,
                "text": line.text,
                "emotion": line.emotion,
                "voice": line.voice_id
            }
            for line in lines
        ]
    }
    with open(script_file, 'w', encoding='utf-8') as f:
        json.dump(script_data, f, ensure_ascii=False, indent=2)

    # 生成可读文本文件
    readable_file = output_dir / f"{chapter.number:03d}_{chapter.title}_readable.txt"
    _generate_readable_text_file(script_data, readable_file)

    return output_files


def _generate_readable_text_file(script_data: dict, output_path: Path):
    """
    生成可读性强的文本文件

    Args:
        script_data: 脚本数据（包含chapter和lines）
        output_path: 输出文件路径
    """
    lines = []
    lines.append(f"=== {script_data['chapter']} ===")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"共 {len(script_data['lines'])} 段")
    lines.append("")

    # 统计角色
    characters = set()
    for line in script_data['lines']:
        characters.add(line['character'])

    lines.append(f"角色列表: {', '.join(sorted(characters))}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    # 逐段输出
    for i, line in enumerate(script_data['lines'], 1):
        character = line['character']
        text = line['text']
        emotion = line['emotion']
        voice = line['voice']

        # 格式化输出
        lines.append(f"{i:3d}. [{character}] ({voice})")
        lines.append(f"     {text}")
        if emotion != 'neutral':
            lines.append(f"     情绪: {emotion}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("=== 结束 ===")

    # 写入文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"      📄 可读文本: {output_path.name}")
    except Exception as e:
        print(f"      ⚠️ 生成可读文本失败: {e}")


async def generate_all_audio(
    success_results: List[ChapterSplitResult],
    book_name: str,
    config: ConfigManager,
    skip_existing: bool = True
) -> List[str]:
    """
    阶段2: 生成所有章节的音频

    Args:
        success_results: 成功的分割结果列表
        book_name: 书名
        config: 配置管理器
        skip_existing: 是否跳过已生成的章节

    Returns:
        生成的音频文件路径列表
    """
    print(f"\n{'='*60}")
    print(f"  阶段2: 生成音频")
    print(f"{'='*60}")

    start_time = time.time()
    all_output_files = []

    # 逐章生成音频
    for result in success_results:
        output_files = await generate_chapter_audio(result, book_name, config, skip_existing)
        if output_files:
            all_output_files.extend(output_files)

    elapsed = time.time() - start_time

    print(f"\n  📊 音频统计:")
    print(f"     生成文件: {len(all_output_files)} 个")
    print(f"     耗时: {elapsed:.1f}秒")

    return all_output_files


# ============================================================
# 内容过滤
# ============================================================

def _filter_chapters_content(
    chapters: List[Chapter],
    first_paragraphs: Optional[int] = None,
    first_chars: Optional[int] = None
) -> List[Chapter]:
    """
    过滤章节内容，只保留前N段或前N个字符

    Args:
        chapters: 章节列表
        first_paragraphs: 只保留前N段
        first_chars: 只保留前N个字符

    Returns:
        过滤后的章节列表
    """
    filtered_chapters = []

    for chapter in chapters:
        # 按段落分割
        paragraphs = chapter.text.split('\n\n')
        original_paragraph_count = len(paragraphs)

        # 应用段落数限制
        if first_paragraphs and first_paragraphs > 0:
            paragraphs = paragraphs[:first_paragraphs]

        # 应用字符数限制
        if first_chars and first_chars > 0:
            filtered_text = []
            current_chars = 0
            for para in paragraphs:
                if current_chars + len(para) > first_chars:
                    # 如果加上这段会超限，只取部分
                    remaining = first_chars - current_chars
                    if remaining > 0:
                        filtered_text.append(para[:remaining])
                    break
                filtered_text.append(para)
                current_chars += len(para)
            paragraphs = filtered_text

        # 重新组合文本
        filtered_text = '\n\n'.join(paragraphs)

        # 创建新的Chapter对象
        filtered_chapter = Chapter(
            number=chapter.number,
            title=chapter.title,
            text=filtered_text
        )
        filtered_chapters.append(filtered_chapter)

        # 打印过滤信息
        if first_paragraphs or first_chars:
            print(f"  📝 章节 {chapter.number:03d}: {chapter.title}")
            if first_paragraphs:
                print(f"     段落: {original_paragraph_count} → {len(paragraphs)}")
            if first_chars:
                print(f"     字符: {len(chapter.text)} → {len(filtered_text)}")

    return filtered_chapters


# ============================================================
# 主函数
# ============================================================

async def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = get_config()

    # 应用命令行参数
    if args.chunk_size:
        config.set("generation.chunk_size", args.chunk_size)
    if args.max_duration:
        config.set("generation.max_duration_per_file", args.max_duration)
    if args.min_split_length:
        config.set("generation.min_split_length", args.min_split_length)

    # 测试API连接
    if args.test:
        print("🔍 测试API连接...")

        # 测试LLM API
        print("\n📝 测试LLM API...")
        try:
            from modules.dialogue_splitter import get_llm_client
            llm_client = get_llm_client()
            response = llm_client.chat.completions.create(
                model=config.llm_api.model,
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=10
            )
            print(f"  ✅ LLM API连接成功 (模型: {config.llm_api.model})")
        except Exception as e:
            print(f"  ❌ LLM API连接失败: {e}")

        # 测试TTS API
        print("\n🎤 测试TTS API...")
        try:
            from modules.tts_generator import get_tts_client
            tts_client = get_tts_client()
            response = tts_client.chat.completions.create(
                model=config.tts_api.model,
                messages=[
                    {"role": "user", "content": "用平静的语气朗读"},
                    {"role": "assistant", "content": "你好，这是测试。"}
                ],
                audio={"format": "wav", "voice": "冰糖"}
            )
            print(f"  ✅ TTS API连接成功 (模型: {config.tts_api.model})")
        except Exception as e:
            print(f"  ❌ TTS API连接失败: {e}")

        return

    # 读取小说
    print("\n📖 读取小说...")
    try:
        book_name, content = read_novel(args.novel)
    except Exception as e:
        print(f"❌ 读取小说失败: {e}")
        return

    # 清空全局人物映射表（开始新书时重置）
    clear_global_character_map()
    print(f"  🔄 已清空全局人物映射表")

    # 清空输出目录（确保每次都重新生成）
    output_dir = config.get_output_dir(book_name)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print(f"  🗑️ 已清空输出目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清空LLM原始响应目录
    llm_raw_dir = config.get_llm_raw_dir(book_name)
    if llm_raw_dir.exists():
        import shutil
        shutil.rmtree(llm_raw_dir)
        print(f"  🗑️ 已清空LLM原始响应目录: {llm_raw_dir}")

    # 分割章节
    print("\n📑 分割章节...")
    chapters = split_chapters(content)

    # 限制章节数
    if args.chapters > 0:
        chapters = chapters[:args.chapters]
        print(f"  限制处理前 {args.chapters} 章")

    # 部分处理选项：限制每个章节的段落数或字符数
    if args.first_paragraphs or args.first_chars:
        chapters = _filter_chapters_content(chapters, args.first_paragraphs, args.first_chars)

    # ============================================================
    # 交互式选择：让用户选择要处理的章节
    # ============================================================
    from modules.dialogue_splitter import _split_text_chunks

    print(f"\n📋 章节列表:")
    for i, chapter in enumerate(chapters, 1):
        print(f"  {i}. {chapter.title} ({len(chapter.text)}字)")

    if len(chapters) == 1:
        # 没有章节标题，让用户输入块数
        chunk_size = config.generation.chunk_size
        all_chunks = _split_text_chunks(chapters[0].text, chunk_size)
        print(f"\n  检测到没有章节标题")
        print(f"  按{chunk_size}字分块，共{len(all_chunks)}块")
        print(f"  请输入要处理的块数（例如：2，直接回车处理全部）:")

        user_input = input("  > ").strip()
        if user_input:
            try:
                num_chunks = int(user_input)
                if num_chunks < 1:
                    print("❌ 块数必须大于0")
                    return
                # 取前N块
                selected_chunks = all_chunks[:num_chunks]
                print(f"\n  ✅ 选择处理前{num_chunks}块")
                for i, chunk in enumerate(selected_chunks, 1):
                    print(f"    块{i}: {len(chunk)}字")
            except ValueError:
                print("❌ 请输入有效的数字")
                return
        else:
            selected_chunks = all_chunks
            print(f"\n  ✅ 处理全部{len(selected_chunks)}块")

        # 每块作为独立章节处理
        from modules.novel_reader import Chapter
        chapters = []
        for i, chunk in enumerate(selected_chunks, 1):
            chapters.append(Chapter(
                number=i,
                title=f"块{i}",
                text=chunk
            ))
    else:
        # 有章节，让用户选择章节
        print(f"\n  检测到{len(chapters)}个章节")
        print(f"  请输入要处理的章节号（例如：2,3,4,6，直接回车处理全部）:")

        user_input = input("  > ").strip()
        if user_input:
            try:
                chapter_indices = [int(x.strip()) for x in user_input.split(",")]
                # 验证章节号
                for idx in chapter_indices:
                    if idx < 1 or idx > len(chapters):
                        print(f"❌ 章节号 {idx} 无效，有效范围: 1-{len(chapters)}")
                        return
                # 筛选章节
                chapters = [chapters[i-1] for i in chapter_indices]
                print(f"\n  ✅ 选择处理{len(chapters)}个章节")
                for i, chapter in enumerate(chapters, 1):
                    print(f"    {i}. {chapter.title} ({len(chapter.text)}字)")
            except ValueError:
                print("❌ 请输入有效的章节号（用逗号分隔）")
                return
        else:
            print(f"\n  ✅ 处理全部{len(chapters)}个章节")

    # ============================================================
    # 阶段1: LLM分割
    # ============================================================
    success_results, failed_results = await split_all_chapters(chapters, book_name, config)

    # 检查是否有失败的章节
    if failed_results:
        print(f"\n⚠️ 有 {len(failed_results)} 个章节分割失败，是否继续生成音频？")
        print(f"   失败章节将被跳过。")

    if not success_results:
        print(f"\n❌ 没有成功分割的章节，无法生成音频")
        return

    # 打印全局人物映射表
    character_map = get_global_character_map()
    if character_map:
        print(f"\n🎭 全局人物映射表:")
        for char, voice in character_map.items():
            print(f"   {char} -> {voice}")

    # ============================================================
    # 覆盖率分析
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  覆盖率详细分析")
    print(f"{'='*60}")

    coverage_reports = []
    for result in success_results:
        report = analyze_coverage(
            original_text=result.chapter.text,
            lines=result.lines,
            chapter_title=result.chapter.title
        )
        coverage_reports.append(report)
        print_coverage_report(report, verbose=(report.coverage < 0.9))

    # 打印汇总
    print_summary(coverage_reports)

    # 如果只分割不生成
    if args.split_only:
        print(f"\n✅ 分割完成（--split-only 模式）")
        return

    # ============================================================
    # 阶段2: TTS生成
    # ============================================================
    all_output_files = await generate_all_audio(
        success_results,
        book_name,
        config,
        skip_existing=False  # 强制不跳过，因为已经清空了目录
    )

    # 完成
    print(f"\n{'='*60}")
    print(f"  生成完成!")
    print(f"{'='*60}")
    print(f"  成功章节: {len(success_results)}/{len(chapters)}")
    print(f"  音频文件: {len(all_output_files)} 个")
    print(f"  输出目录: {output_dir}")
    print(f"\n🎉 有声书生成完成！")


if __name__ == "__main__":
    asyncio.run(main())
