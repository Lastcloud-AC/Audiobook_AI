"""
对话分割模块
负责使用LLM将小说文本分割成角色对话和旁白
"""

import os
import re
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from openai import OpenAI, AsyncOpenAI

from .config_manager import get_config, LLM_RAW_DIR


@dataclass
class DialogueLine:
    """对话行数据"""
    character: str   # 角色名
    text: str        # 文本内容
    emotion: str     # 情绪标签
    voice_id: str = ""  # 音色ID（后续填充）


# 全局人物映射表：角色名 -> 音色名
_global_character_map: Dict[str, str] = {}

# 无效角色名列表（代词、描述性词语等）
INVALID_CHARACTER_NAMES: Set[str] = {
    # 代词
    "他", "她", "它", "我", "你", "我们", "你们", "他们", "她们", "它们",
    "这个", "那个", "这些", "那些", "自己", "别人",
    # 描述性词语
    "些调皮地", "没好气地", "开心地", "生气地", "愤怒地", "悲伤地",
    "温柔地", "严厉地", "惊讶地", "恐惧地", "厌恶地",
    # 动作描述
    "说", "道", "喊", "问", "答", "叫", "笑", "哭", "叹", "哼",
    "说道", "喊道", "问道", "答道", "叫道", "笑道", "哭道", "叹道",
    # 其他
    "声音", "语气", "态度", "表情", "样子",
}


def get_global_character_map() -> Dict[str, str]:
    """获取全局人物映射表"""
    return _global_character_map.copy()


def update_global_character_map(new_map: Dict[str, str]):
    """更新全局人物映射表"""
    global _global_character_map
    _global_character_map.update(new_map)


def clear_global_character_map():
    """清空全局人物映射表"""
    global _global_character_map
    _global_character_map.clear()


def is_valid_character_name(name: str) -> bool:
    """
    验证角色名是否有效

    Args:
        name: 角色名

    Returns:
        是否有效
    """
    if not name or not name.strip():
        return False

    name = name.strip()

    # 检查是否在无效列表中
    if name in INVALID_CHARACTER_NAMES:
        return False

    # 检查是否是纯代词
    if re.match(r'^[他她它我你我们你们他们她们它们自己别人]+$', name):
        return False

    # 检查是否是描述性词语（以"地"结尾的副词）
    if name.endswith("地") and len(name) <= 6:
        return False

    # 检查是否是动作描述
    if name in ["说", "道", "喊", "问", "答", "叫", "笑", "哭", "叹", "哼"]:
        return False

    return True


def sanitize_character_name(name: str) -> str:
    """
    清理角色名，如果无效则返回"旁白"

    Args:
        name: 原始角色名

    Returns:
        清理后的角色名
    """
    if not name:
        return "旁白"

    name = name.strip()

    # 如果是"旁白"，直接返回
    if name == "旁白":
        return "旁白"

    # 检查是否在全局映射中（已确认的角色）
    if name in _global_character_map:
        return name

    # 验证角色名有效性
    if not is_valid_character_name(name):
        return "旁白"

    return name


class RateLimiter:
    """速率限制器"""

    def __init__(self, rpm_limit: int):
        self.rpm_limit = rpm_limit
        self.interval = 60.0 / rpm_limit
        self.last_request_time = 0
        self._lock = None
        self._loop = None

    def _get_lock(self):
        """延迟创建 Lock，确保绑定到当前事件循环"""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._lock is None or self._loop != current_loop:
            self._lock = asyncio.Lock()
            self._loop = current_loop
        return self._lock

    async def acquire(self):
        """获取请求许可"""
        lock = self._get_lock()
        async with lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_request_time = time.time()


# 全局速率限制器
_llm_rate_limiter: Optional[RateLimiter] = None


def get_llm_rate_limiter() -> RateLimiter:
    """获取LLM速率限制器"""
    global _llm_rate_limiter
    config = get_config()
    if _llm_rate_limiter is None:
        _llm_rate_limiter = RateLimiter(config.rate_limit.llm_rpm)
    return _llm_rate_limiter


def get_llm_client() -> OpenAI:
    """获取LLM客户端"""
    config = get_config()
    return OpenAI(
        base_url=config.llm_api.base_url,
        api_key=config.llm_api.api_key
    )


def get_async_llm_client() -> AsyncOpenAI:
    """获取异步LLM客户端"""
    config = get_config()
    return AsyncOpenAI(
        base_url=config.llm_api.base_url,
        api_key=config.llm_api.api_key
    )


def split_dialogues(text: str, chapter_title: str, book_name: str) -> List[DialogueLine]:
    """
    将文本分割成对话行（同步版本）

    Args:
        text: 要分割的文本
        chapter_title: 章节标题
        book_name: 书名

    Returns:
        对话行列表
    """
    client = get_llm_client()
    return _split_dialogues_impl(client, text, chapter_title, book_name, is_async=False)


async def split_dialogues_async(text: str, chapter_title: str, book_name: str) -> List[DialogueLine]:
    """
    将文本分割成对话行（异步版本）

    Args:
        text: 要分割的文本
        chapter_title: 章节标题
        book_name: 书名

    Returns:
        对话行列表
    """
    client = get_async_llm_client()
    return await _split_dialogues_impl_async(client, text, chapter_title, book_name)


def _split_dialogues_impl(client, text: str, chapter_title: str, book_name: str, is_async: bool = False) -> List[DialogueLine]:
    """分割对话的实现"""
    config = get_config()

    # 第一步：按chunk_size分块（章节级别）
    chunks = _split_text_chunks(text, config.generation.chunk_size)
    all_lines = []

    for i, chunk in enumerate(chunks):
        # 每个块固定分成N段（N=llm_concurrency）
        segments = _split_fixed_segments(chunk, config.concurrency.llm_concurrency)
        for si, segment in enumerate(segments):
            chunk_id = f"{chapter_title}_block{i:03d}_seg{si:03d}"
            lines = _split_single_chunk(client, segment, chunk_id, book_name, chapter_title, is_async)
            all_lines.extend(lines)

    return all_lines


async def _split_dialogues_impl_async(client: AsyncOpenAI, text: str, chapter_title: str, book_name: str) -> List[DialogueLine]:
    """异步分割对话的实现"""
    config = get_config()

    # 第一步：按chunk_size分块（章节级别）
    chunks = _split_text_chunks(text, config.generation.chunk_size)

    # 显示分块信息
    if len(chunks) > 1:
        print(f"    📦 章节分为{len(chunks)}块处理:")
        for i, chunk in enumerate(chunks):
            print(f"       块{i+1}: {len(chunk)}字")
    else:
        print(f"    📦 章节作为1块处理: {len(chunks[0])}字")

    # 获取当前全局人物映射
    current_character_map = get_global_character_map()

    # 第二步：对每个块，固定分成N段并发处理
    all_lines = []
    chunk_times = {}
    chapter_start = time.time()

    for chunk_idx, chunk in enumerate(chunks):
        # 每个块固定分成N段（N=llm_concurrency）
        segments = _split_fixed_segments(chunk, config.concurrency.llm_concurrency)
        seg_count = len(segments)

        if seg_count > 1:
            print(f"    📦 块{chunk_idx+1}分为{seg_count}段并发处理:")
            for si, seg in enumerate(segments):
                print(f"       段{si+1}: {len(seg)}字")

        # 并发处理这个块的所有段
        semaphore = asyncio.Semaphore(config.concurrency.llm_concurrency)

        async def process_segment(seg_idx: int, segment: str, block_idx: int, total_segs: int) -> List[DialogueLine]:
            async with semaphore:
                chunk_id = f"{chapter_title}_block{block_idx:03d}_seg{seg_idx:03d}"
                start_time = time.time()
                start_str = time.strftime("%H:%M:%S.") + f"{start_time % 1:.3f}"[2:]
                print(f"    ⏳ [{start_str}] 块{block_idx}-段{seg_idx+1}/{total_segs} 开始处理 ({len(segment)}字)")
                result = await _split_single_chunk_async(client, segment, chunk_id, book_name, chapter_title, current_character_map)
                end_time = time.time()
                end_str = time.strftime("%H:%M:%S.") + f"{end_time % 1:.3f}"[2:]
                elapsed = end_time - start_time
                time_key = f"block{block_idx}_seg{seg_idx}"
                chunk_times[time_key] = {"start": start_str, "end": end_str, "elapsed": elapsed, "start_ts": start_time, "end_ts": end_time, "block": block_idx}
                print(f"    ✅ [{end_str}] 块{block_idx}-段{seg_idx+1}/{total_segs} 处理完成: {len(result)}段 (耗时{elapsed:.1f}秒)")
                return result

        block_start = time.time()
        tasks = [process_segment(si, seg, chunk_idx + 1, seg_count) for si, seg in enumerate(segments)]
        seg_results = await asyncio.gather(*tasks, return_exceptions=True)
        block_elapsed = time.time() - block_start

        # 处理结果
        for si, result in enumerate(seg_results):
            if isinstance(result, Exception):
                print(f"    ❌ 块{chunk_idx+1}-段{si+1} 处理失败: {str(result)[:100]}")
            elif result:
                all_lines.extend(result)

    overall_elapsed = time.time() - chapter_start

    # 打印并行情况汇总
    print(f"\n    📊 处理汇总 (总耗时{overall_elapsed:.1f}秒, {len(chunks)}块, {len(chunk_times)}段):")
    print(f"    {'段':>12} | {'开始时间':>12} | {'结束时间':>12} | {'耗时':>6}")
    print(f"    {'------------':>12} | {'--------':>12} | {'--------':>12} | {'------':>6}")
    for key in sorted(chunk_times.keys()):
        t = chunk_times[key]
        print(f"    {key:>12} | {t['start']:>12} | {t['end']:>12} | {t['elapsed']:>5.1f}s")

    # 计算并行效率
    sum_elapsed = sum(t['elapsed'] for t in chunk_times.values())
    if sum_elapsed > 0:
        parallel_ratio = overall_elapsed / sum_elapsed * 100
        print(f"    并行效率: {parallel_ratio:.0f}% (越低越好, 100%表示完全串行)")

    # 检查是否有重叠（并行证据）
    if len(chunk_times) >= 2:
        sorted_chunks = sorted(chunk_times.items(), key=lambda x: x[1]['start_ts'])
        overlaps = []
        for j in range(len(sorted_chunks) - 1):
            idx1, t1 = sorted_chunks[j]
            idx2, t2 = sorted_chunks[j + 1]
            if t2['start_ts'] < t1['end_ts']:
                overlap = t1['end_ts'] - t2['start_ts']
                overlaps.append((idx1, idx2, overlap))

        if overlaps:
            print(f"    ✅ 检测到并行执行:")
            for idx1, idx2, overlap in overlaps:
                print(f"       {idx1}和{idx2}重叠 {overlap:.2f}秒")
        else:
            print(f"    ⚠️ 未检测到并行重叠，可能是串行执行")

    return all_lines


def _split_text_chunks(text: str, chunk_size: int, fixed_segments: int = 0) -> List[str]:
    """
    将文本分割成chunks
    
    参数：
        text: 要分割的文本
        chunk_size: 每块最大字数（当fixed_segments=0时使用）
        fixed_segments: 固定分段数（0表示按chunk_size分割，>0表示固定分N段）
    
    逻辑：
    - fixed_segments > 0: 固定分成N段，在段落边界处切割
    - fixed_segments = 0: 按chunk_size分割，尽可能接近chunk_size
    """
    # 固定分段模式
    if fixed_segments > 0:
        return _split_fixed_segments(text, fixed_segments)
    
    # 按chunk_size分割模式
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    
    while start < len(text):
        # 如果剩余文本不超过chunk_size，直接作为一个块
        if start + chunk_size >= len(text):
            chunks.append(text[start:])
            break
        
        # 在chunk_size位置往前找分块点
        end = start + chunk_size
        
        # 1. 先找100个字符内的\n\n
        search_start = max(start, end - 100)
        newline_pos = text.rfind('\n\n', search_start, end)
        
        if newline_pos > start:
            # 找到了\n\n，在这个位置分块
            chunks.append(text[start:newline_pos])
            start = newline_pos + 2  # 跳过\n\n
        else:
            # 2. 找不到\n\n，找离chunk_size往前最近的\n
            newline_pos = text.rfind('\n', start, end)
            
            if newline_pos > start:
                # 找到了\n，在这个位置分块
                chunks.append(text[start:newline_pos])
                start = newline_pos + 1  # 跳过\n
            else:
                # 3. 都找不到，强制在chunk_size位置分块
                chunks.append(text[start:end])
                start = end
    
    return chunks


def _split_fixed_segments(text: str, n: int) -> List[str]:
    """
    将文本固定分成N段，在段落边界处切割
    
    逻辑：
    1. 计算每段的理想长度 = len(text) / n
    2. 第1段：从开头到理想长度，往前找最近的\n
    3. 第2段：从第1段结尾到理想长度*2，往前找最近的\n
    4. 第N段：从上一段结尾到文本最后
    
    参数：
        text: 要分割的文本
        n: 固定分段数
    """
    if n <= 1 or len(text) <= 100:
        return [text]
    
    segments = []
    ideal_len = len(text) / n
    start = 0
    
    for i in range(n - 1):  # 前n-1段需要找切割点，最后一段直接取剩余
        target_pos = int(ideal_len * (i + 1))
        
        # 确保不超过文本长度
        if target_pos >= len(text):
            break
        
        # 从target_pos往前找最近的\n
        search_start = max(start, target_pos - 200)  # 最多往前找200字
        newline_pos = text.rfind('\n', search_start, target_pos + 100)
        
        if newline_pos > start:
            # 找到了\n，在这个位置分段
            segments.append(text[start:newline_pos])
            start = newline_pos + 1  # 跳过\n
        else:
            # 找不到\n，强制在target_pos分段
            segments.append(text[start:target_pos])
            start = target_pos
    
    # 最后一段
    if start < len(text):
        segments.append(text[start:])
    
    return segments


def _split_single_chunk(client: OpenAI, text: str, chunk_id: str, book_name: str, chapter_title: str, is_async: bool = False) -> List[DialogueLine]:
    """处理单个chunk（同步）- 使用递归二分法处理内容审核拒绝"""
    return _recursive_split_by_moderation(client, text, chapter_title, chunk_id, book_name)


async def _split_single_chunk_async(client: AsyncOpenAI, text: str, chunk_id: str, book_name: str, chapter_title: str, character_map: Optional[Dict[str, str]] = None) -> List[DialogueLine]:
    """处理单个chunk（异步）- 使用递归二分法处理内容审核拒绝"""
    return await _recursive_split_by_moderation_async(client, text, chapter_title, chunk_id, book_name)


def _parse_llm_response(raw_response: str, original_text: str) -> List[DialogueLine]:
    """
    解析LLM响应
    
    处理情况：
    1. 空字符串 → fallback
    2. 非JSON格式 → fallback
    3. JSON解析失败 → fallback
    4. 空数组 → fallback
    5. 缺少text字段 → 跳过该项
    6. text为空 → 跳过该项
    7. 正常解析 → 返回结果
    """
    # 情况1：空字符串
    if not raw_response or not raw_response.strip():
        print(f"    ⚠️ 模型返回空字符串，使用fallback分割")
        return _fallback_split(original_text)
    
    # 情况2：检查是否是错误信息（非JSON）
    if raw_response.strip().startswith('{') and '"error"' in raw_response:
        print(f"    ⚠️ 模型返回错误信息: {raw_response[:100]}，使用fallback分割")
        return _fallback_split(original_text)
    
    # 提取JSON数组
    json_match = re.search(r'\[[\s\S]*\]', raw_response)
    if not json_match:
        print(f"    ⚠️ 模型返回非JSON格式: {raw_response[:100]}，使用fallback分割")
        return _fallback_split(original_text)

    try:
        data = json.loads(json_match.group())
        
        # 情况3：空数组
        if not data:
            print(f"    ⚠️ 模型返回空数组，使用fallback分割")
            return _fallback_split(original_text)
        
        lines = []
        skipped_count = 0
        
        for item in data:
            if isinstance(item, dict):
                # 情况4：缺少text字段
                if "text" not in item:
                    skipped_count += 1
                    continue
                
                # 情况5：text为空
                text = item["text"].strip()
                if not text:
                    skipped_count += 1
                    continue
                
                # 清理角色名
                raw_character = item.get("character", "旁白")
                character = sanitize_character_name(raw_character)

                # 如果角色名被清理为"旁白"，打印警告
                if raw_character != "旁白" and character == "旁白":
                    print(f"    ⚠️ 角色名 '{raw_character}' 无效，已归为旁白")

                lines.append(DialogueLine(
                    character=character,
                    text=text,
                    emotion=item.get("emotion", "neutral")
                ))
        
        # 如果跳过了项目，打印提示
        if skipped_count > 0:
            print(f"    ⚠️ 跳过了 {skipped_count} 个无效项目（缺少text字段或text为空）")
        
        # 如果所有项目都被跳过
        if not lines:
            print(f"    ⚠️ 所有项目都无效，使用fallback分割")
            return _fallback_split(original_text)
        
        return lines
    except json.JSONDecodeError as e:
        print(f"    ⚠️ JSON解析失败: {e}，使用fallback分割")
        return _fallback_split(original_text)


def _try_simplified_prompt_sync(client: OpenAI, text: str, chapter_title: str, chunk_id: str, book_name: str) -> Optional[List[DialogueLine]]:
    """
    同步版：使用简化prompt重试
    
    当正常prompt失败时，用更简单的prompt再试一次，提高成功率
    只在文本>500字时尝试（太短的文本简化prompt效果不好）
    
    Returns:
        成功返回对话行列表，失败返回None
    """
    if len(text) <= 500:
        return None
    
    config = get_config()
    simplified_prompt = f"""请将以下文本按说话人分段，输出JSON数组。
格式：[{{"character":"角色名","text":"内容"}}]
如果是旁白或描述，character写"旁白"。

文本：
{text}"""
    
    try:
        response = client.chat.completions.create(
            model=config.llm_api.model,
            messages=[
                {"role": "system", "content": "只输出JSON数组，不要其他内容。"},
                {"role": "user", "content": simplified_prompt}
            ],
            temperature=0.3
        )
        
        raw_response = response.choices[0].message.content.strip()
        print(f"    简化prompt重试: 收到响应({len(raw_response)}字)")
        
        # 解析响应
        lines = _parse_llm_response(raw_response, text)
        
        # 验证覆盖率
        coverage = _verify_coverage(text, lines)
        if coverage >= 0.7:
            print(f"    ✓ 简化prompt重试成功，覆盖率: {coverage:.1%}")
            return lines
        else:
            print(f"    ✗ 简化prompt重试失败，覆盖率过低: {coverage:.1%}")
            return None
    except Exception as e:
        print(f"    ✗ 简化prompt重试异常: {e}")
        return None


async def _try_simplified_prompt_async(client: AsyncOpenAI, text: str, chapter_title: str, chunk_id: str, book_name: str) -> Optional[List[DialogueLine]]:
    """
    异步版：使用简化prompt重试
    
    当正常prompt失败时，用更简单的prompt再试一次，提高成功率
    只在文本>500字时尝试（太短的文本简化prompt效果不好）
    
    Returns:
        成功返回对话行列表，失败返回None
    """
    if len(text) <= 500:
        return None
    
    config = get_config()
    rate_limiter = get_llm_rate_limiter()
    
    simplified_prompt = f"""请将以下文本按说话人分段，输出JSON数组。
格式：[{{"character":"角色名","text":"内容"}}]
如果是旁白或描述，character写"旁白"。

文本：
{text}"""
    
    try:
        await rate_limiter.acquire()
        
        response = await client.chat.completions.create(
            model=config.llm_api.model,
            messages=[
                {"role": "system", "content": "只输出JSON数组，不要其他内容。"},
                {"role": "user", "content": simplified_prompt}
            ],
            temperature=0.3
        )
        
        raw_response = response.choices[0].message.content.strip()
        print(f"    简化prompt重试: 收到响应({len(raw_response)}字)")
        
        # 检查内容审核拒绝
        if is_content_moderation_error(raw_response):
            print(f"    ✗ 简化prompt重试也被内容审核拒绝")
            return None
        
        # 解析响应
        lines = _parse_llm_response(raw_response, text)
        
        # 验证覆盖率
        coverage = _verify_coverage(text, lines)
        if coverage >= 0.7:
            print(f"    ✓ 简化prompt重试成功，覆盖率: {coverage:.1%}")
            return lines
        else:
            print(f"    ✗ 简化prompt重试失败，覆盖率过低: {coverage:.1%}")
            return None
    except Exception as e:
        print(f"    ✗ 简化prompt重试异常: {e}")
        return None


def _verify_coverage(original_text: str, lines: List[DialogueLine], min_ratio: float = 0.7) -> float:
    """验证分割后的文本覆盖率"""
    if not lines:
        return 0.0

    original_len = len(original_text.strip())
    if original_len == 0:
        return 1.0

    # 计算分割后文本的总长度
    split_len = sum(len(line.text) for line in lines)

    return split_len / original_len


def _fallback_split(text: str) -> List[DialogueLine]:
    """
    fallback分割：最大程度保留内容完整性
    
    优先级：
    1. 内容完整性（不丢失任何文字）
    2. 在标点处切分（保证同一段话是同一个人物说的）
    3. 尝试提取角色名（次要）
    
    策略：
    - 先尝试按段落（\n\n）分割
    - 如果段落太长（>500字），在标点处二次切分
    - 尝试从文本中提取角色名
    """
    # 按段落分割
    paragraphs = text.split('\n\n')
    lines = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果段落太长，在标点处二次切分
        if len(para) > 500:
            # 按句号、问号、感叹号切分
            sentences = re.split(r'([。！？])', para)
            current_text = ""
            for i, sent in enumerate(sentences):
                if not sent:
                    continue
                current_text += sent
                # 遇到标点符号且累积文本足够长时，生成一个DialogueLine
                if sent in '。！？' and len(current_text) > 100:
                    character, content = _extract_character_from_text(current_text)
                    lines.append(DialogueLine(
                        character=character,
                        text=content,
                        emotion="neutral"
                    ))
                    current_text = ""
            # 剩余文本
            if current_text.strip():
                character, content = _extract_character_from_text(current_text)
                lines.append(DialogueLine(
                    character=character,
                    text=content,
                    emotion="neutral"
                ))
        else:
            # 段落不长，直接处理
            character, content = _extract_character_from_text(para)
            lines.append(DialogueLine(
                character=character,
                text=content,
                emotion="neutral"
            ))

    return lines


def _extract_character_from_text(text: str) -> Tuple[str, str]:
    """
    从文本中提取角色名和内容
    
    Returns:
        (character, content): 角色名和内容
    """
    text = text.strip()
    if not text:
        return "旁白", ""

    # 模式1: "XXX说/道/喊/问：'对话内容'" 或 "XXX说/道/喊/问：「对话内容」"
    match1 = re.match(r'^([\u4e00-\u9fa5]{1,10}?)(?:说|道|喊|问|答|叫|笑|哭|叹|哼)[：:]\s*["「](.*?)["」]\s*$', text)
    if match1:
        return match1.group(1), match1.group(2)

    # 模式2: "「对话内容」XXX说"
    match2 = re.match(r'^["「](.*?)["」]\s*([\u4e00-\u9fa5]{1,10}?)(?:说|道|喊|问|答|叫|笑|哭|叹|哼)\s*$', text)
    if match2:
        return match2.group(2), match2.group(1)

    # 模式3: "「对话内容」"（无角色名）
    match3 = re.match(r'^["「](.*?)["」]\s*$', text)
    if match3:
        return "未知角色", match3.group(1)

    # 模式4: 纯文本 → 旁白
    return "旁白", text


def _save_llm_response(chunk_id: str, raw_response: str, book_name: str, chapter_title: str, error: str = None):
    """保存LLM原始响应"""
    config = get_config()

    # 获取分类目录
    response_dir = config.get_llm_raw_dir(book_name, chapter_title)

    # 生成文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{chunk_id}.json"
    filepath = response_dir / filename

    # 准备数据
    data = {
        "chunk_id": chunk_id,
        "timestamp": timestamp,
        "raw_response": raw_response,
        "error": error
    }

    # 保存JSON文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"    ⚠️ 保存LLM响应失败: {e}")

    # 生成可读文本文件
    try:
        readable_filename = f"{timestamp}_{chunk_id}_readable.txt"
        readable_filepath = response_dir / readable_filename

        # 解析JSON并生成可读文本
        readable_text = _generate_readable_text(raw_response, chunk_id)

        with open(readable_filepath, 'w', encoding='utf-8') as f:
            f.write(readable_text)
    except Exception as e:
        print(f"    ⚠️ 保存可读文本失败: {e}")


def _generate_readable_text(raw_response: str, chunk_id: str) -> str:
    """
    生成可读性强的文本格式

    Args:
        raw_response: LLM原始响应（JSON字符串）
        chunk_id: chunk标识

    Returns:
        可读的文本内容
    """
    lines = []
    lines.append(f"=== {chunk_id} ===")
    lines.append(f"解析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 尝试解析JSON
    json_match = re.search(r'\[[\s\S]*\]', raw_response)
    if not json_match:
        lines.append("⚠️ 无法解析JSON响应")
        lines.append("")
        lines.append("原始响应:")
        lines.append(raw_response)
        return "\n".join(lines)

    try:
        data = json.loads(json_match.group())
        lines.append(f"共 {len(data)} 段:")
        lines.append("")

        for i, item in enumerate(data, 1):
            if isinstance(item, dict) and "text" in item:
                character = item.get("character", "旁白")
                text = item["text"].strip()
                emotion = item.get("emotion", "neutral")

                # 格式化输出
                lines.append(f"{i:3d}. [{character}] {text}")
                if emotion != "neutral":
                    lines.append(f"     情绪: {emotion}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)

    except json.JSONDecodeError as e:
        lines.append(f"⚠️ JSON解析失败: {e}")
        lines.append("")
        lines.append("原始响应:")
        lines.append(raw_response)
        return "\n".join(lines)


def is_content_moderation_error(content: str) -> bool:
    """检查是否是API平台级内容审核拒绝（而非LLM的正常回复）"""
    # 只匹配API平台级的审核拒绝，不匹配LLM的正常回复
    # API平台审核拒绝的典型特征：英文短句，不包含JSON
    platform_rejection_patterns = [
        "the request was rejected because it was considered high risk",
        "content blocked due to safety",
        "content moderation policy violation",
        "your request has been blocked",
    ]
    content_lower = content.lower().strip()

    # 匹配API平台级拒绝
    for pattern in platform_rejection_patterns:
        if pattern in content_lower:
            return True

    # 额外检查：如果返回内容很短且不包含JSON特征，可能是审核拒绝
    if len(content_lower) < 100 and not content_lower.startswith('[') and not content_lower.startswith('{'):
        # 检查是否包含明确的审核拒绝关键词
        explicit_keywords = ["high risk", "blocked", "moderation", "rejected", "违规", "封禁"]
        if any(kw in content_lower for kw in explicit_keywords):
            return True

    return False


def _extract_json_array(raw: str) -> Optional[list]:
    """从LLM原始返回中稳健提取JSON数组"""
    # 1. 去除markdown代码块
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

    # 2. 直接尝试解析
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # API错误返回
            if "error" in data:
                return None
            # dict包装数组：按优先级查找
            for key in ["lines", "dialogues", "segments", "result", "content",
                         "items", "data", "output", "text", "texts", "speeches",
                         "clips", "parts", "chapters", "dialogue_lines", "thoughts"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # 遍历所有value找数组
            for v in data.values():
                if isinstance(v, list):
                    return v
            return None
    except json.JSONDecodeError:
        pass

    # 3. 提取最外层 [...] 数组
    match = re.search(r'\[[\s\S]*\]', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # 修复常见问题：尾逗号
            fixed = re.sub(r',\s*([\]\}])', r'\1', match.group())
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    # 4. 拼接逗号分隔的独立JSON对象为数组
    obj_pattern = re.compile(r'\{[^{}]*\}')
    objects = obj_pattern.findall(cleaned)
    if len(objects) >= 2:
        try:
            arr = [json.loads(obj) for obj in objects]
            return arr
        except json.JSONDecodeError:
            pass

    return None


def _split_text_for_moderation(text: str) -> Tuple[str, str]:
    """
    智能拆分文本（优先在标点符号处切割，保持句子完整性）
    
    优先级：
    1. 强标点：。！？；……（句子自然结束）
    2. 弱标点：，、：（子句边界）
    3. 空白字符：空格、换行
    4. 最后手段：直接对半切
    
    Returns:
        (first_half, second_half): 前半部分和后半部分，保持原始顺序
    """
    mid = len(text) // 2
    search_range = min(200, len(text) // 3)  # 搜索范围扩大到200字或1/3长度
    
    # 标点优先级（从高到低）
    strong_punct = '。！？；…\n'  # 强标点：句子结束
    weak_punct = '，、："'        # 弱标点：子句边界
    whitespace = ' \t\r'          # 空白字符
    
    # 存储候选位置：(位置, 优先级)
    candidates = []
    
    for offset in range(search_range):
        # 向右搜索
        pos_right = mid + offset
        if pos_right < len(text):
            char = text[pos_right]
            if char in strong_punct:
                candidates.append((pos_right + 1, 3, offset))  # 强标点，切在标点后
            elif char in weak_punct:
                candidates.append((pos_right + 1, 2, offset))  # 弱标点，切在标点后
            elif char in whitespace:
                candidates.append((pos_right + 1, 1, offset))  # 空白，切在空白后
        
        # 向左搜索
        pos_left = mid - offset
        if pos_left > 0:
            char_before = text[pos_left - 1]
            if char_before in strong_punct:
                candidates.append((pos_left, 3, offset))  # 强标点，切在标点后
            elif char_before in weak_punct:
                candidates.append((pos_left, 2, offset))  # 弱标点，切在标点后
            elif pos_left < len(text) and text[pos_left] in whitespace:
                candidates.append((pos_left, 1, offset))  # 空白，切在空白处
    
    if candidates:
        # 按优先级降序、距离中点升序排序
        candidates.sort(key=lambda x: (-x[1], x[2]))
        best_pos = candidates[0][0]
    else:
        # 最后手段：直接对半切
        best_pos = mid
    
    return text[:best_pos], text[best_pos:]


def _handle_moderation_split(
    client: OpenAI,
    text: str,
    chapter_title: str,
    chunk_prefix: str,
    book_name: str,
    depth: int,
    max_depth: int
) -> List[DialogueLine]:
    """同步版处理内容审核拒绝的拆分"""
    config = get_config()
    min_split_length = config.generation.min_split_length

    first_half, second_half = _split_text_for_moderation(text)
    print(f"  {'  ' * depth}  拆分位置: {len(first_half)}/{len(text)}字")

    # 递归前检查文本长度，过短则使用fallback分割
    first_lines = []
    if len(first_half) < min_split_length:
        print(f"  {'  ' * depth}  ⚠ 前半部分过短({len(first_half)}字<{min_split_length})，使用fallback分割")
        first_lines = _fallback_split(first_half)
    else:
        first_lines = _recursive_split_by_moderation(
            client, first_half, chapter_title,
            f"{chunk_prefix}_first", book_name, depth + 1, max_depth
        )

    second_lines = []
    if len(second_half) < min_split_length:
        print(f"  {'  ' * depth}  ⚠ 后半部分过短({len(second_half)}字<{min_split_length})，使用fallback分割")
        second_lines = _fallback_split(second_half)
    else:
        second_lines = _recursive_split_by_moderation(
            client, second_half, chapter_title,
            f"{chunk_prefix}_second", book_name, depth + 1, max_depth
        )

    return first_lines + second_lines


async def _handle_moderation_split_async(
    client: AsyncOpenAI,
    text: str,
    chapter_title: str,
    chunk_prefix: str,
    book_name: str,
    depth: int,
    max_depth: int
) -> List[DialogueLine]:
    """异步版处理内容审核拒绝的拆分（递归时使用异步调用，受Semaphore控制并发）"""
    config = get_config()
    min_split_length = config.generation.min_split_length

    first_half, second_half = _split_text_for_moderation(text)
    print(f"  {'  ' * depth}  拆分位置: {len(first_half)}/{len(text)}字")

    # 递归前检查文本长度，过短则使用fallback分割
    first_lines = []
    if len(first_half) < min_split_length:
        print(f"  {'  ' * depth}  ⚠ 前半部分过短({len(first_half)}字<{min_split_length})，使用fallback分割")
        first_lines = _fallback_split(first_half)
    else:
        first_lines = await _recursive_split_by_moderation_async(
            client, first_half, chapter_title,
            f"{chunk_prefix}_first", book_name, depth + 1, max_depth
        )

    second_lines = []
    if len(second_half) < min_split_length:
        print(f"  {'  ' * depth}  ⚠ 后半部分过短({len(second_half)}字<{min_split_length})，使用fallback分割")
        second_lines = _fallback_split(second_half)
    else:
        second_lines = await _recursive_split_by_moderation_async(
            client, second_half, chapter_title,
            f"{chunk_prefix}_second", book_name, depth + 1, max_depth
        )

    return first_lines + second_lines


def _recursive_split_by_moderation(
    client: OpenAI,
    text: str,
    chapter_title: str,
    chunk_prefix: str,
    book_name: str,
    depth: int = 0,
    max_depth: int = 10
) -> List[DialogueLine]:
    """同步版递归二分法处理内容审核拒绝"""
    config = get_config()

    if depth >= max_depth:
        print(f"  {'  ' * depth}⚠ 达到最大递归深度({max_depth})，使用fallback")
        return _fallback_split(text)

    min_split_length = config.generation.min_split_length
    if len(text) < min_split_length:
        print(f"  {'  ' * depth}⚠ 文本过短({len(text)}字<{min_split_length})，使用fallback分割")
        return _fallback_split(text)

    # 构建提示词
    prompt = config.config.get("split_prompt", "").replace("{text}", text)
    # 替换角色映射占位符
    character_map = get_global_character_map()
    if character_map:
        character_map_str = "\n".join([f"- {name}: {voice}" for name, voice in character_map.items()])
    else:
        character_map_str = "（暂无已知角色）"
    prompt = prompt.replace("{character_map}", character_map_str)
    chunk_id = f"{chunk_prefix}_d{depth}"

    # 重试配置
    max_api_retries = 3  # API调用异常最大重试次数
    max_parse_retries = 1  # JSON解析/覆盖率失败最大重试次数

    # API调用异常重试循环
    for api_attempt in range(max_api_retries):
        try:
            response = client.chat.completions.create(
                model=config.llm_api.model,
                messages=[
                    {"role": "system", "content": "你是一个JSON输出助手，只输出JSON数组，不要其他内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            raw_response = response.choices[0].message.content.strip()

            # 检查内容审核拒绝
            if is_content_moderation_error(raw_response):
                print(f"  {'  ' * depth}⚠ 内容审核拒绝(深度{depth})，对半拆分处理...")
                _save_llm_response(chunk_id, raw_response, book_name, chapter_title, error="content_moderation_rejected")
                return _handle_moderation_split(client, text, chapter_title, chunk_prefix, book_name, depth, max_depth)

            # 保存原始响应
            _save_llm_response(chunk_id, raw_response, book_name, chapter_title)

            # JSON解析/覆盖率重试循环
            for parse_attempt in range(max_parse_retries + 1):
                # 解析响应
                lines = _parse_llm_response(raw_response, text)

                # 验证覆盖率
                coverage = _verify_coverage(text, lines)
                if coverage >= 0.7:
                    return lines

                # 覆盖率过低，判断是否重试
                if parse_attempt < max_parse_retries:
                    print(f"  {'  ' * depth}⚠ 覆盖率过低({coverage:.1%})，重试解析({parse_attempt + 1}/{max_parse_retries})...")
                    continue
                else:
                    print(f"  {'  ' * depth}⚠ 覆盖率过低({coverage:.1%})，已达重试上限，尝试简化prompt重试...")
                    # 简化prompt重试
                    simplified_result = _try_simplified_prompt_sync(client, text, chapter_title, chunk_id, book_name)
                    if simplified_result:
                        return simplified_result
                    print(f"  {'  ' * depth}⚠ 简化prompt重试失败，使用fallback分割")
                    return _fallback_split(text)

        except Exception as e:
            error_msg = str(e).lower()
            
            # 超时且字数多，直接拆分（明显是字数太多导致处理慢）
            is_timeout = "timeout" in error_msg or "timed out" in error_msg or "90" in error_msg
            if is_timeout and len(text) > 1000:
                print(f"  {'  ' * depth}⚠ API超时({len(text)}字>1000)，直接拆分处理...")
                _save_llm_response(chunk_id, "", book_name, chapter_title, error=f"timeout_with_long_text({len(text)}字)")
                return _handle_moderation_split(client, text, chapter_title, chunk_prefix, book_name, depth, max_depth)
            
            # 其他错误，按原有逻辑重试
            if api_attempt < max_api_retries - 1:
                wait_time = (api_attempt + 1) * 2  # 递增等待：2秒、4秒
                print(f"  {'  ' * depth}⚠ API调用异常: {e}，{wait_time}秒后重试({api_attempt + 1}/{max_api_retries})...")
                time.sleep(wait_time)
            else:
                print(f"  {'  ' * depth}⚠ API调用异常: {e}，已达重试上限，尝试简化prompt重试...")
                # 简化prompt重试
                simplified_result = _try_simplified_prompt_sync(client, text, chapter_title, chunk_id, book_name)
                if simplified_result:
                    return simplified_result
                print(f"  {'  ' * depth}⚠ 简化prompt重试失败，使用fallback分割")
                _save_llm_response(chunk_id, "", book_name, chapter_title, error=str(e))
                return _fallback_split(text)

    # 理论上不会到这里，但作为保底
    return _fallback_split(text)


async def _recursive_split_by_moderation_async(
    client: AsyncOpenAI,
    text: str,
    chapter_title: str,
    chunk_prefix: str,
    book_name: str,
    depth: int = 0,
    max_depth: int = 10
) -> List[DialogueLine]:
    """异步版递归二分法处理内容审核拒绝"""
    config = get_config()

    if depth >= max_depth:
        print(f"  {'  ' * depth}⚠ 达到最大递归深度({max_depth})，使用fallback")
        return _fallback_split(text)

    min_split_length = config.generation.min_split_length
    if len(text) < min_split_length:
        print(f"  {'  ' * depth}⚠ 文本过短({len(text)}字<{min_split_length})，使用fallback分割")
        return _fallback_split(text)

    # 构建提示词
    prompt = config.config.get("split_prompt", "").replace("{text}", text)
    # 替换角色映射占位符
    character_map = get_global_character_map()
    if character_map:
        character_map_str = "\n".join([f"- {name}: {voice}" for name, voice in character_map.items()])
    else:
        character_map_str = "（暂无已知角色）"
    prompt = prompt.replace("{character_map}", character_map_str)
    chunk_id = f"{chunk_prefix}_d{depth}"
    rate_limiter = get_llm_rate_limiter()

    # 重试配置
    max_api_retries = 3  # API调用异常最大重试次数
    max_coverage_retries = 1  # 覆盖率失败最大重试次数

    # 重试计数器
    api_error_count = 0  # API异常次数
    coverage_fail_count = 0  # 覆盖率失败次数

    # 计时统计
    total_rate_limit_time = 0
    total_api_time = 0
    total_parse_time = 0

    # API调用重试循环（最多调用 max_api_retries + max_coverage_retries 次）
    total_attempts = max_api_retries + max_coverage_retries
    for attempt in range(total_attempts):
        try:
            # 等待速率限制
            t0 = time.time()
            await rate_limiter.acquire()
            rate_limit_time = time.time() - t0
            total_rate_limit_time += rate_limit_time
            if rate_limit_time > 0.1:
                print(f"  {'  ' * depth}⏱ 速率限制等待: {rate_limit_time:.2f}秒")

            # API调用
            t0 = time.time()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.llm_api.model,
                    messages=[
                        {"role": "system", "content": "你是一个JSON输出助手，只输出JSON数组，不要其他内容。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                ),
                timeout=90  # 90秒超时
            )
            api_time = time.time() - t0
            total_api_time += api_time
            print(f"  {'  ' * depth}⏱ API调用: {api_time:.2f}秒")

            raw_response = response.choices[0].message.content.strip()

            # 检查内容审核拒绝 → 直接对半拆分，不重试（内容有问题，重试没用）
            if is_content_moderation_error(raw_response):
                print(f"  {'  ' * depth}⚠ 内容审核拒绝(深度{depth})，直接对半拆分处理...")
                _save_llm_response(chunk_id, raw_response, book_name, chapter_title, error="content_moderation_rejected")
                return await _handle_moderation_split_async(client, text, chapter_title, chunk_prefix, book_name, depth, max_depth)

            # 保存原始响应
            _save_llm_response(chunk_id, raw_response, book_name, chapter_title)

            # 解析响应
            t0 = time.time()
            lines = _parse_llm_response(raw_response, text)
            parse_time = time.time() - t0
            total_parse_time += parse_time
            print(f"  {'  ' * depth}⏱ 解析响应: {parse_time:.3f}秒, 生成{len(lines)}段")

            # 验证覆盖率
            coverage = _verify_coverage(text, lines)
            if coverage >= 0.7:
                print(f"  {'  ' * depth}✅ 覆盖率通过: {coverage:.1%}")
                print(f"  {'  ' * depth}📊 时间统计 - 速率限制: {total_rate_limit_time:.2f}秒, API调用: {total_api_time:.2f}秒, 解析: {total_parse_time:.3f}秒")
                return lines

            # 覆盖率过低，判断是否重试（只重试1次，重新调用API）
            coverage_fail_count += 1
            if coverage_fail_count <= max_coverage_retries:
                print(f"  {'  ' * depth}⚠ 覆盖率过低({coverage:.1%})，重新调用API重试({coverage_fail_count}/{max_coverage_retries})...")
                continue
            else:
                # 覆盖率重试都失败，对半拆分递归
                print(f"  {'  ' * depth}⚠ 覆盖率过低({coverage:.1%})，重试{max_coverage_retries}次都失败，对半拆分递归处理...")
                print(f"  {'  ' * depth}📊 时间统计 - 速率限制: {total_rate_limit_time:.2f}秒, API调用: {total_api_time:.2f}秒, 解析: {total_parse_time:.3f}秒")
                return await _handle_moderation_split_async(client, text, chapter_title, chunk_prefix, book_name, depth, max_depth)

        except asyncio.TimeoutError:
            # 超时且字数多，直接拆分（明显是字数太多导致处理慢）
            if len(text) > 1000:
                print(f"  {'  ' * depth}⚠ API超时({len(text)}字>1000)，直接拆分处理...")
                _save_llm_response(chunk_id, "", book_name, chapter_title, error=f"timeout_with_long_text({len(text)}字)")
                return await _handle_moderation_split_async(client, text, chapter_title, chunk_prefix, book_name, depth, max_depth)
            
            # 字数不多，按原有逻辑重试
            api_error_count += 1
            if api_error_count < max_api_retries:
                wait_time = api_error_count * 5  # 递增等待：5秒、10秒
                print(f"  {'  ' * depth}⚠ API调用超时(90秒)，{wait_time}秒后重试({api_error_count}/{max_api_retries})...")
                await asyncio.sleep(wait_time)
            else:
                # 重试都失败，直接对半拆分递归
                print(f"  {'  ' * depth}⚠ API调用超时(90秒)，重试{max_api_retries}次都失败，对半拆分递归处理...")
                print(f"  {'  ' * depth}📊 时间统计 - 速率限制: {total_rate_limit_time:.2f}秒, API调用: {total_api_time:.2f}秒, 解析: {total_parse_time:.3f}秒")
                _save_llm_response(chunk_id, "", book_name, chapter_title, error="timeout")
                return await _handle_moderation_split_async(client, text, chapter_title, chunk_prefix, book_name, depth, max_depth)
        except Exception as e:
            error_type = type(e).__name__
            api_error_count += 1
            if api_error_count < max_api_retries:
                wait_time = api_error_count * 5  # 递增等待：5秒、10秒
                print(f"  {'  ' * depth}⚠ API调用异常({error_type}): {e}，{wait_time}秒后重试({api_error_count}/{max_api_retries})...")
                await asyncio.sleep(wait_time)
            else:
                # 重试都失败，直接对半拆分递归
                print(f"  {'  ' * depth}⚠ API调用异常({error_type}): {e}，重试{max_api_retries}次都失败，对半拆分递归处理...")
                print(f"  {'  ' * depth}📊 时间统计 - 速率限制: {total_rate_limit_time:.2f}秒, API调用: {total_api_time:.2f}秒, 解析: {total_parse_time:.3f}秒")
                _save_llm_response(chunk_id, "", book_name, chapter_title, error=f"{error_type}: {e}")
                return await _handle_moderation_split_async(client, text, chapter_title, chunk_prefix, book_name, depth, max_depth)

    # 理论上不会到这里，但作为保底
    return _fallback_split(text)
