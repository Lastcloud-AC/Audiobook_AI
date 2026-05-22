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
    chunks = _split_text_chunks(text, config.generation.chunk_size)
    all_lines = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{chapter_title}_chunk{i:03d}"
        lines = _split_single_chunk(client, chunk, chunk_id, book_name, chapter_title, is_async)
        all_lines.extend(lines)

    return all_lines


async def _split_dialogues_impl_async(client: AsyncOpenAI, text: str, chapter_title: str, book_name: str) -> List[DialogueLine]:
    """异步分割对话的实现"""
    config = get_config()
    chunks = _split_text_chunks(text, config.generation.chunk_size)

    # 获取当前全局人物映射
    current_character_map = get_global_character_map()

    # 并发处理所有chunks
    semaphore = asyncio.Semaphore(config.concurrency.llm_concurrency)

    async def process_chunk(i: int, chunk: str) -> List[DialogueLine]:
        async with semaphore:
            chunk_id = f"{chapter_title}_chunk{i:03d}"
            return await _split_single_chunk_async(client, chunk, chunk_id, book_name, chapter_title, current_character_map)

    tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_lines = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"    ❌ Chunk {i} 处理失败: {result}")
            # 使用fallback分割
            all_lines.extend(_fallback_split(chunks[i]))
        else:
            all_lines.extend(result)

    return all_lines


def _split_text_chunks(text: str, chunk_size: int) -> List[str]:
    """将文本分割成chunks"""
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []

    for para in paragraphs:
        if len('\n\n'.join(current_chunk + [para])) > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
        else:
            current_chunk.append(para)

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def _split_single_chunk(client: OpenAI, text: str, chunk_id: str, book_name: str, chapter_title: str, is_async: bool = False) -> List[DialogueLine]:
    """处理单个chunk（同步）- 使用递归二分法处理内容审核拒绝"""
    return _recursive_split_by_moderation(client, text, chapter_title, chunk_id, book_name)


async def _split_single_chunk_async(client: AsyncOpenAI, text: str, chunk_id: str, book_name: str, chapter_title: str, character_map: Optional[Dict[str, str]] = None) -> List[DialogueLine]:
    """处理单个chunk（异步）- 使用递归二分法处理内容审核拒绝"""
    return await _recursive_split_by_moderation_async(client, text, chapter_title, chunk_id, book_name)


def _parse_llm_response(raw_response: str, original_text: str) -> List[DialogueLine]:
    """解析LLM响应"""
    # 提取JSON数组
    json_match = re.search(r'\[[\s\S]*\]', raw_response)
    if not json_match:
        return _fallback_split(original_text)

    try:
        data = json.loads(json_match.group())
        lines = []
        for item in data:
            if isinstance(item, dict) and "text" in item:
                # 清理角色名
                raw_character = item.get("character", "旁白")
                character = sanitize_character_name(raw_character)

                # 如果角色名被清理为"旁白"，打印警告
                if raw_character != "旁白" and character == "旁白":
                    print(f"    ⚠️ 角色名 '{raw_character}' 无效，已归为旁白")

                lines.append(DialogueLine(
                    character=character,
                    text=item["text"].strip(),
                    emotion=item.get("emotion", "neutral")
                ))
        return lines
    except json.JSONDecodeError:
        return _fallback_split(original_text)


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
    """fallback分割：按段落分割"""
    paragraphs = text.split('\n\n')
    lines = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 尝试检测对话
        dialogue_match = re.match(r'^["「](.*?)["」]\s*(.*?)(?:\s*说|\s*道|\s*喊|\s*问)?$', para)
        if dialogue_match:
            # 检测到对话
            content = dialogue_match.group(1)
            # 尝试从上下文推断角色
            lines.append(DialogueLine(
                character="未知角色",
                text=content,
                emotion="neutral"
            ))
        else:
            # 旁白
            lines.append(DialogueLine(
                character="旁白",
                text=para,
                emotion="neutral"
            ))

    return lines


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
    """检查是否是内容审核错误"""
    moderation_keywords = [
        "high risk", "rejected", "safety", "moderation", "inappropriate", "blocked",
        "I'm sorry", "I cannot", "I can't", "content policy",
        "harmful", "offensive",
        "抱歉", "无法", "不能", "违规", "不当"
    ]
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in moderation_keywords)


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
    first_half, second_half = _split_text_for_moderation(text)
    print(f"  {'  ' * depth}  拆分位置: {len(first_half)}/{len(text)}字")

    first_lines = _recursive_split_by_moderation(
        client, first_half, chapter_title,
        f"{chunk_prefix}_first", book_name, depth + 1, max_depth
    )
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
    """异步版处理内容审核拒绝的拆分（递归时使用同步调用，不占用异步并发池）"""
    first_half, second_half = _split_text_for_moderation(text)
    print(f"  {'  ' * depth}  拆分位置: {len(first_half)}/{len(text)}字")

    # 递归时使用同步客户端，避免占用异步并发池
    sync_client = get_llm_client()
    
    # 顺序处理两半（同步调用）
    first_lines = _recursive_split_by_moderation(
        sync_client, first_half, chapter_title,
        f"{chunk_prefix}_first", book_name, depth + 1, max_depth
    )
    second_lines = _recursive_split_by_moderation(
        sync_client, second_half, chapter_title,
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
        print(f"  {'  ' * depth}⚠ 文本过短({len(text)}字<{min_split_length})，跳过")
        return []

    # 构建提示词
    prompt = config.config.get("split_prompt", "").replace("{text}", text)
    chunk_id = f"{chunk_prefix}_d{depth}"

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

        # 解析响应
        lines = _parse_llm_response(raw_response, text)

        # 验证覆盖率
        coverage = _verify_coverage(text, lines)
        if coverage < 0.7:
            print(f"    ⚠️ 覆盖率过低({coverage:.1%})，使用fallback分割")
            return _fallback_split(text)

        return lines

    except Exception as e:
        print(f"  {'  ' * depth}⚠ API调用异常: {e}")
        _save_llm_response(chunk_id, "", book_name, chapter_title, error=str(e))
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
        print(f"  {'  ' * depth}⚠ 文本过短({len(text)}字<{min_split_length})，跳过")
        return []

    # 构建提示词
    prompt = config.config.get("split_prompt", "").replace("{text}", text)
    chunk_id = f"{chunk_prefix}_d{depth}"
    rate_limiter = get_llm_rate_limiter()

    try:
        # 等待速率限制
        await rate_limiter.acquire()

        response = await client.chat.completions.create(
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
            return await _handle_moderation_split_async(client, text, chapter_title, chunk_prefix, book_name, depth, max_depth)

        # 保存原始响应
        _save_llm_response(chunk_id, raw_response, book_name, chapter_title)

        # 解析响应
        lines = _parse_llm_response(raw_response, text)

        # 验证覆盖率
        coverage = _verify_coverage(text, lines)
        if coverage < 0.7:
            print(f"    ⚠️ 覆盖率过低({coverage:.1%})，使用fallback分割")
            return _fallback_split(text)

        return lines

    except Exception as e:
        print(f"  {'  ' * depth}⚠ API调用异常: {e}")
        _save_llm_response(chunk_id, "", book_name, chapter_title, error=str(e))
        return _fallback_split(text)
