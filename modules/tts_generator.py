"""
TTS生成模块
负责调用TTS API生成语音
"""

import os
import re
import time
import base64
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI

from .config_manager import get_config
from .dialogue_splitter import DialogueLine, RateLimiter, is_content_moderation_error


# 全局速率限制器
_tts_rate_limiter: Optional[RateLimiter] = None


def get_tts_rate_limiter() -> RateLimiter:
    """获取TTS速率限制器"""
    global _tts_rate_limiter
    config = get_config()
    if _tts_rate_limiter is None:
        _tts_rate_limiter = RateLimiter(config.rate_limit.tts_rpm)
    return _tts_rate_limiter


def get_tts_client() -> OpenAI:
    """获取TTS客户端"""
    config = get_config()
    return OpenAI(
        base_url=config.tts_api.base_url,
        api_key=config.tts_api.api_key
    )


def get_async_tts_client() -> AsyncOpenAI:
    """获取异步TTS客户端"""
    config = get_config()
    return AsyncOpenAI(
        base_url=config.tts_api.base_url,
        api_key=config.tts_api.api_key
    )


def generate_speech(text: str, voice: str, emotion: str = "neutral") -> Optional[bytes]:
    """
    生成单段语音（同步）

    Args:
        text: 要转换的文本
        voice: 音色名称
        emotion: 情绪标签

    Returns:
        音频字节数据，失败返回None
    """
    client = get_tts_client()
    return _generate_speech_impl(client, text, voice, emotion)


async def generate_speech_async(text: str, voice: str, emotion: str = "neutral") -> Optional[bytes]:
    """
    生成单段语音（异步）

    Args:
        text: 要转换的文本
        voice: 音色名称
        emotion: 情绪标签

    Returns:
        音频字节数据，失败返回None
    """
    client = get_async_tts_client()
    return await _generate_speech_impl_async(client, text, voice, emotion)


def _generate_speech_impl(client: OpenAI, text: str, voice: str, emotion: str) -> Optional[bytes]:
    """生成语音的实现（同步）"""
    config = get_config()

    # 检查文本长度
    if len(text) < config.tts.min_chars:
        print(f"    ⚠️ TTS文本过短({len(text)}字)，跳过")
        return None

    if len(text) > config.tts.max_chars:
        # 文本过长，需要拆分
        return _generate_long_text(client, text, voice, emotion)

    # 获取情绪提示词
    emotion_prompts = config.config.get("emotion_prompts", {})
    emotion_hint = emotion_prompts.get(emotion, emotion_prompts.get("default", "用自然的语气朗读"))

    try:
        response = client.chat.completions.create(
            model=config.tts_api.model,
            messages=[
                {"role": "user", "content": emotion_hint},
                {"role": "assistant", "content": text}
            ],
            audio={"format": "wav", "voice": voice}
        )

        # 检查响应
        if not response.choices or not response.choices[0].message.audio:
            print(f"    ❌ TTS响应无音频数据")
            return None

        audio_data = response.choices[0].message.audio.data
        if not audio_data:
            print(f"    ❌ TTS音频数据为空")
            return None

        return base64.b64decode(audio_data)

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # 常见错误分类
        if "content" in error_msg.lower() and ("moderation" in error_msg.lower() or "filter" in error_msg.lower()):
            print(f"    ❌ TTS内容审核拒绝: {error_msg[:100]}")
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            print(f"    ❌ TTS速率限制: {error_msg[:100]}")
        elif "timeout" in error_msg.lower():
            print(f"    ❌ TTS请求超时: {error_msg[:100]}")
        elif "auth" in error_msg.lower() or "key" in error_msg.lower():
            print(f"    ❌ TTS认证失败: {error_msg[:100]}")
        else:
            print(f"    ❌ TTS生成失败({error_type}): {error_msg[:100]}")
        
        return None


async def _generate_speech_impl_async(client: AsyncOpenAI, text: str, voice: str, emotion: str) -> Optional[bytes]:
    """生成语音的实现（异步）"""
    config = get_config()
    rate_limiter = get_tts_rate_limiter()

    # 检查文本长度
    if len(text) < config.tts.min_chars:
        print(f"    ⚠️ TTS文本过短({len(text)}字)，跳过")
        return None

    if len(text) > config.tts.max_chars:
        # 文本过长，需要拆分
        return await _generate_long_text_async(client, text, voice, emotion)

    # 获取情绪提示词
    emotion_prompts = config.config.get("emotion_prompts", {})
    emotion_hint = emotion_prompts.get(emotion, emotion_prompts.get("default", "用自然的语气朗读"))

    try:
        # 等待速率限制
        await rate_limiter.acquire()

        response = await client.chat.completions.create(
            model=config.tts_api.model,
            messages=[
                {"role": "user", "content": emotion_hint},
                {"role": "assistant", "content": text}
            ],
            audio={"format": "wav", "voice": voice}
        )

        # 检查响应
        if not response.choices or not response.choices[0].message.audio:
            print(f"    ❌ TTS响应无音频数据")
            return None

        audio_data = response.choices[0].message.audio.data
        if not audio_data:
            print(f"    ❌ TTS音频数据为空")
            return None

        return base64.b64decode(audio_data)

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # 常见错误分类
        if "content" in error_msg.lower() and ("moderation" in error_msg.lower() or "filter" in error_msg.lower()):
            print(f"    ❌ TTS内容审核拒绝: {error_msg[:100]}")
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            print(f"    ❌ TTS速率限制: {error_msg[:100]}")
        elif "timeout" in error_msg.lower():
            print(f"    ❌ TTS请求超时: {error_msg[:100]}")
        elif "auth" in error_msg.lower() or "key" in error_msg.lower():
            print(f"    ❌ TTS认证失败: {error_msg[:100]}")
        else:
            print(f"    ❌ TTS生成失败({error_type}): {error_msg[:100]}")
        
        return None


def _split_text_for_moderation(text: str) -> Tuple[str, str]:
    """
    智能拆分文本（优先在标点符号处切割，保持句子完整性）
    """
    mid = len(text) // 2
    search_range = min(200, len(text) // 3)

    strong_punct = '。！？；…\n'
    weak_punct = '，、："'
    whitespace = ' \t\r'

    candidates = []

    for offset in range(search_range):
        pos_right = mid + offset
        if pos_right < len(text):
            char = text[pos_right]
            if char in strong_punct:
                candidates.append((pos_right + 1, 3, offset))
            elif char in weak_punct:
                candidates.append((pos_right + 1, 2, offset))
            elif char in whitespace:
                candidates.append((pos_right + 1, 1, offset))

        pos_left = mid - offset
        if pos_left > 0:
            char_before = text[pos_left - 1]
            if char_before in strong_punct:
                candidates.append((pos_left, 3, offset))
            elif char_before in weak_punct:
                candidates.append((pos_left, 2, offset))
            elif pos_left < len(text) and text[pos_left] in whitespace:
                candidates.append((pos_left, 1, offset))

    if candidates:
        candidates.sort(key=lambda x: (-x[1], x[2]))
        best_pos = candidates[0][0]
    else:
        best_pos = mid

    return text[:best_pos], text[best_pos:]


async def _generate_speech_with_moderation_retry_async(
    client: AsyncOpenAI,
    text: str,
    voice: str,
    emotion: str,
    depth: int = 0,
    max_depth: int = 5
) -> Optional[bytes]:
    """
    TTS生成 + 递归二分法处理内容审核拒绝

    策略：
    1. 尝试生成音频
    2. 如果失败（可能是内容审核），对半切
    3. 递归时使用同步调用，避免占用异步并发池
    4. 拼接返回
    5. 直到文本长度小于配置的最小分割长度时放弃
    """
    if not text.strip():
        return None

    config = get_config()
    min_chars = config.tts.min_chars

    # 文本太短，放弃
    if len(text) < min_chars:
        print(f"  {'  ' * depth}⚠ TTS文本过短({len(text)}字<{min_chars})，跳过")
        return None

    # 尝试生成
    audio = await _generate_speech_impl_async(client, text, voice, emotion)

    # 成功则直接返回
    if audio:
        return audio

    # 失败，尝试递归二分（使用同步调用）
    if depth >= max_depth:
        print(f"  {'  ' * depth}⚠ TTS达到最大递归深度({max_depth})，放弃")
        return None

    print(f"  {'  ' * depth}⚠ TTS生成失败，对半拆分重试...")

    # 对半切分
    upper_half, lower_half = _split_text_for_moderation(text)
    print(f"  {'  ' * depth}  拆分: {len(upper_half)}字 + {len(lower_half)}字")

    # 递归时使用同步客户端，避免占用异步并发池
    sync_client = get_tts_client()
    
    # 顺序生成两半（同步调用）
    upper_audio = _generate_speech_with_moderation_retry_sync(
        sync_client, upper_half, voice, emotion, depth + 1, max_depth
    )
    lower_audio = _generate_speech_with_moderation_retry_sync(
        sync_client, lower_half, voice, emotion, depth + 1, max_depth
    )

    # 合并结果
    audio_parts = []
    if upper_audio:
        audio_parts.append(upper_audio)
    if lower_audio:
        audio_parts.append(lower_audio)

    if not audio_parts:
        return None
    if len(audio_parts) == 1:
        return audio_parts[0]

    # 两半之间加短静音
    silence = create_silence(0.3)
    return _concat_wav_files([audio_parts[0], silence, audio_parts[1]])


def _generate_speech_with_moderation_retry_sync(
    client: OpenAI,
    text: str,
    voice: str,
    emotion: str,
    depth: int = 0,
    max_depth: int = 5
) -> Optional[bytes]:
    """
    TTS生成 + 递归二分法处理内容审核拒绝（同步版本）
    
    用于递归调用，避免占用异步并发池
    """
    if not text.strip():
        return None

    config = get_config()
    min_chars = config.tts.min_chars

    # 文本太短，放弃
    if len(text) < min_chars:
        print(f"  {'  ' * depth}⚠ TTS文本过短({len(text)}字<{min_chars})，跳过")
        return None

    # 尝试生成（同步调用）
    audio = _generate_speech_impl(client, text, voice, emotion)

    # 成功则直接返回
    if audio:
        return audio

    # 失败，尝试递归二分
    if depth >= max_depth:
        print(f"  {'  ' * depth}⚠ TTS达到最大递归深度({max_depth})，放弃")
        return None

    print(f"  {'  ' * depth}⚠ TTS生成失败，对半拆分重试...")

    # 对半切分
    upper_half, lower_half = _split_text_for_moderation(text)
    print(f"  {'  ' * depth}  拆分: {len(upper_half)}字 + {len(lower_half)}字")

    # 顺序生成两半（同步调用）
    upper_audio = _generate_speech_with_moderation_retry_sync(
        client, upper_half, voice, emotion, depth + 1, max_depth
    )
    lower_audio = _generate_speech_with_moderation_retry_sync(
        client, lower_half, voice, emotion, depth + 1, max_depth
    )

    # 合并结果
    audio_parts = []
    if upper_audio:
        audio_parts.append(upper_audio)
    if lower_audio:
        audio_parts.append(lower_audio)

    if not audio_parts:
        return None
    if len(audio_parts) == 1:
        return audio_parts[0]

    # 两半之间加短静音
    silence = create_silence(0.3)
    return _concat_wav_files([audio_parts[0], silence, audio_parts[1]])


def create_silence(duration_sec: float, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """创建静音音频"""
    import wave
    import io
    num_samples = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b'\x00' * (num_samples * sample_width * channels))
    return buf.getvalue()


def _generate_long_text(client: OpenAI, text: str, voice: str, emotion: str) -> Optional[bytes]:
    """生成长文本语音（同步）"""
    config = get_config()
    chunks = _split_for_tts(text, config.tts.max_chars)

    all_audio = []
    for chunk in chunks:
        audio = _generate_speech_impl(client, chunk, voice, emotion)
        if audio:
            all_audio.append(audio)
        else:
            return None

    # 合并音频
    if len(all_audio) == 1:
        return all_audio[0]
    return _concat_wav_files(all_audio)


async def _generate_long_text_async(client: AsyncOpenAI, text: str, voice: str, emotion: str) -> Optional[bytes]:
    """生成长文本语音（异步）"""
    config = get_config()
    chunks = _split_for_tts(text, config.tts.max_chars)

    # 并发生成所有子块
    tasks = [_generate_speech_impl_async(client, chunk, voice, emotion) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_audio = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"    ❌ 子块 {i} 生成失败: {result}")
            return None
        if result is None:
            return None
        all_audio.append(result)

    # 合并音频
    if len(all_audio) == 1:
        return all_audio[0]
    return _concat_wav_files(all_audio)


def _split_for_tts(text: str, max_chars: int) -> List[str]:
    """按句子拆分文本用于TTS"""
    sentences = re.split(r'([。！？；\n])', text)
    chunks = []
    current = ""

    for seg in sentences:
        if len(current + seg) > max_chars and current:
            chunks.append(current)
            current = seg
        else:
            current += seg

    if current:
        chunks.append(current)

    return chunks


def _concat_wav_files(audio_list: List[bytes]) -> bytes:
    """合并多个WAV文件"""
    import wave
    import io

    if not audio_list:
        return b""

    if len(audio_list) == 1:
        return audio_list[0]

    # 读取第一个文件的参数
    with wave.open(io.BytesIO(audio_list[0]), 'rb') as first_wav:
        params = first_wav.getparams()

    # 合并所有音频数据
    output = io.BytesIO()
    with wave.open(output, 'wb') as out_wav:
        out_wav.setparams(params)
        for audio_bytes in audio_list:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                out_wav.writeframes(wav_file.readframes(wav_file.getnframes()))

    return output.getvalue()


async def generate_tts_batch(tasks: List[Dict]) -> List[Dict]:
    """
    批量生成TTS音频（异步，带递归二分法处理审核拒绝）

    Args:
        tasks: 任务列表，每个任务包含 idx, text, voice, emotion, output_file

    Returns:
        结果列表
    """
    config = get_config()
    client = get_async_tts_client()
    semaphore = asyncio.Semaphore(config.concurrency.tts_concurrency)

    async def process_task(task: Dict) -> Dict:
        async with semaphore:
            idx = task["idx"]
            text = task["text"]
            voice = task["voice"]
            emotion = task["emotion"]
            output_file = task["output_file"]

            # 使用带递归二分法的版本
            audio_bytes = await _generate_speech_with_moderation_retry_async(
                client, text, voice, emotion
            )

            if audio_bytes and save_wav(audio_bytes, output_file):
                return {"idx": idx, "success": True, "file": output_file}
            
            # 判断失败原因
            config = get_config()
            if len(text) < config.tts.min_chars:
                error_msg = f"文本过短({len(text)}字<{config.tts.min_chars})"
            else:
                error_msg = f"TTS生成失败(文本: {text[:20]}...)"
            return {"idx": idx, "success": False, "file": output_file, "error": error_msg}

    # 并发执行所有任务
    results = await asyncio.gather(*[process_task(t) for t in tasks], return_exceptions=True)

    # 处理异常结果
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({
                "idx": tasks[i]["idx"],
                "success": False,
                "file": tasks[i]["output_file"],
                "error": str(result)
            })
        else:
            final_results.append(result)

    return final_results


def save_wav(audio_bytes: bytes, output_path: str) -> bool:
    """保存WAV文件"""
    try:
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        return True
    except Exception as e:
        print(f"    ❌ 保存音频失败: {e}")
        return False
