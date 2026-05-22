"""
音频处理模块
负责音频文件的合并、切分和处理
"""

import os
import wave
import struct
import io
from pathlib import Path
from typing import List, Tuple

from .config_manager import get_config


def get_wav_duration(wav_path: str) -> float:
    """
    获取WAV文件时长（秒）

    Args:
        wav_path: WAV文件路径

    Returns:
        时长（秒）
    """
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / rate
    except Exception:
        return 0.0


def create_silence(duration_sec: float, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """
    创建静音音频

    Args:
        duration_sec: 静音时长（秒）
        sample_rate: 采样率
        channels: 声道数
        sample_width: 采样位宽（字节）

    Returns:
        WAV格式的静音音频字节
    """
    num_frames = int(duration_sec * sample_rate)
    num_samples = num_frames * channels

    # 创建WAV格式的静音
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00' * (num_samples * sample_width))

    return output.getvalue()


def merge_wav_files(wav_files: List[str], output_path: str, silence_sec: float = 0.5) -> bool:
    """
    合并多个WAV文件

    Args:
        wav_files: WAV文件路径列表
        output_path: 输出文件路径
        silence_sec: 文件间静音时长（秒）

    Returns:
        是否成功
    """
    if not wav_files:
        return False

    try:
        # 读取第一个文件的参数
        with wave.open(wav_files[0], 'rb') as first_wav:
            params = first_wav.getparams()
            sample_rate = params.framerate
            channels = params.nchannels
            sample_width = params.sampwidth

        # 创建静音
        silence = create_silence(silence_sec, sample_rate, channels, sample_width)

        # 合并所有文件
        output = io.BytesIO()
        with wave.open(output, 'wb') as out_wav:
            out_wav.setparams(params)

            for i, wav_file in enumerate(wav_files):
                with wave.open(wav_file, 'rb') as wav:
                    out_wav.writeframes(wav.readframes(wav.getnframes()))

                # 添加静音（除了最后一个文件）
                if i < len(wav_files) - 1:
                    out_wav.writeframes(silence)

        # 保存到文件
        with open(output_path, 'wb') as f:
            f.write(output.getvalue())

        return True

    except Exception as e:
        print(f"    ❌ 合并音频失败: {e}")
        return False


def split_audio_by_duration(wav_files: List[str], max_duration: float) -> List[List[str]]:
    """
    按时长切分音频文件列表

    Args:
        wav_files: WAV文件路径列表
        max_duration: 最大时长（秒）

    Returns:
        分组后的文件列表
    """
    config = get_config()
    silence_sec = config.generation.silence_between_segments

    batches = []
    current_batch = []
    current_duration = 0.0

    for wav_file in wav_files:
        duration = get_wav_duration(wav_file)

        # 如果加上这个文件会超限，且当前批次不为空，则切分
        if current_duration + duration > max_duration and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_duration = 0.0

        current_batch.append(wav_file)
        current_duration += duration + silence_sec

    # 添加最后一个批次
    if current_batch:
        batches.append(current_batch)

    return batches


def format_duration(seconds: float) -> str:
    """
    格式化时长显示

    Args:
        seconds: 时长（秒）

    Returns:
        格式化的时长字符串，如 "3:45" 或 "1:02:30"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def get_audio_info(wav_path: str) -> dict:
    """
    获取音频文件信息

    Args:
        wav_path: WAV文件路径

    Returns:
        音频信息字典
    """
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            return {
                "channels": wav_file.getnchannels(),
                "sample_width": wav_file.getsampwidth(),
                "sample_rate": wav_file.getframerate(),
                "frames": wav_file.getnframes(),
                "duration": wav_file.getnframes() / wav_file.getframerate()
            }
    except Exception as e:
        return {"error": str(e)}
