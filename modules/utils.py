"""
工具函数模块
"""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any


def safe_filename(name: str) -> str:
    """
    生成安全的文件名

    Args:
        name: 原始名称

    Returns:
        安全的文件名
    """
    # 替换不安全的字符
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        name = name.replace(char, '_')
    # 限制长度
    if len(name) > 100:
        name = name[:100]
    return name.strip()


def calculate_hash(data: Any) -> str:
    """
    计算数据的MD5哈希值

    Args:
        data: 要计算哈希的数据

    Returns:
        MD5哈希字符串
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, (list, dict)):
        data = json.dumps(data, ensure_ascii=False).encode('utf-8')

    return hashlib.md5(data).hexdigest()


def format_duration(seconds: float) -> str:
    """
    格式化时长

    Args:
        seconds: 秒数

    Returns:
        格式化的时长字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}时{minutes}分"


def format_size(bytes_size: int) -> str:
    """
    格式化文件大小

    Args:
        bytes_size: 字节数

    Returns:
        格式化的大小字符串
    """
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f}GB"


def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Path, pattern: str = "*") -> List[Path]:
    """
    列出目录下的文件

    Args:
        directory: 目录路径
        pattern: 文件匹配模式

    Returns:
        文件路径列表
    """
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def read_json(filepath: Path) -> Dict:
    """读取JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(filepath: Path, data: Dict):
    """写入JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
