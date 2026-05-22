"""
音色分配模块
负责为角色分配音色
"""

import re
from typing import List, Dict
from collections import Counter

from .config_manager import get_config
from .dialogue_splitter import DialogueLine, update_global_character_map, get_global_character_map


def assign_voices(lines: List[DialogueLine]) -> Dict[str, str]:
    """
    为角色分配音色

    Args:
        lines: 对话行列表

    Returns:
        角色名 -> 音色名的映射
    """
    config = get_config()

    # 统计角色出现次数
    character_counts = Counter(line.character for line in lines)

    # 按出现次数排序
    sorted_characters = [char for char, _ in character_counts.most_common()]

    # 获取可用音色
    voice_presets = config.config.get("voice_presets", {})
    available_voices = list(voice_presets.keys())
    available_voices.remove("default")  # 移除默认音色

    # 获取已有的全局人物映射
    existing_map = get_global_character_map()

    # 角色音色映射
    voice_map = {}

    # 已使用的音色集合（用于避免重复分配）
    used_voices = set()

    # 首先应用已有的全局映射（保持一致性）
    for char in sorted_characters:
        if char in existing_map:
            voice_map[char] = existing_map[char]
            used_voices.add(existing_map[char])

    # 然后应用用户配置的角色音色
    character_voices = config.config.get("character_voices", {})
    for char, voice in character_voices.items():
        if char in sorted_characters and char not in voice_map:
            voice_value = voice_presets.get(voice, voice_presets.get("default", "冰糖"))
            voice_map[char] = voice_value
            used_voices.add(voice_value)

    # 为未分配的角色分配音色（跳过已使用的音色）
    voice_index = 0
    for char in sorted_characters:
        if char not in voice_map:
            # 找到下一个未使用的音色
            assigned = False
            for i in range(len(available_voices)):
                candidate_index = (voice_index + i) % len(available_voices)
                voice_name = available_voices[candidate_index]
                voice_value = voice_presets.get(voice_name, voice_presets.get("default", "冰糖"))

                # 如果音色未被使用，则分配
                if voice_value not in used_voices:
                    voice_map[char] = voice_value
                    used_voices.add(voice_value)
                    voice_index = candidate_index + 1
                    assigned = True
                    break

            # 如果所有音色都已使用，则循环使用（允许重复）
            if not assigned:
                voice_name = available_voices[voice_index % len(available_voices)]
                voice_value = voice_presets.get(voice_name, voice_presets.get("default", "冰糖"))
                voice_map[char] = voice_value
                voice_index += 1

    # 更新全局人物映射表
    update_global_character_map(voice_map)

    # 打印音色分配结果
    print(f"    🎭 角色音色分配:")
    for char, voice in voice_map.items():
        count = character_counts[char]
        # 标记是否是新角色
        is_new = char not in existing_map
        marker = " (新)" if is_new else ""
        print(f"      {char} ({count}段) -> {voice}{marker}")

    return voice_map


def detect_characters(text: str) -> List[str]:
    """
    从文本中检测角色名

    Args:
        text: 小说文本

    Returns:
        角色名列表
    """
    # 常见的角色名模式
    patterns = [
        r'["「](.*?)["」]\s*(\w+)\s*(?:说|道|喊|问|答|叫)',  # "xxx" 角色说
        r'(\w+)\s*(?:说|道|喊|问|答|叫)',  # 角色说
    ]

    characters = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # 取最后一个匹配组作为角色名
            char_name = match[-1] if isinstance(match, tuple) else match
            # 过滤掉常见动词和代词
            if char_name and len(char_name) <= 4 and char_name not in ['他', '她', '它', '我', '你', '说', '道', '喊', '问', '答']:
                characters.add(char_name)

    return list(characters)
