#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试人物映射功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.dialogue_splitter import (
    is_valid_character_name,
    sanitize_character_name,
    clear_global_character_map,
    update_global_character_map,
    get_global_character_map,
    INVALID_CHARACTER_NAMES
)


def test_character_name_validation():
    """测试角色名验证"""
    print("=" * 60)
    print("测试角色名验证")
    print("=" * 60)

    # 有效的角色名
    valid_names = ["小明", "妈妈", "老师", "张三", "李四", "旁白", "小雅", "小雅妈妈"]
    for name in valid_names:
        result = is_valid_character_name(name)
        status = "✅" if result else "❌"
        print(f"  {status} '{name}' -> {result}")

    print()

    # 无效的角色名
    invalid_names = [
        "他", "她", "我", "你", "我们", "他们",
        "些调皮地", "没好气地", "开心地", "生气地",
        "说", "道", "喊", "问",
        "", None
    ]
    for name in invalid_names:
        result = is_valid_character_name(name)
        status = "✅" if not result else "❌"
        print(f"  {status} '{name}' -> {result}")


def test_character_name_sanitization():
    """测试角色名清理"""
    print("\n" + "=" * 60)
    print("测试角色名清理")
    print("=" * 60)

    test_cases = [
        ("小明", "小明"),
        ("妈妈", "妈妈"),
        ("旁白", "旁白"),
        ("他", "旁白"),
        ("她", "旁白"),
        ("我", "旁白"),
        ("些调皮地", "旁白"),
        ("没好气地", "旁白"),
        ("说", "旁白"),
        ("", "旁白"),
        (None, "旁白"),
    ]

    for input_name, expected in test_cases:
        result = sanitize_character_name(input_name)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{input_name}' -> '{result}' (期望: '{expected}')")


def test_global_character_map():
    """测试全局人物映射表"""
    print("\n" + "=" * 60)
    print("测试全局人物映射表")
    print("=" * 60)

    # 清空映射表
    clear_global_character_map()
    print("  清空映射表:", get_global_character_map())

    # 更新映射表
    new_map = {
        "小明": "冰糖",
        "妈妈": "茉莉",
        "爸爸": "苏打",
    }
    update_global_character_map(new_map)
    print("  更新后:", get_global_character_map())

    # 再次更新（应该合并）
    another_map = {
        "老师": "白桦",
        "小明": "冰糖",  # 已存在，应该保持
    }
    update_global_character_map(another_map)
    print("  再次更新后:", get_global_character_map())


def test_invalid_character_names_set():
    """测试无效角色名集合"""
    print("\n" + "=" * 60)
    print("无效角色名集合")
    print("=" * 60)

    print(f"  共 {len(INVALID_CHARACTER_NAMES)} 个无效角色名:")
    for i, name in enumerate(sorted(INVALID_CHARACTER_NAMES), 1):
        print(f"    {i:2d}. {name}")


if __name__ == "__main__":
    test_character_name_validation()
    test_character_name_sanitization()
    test_global_character_map()
    test_invalid_character_names_set()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
