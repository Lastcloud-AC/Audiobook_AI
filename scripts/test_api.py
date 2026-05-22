#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API测试脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import get_config
from modules.dialogue_splitter import get_llm_client, get_async_llm_client
from modules.tts_generator import get_tts_client, get_async_tts_client


def test_llm_sync():
    """测试LLM同步API"""
    print("\n🔍 测试LLM同步API...")
    config = get_config()

    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=config.llm_api.model,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10
        )

        print("✅ LLM同步API测试成功")
        print(f"   模型: {config.llm_api.model}")
        print(f"   响应: {response.choices[0].message.content}")

        return True

    except Exception as e:
        print(f"❌ LLM同步API测试失败: {e}")
        return False


def test_tts_sync():
    """测试TTS同步API"""
    print("\n🔍 测试TTS同步API...")
    config = get_config()

    try:
        client = get_tts_client()
        response = client.chat.completions.create(
            model=config.tts_api.model,
            messages=[
                {"role": "user", "content": "用平静的语气朗读"},
                {"role": "assistant", "content": "你好，这是一个测试。"}
            ],
            audio={"format": "wav", "voice": "冰糖"}
        )

        print("✅ TTS同步API测试成功")
        print(f"   模型: {config.tts_api.model}")

        return True

    except Exception as e:
        print(f"❌ TTS同步API测试失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  有声书生成器 - API测试")
    print("="*60)

    config = get_config()

    print(f"\n当前配置:")
    print(f"  LLM API:")
    print(f"    URL: {config.llm_api.base_url}")
    print(f"    密钥: {'*' * 8 if config.llm_api.api_key else '未设置'}")
    print(f"    模型: {config.llm_api.model}")
    print(f"  TTS API:")
    print(f"    URL: {config.tts_api.base_url}")
    print(f"    密钥: {'*' * 8 if config.tts_api.api_key else '未设置'}")
    print(f"    模型: {config.tts_api.model}")

    # 检查密钥
    if not config.llm_api.api_key and not config.tts_api.api_key:
        print("\n❌ 未设置任何API密钥，请先运行 configure.py 设置")
        return

    results = {}

    # 测试LLM API
    if config.llm_api.api_key:
        results["llm"] = test_llm_sync()
    else:
        print("\n⚠️ LLM API密钥未设置，跳过测试")
        results["llm"] = None

    # 测试TTS API
    if config.tts_api.api_key:
        results["tts"] = test_tts_sync()
    else:
        print("\n⚠️ TTS API密钥未设置，跳过测试")
        results["tts"] = None

    # 总结
    print("\n" + "="*60)
    print("  测试结果")
    print("="*60)

    if results["llm"] is not None:
        print(f"  LLM API: {'✅ 正常' if results['llm'] else '❌ 失败'}")
    else:
        print(f"  LLM API: ⏭ 跳过")

    if results["tts"] is not None:
        print(f"  TTS API: {'✅ 正常' if results['tts'] else '❌ 失败'}")
    else:
        print(f"  TTS API: ⏭ 跳过")

    # 判断是否可以开始生成
    can_generate = True
    if results["llm"] is False:
        can_generate = False
    if results["tts"] is False:
        can_generate = False

    if can_generate:
        print("\n🎉 测试通过！可以开始生成有声书了。")
    else:
        print("\n⚠️ 存在测试失败，请检查配置和网络连接。")


if __name__ == "__main__":
    main()
