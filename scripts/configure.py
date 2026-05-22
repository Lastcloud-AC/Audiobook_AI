#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置脚本
用于设置API密钥和其他配置项
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import get_config, ConfigManager


def main():
    """主函数"""
    config = get_config()

    print("\n" + "="*60)
    print("  有声书生成器 - 配置工具")
    print("="*60)

    while True:
        print("\n当前配置:")
        print(f"  --- LLM API（对话分割）---")
        print(f"  1. LLM API URL: {config.llm_api.base_url}")
        print(f"  2. LLM API密钥: {'*' * 8 if config.llm_api.api_key else '未设置'}")
        print(f"  3. LLM模型: {config.llm_api.model}")
        print(f"  --- TTS API（语音合成）---")
        print(f"  4. TTS API URL: {config.tts_api.base_url}")
        print(f"  5. TTS API密钥: {'*' * 8 if config.tts_api.api_key else '未设置'}")
        print(f"  6. TTS模型: {config.tts_api.model}")
        print(f"  --- 生成参数 ---")
        print(f"  7. Chunk大小: {config.generation.chunk_size}字")
        print(f"  8. 最大音频时长: {config.generation.max_duration_per_file}秒")
        print(f"  9. LLM并发数: {config.concurrency.llm_concurrency}")
        print(f"  10. TTS并发数: {config.concurrency.tts_concurrency}")
        print(f"  0. 退出")

        choice = input("\n请选择要修改的配置项 (0-10): ").strip()

        if choice == "0":
            break
        elif choice == "1":
            new_url = input(f"请输入LLM API URL [{config.llm_api.base_url}]: ").strip()
            if new_url:
                config.set("llm_api.base_url", new_url)
                config.save_config()
                print("✅ 已更新")
        elif choice == "2":
            new_key = input("请输入LLM API密钥: ").strip()
            if new_key:
                config.update_llm_api_key(new_key)
                print("✅ 已更新")
        elif choice == "3":
            new_model = input(f"请输入LLM模型 [{config.llm_api.model}]: ").strip()
            if new_model:
                config.set("llm_api.model", new_model)
                config.save_config()
                print("✅ 已更新")
        elif choice == "4":
            new_url = input(f"请输入TTS API URL [{config.tts_api.base_url}]: ").strip()
            if new_url:
                config.set("tts_api.base_url", new_url)
                config.save_config()
                print("✅ 已更新")
        elif choice == "5":
            new_key = input("请输入TTS API密钥: ").strip()
            if new_key:
                config.update_tts_api_key(new_key)
                print("✅ 已更新")
        elif choice == "6":
            new_model = input(f"请输入TTS模型 [{config.tts_api.model}]: ").strip()
            if new_model:
                config.set("tts_api.model", new_model)
                config.save_config()
                print("✅ 已更新")
        elif choice == "7":
            new_size = input(f"请输入新的Chunk大小 [{config.generation.chunk_size}]: ").strip()
            if new_size.isdigit():
                config.set("generation.chunk_size", int(new_size))
                config.save_config()
                print("✅ 已更新")
        elif choice == "8":
            new_duration = input(f"请输入新的最大音频时长（秒）[{config.generation.max_duration_per_file}]: ").strip()
            if new_duration.isdigit():
                config.set("generation.max_duration_per_file", int(new_duration))
                config.save_config()
                print("✅ 已更新")
        elif choice == "9":
            new_concurrency = input(f"请输入新的LLM并发数 [{config.concurrency.llm_concurrency}]: ").strip()
            if new_concurrency.isdigit():
                config.set("concurrency.llm_concurrency", int(new_concurrency))
                config.save_config()
                print("✅ 已更新")
        elif choice == "10":
            new_concurrency = input(f"请输入新的TTS并发数 [{config.concurrency.tts_concurrency}]: ").strip()
            if new_concurrency.isdigit():
                config.set("concurrency.tts_concurrency", int(new_concurrency))
                config.save_config()
                print("✅ 已更新")
        else:
            print("❌ 无效的选择")

    print("\n👋 再见！")


if __name__ == "__main__":
    main()
