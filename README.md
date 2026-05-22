# 有声书生成器 - 模块化版本

## 项目结构

```
audiobook/
├── modules/                    # 核心模块
│   ├── __init__.py
│   ├── config_manager.py      # 配置管理
│   ├── novel_reader.py        # 小说读取与章节分割
│   ├── dialogue_splitter.py   # LLM对话分割
│   ├── voice_assigner.py      # 角色音色分配
│   ├── tts_generator.py       # TTS音频生成
│   ├── audio_processor.py     # 音频处理与合并
│   └── utils.py               # 工具函数
├── scripts/                    # 脚本工具
│   ├── generate.py            # 主生成脚本
│   ├── configure.py           # 配置管理脚本
│   └── test_api.py            # API测试脚本
├── config/                     # 配置文件
│   ├── settings.json          # 主配置文件（含密钥，不提交）
│   └── settings.example.json  # 配置示例（提交到Git）
├── input/                      # 输入文件
├── output/                     # 输出目录
├── llm_raw_responses/          # LLM原始响应（按书名组织）
├── .gitignore                  # Git忽略文件
├── README.md                   # 项目说明
└── audiobook_generator.py      # 原始单文件版本（保留）
```

## 快速开始

### 1. 配置API

从示例文件创建配置：

```bash
cp config/settings.example.json config/settings.json
```

编辑 `config/settings.json`，填入你的API配置：

```json
{
  "llm_api": {
    "base_url": "https://api.mimoai.com/v1",
    "api_key": "你的LLM API密钥",
    "model": "mimo-v2.5-pro"
  },
  "tts_api": {
    "base_url": "https://api.mimoai.com/v1",
    "api_key": "你的TTS API密钥",
    "model": "mimo-v2.5-pro"
  }
}
```

> **注意**：LLM和TTS的API配置是分开的，可以使用不同的服务商和密钥。如果你使用同一个服务商，可以填相同的URL和密钥。

### 2. 测试API连接

```bash
python scripts/test_api.py
```

### 3. 生成有声书

```bash
# 生成全部章节
python scripts/generate.py input/测试短文.txt

# 只生成前2章
python scripts/generate.py input/测试短文.txt -c 2

# 自定义chunk大小
python scripts/generate.py input/测试短文.txt --chunk-size 2000

# 不跳过已生成的章节
python scripts/generate.py input/测试短文.txt --no-skip
```

## 配置说明

### 核心配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `generation.chunk_size` | LLM分割的chunk大小（字符数） | 1500 |
| `generation.max_duration_per_file` | 单个音频文件最大时长（秒） | 900 (15分钟) |
| `generation.silence_between_segments` | 段落间静音时长（秒） | 0.5 |
| `concurrency.llm_concurrency` | LLM并发数 | 3 |
| `concurrency.tts_concurrency` | TTS并发数 | 5 |
| `rate_limit.llm_rpm` | LLM每分钟请求数限制 | 30 |
| `rate_limit.tts_rpm` | TTS每分钟请求数限制 | 60 |

### 音色配置

```json
{
  "voice_presets": {
    "冰糖": "ice_candy",
    "苏打": "soda",
    "default": "ice_candy"
  }
}
```

## 人物映射功能

### 问题背景
由于小说会被分割成多个chunk分别处理，可能会出现以下问题：
1. 同一个人物在不同chunk中被分配不同的音色
2. LLM错误地将描述性词语（如"些调皮地"、"没好气地"）当作角色名
3. 代词（如"他"、"我"）被当作角色处理

### 解决方案

#### 1. 全局人物映射表
- 第一次分割后自动建立人物映射表
- 后续分割时传入已知映射，保持一致性
- 映射表在生成新书时自动清空

#### 2. 角色名验证
系统会自动过滤以下无效角色名：
- **代词**：他、她、它、我、你、我们、他们
- **描述性词语**：些调皮地、没好气地、开心地、生气地
- **动作描述**：说、道、喊、问、笑、哭

#### 3. 改进的提示词
LLM提示词中明确说明了角色识别规则，减少错误识别。

### 查看人物映射
生成完成后，系统会打印全局人物映射表：
```
🎭 全局人物映射表:
   小明 -> 冰糖
   妈妈 -> 茉莉
   爸爸 -> 苏打
   老师 -> 白桦
```

### 测试人物映射
```bash
python scripts/test_character_map.py
```

## 输出结构

```
output/
└── {书名}_有声书/
    ├── 001_章节标题.wav          # 章节音频（按时长切分）
    ├── 001_章节标题_part02.wav   # 超过15分钟会自动切分
    ├── 001_章节标题_script.json  # 对话脚本（用于字幕）
    ├── 002_章节标题.wav
    └── ...

llm_raw_responses/
└── {书名}/
    ├── chapter_001/
    │   ├── chunk_000.json              # LLM原始响应（JSON格式）
    │   ├── chunk_000_readable.txt      # 可读文本（方便对照检查）
    │   └── chunk_001.json
    └── chapter_002/
        └── ...
```

### 可读文本格式

`*_readable.txt` 文件提供清晰的对照格式：

```
=== 第一章 相遇_chunk000 ===
解析时间: 2026-05-22 12:29:30

共 11 段:

  1. [旁白] 小明走在放学的路上，看到妈妈在门口等他。
  2. [旁白] 小明开心地喊道
     情绪: happy
  3. [小明] 妈妈，我回来了！
     情绪: happy
  4. [妈妈] 今天在学校怎么样？
     情绪: happy
  ...
```

**用途**：
- 快速检查LLM分割是否正确
- 对照原文查看是否有遗漏
- 验证角色识别是否准确

## 模块说明

### config_manager.py
- 统一配置管理
- 支持JSON配置文件
- 提供默认配置和用户配置合并

### novel_reader.py
- 支持 `.txt`, `.docx`, `.doc` 格式
- 自动识别章节标题
- 返回结构化的章节数据

### dialogue_splitter.py
- 使用LLM分割对话和旁白
- 支持异步并发处理（Semaphore控制并发数）
- 递归二分法处理内容审核拒绝（异步版本，不阻塞事件循环）
- 自动验证覆盖率（最低70%）
- 失败时自动降级到简单分割

### voice_assigner.py
- 自动识别角色
- 分配音色ID
- 支持自定义角色音色映射
- **全局人物映射表**：保持跨章节角色一致性

### tts_generator.py
- 异步批量生成TTS音频
- 自动处理长文本分割
- 速率限制和错误重试

### audio_processor.py
- WAV文件合并
- 按时长自动切分
- 静音插入

## 与原版本对比

| 特性 | 原版本 | 模块化版本 |
|------|--------|------------|
| 代码组织 | 单文件1711行 | 7个独立模块 |
| 配置管理 | 硬编码 | JSON配置文件 |
| 输出组织 | 按时间戳 | 按书名/章节 |
| 章节处理 | 全文处理 | 逐章处理 |
| 时长控制 | 无限制 | 自动切分（默认15分钟） |
| LLM响应 | 不保存 | 按书名/章节保存 |
| 可扩展性 | 差 | 良好 |

## 故障排除

### asyncio.Lock() 错误
已修复。使用延迟创建Lock的方案，确保绑定到正确的事件循环。

### chunk并发处理被阻塞（2026-05-22修复）
**问题**：多个chunk没有并发处理，而是顺序执行。

**原因**：`_handle_moderation_split_async` 函数在遇到内容审核拒绝时，调用了同步版本的 `_recursive_split_by_moderation`，阻塞了整个事件循环。

**修复**：改为调用异步版本 `_recursive_split_by_moderation_async`，使用 `await` 等待结果。

**效果**：chunk遇到审核拒绝时不再阻塞其他chunk的并发处理。

### API连接失败
1. 检查 `config/settings.json` 中的API配置
2. 运行 `python scripts/test_api.py` 测试连接
3. 检查网络连接和API密钥

### 覆盖率过低
如果LLM分割覆盖率低于70%，系统会自动降级到简单分割。可以尝试：
1. 减小 `chunk_size` 配置
2. 检查原文格式是否规范

## 开发说明

### 添加新模块
1. 在 `modules/` 目录创建新文件
2. 在 `modules/__init__.py` 中导出
3. 在需要的地方导入使用

### 修改配置
使用 `scripts/configure.py` 或直接编辑 `config/settings.json`

## 许可证

MIT License
