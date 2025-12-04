# astrbot_plugin_smart_translator

专注文本翻译的 AstrBot 插件，最小化上下文干扰：仅把用户输入和极短 system prompt 丢给 LLM，实现"文本进、译文出"。

## 功能亮点

- **50+ 语言支持**：中/英/日/韩/俄/法/德/西/意/葡等常见语言，自动检测源语言。
- **自然语言触发**：无需命令前缀，直接说"翻译成日语：xxx"或"xxx 翻译成英文"即可。
- **二次翻译**：首次翻译后，回复该消息并说"再翻译成德语"，插件会使用缓存的**原文**进行翻译，避免翻译链质量损失。
- **上下文隔离**：仅使用翻译所需的文本，完全与聊天历史解耦，避免上下文污染。
- **Provider 灵活配置**：直接复用 AstrBot 已连接的在线/本地 LLM，支持主备切换。
- **多种输出格式**：plain（纯译文）、tagged（带语言标签）、bilingual（原文+译文对照）。

## 触发方式

### 首次翻译

| 格式 | 示例 |
|------|------|
| `翻译成<语言>: <文本>` | `翻译成日语: Hello world` |
| `<语言>: <文本>` | `英文: 今天天气不错` |
| `<文本> 翻译成<语言>` | `这是一个测试 翻译成韩语` |

### 二次翻译（引用回复）

1. 发送首次翻译请求，如：`今天心情很好 翻译成英文`
2. 收到翻译结果后，**回复该消息**并输入：
   - `翻译成日语` 或 `再翻译成法语`
3. 插件会使用**缓存的原文**（"今天心情很好"）进行翻译，而非翻译后的英文

> 缓存有效期可在配置中调整（默认 15 分钟），过期后需重新发送原文。

## 支持的语言

支持 50+ 种语言及其别名，包括但不限于：

| 语言 | 别名示例 |
|------|----------|
| 中文 | `中文`、`中`、`汉语`、`zh`、`chinese` |
| 英文 | `英文`、`英语`、`en`、`english` |
| 日语 | `日语`、`日文`、`ja`、`japanese` |
| 韩语 | `韩语`、`韩文`、`ko`、`korean` |
| 法语 | `法语`、`法文`、`fr`、`french` |
| 德语 | `德语`、`德文`、`de`、`german` |
| 俄语 | `俄语`、`俄文`、`ru`、`russian` |
| 西班牙语 | `西班牙语`、`es`、`spanish` |
| 意大利语 | `意大利语`、`it`、`italian` |
| 拉丁语 | `拉丁语`、`la`、`latin` |
| 阿拉伯语 | `阿拉伯语`、`ar`、`arabic` |

## 配置项

插件在根目录提供 `_conf_schema.json`，AstrBot 会据此自动渲染配置 UI。

### `api_settings` - LLM 提供商设置

| 配置项 | 说明 |
|--------|------|
| `provider_id` | 绑定 AstrBot 中已连接的 Provider（必填） |
| `fallback_provider_id` | 主 Provider 失败时的备用 Provider |
| `provider_is_local` | 所选 Provider 是否为本地 LLM |

### `interaction_settings` - 交互与预设

| 配置项 | 说明 | 默认值 | 范围 |
|--------|------|--------|------|
| `cache_ttl_minutes` | 二次翻译缓存时间（分钟） | 15 | 5-1440 |
| `default_target_lang` | 默认目标语言 | zh | - |
| `system_prompt` | LLM 系统提示词 | （内置） | - |

### `formatter_settings` - 输出格式

| 配置项 | 说明 | 可选值 |
|--------|------|--------|
| `output_mode` | 响应模式 | `plain` / `tagged` / `bilingual` |

- `plain`：仅输出译文
- `tagged`：输出 `[src->tgt] 译文` 格式
- `bilingual`：输出 `原文：xxx / 译文：xxx` 对照格式

### `logging_settings` - 日志与调试

| 配置项 | 说明 |
|--------|------|
| `log_level` | 日志级别：`error` / `warning` / `info` / `debug` |
| `show_api_exchange` | 是否打印 LLM 请求/响应预览 |

## 技术实现

- **缓存机制**：首次翻译后缓存原文（而非译文），支持多次二次翻译都从原文出发
- **语言检测**：基于字符统计自动识别中/日/韩/俄/英
- **回退策略**：主 Provider 失败时自动切换到配置的备用 Provider

## 文件结构

```
├── main.py           # 插件主体
├── _conf_schema.json # 配置 schema
├── metadata.yaml     # 插件元数据
└── README.md
```

## 参考

- [astrbot_plugin_gemini_image_generation](https://github.com/piexian/astrbot_plugin_gemini_image_generation) - 插件结构参考
- [astrbot_plugin_translate](https://github.com/xu-wish/astrbot_plugin_translate) - 翻译功能参考
- [Claude](https://claude.ai) - 代码开发辅助
