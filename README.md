# 阅阶 CET

基于 AI 的四六级阅读 TUI 刷题工具。

当前架构：

- Rust + Ratatui 前端
- Python 3.12+ AI / 数据桥接层
- SQLite
- DeepSeek API

## 功能范围

- 四级 / 六级切换
- 阅读 4 题型入口
- TUI 多页面导航
- 本地 SQLite 持久化
- AI 出题接口层
- 历史记录、词汇表、薄弱项基础闭环

## 快速开始

```bash
cd /data/YueJie-CET
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
cargo build --release
yuejie-cet
```

默认模型使用 `deepseek-v4-pro`。如果没有配置 `DEEPSEEK_API_KEY`，程序会自动退回到内置示例题，方便先联调界面和流程。`yuejie-cet` 会优先启动 `target/release/yuejie-cet-rs`，找不到时再回退到 Python 版界面。

## 目录结构

```text
src/
  main.rs      Rust TUI 前端
app/
  bridge.py    Rust <-> Python JSON CLI bridge
  ai/          DeepSeek 客户端、Prompt 和出题流水线
  data/        SQLite 初始化与查询
  domain/      枚举、数据模型、评分逻辑
  services/    业务服务
  ui/          旧的 Python TUI 兜底实现
tests/         基础单元测试
```
