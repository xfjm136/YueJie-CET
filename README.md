# 阅阶 CET

基于 AI 的四六级 TUI 刷题工具，当前已覆盖阅读、写作、翻译三大方向，并以 Rust TUI + Python AI 服务的混合架构实现真题风格训练。

## 当前能力

- 四级 / 六级切换
- 阅读 4 题型
  - `Section A` 选词填空
  - `Section B` 长篇阅读
  - `Section C` 仔细阅读 1
  - `Section D` 仔细阅读 2
- 主观题 2 题型
  - `Part I` 写作
  - `Part IV` 翻译
- DeepSeek 真实出题
  - 不再使用本地题目兜底
  - 未配置 `DEEPSEEK_API_KEY` 时会直接拒绝生成 / 评阅
- 主观题 AI 评阅
  - CET 风格分数
  - 错词纠正
  - 病句改写
  - 高分范文 / 高分译文
  - 语法与表达问题自动纳入薄弱项
- 历史记录、词汇表、薄弱项闭环
  - 删除历史后会重算统计、词汇和薄弱项
- 键盘 + 鼠标混合作答
- 深浅主题、配色方案、背景模式切换

## 界面说明

- Rust Ratatui 前端为默认主界面。
- 阅读题使用偏试卷式布局：
  - 左侧为计时器与菜单
  - 中间为文章
  - 右侧为题目与答题区
- 写作 / 翻译使用两列布局：
  - 左侧为计时器与菜单
  - 右侧上方为题面
  - 右侧下方为答题卡
- 生成题目、评分、复盘均带有等待动画和阶段状态提示。
- 结果页先展示居中分数 / 正确率，再进入完整复盘页查看解析细节。

## 技术架构

- 前端：Rust + `ratatui` + `crossterm`
- AI / 业务桥接：Python 3.12+
- 数据库：SQLite
- 模型：DeepSeek，默认 `deepseek-v4-pro`

数据流大致如下：

1. Rust TUI 负责交互、动画、布局和本地事件处理。
2. Rust 通过 `app/bridge.py` 与 Python 层交换 JSON。
3. Python 负责：
   - 调用 DeepSeek 出题
   - 严格校验 JSON 结构与 CET 约束
   - 评分与生成主观题反馈
   - 落库存储题目、记录、词汇和薄弱项

## 快速开始

### 1. 环境要求

- Python `3.12+`
- Rust 稳定版工具链
- 可访问 DeepSeek API 的网络环境

### 2. 安装

```bash
cd /data/YueJie-CET
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
cargo build --release
```

### 3. 配置 `.env`

`.env.example`：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
YUEJIE_DB_PATH=
YUEJIE_REQUEST_TIMEOUT=120
```

说明：

- `DEEPSEEK_API_KEY`
  - 必填，不填无法出题或评阅
- `DEEPSEEK_BASE_URL`
  - 默认为官方地址
- `DEEPSEEK_MODEL`
  - 默认 `deepseek-v4-pro`
- `YUEJIE_DB_PATH`
  - 留空时默认使用 `data/yuejie.db`
- `YUEJIE_REQUEST_TIMEOUT`
  - DeepSeek 请求超时秒数

### 4. 启动

```bash
source .venv/bin/activate
yuejie-cet
```

启动逻辑：

- 如果已经编译过 Rust 前端，`yuejie-cet` 会自动在
  - `target/release/yuejie-cet-rs`
  - `target/debug/yuejie-cet-rs`
  中选择最近一次编译的可执行文件启动
- 如果本地没有可执行的 Rust 二进制，则回退到旧的 Python Textual 界面

## 使用流程

1. 进入首页查看总刷题数据、最近正确率、平均用时等信息。
2. 选择四级或六级。
3. 选择题型：
   - 阅读 4 题型
   - 写作
   - 翻译
4. 等待 AI 生成题目。
5. 在 TUI 中作答：
   - 阅读题可用键盘或鼠标选择
   - 写作 / 翻译可直接在答题区输入文本
6. 提交后查看结果页。
7. 进入复盘页查看：
   - 客观题解析与技巧
   - 主观题分项评分
   - 错词纠正
   - 病句改写
   - 高分范文 / 译文
   - 词汇整理
8. 在历史中可重新作答或删除记录。

## 开发与验证

### 运行测试

```bash
cargo test
python -m unittest discover -s tests -v
```

### 构建发布版前端

```bash
cargo build --release
```

### 开发者调试桥接层

真实出题流式调试：

```bash
python -u -m app.bridge generate-live --level cet4 --question-type writing
python -u -m app.bridge generate-live --level cet4 --question-type translation
```

提交评分调试：

```bash
python -m app.bridge submit
```

`submit` 会从标准输入读取 JSON，格式大致为：

```json
{
  "question_set_id": "qs_xxx",
  "started_at": "2026-06-10T17:10:00+08:00",
  "answers": {
    "response_text": "your answer here"
  }
}
```

## 目录结构

```text
src/
  main.rs        Rust TUI 主前端
app/
  main.py        启动入口，优先拉起 Rust 前端
  bridge.py      Rust <-> Python JSON CLI bridge
  ai/            DeepSeek 客户端、Prompt、生成与评阅流水线
  config.py      .env 加载与运行配置
  data/          SQLite 初始化与查询
  domain/        枚举、数据模型、评分结构
  services/      出题、作答、统计、薄弱项服务
  ui/            旧的 Python Textual UI
tests/           单元测试与回归测试
data/            默认数据库目录
```

## 当前实现状态

已完成：

- 写作 / 翻译主观题正式接入
- 主观题两列作答界面
- 主观题真实 AI 评分
- 错词、病句、高分版本入复盘页
- 错词纳入词汇表
- 主观题薄弱项纳入动态统计
- 历史删除后的派生数据重建

当前约束：

- 题目生成与主观题评阅都依赖 DeepSeek，可用性受网络与 API 状态影响
- Python Textual 界面仍保留在仓库中，但默认体验目标是 Rust 前端
