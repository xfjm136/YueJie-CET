# 阅阶 CET

<p align="center">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-TUI-%23ce422b?logo=rust">
  <img alt="Python" src="https://img.shields.io/badge/Python-AI%20Bridge-%233776ab?logo=python&logoColor=white">
  <img alt="SQLite" src="https://img.shields.io/badge/SQLite-Local%20Persistence-%23003b57?logo=sqlite&logoColor=white">
  <img alt="DeepSeek" src="https://img.shields.io/badge/DeepSeek-Generation%20%26%20Review-1f6feb">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Terminal%20%2F%20TUI-0b7285">
</p>

> 面向 CET-4 / CET-6 的 AI 驱动终端练习系统
> Rust 负责 TUI 交互，Python 负责 DeepSeek 出题、评分、校验与数据闭环

`阅阶 CET` 不是简单题库，而是一套围绕四六级真题结构、题面风格、作答体验和复盘成长设计的终端学习工具。

- 普通刷题：按等级与题型单独练习
- 模拟四六级考试：先作文，再完成整套题
- AI 出题：按 CET 风格约束生成阅读、写作、翻译
- AI 评分：主观题逐句批注，客观题逐题解析
- 数据闭环：历史、词汇、薄弱项、趋势图、删除重算

声明：本项目目标是生成尽量贴近 CET 结构、难度与设问偏好的训练题，不代表官方真题或官方阅卷系统。

## 项目预览

### 总览

<p align="center">
  <img src="screenshot/overview-grid.png" alt="YueJie CET overview" width="100%">
</p>

### 典型页面

| 首页总览 | 阅读作答 |
| --- | --- |
| <img src="screenshot/swappy-20260611_202502.png" alt="首页总览" width="100%"> | <img src="screenshot/swappy-20260611_202651.png" alt="仔细阅读作答页" width="100%"> |
| 统计、趋势、快速入口、能力概览 | 三列考卷布局、文章区、答题区、电子表计时 |

| 结果页 | 写作作答 |
| --- | --- |
| <img src="screenshot/swappy-20260611_202811.png" alt="结果页" width="100%"> | <img src="screenshot/swappy-20260611_203421.png" alt="写作作答页" width="100%"> |
| 居中结果卡片、下一步操作、复盘入口 | 两列主观题布局、答题横线、词数与光标信息 |

| 能力与词汇 | 设置页 |
| --- | --- |
| <img src="screenshot/swappy-20260611_202528.png" alt="能力与词汇页" width="100%"> | <img src="screenshot/swappy-20260611_202909.png" alt="设置页" width="100%"> |
| 普通弱势 / 模拟四六级考试弱势 / 词汇表合并入口 | 主题、背景与配色方案切换 |

更多截图见 [`screenshot/`](./screenshot/)。

## 核心能力

### 1. 普通刷题模式

- 选择四级或六级
- 选择题型后由 AI 生成单套练习题
- 阅读使用三列布局
- 写作 / 翻译使用两列布局
- 作答结束后先看结果，再进入完整复盘

### 2. 模拟四六级考试模式

- 从首页直接进入 `模拟四六级考试`
- 先选择四级或六级
- 第一步只能做作文
- 作文生成后，其他部分在后台继续生成
- 若作文提前提交而其余题尚未生成完，会进入等待动画页
- 等待页不计时，后台会自动重试生成，不会把用户带回题型选择
- 作文提交后立即锁定
- 之后可在翻译、选词填空、长篇阅读、仔细阅读 1、仔细阅读 2 之间自由切换
- 最终交卷后生成整套总评、分项得分、弱势分析和提升建议

### 3. AI 评分与复盘

#### 客观题

- 正确率
- 用时
- 逐题正误
- 逐题解析
- 技巧提示
- 重点词汇整理

#### 主观题

- 15 分制评分与估算成绩
- 分项反馈
- 错词纠正
- 病句改写
- 逐句批注
- 高分范文 / 参考译文

### 4. 成长数据系统

- 首页表现指数与节奏匹配
- 普通刷题与模拟四六级考试历史分流
- 能力与词汇页面合并
- 普通弱势 / 模拟四六级考试弱势分开统计
- 词汇表合并累计、去重持久化
- 删除历史后自动重算相关统计、词汇和薄弱项
- 首页趋势图支持普通刷题与模拟四六级考试叠加显示

### 5. 终端交互体验

- 键盘 + 鼠标混合操作
- 大字电子表风格计时器
- 出题中 / 评分中动画
- 深色 / 浅色主题
- 多套配色方案
- 深色模式支持透明 / 不透明背景
- 试卷感布局，而非纯菜单式工具界面

## 支持题型

| 模块 | 题型 | 说明 |
| --- | --- | --- |
| `Section A` | 选词填空 | 10 空 + 15 共享选项，重词性、搭配、逻辑 |
| `Section B` | 长篇阅读 | 段落匹配，重定位与同义改写 |
| `Section C` | 仔细阅读 1 | 更偏事实驱动、研究 / 结果 / 细节 |
| `Section C` | 仔细阅读 2 | 更偏观点驱动、态度 / 推断 / 例证 |
| `Part I` | 写作 | CET 风格英文短文写作 |
| `Part IV` | 翻译 | 中文段落英译 |

## 模拟四六级考试流程

### 时间规则

- 总时长 100 分钟
- 作文阶段 30 分钟
- 作文时间到后强制提交并锁定
- 作文无论提前多久提交，后续部分都固定共享 70 分钟，不会把剩余作文时间累加到后半程
- 题型切换计时，不支持暂停
- 只有“作文提前提交后等待其余题目生成”的页面不计时

### 生成规则

- 先生成作文
- 用户写作文时，翻译与阅读其余部分在后台生成
- 后台生成失败会自动重试
- 不会让用户切到某部分时发现题目不存在

### 提交规则

- 作文提交后不可修改
- 其余部分作答可反复切换与修改，直到最终交卷
- 最终评分失败时会留在评分页自动重试，不会把用户踢回答题界面

## 技术架构

```text
Rust TUI (ratatui / crossterm)
        │
        │ JSON CLI bridge
        ▼
Python bridge / services
        │
        ├─ DeepSeek client
        ├─ Prompt & pipeline
        ├─ CET validator
        ├─ Objective scoring
        ├─ Subjective evaluation
        └─ SQLite persistence
```

### 角色划分

- Rust 前端
  - 页面布局
  - 键鼠事件
  - 动画
  - 考试状态机
  - 结果与复盘展示

- Python 后端
  - DeepSeek 调用
  - Prompt 生成与后处理
  - CET 结构校验
  - 客观题判分
  - 主观题评阅
  - SQLite 数据存储

## 快速开始

### 环境要求

- Python 3.12+
- Rust 稳定版工具链
- 可访问 DeepSeek API 的网络环境

### 安装依赖

```bash
cd /data/YueJie-CET
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
cargo build --release
```

### 配置 `.env`

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
YUEJIE_DB_PATH=
YUEJIE_REQUEST_TIMEOUT=120
```

说明：

- `.env` 已被 `.gitignore` 忽略
- 未配置 `DEEPSEEK_API_KEY` 时，不会进行真实 AI 出题与评阅
- 默认数据库路径为 `data/yuejie.db`

### 启动

```bash
source .venv/bin/activate
yuejie-cet
```

启动逻辑：

1. 优先寻找已编译的 Rust 二进制
2. 若已存在 `target/release/yuejie-cet-rs` 或 `target/debug/yuejie-cet-rs`，直接启动
3. 若本地无可执行文件，则尝试调用 `cargo run --release`
4. 若当前环境无法调用 Cargo，则提示先执行 `cargo build --release`

## 常用操作

### 普通刷题

1. 首页进入 `开始刷题`
2. 选择四级 / 六级
3. 选择题型
4. 等待 AI 生成
5. 作答、提交、查看结果与复盘

### 模拟四六级考试

1. 首页进入 `模拟四六级考试`
2. 选择四级 / 六级
3. 先完成作文
4. 作文提交后切换其余部分继续作答
5. 最终交卷并查看整套总评

### 历史与能力

- `刷题历史`：普通刷题 / 模拟四六级考试双标签页
- `能力与词汇`：普通弱势 / 模拟四六级考试弱势 / 词汇表三标签页

## 开发与验证

### 测试

```bash
cargo test
python -m unittest discover -s tests -v
```

### 构建

```bash
cargo build --release
```

### 桥接层调试

#### 真实出题

```bash
python -u -m app.bridge generate-live --level cet4 --question-type writing
python -u -m app.bridge generate-live --level cet4 --question-type translation
python -u -m app.bridge generate-live --level cet4 --question-type careful_reading --slot 1
```

#### 提交评分

```bash
python -m app.bridge submit
```

`submit` 从标准输入读取 JSON，例如：

```json
{
  "question_set_id": "qs_xxx",
  "started_at": "2026-06-10T17:10:00+08:00",
  "answers": {
    "response_text": "your answer here"
  }
}
```

#### 模拟四六级考试总评提交

```bash
python -m app.bridge submit-mock-exam
```

## 目录结构

```text
src/
  main.rs                  Rust TUI 主前端
app/
  main.py                  启动入口，仅负责拉起 Rust 前端
  bridge.py                Rust <-> Python JSON CLI bridge
  ai/                      DeepSeek 客户端、Prompt、生成与评分流水线
  config.py                .env 加载与运行配置
  constants.py             标签、模型、题型等常量
  data/                    SQLite 初始化与查询
  domain/                  枚举、数据模型、评分结构
  services/                出题、判分、统计、薄弱项、模拟考试服务
tests/                     Python 单元测试
screenshot/                README 截图
data/                      默认数据库目录
真题/                      题面与风格参考资料
```

## 当前状态

### 已完成

- Rust TUI 主前端
- Python AI / 数据后端
- 阅读 4 题型
- 写作 / 翻译
- 普通刷题模式
- 模拟四六级考试模式
- 主观题 AI 评分
- 历史回看 / 重做 / 删除
- 词汇表累计
- 普通弱势与模拟四六级考试弱势分流
- 评分失败原地自动重试

### 当前边界

- 题目仍是 AI 生成的 CET 风格练习题，不是官方原卷复刻
- 出题与评分依赖外部 API，可用性受网络和模型状态影响
- 阅读与写作题面已按真题格式约束，但仍会持续优化“像真题”的细节
- Python 侧只保留 AI、桥接、评分与数据库后端，不再维护独立 Python TUI

## 适合谁

- 想在终端里高频刷四六级题的用户
- 想研究 Rust TUI + Python AI 协作模式的开发者
- 想做 AI 出题 / AI 评阅 / 学习数据闭环产品的项目实践者

## 路线图

- 继续提升阅读与主观题生成稳定性
- 继续压缩题面重复度，提升写作 / 翻译话题多样性
- 增强模拟四六级考试总评的细粒度解析展示
- 打磨更多趋势图与成长视图
- 增加导出、备份与迁移能力
