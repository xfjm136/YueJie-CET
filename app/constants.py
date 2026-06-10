from __future__ import annotations

from app.domain.enums import BackgroundMode, Level, QuestionType, ThemeMode

APP_NAME = "阅阶 CET"
APP_SUBTITLE = "AI 四六级阅读专项训练"
MIN_TERMINAL_WIDTH = 120
MIN_TERMINAL_HEIGHT = 34
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_THEME_MODE = ThemeMode.DARK.value
DEFAULT_BACKGROUND_MODE = BackgroundMode.OPAQUE.value
DEFAULT_PALETTE_MODE = "fern"

LEVEL_LABELS = {
    Level.CET4: "四级",
    Level.CET6: "六级",
}

QUESTION_TYPE_LABELS = {
    QuestionType.BANKED_CLOZE: "选词填空",
    QuestionType.LONG_READING: "长篇阅读",
    QuestionType.CAREFUL_READING: "仔细阅读",
}

THEME_LABELS = {
    ThemeMode.DARK.value: "深色",
    ThemeMode.LIGHT.value: "浅色",
}

BACKGROUND_LABELS = {
    BackgroundMode.OPAQUE.value: "不透明",
    BackgroundMode.TRANSPARENT.value: "透明",
}

PALETTE_LABELS = {
    "fern": "蕨影绿",
    "ink": "墨青蓝",
    "amber": "琥珀米",
    "rose": "灰玫棕",
}

QUESTION_TYPE_SHORT_LABELS = {
    "banked_cloze": "选词",
    "long_reading": "长篇",
    "careful_reading": "仔细",
    "careful_reading:1": "仔细 1",
    "careful_reading:2": "仔细 2",
}

SKILL_LABELS = {
    "main_idea": "主旨理解",
    "detail": "细节定位",
    "inference": "推理判断",
    "attitude": "态度判断",
    "vocabulary": "语境词义",
    "vocabulary_in_context": "语境词义",
    "matching": "段落匹配",
    "speed": "作答速度",
    "logic": "上下文逻辑",
    "general": "综合理解",
}

QUESTION_TIME_HINTS = {
    QuestionType.BANKED_CLOZE: 8 * 60,
    QuestionType.LONG_READING: 15 * 60,
    QuestionType.CAREFUL_READING: 12 * 60,
}

SKILL_ADVICE = {
    "main_idea": "先抓首段和各段主题句，再判断文章中心。",
    "detail": "回到原文定位信号词，避免凭印象作答。",
    "inference": "关注转折和隐含态度，别只看表层字面义。",
    "attitude": "优先判断作者用词的褒贬与语气强弱。",
    "vocabulary": "把词性、搭配和上下文逻辑一起看。",
    "matching": "先扫题干关键词，再快速回文定位段落。",
    "speed": "先易后难，卡住一题时先跳过，保证整体节奏。",
    "logic": "多看因果、转折、并列等连接关系。",
    "general": "先明确题目考点，再回原文核对依据。",
}
