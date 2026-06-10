from __future__ import annotations

from enum import Enum


class Level(str, Enum):
    CET4 = "cet4"
    CET6 = "cet6"


class QuestionType(str, Enum):
    BANKED_CLOZE = "banked_cloze"
    LONG_READING = "long_reading"
    CAREFUL_READING = "careful_reading"
    WRITING = "writing"
    TRANSLATION = "translation"


class ThemeMode(str, Enum):
    DARK = "dark"
    LIGHT = "light"


class BackgroundMode(str, Enum):
    OPAQUE = "opaque"
    TRANSPARENT = "transparent"
