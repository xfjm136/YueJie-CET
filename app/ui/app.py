from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import cast

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.events import Click, Key
from textual.screen import Screen
from textual.widgets import (
    Button,
    Digits,
    Footer,
    Header,
    Input,
    ProgressBar,
    Select,
    Sparkline,
    Static,
)

from app.ai.client import DeepSeekClient
from app.ai.pipelines import QuestionGenerationPipeline
from app.config import get_settings
from app.constants import (
    APP_NAME,
    APP_SUBTITLE,
    BACKGROUND_LABELS,
    DEFAULT_BACKGROUND_MODE,
    DEFAULT_THEME_MODE,
    LEVEL_LABELS,
    MIN_TERMINAL_HEIGHT,
    MIN_TERMINAL_WIDTH,
    QUESTION_TIME_HINTS,
    QUESTION_TYPE_LABELS,
    SKILL_LABELS,
    THEME_LABELS,
)
from app.data.db import Database
from app.domain.enums import BackgroundMode, Level, QuestionType, ThemeMode
from app.domain.schemas import AttemptResult, QuestionSet
from app.services.attempt_service import AttemptService
from app.services.question_service import QuestionService
from app.services.stats_service import StatsService
from app.services.weakness_service import WeaknessService


def seconds_to_text(seconds: int) -> str:
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining:02d}"


def digits_number(value: int | float, width: int = 4, decimals: int = 0) -> str:
    if decimals:
        return f"{value:0{width}.{decimals}f}"
    return f"{int(value):0{width}d}"


def safe_series(values: list[float] | list[int]) -> list[float]:
    return [float(item) for item in values] if values else [0.0]


def level_text(level: Level) -> str:
    return LEVEL_LABELS[level]


def type_text(question_type: QuestionType, slot: int | None = None) -> str:
    if question_type is QuestionType.CAREFUL_READING and slot:
        return f"仔细阅读 {slot}"
    return QUESTION_TYPE_LABELS[question_type]


class BaseScreen(Screen):
    @property
    def yj_app(self) -> "YueJieApp":
        return cast("YueJieApp", self.app)


class HomeScreen(BaseScreen):
    def compose(self) -> ComposeResult:
        yield Container(
            Static(APP_NAME, classes="screen-title hero-title"),
            Static(APP_SUBTITLE, classes="screen-subtitle hero-subtitle"),
            Horizontal(
                self._metric_card("总刷题数", "card-total", Digits("0000", id="home-total-digits")),
                self._metric_card("最近正确率", "card-accuracy", Digits("00.0", id="home-accuracy-digits")),
                self._metric_card("平均用时", "card-duration", Digits("00:00", id="home-duration-digits")),
                self._distribution_card(),
                id="home-top-cards",
            ),
            Horizontal(
                self._chart_card("最近 5 次正确率走势", "home-accuracy-spark", "home-accuracy-caption"),
                self._chart_card("最近 5 次用时走势", "home-duration-spark", "home-duration-caption"),
                id="home-chart-row",
            ),
            Horizontal(
                Button("开始刷题", id="start-practice", variant="primary"),
                Button("刷题历史", id="open-history"),
                Button("我的薄弱项", id="open-weakness"),
                Button("词汇表", id="open-vocabulary"),
                Button("设置", id="open-settings"),
                Button("退出", id="quit-app"),
                classes="button-row center-row",
            ),
            id="home-root",
            classes="center-shell",
        )

    def _metric_card(self, title: str, card_id: str, digits: Digits) -> Container:
        suffix = card_id.split("-")[-1]
        return Container(
            Static(title, classes="card-kicker"),
            digits,
            Static("", id=f"{suffix}-meta", classes="metric-meta"),
            ProgressBar(total=100, show_eta=False, id=f"{suffix}-bar", classes="metric-bar"),
            classes="metric-card panel",
            id=card_id,
        )

    def _distribution_card(self) -> Container:
        return Container(
            Static("训练分布", classes="card-kicker"),
            Static("", id="distribution-common", classes="metric-meta strong-text"),
            Static("四级占比", classes="tiny-label"),
            ProgressBar(total=100, show_eta=False, id="distribution-cet4-bar", classes="metric-bar"),
            Static("六级占比", classes="tiny-label"),
            ProgressBar(total=100, show_eta=False, id="distribution-cet6-bar", classes="metric-bar"),
            Static("", id="distribution-meta", classes="metric-meta"),
            classes="metric-card panel",
            id="card-distribution",
        )

    def _chart_card(self, title: str, spark_id: str, caption_id: str) -> Container:
        return Container(
            Static(title, classes="card-kicker"),
            Sparkline(id=spark_id, classes="chart-spark"),
            Static("", id=caption_id, classes="metric-meta"),
            classes="chart-card panel",
        )

    def on_show(self) -> None:
        overview = self.yj_app.stats_service.overview_data()
        self.query_one("#home-total-digits", Digits).update(
            digits_number(overview["total_attempts"], width=4)
        )
        self.query_one("#home-accuracy-digits", Digits).update(
            digits_number(overview["recent_accuracy_percent"], width=4, decimals=1)
        )
        self.query_one("#home-duration-digits", Digits).update(overview["recent_duration_text"])

        self.query_one("#total-meta", Static).update(
            f"四级 {overview['total_cet4']} 题 / 六级 {overview['total_cet6']} 题"
        )
        self.query_one("#accuracy-meta", Static).update(
            f"最近 5 次平均正确率 {overview['recent_accuracy_percent']:.1f}%"
        )
        self.query_one("#duration-meta", Static).update(
            f"最近 5 次平均用时 {overview['recent_duration_text']}"
        )
        self.query_one("#distribution-common", Static).update(
            f"当前最常练：{overview['most_common_type_label']}"
        )
        self.query_one("#distribution-meta", Static).update(
            f"薄弱项最近更新：{overview['latest_weakness_updated_at'] or '暂无'}"
        )

        self.query_one("#total-bar", ProgressBar).update(
            progress=min(overview["total_attempts"], 100)
        )
        self.query_one("#accuracy-bar", ProgressBar).update(
            progress=overview["recent_accuracy_percent"]
        )
        duration_progress = min(overview["recent_duration_seconds"], 40 * 60)
        self.query_one("#duration-bar", ProgressBar).update(progress=duration_progress, total=40 * 60)
        self.query_one("#distribution-cet4-bar", ProgressBar).update(
            progress=overview["cet4_ratio"] * 100
        )
        self.query_one("#distribution-cet6-bar", ProgressBar).update(
            progress=overview["cet6_ratio"] * 100
        )

        self.query_one("#home-accuracy-spark", Sparkline).data = safe_series(
            overview["recent_accuracy_series"]
        )
        self.query_one("#home-duration-spark", Sparkline).data = safe_series(
            overview["recent_duration_series"]
        )
        self.query_one("#home-accuracy-caption", Static).update(
            "越靠右越新，峰值越高表示最近更稳。"
        )
        self.query_one("#home-duration-caption", Static).update(
            "越靠低越省时，目标是稳定而不是盲目求快。"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "start-practice":
            self.app.push_screen(LevelSelectScreen())
        elif button_id == "open-history":
            self.yj_app.open_history_screen()
        elif button_id == "open-weakness":
            self.yj_app.open_weakness_screen()
        elif button_id == "open-vocabulary":
            self.yj_app.open_vocabulary_screen()
        elif button_id == "open-settings":
            self.app.push_screen(SettingsScreen())
        elif button_id == "quit-app":
            self.app.exit()


class LevelSelectScreen(BaseScreen):
    def compose(self) -> ComposeResult:
        yield Container(
            Static("选择等级", classes="screen-title"),
            Static("先选择四级还是六级，再进入对应题型。", classes="screen-subtitle"),
            Horizontal(
                Button("四级", id="pick-cet4", variant="primary", classes="big-button"),
                Button("六级", id="pick-cet6", variant="primary", classes="big-button"),
                classes="button-row center-row",
            ),
            Button("返回", id="back-home"),
            classes="screen-shell center-shell small-shell",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pick-cet4":
            self.app.push_screen(TypeSelectScreen(Level.CET4))
        elif event.button.id == "pick-cet6":
            self.app.push_screen(TypeSelectScreen(Level.CET6))
        elif event.button.id == "back-home":
            self.app.pop_screen()


class TypeSelectScreen(BaseScreen):
    BINDINGS = [
        ("left", "move_selection_left", "左移"),
        ("right", "move_selection_right", "右移"),
        ("up", "move_selection_up", "上移"),
        ("down", "move_selection_down", "下移"),
        ("enter", "activate_selection", "进入"),
    ]
    CARD_MAP = {
        "banked_cloze": ("选词填空", "banked", "type-banked-cloze"),
        "long_reading": ("长篇阅读", "long", "type-long-reading"),
        "careful_reading:1": ("仔细阅读 1", "careful1", "type-careful-1"),
        "careful_reading:2": ("仔细阅读 2", "careful2", "type-careful-2"),
    }

    def __init__(self, level: Level) -> None:
        super().__init__()
        self.level = level
        self.card_keys = list(self.CARD_MAP.keys())
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        yield Container(
            Static(f"{level_text(self.level)}题型选择", classes="screen-title"),
            Static("每张卡片展示该题型最近数据，并直接进入训练。", classes="screen-subtitle"),
            Horizontal(
                self._type_card("banked_cloze"),
                self._type_card("long_reading"),
                id="type-card-row-1",
                classes="center-row",
            ),
            Horizontal(
                self._type_card("careful_reading:1"),
                self._type_card("careful_reading:2"),
                id="type-card-row-2",
                classes="center-row",
            ),
            Button("返回", id="back-level"),
            classes="screen-shell center-shell",
        )

    def _type_card(self, key: str) -> Container:
        label, token, button_id = self.CARD_MAP[key]
        return Container(
            Static(label, classes="card-kicker"),
            Digits("00.0", id=f"type-digits-{token}"),
            Static("", id=f"type-meta-{token}", classes="metric-meta"),
            ProgressBar(total=100, show_eta=False, id=f"type-bar-{token}", classes="metric-bar"),
            Sparkline(id=f"type-spark-{token}", classes="chart-spark"),
            Button("进入训练", id=button_id, variant="primary"),
            classes="type-card panel",
            id=f"type-card-{token}",
        )

    def on_show(self) -> None:
        token_map = {key: token for key, (_label, token, _button_id) in self.CARD_MAP.items()}
        for card in self.yj_app.stats_service.level_type_data(self.level):
            token = token_map[card["key"]]
            self.query_one(f"#type-digits-{token}", Digits).update(
                digits_number(card["recent_accuracy_percent"], width=4, decimals=1)
            )
            self.query_one(f"#type-meta-{token}", Static).update(
                f"{card['attempt_count']} 次 | 用时 {card['recent_duration_text']} | 近 5 次最高 {card['best_recent_accuracy_percent']:.1f}%"
            )
            self.query_one(f"#type-bar-{token}", ProgressBar).update(
                progress=card["recent_accuracy_percent"]
            )
            self.query_one(f"#type-spark-{token}", Sparkline).data = safe_series(
                card["recent_accuracy_series"]
            )
        self._sync_selection_state()

    def _selected_key(self) -> str:
        return self.card_keys[self.selected_index]

    def _selected_mapping(self) -> tuple[QuestionType, int | None]:
        key = self._selected_key()
        mapping = {
            "banked_cloze": (QuestionType.BANKED_CLOZE, None),
            "long_reading": (QuestionType.LONG_READING, None),
            "careful_reading:1": (QuestionType.CAREFUL_READING, 1),
            "careful_reading:2": (QuestionType.CAREFUL_READING, 2),
        }
        return mapping[key]

    def _sync_selection_state(self) -> None:
        for key, (_label, token, button_id) in self.CARD_MAP.items():
            card = self.query_one(f"#type-card-{token}", Container)
            selected = key == self._selected_key()
            card.set_class(selected, "type-card-selected")
            button = self.query_one(f"#{button_id}", Button)
            if selected:
                button.focus()

    def _move_selection(self, delta: int) -> None:
        self.selected_index = (self.selected_index + delta) % len(self.card_keys)
        self._sync_selection_state()

    def _set_selection_by_key(self, key: str) -> None:
        if key in self.card_keys:
            self.selected_index = self.card_keys.index(key)
            self._sync_selection_state()

    def action_move_selection_left(self) -> None:
        if self.selected_index in (1, 3):
            self._move_selection(-1)

    def action_move_selection_right(self) -> None:
        if self.selected_index in (0, 2):
            self._move_selection(1)

    def action_move_selection_up(self) -> None:
        if self.selected_index >= 2:
            self._move_selection(-2)

    def action_move_selection_down(self) -> None:
        if self.selected_index <= 1:
            self._move_selection(2)

    def action_activate_selection(self) -> None:
        question_type, slot = self._selected_mapping()
        self.app.push_screen(GeneratingScreen(self.level, question_type, slot))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-level":
            self.app.pop_screen()
            return
        reverse = {button_id: key for key, (_label, _token, button_id) in self.CARD_MAP.items()}
        self._set_selection_by_key(reverse[event.button.id])
        question_type, slot = self._selected_mapping()
        self.app.push_screen(GeneratingScreen(self.level, question_type, slot))

    def on_click(self, event: Click) -> None:
        widget = event.widget
        while widget is not None:
            widget_id = getattr(widget, "id", None)
            if widget_id and widget_id.startswith("type-card-"):
                token = widget_id.removeprefix("type-card-")
                for key, (_label, current_token, _button_id) in self.CARD_MAP.items():
                    if current_token == token:
                        self._set_selection_by_key(key)
                        self.action_activate_selection()
                        event.stop()
                        return
            widget = widget.parent


class GeneratingScreen(BaseScreen):
    def __init__(self, level: Level, question_type: QuestionType, slot: int | None = None) -> None:
        super().__init__()
        self.level = level
        self.question_type = question_type
        self.slot = slot
        self._tick = 0
        self._started = False

    def compose(self) -> ComposeResult:
        yield Container(
            Static("AI 正在生成题目", classes="screen-title"),
            Digits("00:00", id="generating-digits"),
            Static("", id="generation-status", classes="panel"),
            ProgressBar(total=5, show_eta=False, show_percentage=False, id="generation-progress"),
            Button("返回", id="cancel-generation"),
            classes="screen-shell center-shell small-shell",
        )

    def on_show(self) -> None:
        self.set_interval(0.45, self._refresh_animation)
        if not self._started:
            self._started = True
            threading.Thread(target=self._generate_in_thread, daemon=True).start()

    def _refresh_animation(self) -> None:
        phases = ["生成篇章", "设计题干", "校验结构", "整理解析", "提取词汇"]
        phase = phases[self._tick % len(phases)]
        self.query_one("#generating-digits", Digits).update(f"00:0{self._tick % 10}")
        self.query_one("#generation-progress", ProgressBar).update(progress=(self._tick % len(phases)) + 1)
        self.query_one("#generation-status", Static).update(
            f"等级：{level_text(self.level)}\n"
            f"题型：{type_text(self.question_type, self.slot)}\n"
            f"阶段：{phase}\n\n"
            "现在仅使用真实 AI 出题；如果结构校验失败，会继续按校验结果做定向修复。"
        )
        self._tick += 1

    def _generate_in_thread(self) -> None:
        try:
            question_set = self.yj_app.question_service.generate_question_set(
                self.level,
                self.question_type,
                self.slot,
            )
            self.app.call_from_thread(self._handle_success, question_set)
        except Exception as exc:
            self.app.call_from_thread(self._handle_failure, str(exc))

    def _handle_success(self, question_set: QuestionSet) -> None:
        self.app.push_screen(PracticeScreen(question_set))

    def _handle_failure(self, message: str) -> None:
        self.query_one("#generation-status", Static).update(f"生成失败：{message}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-generation":
            self.app.pop_screen()


class PracticeScreen(BaseScreen):
    BINDINGS = [("ctrl+s", "submit_attempt", "提交"), ("escape", "go_back", "返回")]

    def __init__(self, question_set: QuestionSet, is_history_retry: bool = False) -> None:
        super().__init__()
        self.question_set = question_set
        self.is_history_retry = is_history_retry
        self.started_at = datetime.now(timezone.utc)
        self.answer_state = {question.id: "" for question in self.question_set.questions}
        self.question_order = [question.id for question in self.question_set.questions]
        self.active_question_id = (
            self.question_order[0] if self.question_set.question_type is QuestionType.BANKED_CLOZE else None
        )
        self.option_letters = self._shared_option_letters()

    def compose(self) -> ComposeResult:
        if self.question_set.layout_mode == "two":
            yield self._compose_two_column()
        else:
            yield self._compose_three_column()

    def on_show(self) -> None:
        self.set_interval(1.0, self._refresh_timer)
        self._refresh_timer()
        if self.question_set.question_type is QuestionType.BANKED_CLOZE:
            self._refresh_banked_cloze_ui()

    def _suggested_duration(self) -> int:
        return QUESTION_TIME_HINTS[self.question_set.question_type]

    def _compose_sidebar(self) -> Vertical:
        return Vertical(
            Static("TIME", classes="paper-kicker"),
            Digits("00:00", id="timer-digits", classes="timer-digits"),
            Static("", id="timer-meta", classes="panel thin-panel"),
            ProgressBar(
                total=self._suggested_duration(),
                show_eta=False,
                show_percentage=False,
                id="timer-progress",
                classes="metric-bar",
            ),
            Static("答题进度", classes="paper-kicker"),
            ProgressBar(
                total=len(self.question_set.questions),
                show_eta=False,
                show_percentage=False,
                id="answer-progress",
                classes="metric-bar",
            ),
            Static("", id="answer-meta", classes="panel thin-panel"),
            Static(
                f"等级：{level_text(self.question_set.level)}\n"
                f"题型：{type_text(self.question_set.question_type, self.question_set.slot)}\n"
                f"词数：{self.question_set.word_count}\n"
                f"来源：{self.question_set.source_type}",
                classes="panel thin-panel",
            ),
            Static("快捷键：Ctrl+S 提交，Esc 返回", classes="panel thin-panel"),
            Button("提交答案", id="submit-attempt", variant="primary"),
            Button("返回上页", id="back-practice"),
            id="sidebar",
            classes="panel sidebar-shell",
        )

    def _paper_instruction(self) -> str:
        if self.question_set.question_type is QuestionType.BANKED_CLOZE:
            return (
                "Directions: Fill each numbered blank with the most suitable word from the shared options. "
                "Each option can be used once only."
            )
        if self.question_set.question_type is QuestionType.LONG_READING:
            return (
                "Directions: Match each statement with the paragraph from which the information is taken. "
                "Some paragraphs may be used more than once."
            )
        return (
            "Directions: Read the passage carefully and choose the best answer to each question. "
            "Base every choice on textual evidence."
        )

    def _compose_passage_panel(self) -> Vertical:
        title = type_text(self.question_set.question_type, self.question_set.slot)
        article = "\n\n".join(self.question_set.passage.paragraphs)
        return Vertical(
            Static("Part III  Reading Comprehension", classes="paper-banner"),
            Static(title, classes="paper-section-title"),
            Static(self._paper_instruction(), classes="paper-instruction"),
            Static(self.question_set.passage.title, classes="paper-article-title"),
            VerticalScroll(Static(article, classes="paper-body"), id="passage-scroll"),
            id="passage-column",
            classes="paper-panel panel",
        )

    def _compose_banked_cloze_panel(self) -> Vertical:
        blank_buttons = [
            Button(f"{index:02d}  [ ]", id=f"blank-{question.id}", classes="blank-chip")
            for index, question in enumerate(self.question_set.questions, start=1)
        ]
        option_buttons = [
            Button(option, id=f"option-{letter}", classes="option-chip")
            for option, letter in zip(self.question_set.shared_options, self.option_letters)
        ]
        return Vertical(
            Static("Answer Sheet", classes="paper-banner"),
            Static("当前焦点空位", classes="paper-kicker"),
            Static("", id="cloze-focus", classes="panel thin-panel"),
            Static("空位选择", classes="paper-kicker"),
            Horizontal(*blank_buttons[:5], classes="blank-grid"),
            Horizontal(*blank_buttons[5:], classes="blank-grid"),
            Static("共享选项 A-O", classes="paper-kicker"),
            Horizontal(*option_buttons[:5], classes="option-row"),
            Horizontal(*option_buttons[5:10], classes="option-row"),
            Horizontal(*option_buttons[10:], classes="option-row"),
            Static(
                "键盘支持：输入 A-O 直接填当前空位，输入 1-9/0 快速切换到对应空位。",
                classes="panel thin-panel",
            ),
            id="question-column",
            classes="paper-panel panel answer-sheet",
        )

    def _compose_standard_question_panel(self) -> Vertical:
        children = [
            Static("Answer Sheet", classes="paper-banner"),
            Static("Questions", classes="paper-section-title"),
        ]
        for index, question in enumerate(self.question_set.questions, start=1):
            option_text = "\n".join(question.options) if question.options else ""
            prompt_block = (
                f"{index}. {question.prompt}\n\n{option_text}"
                if option_text
                else f"{index}. {question.prompt}"
            )
            children.append(
                Container(
                    Static(prompt_block, classes="paper-question"),
                    Input(
                        placeholder="填写答案，例如 A / B / C / D",
                        id=f"answer-{question.id}",
                        classes="answer-input",
                    ),
                    classes="question-card paper-card",
                )
            )
        return Vertical(*children, id="question-column", classes="paper-panel panel")

    def _compose_question_panel(self) -> Vertical:
        if self.question_set.question_type is QuestionType.BANKED_CLOZE:
            return self._compose_banked_cloze_panel()
        return self._compose_standard_question_panel()

    def _compose_three_column(self) -> Horizontal:
        return Horizontal(
            self._compose_sidebar(),
            self._compose_passage_panel(),
            VerticalScroll(self._compose_question_panel(), id="right-scroll"),
            classes="practice-root",
        )

    def _compose_two_column(self) -> Horizontal:
        right = Vertical(
            self._compose_passage_panel(),
            VerticalScroll(self._compose_question_panel(), id="right-scroll"),
            id="two-column-right",
        )
        return Horizontal(self._compose_sidebar(), right, classes="practice-root")

    def _shared_option_letters(self) -> list[str]:
        letters = []
        for option in self.question_set.shared_options:
            letter = option.split(".", 1)[0].strip().upper()
            if letter:
                letters.append(letter)
        return letters

    def _answered_count(self) -> int:
        if self.question_set.question_type is QuestionType.BANKED_CLOZE:
            return sum(1 for value in self.answer_state.values() if value)
        return sum(1 for question in self.question_set.questions if self.query_one(f"#answer-{question.id}", Input).value.strip())

    def _refresh_timer(self) -> None:
        elapsed = int((datetime.now(timezone.utc) - self.started_at).total_seconds())
        self.query_one("#timer-digits", Digits).update(seconds_to_text(elapsed))
        self.query_one("#timer-progress", ProgressBar).update(
            progress=min(elapsed, self._suggested_duration())
        )
        self.query_one("#timer-meta", Static).update(
            f"建议时长：{seconds_to_text(self._suggested_duration())}\n当前用时：{seconds_to_text(elapsed)}"
        )
        answered = self._answered_count()
        self.query_one("#answer-progress", ProgressBar).update(progress=answered)
        self.query_one("#answer-meta", Static).update(
            f"已作答 {answered}/{len(self.question_set.questions)}"
        )

    def _refresh_banked_cloze_ui(self) -> None:
        if self.question_set.question_type is not QuestionType.BANKED_CLOZE:
            return
        active_index = self.question_order.index(self.active_question_id) + 1 if self.active_question_id else 1
        current_answer = self.answer_state.get(self.active_question_id or "", "") or "__"
        self.query_one("#cloze-focus", Static).update(
            f"当前空位：{active_index:02d}\n当前答案：{current_answer}\n点击空位或直接输入 A-O。"
        )

        for index, question_id in enumerate(self.question_order, start=1):
            button = self.query_one(f"#blank-{question_id}", Button)
            answer = self.answer_state.get(question_id, "")
            button.label = f"{index:02d}  [{answer or ' '}]"
            button.set_class(question_id == self.active_question_id, "blank-active")
            button.set_class(bool(answer), "blank-filled")

        used_map = {value: key for key, value in self.answer_state.items() if value}
        for option, letter in zip(self.question_set.shared_options, self.option_letters):
            button = self.query_one(f"#option-{letter}", Button)
            button.label = option
            button.set_class(letter in used_map, "option-used")
            button.set_class(letter == self.answer_state.get(self.active_question_id or "", ""), "option-current")

    def _set_active_blank(self, question_id: str) -> None:
        self.active_question_id = question_id
        self._refresh_banked_cloze_ui()

    def _assign_cloze_answer(self, letter: str) -> None:
        if self.question_set.question_type is not QuestionType.BANKED_CLOZE or not self.active_question_id:
            return
        for question_id, value in self.answer_state.items():
            if question_id != self.active_question_id and value == letter:
                self.answer_state[question_id] = ""
        self.answer_state[self.active_question_id] = letter
        for question_id in self.question_order:
            if not self.answer_state[question_id]:
                self.active_question_id = question_id
                break
        self._refresh_banked_cloze_ui()
        self._refresh_timer()

    def on_key(self, event: Key) -> None:
        if self.question_set.question_type is not QuestionType.BANKED_CLOZE:
            return
        key = event.key.upper()
        if key in self.option_letters:
            self._assign_cloze_answer(key)
            event.stop()
            return
        if key in {str(index) for index in range(10)}:
            target_index = 10 if key == "0" else int(key)
            if 1 <= target_index <= len(self.question_order):
                self._set_active_blank(self.question_order[target_index - 1])
                event.stop()

    def _collect_answers(self) -> dict[str, str]:
        if self.question_set.question_type is QuestionType.BANKED_CLOZE:
            return dict(self.answer_state)
        answers: dict[str, str] = {}
        for question in self.question_set.questions:
            answers[question.id] = self.query_one(f"#answer-{question.id}", Input).value.strip()
        return answers

    def action_submit_attempt(self) -> None:
        answers = self._collect_answers()
        result = self.yj_app.attempt_service.submit_attempt(
            self.question_set,
            answers,
            self.started_at,
            self.is_history_retry,
        )
        self.app.push_screen(ResultScreen(self.question_set, result))

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "submit-attempt":
            self.action_submit_attempt()
        elif button_id == "back-practice":
            self.action_go_back()
        elif button_id and button_id.startswith("blank-"):
            self._set_active_blank(button_id.removeprefix("blank-"))
        elif button_id and button_id.startswith("option-"):
            self._assign_cloze_answer(button_id.removeprefix("option-").upper())


class ResultScreen(BaseScreen):
    def __init__(self, question_set: QuestionSet, result: AttemptResult) -> None:
        super().__init__()
        self.question_set = question_set
        self.result = result

    def compose(self) -> ComposeResult:
        explanation_lines = []
        for index, item in enumerate(self.result.question_results, start=1):
            explanation_lines.append(
                f"{index}. 你的答案：{item.user_answer or '未作答'} | 正确答案：{item.correct_answer}\n"
                f"考点：{SKILL_LABELS.get(item.skill_tag, item.skill_tag)}\n"
                f"解析：{item.explanation}"
            )
        vocab_lines = [
            f"{item.surface_form} - {item.meaning_zh}" for item in self.question_set.vocabulary
        ]
        tip_lines = "\n".join(f"- {tip}" for tip in self.result.recommendations)
        yield Container(
            Static("作答结果", classes="screen-title"),
            Horizontal(
                self._result_card("正确率", Digits(digits_number(self.result.accuracy * 100, 4, 1))),
                self._result_card("得分", Digits(f"{self.result.correct_count:02d}")),
                self._result_card("用时", Digits(seconds_to_text(self.result.duration_seconds))),
                classes="center-row",
            ),
            Static(self.result.summary, classes="panel"),
            Static(
                "解题建议\n" + tip_lines + "\n\n"
                + "文章重点词汇\n"
                + ("\n".join(vocab_lines) if vocab_lines else "暂无"),
                classes="panel",
            ),
            VerticalScroll(
                Static("\n\n".join(explanation_lines), classes="panel"),
                id="result-explanations",
            ),
            Horizontal(
                Button("继续该题型", id="same-type", variant="primary"),
                Button("重做本题", id="redo-current"),
                Button("题型选择", id="back-types"),
                Button("回到首页", id="back-home"),
                classes="button-row center-row",
            ),
            classes="screen-shell center-shell",
        )

    def _result_card(self, title: str, digits: Digits) -> Container:
        return Container(
            Static(title, classes="card-kicker"),
            digits,
            classes="metric-card panel result-card",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "same-type":
            self.app.push_screen(
                GeneratingScreen(
                    self.question_set.level,
                    self.question_set.question_type,
                    self.question_set.slot,
                )
            )
        elif button_id == "redo-current":
            self.app.push_screen(PracticeScreen(self.question_set, is_history_retry=True))
        elif button_id == "back-types":
            self.app.push_screen(TypeSelectScreen(self.question_set.level))
        elif button_id == "back-home":
            self.app.push_screen(HomeScreen())


class HistoryScreen(BaseScreen):
    def __init__(self, records: list[dict]) -> None:
        super().__init__()
        self.records = records

    def compose(self) -> ComposeResult:
        items: list[Container | Static] = []
        if not self.records:
            items.append(Static("还没有历史记录，先去刷一套题吧。", classes="panel"))
        for row in self.records:
            items.append(
                Container(
                    Static(
                        f"{row['submitted_at']}\n"
                        f"{level_text(Level(row['level']))} · {type_text(QuestionType(row['question_type']), row['slot'])}\n"
                        f"{row['title']} | 正确率 {row['accuracy'] * 100:.1f}% | 用时 {seconds_to_text(row['duration_seconds'])}",
                        classes="panel",
                    ),
                    Horizontal(
                        Button("查看解析", id=f"review-{row['attempt_id']}"),
                        Button("重新作答", id=f"retry-{row['question_set_id']}"),
                        classes="button-row",
                    ),
                    classes="history-card",
                )
            )
        yield Container(
            Static("刷题历史", classes="screen-title"),
            VerticalScroll(*items, id="history-scroll"),
            Button("返回", id="back-history"),
            classes="screen-shell center-shell",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "back-history":
            self.app.pop_screen()
        elif button_id.startswith("review-"):
            self.yj_app.open_review_screen(button_id.removeprefix("review-"))
        elif button_id.startswith("retry-"):
            self.yj_app.redo_question_set(button_id.removeprefix("retry-"))


class ReviewScreen(BaseScreen):
    def __init__(self, bundle: dict) -> None:
        super().__init__()
        self.bundle = bundle
        self.question_set: QuestionSet = bundle["question_set"]
        self.result: AttemptResult = bundle["result"]

    def compose(self) -> ComposeResult:
        answer_lines = []
        for index, item in enumerate(self.result.question_results, start=1):
            answer_lines.append(
                f"{index}. 你的答案 {item.user_answer or '未作答'} / 正确答案 {item.correct_answer}\n"
                f"{item.explanation}"
            )
        yield Container(
            Static("历史复盘", classes="screen-title"),
            Static(
                f"{self.question_set.title}\n"
                f"{level_text(self.question_set.level)} · {type_text(self.question_set.question_type, self.question_set.slot)}\n"
                f"最近一次正确率 {self.result.accuracy * 100:.1f}% ，用时 {seconds_to_text(self.result.duration_seconds)}",
                classes="panel",
            ),
            Static(self.question_set.analysis.overall_strategy, classes="panel"),
            VerticalScroll(Static("\n\n".join(answer_lines), classes="panel"), id="review-scroll"),
            Horizontal(
                Button("重新作答", id="review-retry", variant="primary"),
                Button("返回历史", id="review-back"),
                classes="button-row center-row",
            ),
            classes="screen-shell center-shell",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "review-retry":
            self.app.push_screen(PracticeScreen(self.question_set, is_history_retry=True))
        elif event.button.id == "review-back":
            self.app.pop_screen()


class WeaknessScreen(BaseScreen):
    def __init__(self, snapshots: list[dict]) -> None:
        super().__init__()
        self.snapshots = snapshots

    def compose(self) -> ComposeResult:
        items: list[Static] = []
        if not self.snapshots:
            items.append(Static("每个题型累计超过 5 次后，这里会开始生成薄弱项总结。", classes="panel"))
        for row in self.snapshots:
            dimensions = json.loads(row["dimensions_json"])
            dimension_lines = ", ".join(
                f"{SKILL_LABELS.get(key, key)} {value:.1f}"
                for key, value in sorted(dimensions.items(), key=lambda item: item[1])[:4]
            )
            items.append(
                Static(
                    f"{row['updated_at']}\n"
                    f"{level_text(Level(row['level']))} · {type_text(QuestionType(row['question_type']))}\n"
                    f"{row['summary']}\n"
                    f"维度：{dimension_lines}",
                    classes="panel",
                )
            )
        yield Container(
            Static("我的薄弱项", classes="screen-title"),
            VerticalScroll(*items, id="weakness-scroll"),
            Button("返回", id="weakness-back"),
            classes="screen-shell center-shell",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "weakness-back":
            self.app.pop_screen()


class VocabularyScreen(BaseScreen):
    def __init__(self, items: list[dict]) -> None:
        super().__init__()
        self.items = items

    def compose(self) -> ComposeResult:
        lines = []
        for item in self.items:
            lines.append(
                f"{item['surface_form']} ({item['level_hint']}) - {item['meaning_zh']} | 频次 {item['frequency_score']}"
            )
        text = "\n".join(lines) if lines else "还没有词汇数据，先做题后会自动累计。"
        yield Container(
            Static("词汇表", classes="screen-title"),
            VerticalScroll(Static(text, classes="panel"), id="vocab-scroll"),
            Button("返回", id="vocab-back"),
            classes="screen-shell center-shell",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "vocab-back":
            self.app.pop_screen()


class SettingsScreen(BaseScreen):
    def compose(self) -> ComposeResult:
        settings = self.yj_app.settings
        api_key_state = "已配置" if settings.deepseek_api_key else "未配置，无法进行真实 AI 出题"
        yield Container(
            Static("设置", classes="screen-title"),
            Static(
                "主题和背景设置会持久化保存，下次启动自动沿用。",
                classes="screen-subtitle",
            ),
            Container(
                Static("界面主题", classes="card-kicker"),
                Select(
                    options=[(label, value) for value, label in THEME_LABELS.items()],
                    value=self.yj_app.theme_mode.value,
                    allow_blank=False,
                    id="theme-select",
                ),
                Static("背景模式", classes="card-kicker"),
                Select(
                    options=[(label, value) for value, label in BACKGROUND_LABELS.items()],
                    value=self.yj_app.background_mode.value,
                    allow_blank=False,
                    id="background-select",
                ),
                classes="panel settings-card",
            ),
            Static(
                f"数据库：{settings.db_path}\n"
                f"DeepSeek Base URL：{settings.deepseek_base_url}\n"
                f"DeepSeek Model：{settings.deepseek_model}\n"
                f"API Key：{api_key_state}\n\n"
                "程序会优先尝试真实 AI 出题，并进行本地 JSON 结构校验。",
                classes="panel",
            ),
            Button("返回", id="settings-back"),
            classes="screen-shell center-shell small-shell",
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "theme-select" and isinstance(event.value, str):
            self.yj_app.update_theme_mode(event.value)
        elif event.select.id == "background-select" and isinstance(event.value, str):
            self.yj_app.update_background_mode(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-back":
            self.app.pop_screen()


class YueJieApp(App):
    CSS_PATH = "styles/main.tcss"
    TITLE = APP_NAME
    SUB_TITLE = APP_SUBTITLE

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.db = Database(self.settings.db_path)
        self.db.init_schema()

        client = (
            DeepSeekClient(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                model=self.settings.deepseek_model,
                timeout=self.settings.request_timeout,
            )
            if self.settings.deepseek_api_key
            else None
        )
        pipeline = QuestionGenerationPipeline(client, self.settings.deepseek_model)
        self.weakness_service = WeaknessService(self.db)
        self.question_service = QuestionService(self.db, pipeline)
        self.attempt_service = AttemptService(self.db, self.weakness_service)
        self.stats_service = StatsService(self.db)
        self.theme_mode = self._load_theme_mode()
        self.background_mode = self._load_background_mode()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()

    def _load_theme_mode(self) -> ThemeMode:
        stored = self.db.get_app_setting("theme_mode", DEFAULT_THEME_MODE) or DEFAULT_THEME_MODE
        try:
            return ThemeMode(stored)
        except ValueError:
            self.db.set_app_setting("theme_mode", DEFAULT_THEME_MODE)
            return ThemeMode(DEFAULT_THEME_MODE)

    def _load_background_mode(self) -> BackgroundMode:
        stored = self.db.get_app_setting("background_mode", DEFAULT_BACKGROUND_MODE) or DEFAULT_BACKGROUND_MODE
        try:
            return BackgroundMode(stored)
        except ValueError:
            self.db.set_app_setting("background_mode", DEFAULT_BACKGROUND_MODE)
            return BackgroundMode(DEFAULT_BACKGROUND_MODE)

    def apply_appearance_classes(self) -> None:
        for class_name in ("theme-dark", "theme-light", "bg-opaque", "bg-transparent"):
            self.remove_class(class_name)
        self.add_class(f"theme-{self.theme_mode.value}")
        self.add_class(f"bg-{self.background_mode.value}")

    def update_theme_mode(self, value: str) -> None:
        self.theme_mode = ThemeMode(value)
        self.db.set_app_setting("theme_mode", self.theme_mode.value)
        self.apply_appearance_classes()
        self.notify(f"已切换到 {THEME_LABELS[self.theme_mode.value]}主题。", timeout=3)

    def update_background_mode(self, value: str) -> None:
        self.background_mode = BackgroundMode(value)
        self.db.set_app_setting("background_mode", self.background_mode.value)
        self.apply_appearance_classes()
        self.notify(f"背景已切换为 {BACKGROUND_LABELS[self.background_mode.value]}。", timeout=3)

    def on_mount(self) -> None:
        self.apply_appearance_classes()
        if self.size.width < MIN_TERMINAL_WIDTH or self.size.height < MIN_TERMINAL_HEIGHT:
            self.notify(
                f"建议终端尺寸至少 {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT}，当前界面可能较拥挤。",
                timeout=8,
            )
        self.push_screen(HomeScreen())

    def open_history_screen(self) -> None:
        self.push_screen(HistoryScreen(self.stats_service.list_history(limit=30)))

    def open_review_screen(self, attempt_id: str) -> None:
        bundle = self.db.get_attempt_bundle(attempt_id)
        if bundle is None:
            self.notify("找不到该历史记录。", severity="error")
            return
        self.push_screen(ReviewScreen(bundle))

    def redo_question_set(self, question_set_id: str) -> None:
        question_set = self.db.get_question_set(question_set_id)
        if question_set is None:
            self.notify("找不到这套题。", severity="error")
            return
        self.push_screen(PracticeScreen(question_set, is_history_retry=True))

    def open_weakness_screen(self) -> None:
        self.push_screen(WeaknessScreen(self.stats_service.list_weakness_snapshots(limit=20)))

    def open_vocabulary_screen(self) -> None:
        self.push_screen(VocabularyScreen(self.stats_service.list_vocabulary(limit=100)))
