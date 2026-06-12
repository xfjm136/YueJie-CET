#![allow(dead_code)]

use std::collections::HashMap;
use std::env;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use chrono::Local;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
    MouseButton, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::prelude::{Color, Line, Modifier, Span, Style};
use ratatui::symbols;
use ratatui::text::Text;
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Chart, Dataset, Gauge, GraphType, List, ListItem, ListState,
    Paragraph, Sparkline, Wrap,
};
use ratatui::{Frame, Terminal};
use serde::Deserialize;
use serde_json::{Value, json};

const MIN_TERMINAL_WIDTH: u16 = 120;
const MIN_TERMINAL_HEIGHT: u16 = 34;
const SUBMISSION_AUTO_RETRY_LIMIT: usize = 3;

fn main() -> Result<()> {
    let mut app = YueJieRustApp::new()?;
    app.run()
}

#[derive(Clone, Debug, Deserialize)]
struct OverviewResponse {
    ok: bool,
    overview: OverviewData,
    settings: BridgeSettings,
}

#[derive(Clone, Debug, Deserialize)]
struct TypeStatsResponse {
    ok: bool,
    cards: Vec<TypeCardData>,
}

#[derive(Clone, Debug, Deserialize)]
struct GenerateResponse {
    ok: bool,
    question_set: QuestionSet,
}

#[derive(Clone, Debug, Deserialize)]
struct SubmitResponse {
    ok: bool,
    result: AttemptResult,
}

#[derive(Clone, Debug, Deserialize)]
struct HistoryResponse {
    ok: bool,
    history: Vec<HistoryEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct ReviewResponse {
    ok: bool,
    question_set: QuestionSet,
    result: AttemptResult,
    answers: HashMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
struct WeaknessResponse {
    ok: bool,
    weakness: Vec<WeaknessEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct DeleteHistoryPayload {
    attempt_id: String,
    question_set_id: String,
    question_set_deleted: bool,
    level: String,
    question_type: String,
    title: String,
    topic: String,
}

#[derive(Clone, Debug, Deserialize)]
struct DeleteHistoryResponse {
    ok: bool,
    deleted: DeleteHistoryPayload,
}

#[derive(Clone, Debug, Deserialize)]
struct VocabularyResponse {
    ok: bool,
    vocabulary: Vec<VocabularyEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamHistoryResponse {
    ok: bool,
    history: Vec<MockExamHistoryEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamReviewResponse {
    ok: bool,
    mock_exam: MockExamRecord,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamWeaknessResponse {
    ok: bool,
    weakness: Vec<MockExamWeaknessEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamDeletePayload {
    exam_id: String,
    level: String,
    submitted_at: String,
    total_score: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamDeleteResponse {
    ok: bool,
    deleted: MockExamDeletePayload,
}

#[derive(Clone, Debug, Deserialize)]
struct SettingsResponse {
    ok: bool,
    settings: BridgeSettings,
}

#[derive(Clone, Debug, Deserialize)]
struct OverviewData {
    total_attempts: usize,
    total_mock_exams: usize,
    total_cet4: usize,
    total_cet6: usize,
    recent_accuracy_percent: f64,
    recent_duration_text: String,
    recent_accuracy_series: Vec<f64>,
    recent_duration_series: Vec<i64>,
    recent_performance_percent: f64,
    recent_pace_percent: f64,
    recent_performance_series: Vec<f64>,
    recent_pace_series: Vec<f64>,
    #[serde(default)]
    recent_mock_exam_score_series: Vec<f64>,
    #[serde(default)]
    recent_mock_exam_pace_series: Vec<f64>,
    raw_recent_accuracy_percent: f64,
    raw_recent_duration_text: String,
    most_common_type_label: String,
    latest_weakness_updated_at: Option<String>,
    cet4_ratio: f64,
    cet6_ratio: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct BridgeSettings {
    theme_mode: String,
    background_mode: String,
    #[serde(default)]
    palette_mode: String,
    deepseek_model: String,
    #[serde(default)]
    db_path: String,
    #[serde(default)]
    deepseek_base_url: String,
    #[serde(default)]
    api_key_configured: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct TypeCardData {
    key: String,
    short_label: String,
    label: String,
    attempt_count: usize,
    recent_accuracy_percent: f64,
    recent_duration_text: String,
    best_recent_accuracy_percent: f64,
    recent_accuracy_series: Vec<f64>,
    recent_duration_series: Vec<i64>,
}

#[derive(Clone, Debug, Deserialize)]
struct Passage {
    title: String,
    paragraphs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct Question {
    id: String,
    prompt: String,
    #[serde(default)]
    options: Vec<String>,
    #[serde(default)]
    skill_tag: String,
    hint: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ItemExplanation {
    question_id: String,
    correct_answer: String,
    explanation: String,
    #[serde(default)]
    skill_tag: String,
}

#[derive(Clone, Debug, Deserialize)]
struct AnalysisReport {
    overall_strategy: String,
    overall_summary: String,
    #[serde(default)]
    item_explanations: Vec<ItemExplanation>,
    #[serde(default)]
    test_tips: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct VocabularyItem {
    lemma: String,
    surface_form: String,
    level_hint: String,
    meaning_zh: String,
    #[serde(default)]
    example_en: String,
}

#[derive(Clone, Debug, Deserialize)]
struct QuestionSet {
    id: String,
    level: String,
    question_type: String,
    title: String,
    topic: String,
    passage: Passage,
    questions: Vec<Question>,
    answer_key: Vec<String>,
    analysis: AnalysisReport,
    vocabulary: Vec<VocabularyItem>,
    #[serde(default)]
    shared_options: Vec<String>,
    #[serde(default)]
    task_prompt: String,
    #[serde(default)]
    reference_answer: String,
    #[serde(default)]
    rubric_focus: Vec<String>,
    #[serde(default)]
    min_response_words: i32,
    #[serde(default)]
    max_response_words: i32,
    slot: Option<i32>,
    word_count: i32,
    created_at: String,
    generator_model: String,
    source_type: String,
}

#[derive(Clone, Debug, Deserialize)]
struct AttemptQuestionResult {
    question_id: String,
    user_answer: String,
    correct_answer: String,
    is_correct: bool,
    explanation: String,
    skill_tag: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ScoreDimension {
    name: String,
    score: f64,
    max_score: f64,
    feedback_zh: String,
}

#[derive(Clone, Debug, Deserialize)]
struct WordCorrection {
    original: String,
    corrected: String,
    #[serde(default)]
    meaning_zh: String,
    reason_zh: String,
    skill_tag: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SentenceRewrite {
    original_sentence: String,
    revised_sentence: String,
    reason_zh: String,
    skill_tag: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SentenceAnnotation {
    original_sentence: String,
    #[serde(default)]
    strengths_zh: String,
    #[serde(default)]
    issues_zh: String,
    #[serde(default)]
    revised_sentence: String,
    #[serde(default)]
    skill_tag: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SubjectiveEvaluation {
    score_15: f64,
    estimated_reported_score: f64,
    grade_band: String,
    overall_feedback_zh: String,
    #[serde(default)]
    score_dimensions: Vec<ScoreDimension>,
    #[serde(default)]
    wrong_words: Vec<WordCorrection>,
    #[serde(default)]
    sentence_rewrites: Vec<SentenceRewrite>,
    #[serde(default)]
    sentence_annotations: Vec<SentenceAnnotation>,
    #[serde(default)]
    high_score_version: String,
    #[serde(default)]
    weakness_tags: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct AttemptResult {
    id: String,
    question_set_id: String,
    correct_count: usize,
    total_count: usize,
    accuracy: f64,
    duration_seconds: i64,
    summary: String,
    recommendations: Vec<String>,
    question_results: Vec<AttemptQuestionResult>,
    #[serde(default)]
    subjective_evaluation: Option<SubjectiveEvaluation>,
    created_at: String,
}

#[derive(Clone, Debug, Deserialize)]
struct HistoryEntry {
    attempt_id: String,
    submitted_at: String,
    duration_seconds: i64,
    accuracy: f64,
    correct_count: usize,
    total_count: usize,
    is_history_retry: i64,
    question_set_id: String,
    level: String,
    question_type: String,
    slot: Option<i32>,
    title: String,
    topic: String,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamHistoryEntry {
    exam_id: String,
    level: String,
    started_at: String,
    submitted_at: String,
    duration_seconds: i64,
    total_score: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct WeaknessEntry {
    id: i64,
    level: String,
    question_type: String,
    summary: String,
    dimensions_json: String,
    based_on_attempt_count: i64,
    updated_at: String,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamWeaknessEntry {
    id: i64,
    level: String,
    summary: String,
    dimensions_json: String,
    based_on_exam_count: i64,
    updated_at: String,
}

#[derive(Clone, Debug, Deserialize)]
struct VocabularyEntry {
    lemma: String,
    surface_form: String,
    level_hint: String,
    meaning_zh: String,
    example_en: String,
    frequency_score: i64,
    error_related_score: i64,
    last_seen_at: String,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamSectionRecord {
    question_type: String,
    slot: Option<i32>,
    question_set: QuestionSet,
    answers: HashMap<String, String>,
    result: AttemptResult,
}

#[derive(Clone, Debug, Deserialize)]
struct MockExamRecord {
    id: String,
    level: String,
    started_at: String,
    submitted_at: String,
    duration_seconds: i64,
    total_score: f64,
    score_breakdown: HashMap<String, f64>,
    summary: String,
    recommendations: Vec<String>,
    weakness_tags: Vec<String>,
    sections: Vec<MockExamSectionRecord>,
    created_at: String,
}

#[derive(Clone)]
struct ReviewBundle {
    question_set: QuestionSet,
    result: AttemptResult,
    answers: HashMap<String, String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Home,
    LevelSelect,
    TypeSelect,
    Generating,
    Submitting,
    Practice,
    Result,
    History,
    Review,
    Weakness,
    Vocabulary,
    Insights,
    MockExamWaiting,
    MockExamReview,
    Settings,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HomeMode {
    Practice,
    MockExam,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HistoryTab {
    Practice,
    MockExam,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InsightsTab {
    PracticeWeakness,
    MockExamWeakness,
    Vocabulary,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LevelChoice {
    Cet4,
    Cet6,
}

impl LevelChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cet4 => "cet4",
            Self::Cet6 => "cet6",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cet4 => "四级",
            Self::Cet6 => "六级",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TypeChoice {
    BankedCloze,
    LongReading,
    Careful1,
    Careful2,
    Writing,
    Translation,
}

impl TypeChoice {
    fn all() -> [TypeChoice; 6] {
        [
            Self::BankedCloze,
            Self::LongReading,
            Self::Careful1,
            Self::Careful2,
            Self::Writing,
            Self::Translation,
        ]
    }

    fn key(self) -> &'static str {
        match self {
            Self::BankedCloze => "banked_cloze",
            Self::LongReading => "long_reading",
            Self::Careful1 => "careful_reading:1",
            Self::Careful2 => "careful_reading:2",
            Self::Writing => "writing",
            Self::Translation => "translation",
        }
    }

    fn question_type(self) -> &'static str {
        match self {
            Self::BankedCloze => "banked_cloze",
            Self::LongReading => "long_reading",
            Self::Careful1 | Self::Careful2 => "careful_reading",
            Self::Writing => "writing",
            Self::Translation => "translation",
        }
    }

    fn slot(self) -> Option<i32> {
        match self {
            Self::Careful1 => Some(1),
            Self::Careful2 => Some(2),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::BankedCloze => "选词填空",
            Self::LongReading => "长篇阅读",
            Self::Careful1 => "仔细阅读 1",
            Self::Careful2 => "仔细阅读 2",
            Self::Writing => "写作",
            Self::Translation => "翻译",
        }
    }

    fn section_label(self) -> &'static str {
        match self {
            Self::Writing => "Part I",
            Self::BankedCloze => "Section A",
            Self::LongReading => "Section B",
            Self::Careful1 => "Section C",
            Self::Careful2 => "Section C",
            Self::Translation => "Part IV",
        }
    }

    fn passage_label(self) -> Option<&'static str> {
        match self {
            Self::Careful1 => Some("Passage One"),
            Self::Careful2 => Some("Passage Two"),
            _ => None,
        }
    }

    fn recommended_time(self) -> &'static str {
        match self {
            Self::BankedCloze => "建议 8 分钟",
            Self::LongReading => "建议 15 分钟",
            Self::Careful1 | Self::Careful2 => "建议 12 分钟",
            Self::Writing | Self::Translation => "建议 30 分钟",
        }
    }

    fn estimated_generation_time(self, level: LevelChoice) -> &'static str {
        match (level, self) {
            (LevelChoice::Cet4, Self::BankedCloze) => "预计生成 20-35 秒",
            (LevelChoice::Cet4, Self::LongReading) => "预计生成 45-90 秒",
            (LevelChoice::Cet4, Self::Careful1) => "预计生成 30-60 秒",
            (LevelChoice::Cet4, Self::Careful2) => "预计生成 30-60 秒",
            (LevelChoice::Cet4, Self::Writing) => "预计生成 20-40 秒",
            (LevelChoice::Cet4, Self::Translation) => "预计生成 20-40 秒",
            (LevelChoice::Cet6, Self::BankedCloze) => "预计生成 25-45 秒",
            (LevelChoice::Cet6, Self::LongReading) => "预计生成 60-120 秒",
            (LevelChoice::Cet6, Self::Careful1) => "预计生成 40-80 秒",
            (LevelChoice::Cet6, Self::Careful2) => "预计生成 45-90 秒",
            (LevelChoice::Cet6, Self::Writing) => "预计生成 25-45 秒",
            (LevelChoice::Cet6, Self::Translation) => "预计生成 25-45 秒",
        }
    }

    fn brief(self, level: LevelChoice) -> &'static str {
        match self {
            Self::BankedCloze => match level {
                LevelChoice::Cet4 => "熟悉社会/校园题材下的词性、搭配与逻辑恢复",
                LevelChoice::Cet6 => "更抽象语篇里的词汇辨析、搭配竞争与逻辑衔接",
            },
            Self::LongReading => match level {
                LevelChoice::Cet4 => "略读查读、信息定位与基础同义改写匹配",
                LevelChoice::Cet6 => "更长语篇中的论点追踪、证据匹配与深层改写",
            },
            Self::Careful1 => match level {
                LevelChoice::Cet4 => "偏研究/实验/科普，重目的、过程、结果与细节",
                LevelChoice::Cet6 => "偏商业/市场/职场，重动机、后果、比较与报告发现",
            },
            Self::Careful2 => match level {
                LevelChoice::Cet4 => "偏社会现象/校园生活，重原因、反应、建议与态度",
                LevelChoice::Cet6 => "偏心理/伦理/科技反思，重推断、例证作用与立场",
            },
            Self::Writing => "立意、结构、语法与词汇表达",
            Self::Translation => "信息准确、语言自然、语法稳定",
        }
    }
}

#[derive(Clone)]
enum Action {
    HomeMenu(usize),
    Level(LevelChoice),
    Type(TypeChoice),
    OpenInsightsTab(usize),
    OpenHistoryTab(usize),
    RetryGeneration,
    RetrySubmission,
    BackHome,
    BackType,
    BackHistory,
    BackWeakness,
    BackVocabulary,
    BackSettings,
    BackReview,
    ReviewRedo,
    ResultViewAnalysis,
    HistoryReview(usize),
    HistoryRedo(usize),
    HistoryDelete(usize),
    HistorySelect(usize),
    MockExamSelectSection(TypeChoice),
    MockExamSubmit,
    MockExamHistoryReview(usize),
    MockExamHistoryDelete(usize),
    MockExamHistorySelect(usize),
    SubmitPractice,
    PracticeBack,
    PracticeSelectBlank(usize),
    PracticeSelectQuestion(usize),
    PracticeAssign(String),
    ResultContinue,
    ResultRedo,
    ResultBackTypes,
    ResultBackHome,
    ToggleTheme,
    ToggleBackground,
    TogglePalette,
    WeaknessSelect(usize),
    VocabularySelect(usize),
    MockWeaknessSelect(usize),
}

#[derive(Clone)]
struct ClickArea {
    rect: Rect,
    action: Action,
}

enum GenerationMessage {
    Progress {
        job_id: u64,
        phase: String,
        message: String,
    },
    Finished {
        job_id: u64,
        result: Result<QuestionSet, String>,
    },
}

#[derive(Clone, Copy)]
enum TrendMetric {
    Index,
    Percentage,
    Duration,
}

struct GeneratingTask {
    job_id: u64,
    started_at: Instant,
    receiver: Receiver<GenerationMessage>,
    cancel_flag: Arc<AtomicBool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GenerationContext {
    Practice,
    MockExamBootstrap,
}

enum SubmissionMessage {
    Progress {
        job_id: u64,
        phase: String,
        message: String,
    },
    Finished {
        job_id: u64,
        result: Result<AttemptResult, String>,
    },
}

struct SubmittingTask {
    job_id: u64,
    started_at: Instant,
    receiver: Receiver<SubmissionMessage>,
    cancel_flag: Arc<AtomicBool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SubmissionContext {
    Practice,
    MockExam,
}

enum MockExamGenerationMessage {
    Progress {
        type_choice: TypeChoice,
        message: String,
    },
    Retry {
        type_choice: TypeChoice,
        attempt: usize,
        error: String,
    },
    Ready {
        type_choice: TypeChoice,
        question_set: QuestionSet,
    },
    Finished,
}

struct MockExamGenerationTask {
    started_at: Instant,
    receiver: Receiver<MockExamGenerationMessage>,
    cancel_flag: Arc<AtomicBool>,
}

enum MockExamSubmissionMessage {
    Progress {
        job_id: u64,
        phase: String,
        message: String,
    },
    Finished {
        job_id: u64,
        result: Result<MockExamRecord, String>,
    },
}

struct MockExamSubmittingTask {
    job_id: u64,
    started_at: Instant,
    receiver: Receiver<MockExamSubmissionMessage>,
    cancel_flag: Arc<AtomicBool>,
}

struct BackendBridge {
    root: PathBuf,
    python_bin: PathBuf,
}

impl BackendBridge {
    fn new() -> Result<Self> {
        let root = find_project_root().context("failed to locate project root")?;
        let preferred = root.join(".venv").join("bin").join("python");
        let python_bin = if preferred.exists() {
            preferred
        } else {
            PathBuf::from("python")
        };
        Ok(Self { root, python_bin })
    }

    fn run_bridge(&self, args: &[&str], stdin_payload: Option<Value>) -> Result<Value> {
        self.run_bridge_with_timeout(args, stdin_payload, Duration::from_secs(15))
    }

    fn run_bridge_with_timeout(
        &self,
        args: &[&str],
        stdin_payload: Option<Value>,
        timeout: Duration,
    ) -> Result<Value> {
        let mut command = Command::new(&self.python_bin);
        command
            .current_dir(&self.root)
            .arg("-u")
            .arg("-m")
            .arg("app.bridge");
        for arg in args {
            command.arg(arg);
        }
        if stdin_payload.is_some() {
            command.stdin(Stdio::piped());
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to start bridge command: {:?}", args))?;
        if let Some(payload) = stdin_payload {
            let mut stdin = child.stdin.take().context("failed to open bridge stdin")?;
            stdin.write_all(payload.to_string().as_bytes())?;
        }
        let start = Instant::now();
        let output = loop {
            if let Some(status) = child.try_wait()? {
                let output = child.wait_with_output()?;
                if !status.success() && output.stdout.is_empty() && !output.stderr.is_empty() {
                    return Err(anyhow!(
                        "bridge command failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
                break output;
            }
            if start.elapsed() >= timeout {
                let _ = child.kill();
                let _ = child.wait();
                return Err(anyhow!(
                    "bridge command timed out after {}s",
                    timeout.as_secs()
                ));
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if stdout.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("bridge returned empty output: {}", stderr));
        }
        let value: Value = serde_json::from_str(&stdout)
            .with_context(|| format!("failed to parse bridge output: {}", stdout))?;
        if !output.status.success() {
            let err = value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("bridge command failed");
            return Err(anyhow!(err.to_string()));
        }
        if !value.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            let err = value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("bridge command failed");
            return Err(anyhow!(err.to_string()));
        }
        Ok(value)
    }

    fn overview(&self) -> Result<OverviewResponse> {
        let value = self.run_bridge(&["overview"], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn type_stats(&self, level: LevelChoice) -> Result<TypeStatsResponse> {
        let value = self.run_bridge(&["type-stats", "--level", level.as_str()], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn generate(&self, level: LevelChoice, type_choice: TypeChoice) -> Result<GenerateResponse> {
        let mut args = vec![
            "generate",
            "--level",
            level.as_str(),
            "--question-type",
            type_choice.question_type(),
        ];
        let slot_holder = type_choice.slot().map(|slot| slot.to_string());
        if let Some(slot) = slot_holder.as_deref() {
            args.push("--slot");
            args.push(slot);
        }
        let value = self.run_bridge_with_timeout(&args, None, Duration::from_secs(180))?;
        Ok(serde_json::from_value(value)?)
    }

    fn stream_generate(
        &self,
        level: LevelChoice,
        type_choice: TypeChoice,
        job_id: u64,
        sender: Sender<GenerationMessage>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<()> {
        let mut command = Command::new(&self.python_bin);
        command
            .current_dir(&self.root)
            .arg("-u")
            .arg("-m")
            .arg("app.bridge")
            .arg("generate-live")
            .arg("--level")
            .arg(level.as_str())
            .arg("--question-type")
            .arg(type_choice.question_type())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let slot_holder = type_choice.slot().map(|slot| slot.to_string());
        if let Some(slot) = slot_holder.as_deref() {
            command.arg("--slot").arg(slot);
        }
        let mut child = command
            .spawn()
            .context("failed to start live generation command")?;
        let stdout = child
            .stdout
            .take()
            .context("failed to capture live generation stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("failed to capture live generation stderr")?;
        let (line_tx, line_rx) = mpsc::channel::<Result<String, String>>();
        let stdout_handle = std::thread::spawn(move || {
            for raw_line in BufReader::new(stdout).lines() {
                let _ = line_tx.send(raw_line.map_err(|err| err.to_string()));
            }
        });
        let stderr_handle = std::thread::spawn(move || {
            let mut stderr_text = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut stderr_text);
            stderr_text
        });

        let mut saw_terminal_event = false;
        loop {
            if cancel_flag.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_handle.join();
                let _ = stderr_handle.join();
                return Ok(());
            }

            match line_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(&line).with_context(|| {
                        format!("failed to parse live generation output: {}", line)
                    })?;
                    match value
                        .get("event")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                    {
                        "progress" => {
                            let phase = value
                                .get("phase")
                                .and_then(Value::as_str)
                                .unwrap_or("progress")
                                .to_string();
                            let message = value
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("正在生成题目。")
                                .to_string();
                            let _ = sender.send(GenerationMessage::Progress {
                                job_id,
                                phase,
                                message,
                            });
                        }
                        "result" => {
                            let question_value = value
                                .get("question_set")
                                .cloned()
                                .context("live generation result missing question_set")?;
                            let question_set: QuestionSet = serde_json::from_value(question_value)?;
                            let _ = sender.send(GenerationMessage::Finished {
                                job_id,
                                result: Ok(question_set),
                            });
                            saw_terminal_event = true;
                        }
                        "error" => {
                            let error = value
                                .get("error")
                                .and_then(Value::as_str)
                                .unwrap_or("bridge command failed")
                                .to_string();
                            let _ = sender.send(GenerationMessage::Finished {
                                job_id,
                                result: Err(error),
                            });
                            saw_terminal_event = true;
                        }
                        _ => {}
                    }
                }
                Ok(Err(err)) => {
                    let _ = sender.send(GenerationMessage::Finished {
                        job_id,
                        result: Err(err),
                    });
                    saw_terminal_event = true;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {}
            }

            if let Some(status) = child.try_wait()? {
                let _ = stdout_handle.join();
                let stderr_text = stderr_handle.join().unwrap_or_default();
                while let Ok(Ok(line)) = line_rx.try_recv() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(&line).with_context(|| {
                        format!("failed to parse live generation output: {}", line)
                    })?;
                    if let Some("result") = value.get("event").and_then(Value::as_str) {
                        let question_value = value
                            .get("question_set")
                            .cloned()
                            .context("live generation result missing question_set")?;
                        let question_set: QuestionSet = serde_json::from_value(question_value)?;
                        let _ = sender.send(GenerationMessage::Finished {
                            job_id,
                            result: Ok(question_set),
                        });
                        saw_terminal_event = true;
                    }
                }
                if !saw_terminal_event {
                    if status.success() {
                        let _ = sender.send(GenerationMessage::Finished {
                            job_id,
                            result: Err("生成流程提前结束，未收到结果。".to_string()),
                        });
                    } else {
                        let detail = if stderr_text.trim().is_empty() {
                            "bridge command failed".to_string()
                        } else {
                            stderr_text.trim().to_string()
                        };
                        let _ = sender.send(GenerationMessage::Finished {
                            job_id,
                            result: Err(detail),
                        });
                    }
                }
                return Ok(());
            }
        }
    }

    fn submit(
        &self,
        question_set_id: &str,
        started_at: &str,
        answers: &HashMap<String, String>,
        is_history_retry: bool,
    ) -> Result<SubmitResponse> {
        let payload = json!({
            "question_set_id": question_set_id,
            "started_at": started_at,
            "answers": answers,
            "is_history_retry": is_history_retry,
        });
        let timeout = submit_timeout_for_answers(answers);
        let value = self.run_bridge_with_timeout(&["submit"], Some(payload), timeout)?;
        Ok(serde_json::from_value(value)?)
    }

    fn stream_submit(
        &self,
        question_set_id: &str,
        started_at: &str,
        answers: &HashMap<String, String>,
        is_history_retry: bool,
        job_id: u64,
        sender: Sender<SubmissionMessage>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<()> {
        let payload = json!({
            "question_set_id": question_set_id,
            "started_at": started_at,
            "answers": answers,
            "is_history_retry": is_history_retry,
        });
        let mut command = Command::new(&self.python_bin);
        command
            .current_dir(&self.root)
            .arg("-u")
            .arg("-m")
            .arg("app.bridge")
            .arg("submit-live")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command
            .spawn()
            .context("failed to start live submit command")?;
        {
            let mut stdin = child
                .stdin
                .take()
                .context("failed to open live submit stdin")?;
            stdin.write_all(payload.to_string().as_bytes())?;
        }

        let stdout = child
            .stdout
            .take()
            .context("failed to capture live submit stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("failed to capture live submit stderr")?;
        let (line_tx, line_rx) = mpsc::channel::<Result<String, String>>();
        let stdout_handle = std::thread::spawn(move || {
            for raw_line in BufReader::new(stdout).lines() {
                let _ = line_tx.send(raw_line.map_err(|err| err.to_string()));
            }
        });
        let stderr_handle = std::thread::spawn(move || {
            let mut stderr_text = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut stderr_text);
            stderr_text
        });

        let mut saw_terminal_event = false;
        loop {
            if cancel_flag.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_handle.join();
                let _ = stderr_handle.join();
                return Ok(());
            }

            match line_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(&line)
                        .with_context(|| format!("failed to parse live submit output: {}", line))?;
                    match value
                        .get("event")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                    {
                        "progress" => {
                            let phase = value
                                .get("phase")
                                .and_then(Value::as_str)
                                .unwrap_or("progress")
                                .to_string();
                            let message = value
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("正在评分。")
                                .to_string();
                            let _ = sender.send(SubmissionMessage::Progress {
                                job_id,
                                phase,
                                message,
                            });
                        }
                        "result" => {
                            let result_value = value
                                .get("result")
                                .cloned()
                                .context("live submit result missing result")?;
                            let attempt_result: AttemptResult =
                                serde_json::from_value(result_value)?;
                            let _ = sender.send(SubmissionMessage::Finished {
                                job_id,
                                result: Ok(attempt_result),
                            });
                            saw_terminal_event = true;
                        }
                        "error" => {
                            let error = value
                                .get("error")
                                .and_then(Value::as_str)
                                .unwrap_or("bridge command failed")
                                .to_string();
                            let _ = sender.send(SubmissionMessage::Finished {
                                job_id,
                                result: Err(error),
                            });
                            saw_terminal_event = true;
                        }
                        _ => {}
                    }
                }
                Ok(Err(err)) => {
                    let _ = sender.send(SubmissionMessage::Finished {
                        job_id,
                        result: Err(err),
                    });
                    saw_terminal_event = true;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {}
            }

            if let Some(status) = child.try_wait()? {
                let _ = stdout_handle.join();
                let stderr_text = stderr_handle.join().unwrap_or_default();
                while let Ok(Ok(line)) = line_rx.try_recv() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(&line)
                        .with_context(|| format!("failed to parse live submit output: {}", line))?;
                    if let Some("result") = value.get("event").and_then(Value::as_str) {
                        let result_value = value
                            .get("result")
                            .cloned()
                            .context("live submit result missing result")?;
                        let attempt_result: AttemptResult = serde_json::from_value(result_value)?;
                        let _ = sender.send(SubmissionMessage::Finished {
                            job_id,
                            result: Ok(attempt_result),
                        });
                        saw_terminal_event = true;
                    }
                }
                if !saw_terminal_event {
                    if status.success() {
                        let _ = sender.send(SubmissionMessage::Finished {
                            job_id,
                            result: Err("评分流程提前结束，未收到结果。".to_string()),
                        });
                    } else {
                        let detail = if stderr_text.trim().is_empty() {
                            "bridge command failed".to_string()
                        } else {
                            stderr_text.trim().to_string()
                        };
                        let _ = sender.send(SubmissionMessage::Finished {
                            job_id,
                            result: Err(detail),
                        });
                    }
                }
                return Ok(());
            }
        }
    }

    fn stream_submit_mock_exam(
        &self,
        level: LevelChoice,
        started_at: &str,
        duration_seconds: i64,
        sections: &[Value],
        job_id: u64,
        sender: Sender<MockExamSubmissionMessage>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<()> {
        let payload = json!({
            "level": level.as_str(),
            "started_at": started_at,
            "duration_seconds": duration_seconds,
            "sections": sections,
        });
        let mut command = Command::new(&self.python_bin);
        command
            .current_dir(&self.root)
            .arg("-u")
            .arg("-m")
            .arg("app.bridge")
            .arg("submit-mock-exam-live")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command
            .spawn()
            .context("failed to start live mock exam submit command")?;
        {
            let mut stdin = child
                .stdin
                .take()
                .context("failed to open live mock exam submit stdin")?;
            stdin.write_all(payload.to_string().as_bytes())?;
        }

        let stdout = child
            .stdout
            .take()
            .context("failed to capture live mock exam submit stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("failed to capture live mock exam submit stderr")?;
        let (line_tx, line_rx) = mpsc::channel::<Result<String, String>>();
        let stdout_handle = std::thread::spawn(move || {
            for raw_line in BufReader::new(stdout).lines() {
                let _ = line_tx.send(raw_line.map_err(|err| err.to_string()));
            }
        });
        let stderr_handle = std::thread::spawn(move || {
            let mut stderr_text = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut stderr_text);
            stderr_text
        });

        let mut saw_terminal_event = false;
        loop {
            if cancel_flag.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_handle.join();
                let _ = stderr_handle.join();
                return Ok(());
            }

            match line_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(&line).with_context(|| {
                        format!("failed to parse live mock exam submit output: {}", line)
                    })?;
                    match value
                        .get("event")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                    {
                        "progress" => {
                            let phase = value
                                .get("phase")
                                .and_then(Value::as_str)
                                .unwrap_or("progress")
                                .to_string();
                            let message = value
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("正在评分整套模拟四六级考试。")
                                .to_string();
                            let _ = sender.send(MockExamSubmissionMessage::Progress {
                                job_id,
                                phase,
                                message,
                            });
                        }
                        "result" => {
                            let record_value = value
                                .get("mock_exam")
                                .cloned()
                                .context("live mock exam submit result missing mock_exam")?;
                            let record: MockExamRecord = serde_json::from_value(record_value)?;
                            let _ = sender.send(MockExamSubmissionMessage::Finished {
                                job_id,
                                result: Ok(record),
                            });
                            saw_terminal_event = true;
                        }
                        "error" => {
                            let error = value
                                .get("error")
                                .and_then(Value::as_str)
                                .unwrap_or("bridge command failed")
                                .to_string();
                            let _ = sender.send(MockExamSubmissionMessage::Finished {
                                job_id,
                                result: Err(error),
                            });
                            saw_terminal_event = true;
                        }
                        _ => {}
                    }
                }
                Ok(Err(err)) => {
                    let _ = sender.send(MockExamSubmissionMessage::Finished {
                        job_id,
                        result: Err(err),
                    });
                    saw_terminal_event = true;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {}
            }

            if let Some(status) = child.try_wait()? {
                let _ = stdout_handle.join();
                let stderr_text = stderr_handle.join().unwrap_or_default();
                while let Ok(Ok(line)) = line_rx.try_recv() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(&line).with_context(|| {
                        format!("failed to parse live mock exam submit output: {}", line)
                    })?;
                    if let Some("result") = value.get("event").and_then(Value::as_str) {
                        let record_value = value
                            .get("mock_exam")
                            .cloned()
                            .context("live mock exam submit result missing mock_exam")?;
                        let record: MockExamRecord = serde_json::from_value(record_value)?;
                        let _ = sender.send(MockExamSubmissionMessage::Finished {
                            job_id,
                            result: Ok(record),
                        });
                        saw_terminal_event = true;
                    }
                }
                if !saw_terminal_event {
                    if status.success() {
                        let _ = sender.send(MockExamSubmissionMessage::Finished {
                            job_id,
                            result: Err("模拟四六级考试评分流程提前结束，未收到结果。".to_string()),
                        });
                    } else {
                        let detail = if stderr_text.trim().is_empty() {
                            "bridge command failed".to_string()
                        } else {
                            stderr_text.trim().to_string()
                        };
                        let _ = sender.send(MockExamSubmissionMessage::Finished {
                            job_id,
                            result: Err(detail),
                        });
                    }
                }
                return Ok(());
            }
        }
    }

    fn history(&self) -> Result<HistoryResponse> {
        let value = self.run_bridge(&["history", "--limit", "30"], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn delete_history(&self, attempt_id: &str) -> Result<DeleteHistoryResponse> {
        let value = self.run_bridge(&["delete-history", "--attempt-id", attempt_id], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn review(&self, attempt_id: &str) -> Result<ReviewResponse> {
        let value = self.run_bridge(&["review", "--attempt-id", attempt_id], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn mock_exam_history(&self) -> Result<MockExamHistoryResponse> {
        let value = self.run_bridge(&["mock-exam-history", "--limit", "30"], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn mock_exam_review(&self, exam_id: &str) -> Result<MockExamReviewResponse> {
        let value = self.run_bridge_with_timeout(
            &["mock-exam-review", "--exam-id", exam_id],
            None,
            Duration::from_secs(90),
        )?;
        Ok(serde_json::from_value(value)?)
    }

    fn delete_mock_exam(&self, exam_id: &str) -> Result<MockExamDeleteResponse> {
        let value = self.run_bridge(&["mock-exam-delete", "--exam-id", exam_id], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn weakness(&self) -> Result<WeaknessResponse> {
        let value = self.run_bridge(&["weakness", "--limit", "20"], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn mock_exam_weakness(&self) -> Result<MockExamWeaknessResponse> {
        let value = self.run_bridge(&["mock-exam-weakness", "--limit", "20"], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn vocabulary(&self) -> Result<VocabularyResponse> {
        let value = self.run_bridge(&["vocabulary", "--limit", "100"], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn settings(&self) -> Result<SettingsResponse> {
        let value = self.run_bridge(&["settings"], None)?;
        Ok(serde_json::from_value(value)?)
    }

    fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.run_bridge(&["set-setting", "--key", key, "--value", value], None)?;
        Ok(())
    }
}

#[derive(Clone)]
struct PracticeState {
    question_set: QuestionSet,
    answers: HashMap<String, String>,
    started_at_iso: String,
    started_instant: Instant,
    is_history_retry: bool,
    selected_blank: usize,
    selected_question: usize,
    passage_scroll: u16,
    choice_cursor: usize,
    subjective_cursor: usize,
    subjective_scroll: u16,
    submit_confirm_pending: bool,
}

impl PracticeState {
    fn new(question_set: QuestionSet, is_history_retry: bool) -> Self {
        let mut answers = HashMap::new();
        if question_set.questions.is_empty() {
            answers.insert("response_text".to_string(), String::new());
        } else {
            for question in &question_set.questions {
                answers.insert(question.id.clone(), String::new());
            }
        }
        Self {
            question_set,
            answers,
            started_at_iso: Local::now().to_rfc3339(),
            started_instant: Instant::now(),
            is_history_retry,
            selected_blank: 0,
            selected_question: 0,
            passage_scroll: 0,
            choice_cursor: 0,
            subjective_cursor: 0,
            subjective_scroll: 0,
            submit_confirm_pending: false,
        }
    }

    fn answered_count(&self) -> usize {
        if self.question_set.questions.is_empty() {
            return usize::from(
                self.answers
                    .get("response_text")
                    .is_some_and(|value| !value.trim().is_empty()),
            );
        }
        self.answers
            .values()
            .filter(|value| !value.is_empty())
            .count()
    }

    fn elapsed_seconds(&self) -> i64 {
        self.started_instant.elapsed().as_secs() as i64
    }

    fn unanswered_count(&self) -> usize {
        if self.question_set.questions.is_empty() {
            return usize::from(
                self.answers
                    .get("response_text")
                    .is_none_or(|value| value.trim().is_empty()),
            );
        }
        self.question_set
            .questions
            .len()
            .saturating_sub(self.answered_count())
    }

    fn available_labels(&self) -> Vec<String> {
        if self.question_set.questions.is_empty() {
            return vec![];
        }
        if self.question_set.question_type == "banked_cloze" {
            return self
                .question_set
                .shared_options
                .iter()
                .filter_map(|item| item.split('.').next())
                .map(|item| item.trim().to_string())
                .collect();
        }
        if self.question_set.question_type == "careful_reading" {
            return vec!["A".into(), "B".into(), "C".into(), "D".into()];
        }
        self.question_set
            .passage
            .paragraphs
            .iter()
            .filter_map(|paragraph| paragraph.split('.').next())
            .map(|item| item.trim().to_string())
            .filter(|item| item.len() == 1)
            .collect()
    }

    fn sync_choice_cursor_to_current_answer(&mut self) {
        if self.question_set.questions.is_empty()
            || self.question_set.question_type == "banked_cloze"
        {
            return;
        }
        let qid = self.question_set.questions[self.selected_question]
            .id
            .clone();
        if let Some(current_answer) = self.answers.get(&qid) {
            if let Some(index) = self
                .available_labels()
                .iter()
                .position(|label| label == current_answer)
            {
                self.choice_cursor = index;
            }
        }
    }

    fn assign_answer(&mut self, answer: String) {
        self.submit_confirm_pending = false;
        if self.question_set.questions.is_empty() {
            return;
        }
        if self.question_set.question_type == "banked_cloze" {
            for value in self.answers.values_mut() {
                if *value == answer {
                    *value = String::new();
                }
            }
            let qid = self.question_set.questions[self.selected_blank].id.clone();
            self.answers.insert(qid, answer);
            if self.selected_blank + 1 < self.question_set.questions.len() {
                self.selected_blank += 1;
            }
            return;
        }

        let qid = self.question_set.questions[self.selected_question]
            .id
            .clone();
        self.answers.insert(qid, answer.clone());
        if let Some(index) = self
            .available_labels()
            .iter()
            .position(|label| label == &answer)
        {
            self.choice_cursor = index;
        }
        if self.selected_question + 1 < self.question_set.questions.len() {
            self.selected_question += 1;
            self.sync_choice_cursor_to_current_answer();
        }
    }

    fn clear_current_answer(&mut self) {
        self.submit_confirm_pending = false;
        if self.question_set.questions.is_empty() {
            self.answers
                .insert("response_text".to_string(), String::new());
            self.subjective_cursor = 0;
            return;
        }
        if self.question_set.question_type == "banked_cloze" {
            let qid = self.question_set.questions[self.selected_blank].id.clone();
            self.answers.insert(qid, String::new());
            return;
        }
        let qid = self.question_set.questions[self.selected_question]
            .id
            .clone();
        self.answers.insert(qid, String::new());
    }

    fn response_text(&self) -> &str {
        self.answers
            .get("response_text")
            .map(String::as_str)
            .unwrap_or("")
    }

    fn set_response_text(&mut self, text: String) {
        self.answers.insert("response_text".to_string(), text);
    }

    fn insert_response_char(&mut self, ch: char) {
        self.submit_confirm_pending = false;
        let mut text = self.response_text().to_string();
        let insert_at = self.subjective_cursor.min(text.len());
        text.insert(insert_at, ch);
        self.subjective_cursor = insert_at + ch.len_utf8();
        self.set_response_text(text);
    }

    fn insert_response_newline(&mut self) {
        self.insert_response_char('\n');
    }

    fn backspace_response_char(&mut self) {
        self.submit_confirm_pending = false;
        if self.subjective_cursor == 0 {
            return;
        }
        let mut text = self.response_text().to_string();
        let prev = text[..self.subjective_cursor]
            .char_indices()
            .last()
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        text.replace_range(prev..self.subjective_cursor, "");
        self.subjective_cursor = prev;
        self.set_response_text(text);
    }

    fn move_response_cursor_left(&mut self) {
        if self.subjective_cursor == 0 {
            return;
        }
        self.subjective_cursor = self.response_text()[..self.subjective_cursor]
            .char_indices()
            .last()
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    fn move_response_cursor_right(&mut self) {
        let text = self.response_text();
        if self.subjective_cursor >= text.len() {
            return;
        }
        if let Some(ch) = text[self.subjective_cursor..].chars().next() {
            self.subjective_cursor =
                (self.subjective_cursor + ch.len_utf8()).min(self.response_text().len());
        }
    }

    fn move_response_cursor_home(&mut self) {
        self.subjective_cursor = 0;
    }

    fn move_response_cursor_end(&mut self) {
        self.subjective_cursor = self.response_text().len();
    }
}

#[derive(Clone)]
struct MockExamSectionState {
    type_choice: TypeChoice,
    practice: PracticeState,
    locked: bool,
    elapsed_seconds: i64,
    active_since: Option<Instant>,
}

#[derive(Clone)]
struct MockExamSession {
    level: LevelChoice,
    started_at_iso: String,
    started_instant: Instant,
    paused_wait_seconds: i64,
    waiting_started_at: Option<Instant>,
    writing_locked: bool,
    writing_locked_at_exam_seconds: Option<i64>,
    active_section: TypeChoice,
    sections: HashMap<String, MockExamSectionState>,
    pending_types: Vec<TypeChoice>,
    failed_types: Vec<TypeChoice>,
    writing_force_submitted: bool,
    final_force_submitted: bool,
}

impl MockExamSession {
    const TOTAL_SECONDS: i64 = 100 * 60;
    const WRITING_SECONDS: i64 = 30 * 60;

    fn new(level: LevelChoice, writing_question_set: QuestionSet) -> Self {
        let started_at_iso = Local::now().to_rfc3339();
        let started_instant = Instant::now();
        let mut sections = HashMap::new();
        sections.insert(
            Self::key_for(TypeChoice::Writing).to_string(),
            MockExamSectionState {
                type_choice: TypeChoice::Writing,
                practice: PracticeState::new(writing_question_set, false),
                locked: false,
                elapsed_seconds: 0,
                active_since: Some(Instant::now()),
            },
        );
        Self {
            level,
            started_at_iso,
            started_instant,
            paused_wait_seconds: 0,
            waiting_started_at: None,
            writing_locked: false,
            writing_locked_at_exam_seconds: None,
            active_section: TypeChoice::Writing,
            sections,
            pending_types: vec![
                TypeChoice::Translation,
                TypeChoice::BankedCloze,
                TypeChoice::LongReading,
                TypeChoice::Careful1,
                TypeChoice::Careful2,
            ],
            failed_types: Vec::new(),
            writing_force_submitted: false,
            final_force_submitted: false,
        }
    }

    fn key_for(type_choice: TypeChoice) -> &'static str {
        type_choice.key()
    }

    fn elapsed_exam_seconds(&self) -> i64 {
        let base = self.started_instant.elapsed().as_secs() as i64;
        let waiting = self
            .waiting_started_at
            .map(|started| started.elapsed().as_secs() as i64)
            .unwrap_or(0);
        (base - self.paused_wait_seconds - waiting).max(0)
    }

    fn pause_for_waiting(&mut self) {
        if self.waiting_started_at.is_none() {
            if let Some(section) = self.section_mut(self.active_section) {
                if let Some(started) = section.active_since.take() {
                    section.elapsed_seconds += started.elapsed().as_secs() as i64;
                }
            }
            self.waiting_started_at = Some(Instant::now());
        }
    }

    fn resume_from_waiting(&mut self) {
        if let Some(started) = self.waiting_started_at.take() {
            self.paused_wait_seconds += started.elapsed().as_secs() as i64;
            if let Some(section) = self.section_mut(self.active_section) {
                if section.active_since.is_none() {
                    section.active_since = Some(Instant::now());
                }
            }
        }
    }

    fn writing_elapsed_seconds(&self) -> i64 {
        self.section(TypeChoice::Writing)
            .map(|section| {
                let raw = section.practice.started_instant.elapsed().as_secs() as i64;
                let waiting = self
                    .waiting_started_at
                    .map(|started| started.elapsed().as_secs() as i64)
                    .unwrap_or(0);
                (raw - waiting).max(0)
            })
            .unwrap_or(0)
    }

    fn writing_remaining_seconds(&self) -> i64 {
        (Self::WRITING_SECONDS - self.writing_elapsed_seconds()).max(0)
    }

    fn total_remaining_seconds(&self) -> i64 {
        if let Some(locked_at) = self.writing_locked_at_exam_seconds {
            let post_writing_elapsed = (self.elapsed_exam_seconds() - locked_at).max(0);
            (Self::TOTAL_SECONDS - Self::WRITING_SECONDS - post_writing_elapsed).max(0)
        } else {
            Self::TOTAL_SECONDS - Self::WRITING_SECONDS + self.writing_remaining_seconds()
        }
    }

    fn is_waiting(&self) -> bool {
        self.waiting_started_at.is_some()
    }

    fn section(&self, type_choice: TypeChoice) -> Option<&MockExamSectionState> {
        self.sections.get(Self::key_for(type_choice))
    }

    fn section_mut(&mut self, type_choice: TypeChoice) -> Option<&mut MockExamSectionState> {
        self.sections.get_mut(Self::key_for(type_choice))
    }

    fn is_ready(&self, type_choice: TypeChoice) -> bool {
        self.sections.contains_key(Self::key_for(type_choice))
    }

    fn register_ready_section(&mut self, type_choice: TypeChoice, question_set: QuestionSet) {
        self.sections.insert(
            Self::key_for(type_choice).to_string(),
            MockExamSectionState {
                type_choice,
                practice: PracticeState::new(question_set, false),
                locked: false,
                elapsed_seconds: 0,
                active_since: None,
            },
        );
        self.pending_types.retain(|item| *item != type_choice);
        self.failed_types.retain(|item| *item != type_choice);
    }

    fn register_failed_section(&mut self, type_choice: TypeChoice) {
        if !self.failed_types.contains(&type_choice) {
            self.failed_types.push(type_choice);
        }
    }

    fn all_sections_ready(&self) -> bool {
        self.pending_types.is_empty()
    }

    fn switch_section(&mut self, type_choice: TypeChoice) {
        if self.is_ready(type_choice) {
            let not_waiting = self.waiting_started_at.is_none();
            if let Some(section) = self.section_mut(self.active_section) {
                if let Some(started) = section.active_since.take() {
                    section.elapsed_seconds += started.elapsed().as_secs() as i64;
                }
            }
            self.active_section = type_choice;
            if let Some(section) = self.section_mut(type_choice) {
                if section.active_since.is_none() && not_waiting {
                    section.active_since = Some(Instant::now());
                }
            }
        }
    }

    fn active_section(&self) -> Option<&MockExamSectionState> {
        self.section(self.active_section)
    }

    fn active_section_mut(&mut self) -> Option<&mut MockExamSectionState> {
        self.section_mut(self.active_section)
    }

    fn lock_writing(&mut self) {
        self.writing_locked = true;
        if self.writing_locked_at_exam_seconds.is_none() {
            self.writing_locked_at_exam_seconds = Some(self.elapsed_exam_seconds());
        }
        if let Some(section) = self.section_mut(TypeChoice::Writing) {
            section.locked = true;
        }
    }

    fn current_section_elapsed_seconds(&self, type_choice: TypeChoice) -> i64 {
        self.section(type_choice)
            .map(|section| {
                section.elapsed_seconds
                    + section
                        .active_since
                        .map(|started| started.elapsed().as_secs() as i64)
                        .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    fn is_section_locked(&self, type_choice: TypeChoice) -> bool {
        self.section(type_choice).map(|section| section.locked).unwrap_or(false)
    }

    fn completed_sections(&self) -> usize {
        self.sections
            .values()
            .filter(|section| {
                if section.practice.question_set.questions.is_empty() {
                    !section.practice.response_text().trim().is_empty()
                } else {
                    section.practice.unanswered_count() == 0
                }
            })
            .count()
    }

    fn total_sections(&self) -> usize {
        6
    }
}

struct YueJieRustApp {
    backend: BackendBridge,
    screen: Screen,
    overview: OverviewData,
    settings: BridgeSettings,
    selected_level: LevelChoice,
    selected_type: TypeChoice,
    home_menu_index: usize,
    type_index: usize,
    type_cards: Vec<TypeCardData>,
    click_areas: Vec<ClickArea>,
    status_line: String,
    generating_task: Option<GeneratingTask>,
    generation_context: GenerationContext,
    generating_tick: usize,
    generation_sequence: u64,
    generation_phase: String,
    generation_message: String,
    generation_log: Vec<String>,
    generation_error_message: Option<String>,
    submitting_task: Option<SubmittingTask>,
    submission_context: SubmissionContext,
    mock_exam_generating_task: Option<MockExamGenerationTask>,
    mock_exam_submitting_task: Option<MockExamSubmittingTask>,
    submission_sequence: u64,
    submission_retry_count: usize,
    submission_phase: String,
    submission_message: String,
    submission_log: Vec<String>,
    submission_error_message: Option<String>,
    practice: Option<PracticeState>,
    mock_exam_session: Option<MockExamSession>,
    result: Option<AttemptResult>,
    result_detail_scroll: u16,
    history: Vec<HistoryEntry>,
    mock_exam_history: Vec<MockExamHistoryEntry>,
    history_tab: HistoryTab,
    history_index: usize,
    mock_exam_history_index: usize,
    review: Option<ReviewBundle>,
    mock_exam_review: Option<MockExamRecord>,
    review_back_screen: Screen,
    review_detail_scroll: u16,
    weakness: Vec<WeaknessEntry>,
    mock_exam_weakness: Vec<MockExamWeaknessEntry>,
    weakness_index: usize,
    vocabulary: Vec<VocabularyEntry>,
    vocabulary_index: usize,
    insights_tab: InsightsTab,
    mock_exam_weakness_index: usize,
    home_mode: HomeMode,
    history_action_index: usize,
    result_action_index: usize,
    review_action_index: usize,
    settings_focus: usize,
    pending_history_delete_attempt_id: Option<String>,
    should_quit: bool,
}

impl YueJieRustApp {
    fn new() -> Result<Self> {
        let backend = BackendBridge::new()?;
        let overview = backend.overview()?;
        let mut settings = overview.settings.clone();
        if force_light_mode_opaque(&mut settings) {
            backend.set_setting("background_mode", &settings.background_mode)?;
        }
        Ok(Self {
            backend,
            screen: Screen::Home,
            overview: overview.overview,
            settings,
            selected_level: LevelChoice::Cet4,
            selected_type: TypeChoice::BankedCloze,
            home_menu_index: 0,
            type_index: 0,
            type_cards: Vec::new(),
            click_areas: Vec::new(),
            status_line: String::from("方向键、Enter、Esc、Ctrl+Q 和鼠标都可使用。"),
            generating_task: None,
            generation_context: GenerationContext::Practice,
            generating_tick: 0,
            generation_sequence: 0,
            generation_phase: String::from("idle"),
            generation_message: String::from("尚未开始生成。"),
            generation_log: Vec::new(),
            generation_error_message: None,
            submitting_task: None,
            submission_context: SubmissionContext::Practice,
            mock_exam_generating_task: None,
            mock_exam_submitting_task: None,
            submission_sequence: 0,
            submission_retry_count: 0,
            submission_phase: String::from("idle"),
            submission_message: String::from("尚未开始评分。"),
            submission_log: Vec::new(),
            submission_error_message: None,
            practice: None,
            mock_exam_session: None,
            result: None,
            result_detail_scroll: 0,
            history: Vec::new(),
            mock_exam_history: Vec::new(),
            history_tab: HistoryTab::Practice,
            history_index: 0,
            mock_exam_history_index: 0,
            review: None,
            mock_exam_review: None,
            review_back_screen: Screen::Home,
            review_detail_scroll: 0,
            weakness: Vec::new(),
            mock_exam_weakness: Vec::new(),
            weakness_index: 0,
            vocabulary: Vec::new(),
            vocabulary_index: 0,
            insights_tab: InsightsTab::PracticeWeakness,
            mock_exam_weakness_index: 0,
            home_mode: HomeMode::Practice,
            history_action_index: 0,
            result_action_index: 0,
            review_action_index: 0,
            settings_focus: 0,
            pending_history_delete_attempt_id: None,
            should_quit: false,
        })
    }

    fn run(&mut self) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let outcome = self.event_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;
        outcome
    }

    fn event_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
        loop {
            terminal.draw(|frame| self.draw(frame))?;
            if self.should_quit {
                break;
            }

            if let Some(task) = &self.generating_task {
                let active_job_id = task.job_id;
                let mut pending_messages = Vec::new();
                while let Ok(message) = task.receiver.try_recv() {
                    pending_messages.push(message);
                }
                for message in pending_messages {
                    match message {
                        GenerationMessage::Progress {
                            job_id,
                            phase,
                            message,
                        } => {
                            if job_id != active_job_id {
                                continue;
                            }
                            self.generation_phase = phase;
                            self.generation_message = message.clone();
                            self.push_generation_log(message);
                        }
                        GenerationMessage::Finished { job_id, result } => {
                            if job_id != active_job_id {
                                continue;
                            }
                            self.generating_task = None;
                            match result {
                                Ok(question_set) => {
                                    self.generation_error_message = None;
                                    self.sync_selection_from_question_set(&question_set);
                                    match self.generation_context {
                                        GenerationContext::Practice => {
                                            self.status_line = match question_set.source_type.as_str()
                                            {
                                                "ai" => String::from("题目已生成，开始作答。"),
                                                "ai_repaired" => String::from(
                                                    "题目已生成，并已自动修复结构后进入作答。",
                                                ),
                                                other => format!(
                                                    "题目来源：{}，现在开始作答。",
                                                    display_source_type(other)
                                                ),
                                            };
                                            self.practice =
                                                Some(PracticeState::new(question_set, false));
                                            self.result_action_index = 0;
                                            self.screen = Screen::Practice;
                                        }
                                        GenerationContext::MockExamBootstrap => {
                                            self.status_line = String::from(
                                                "作文已生成，模拟四六级考试开始；其余部分会在后台继续生成。",
                                            );
                                            let session =
                                                MockExamSession::new(self.selected_level, question_set);
                                            self.mock_exam_session = Some(session);
                                            self.practice = None;
                                            self.start_mock_exam_background_generation()?;
                                            self.screen = Screen::Practice;
                                        }
                                    }
                                }
                                Err(message) => {
                                    self.generation_error_message = Some(message.clone());
                                    self.generation_phase = String::from("failed");
                                    self.generation_message = String::from(
                                        "本轮出题未成功，请直接重试或返回题型选择。",
                                    );
                                    self.push_generation_log(format!("失败：{}", message));
                                    self.status_line = String::from(
                                        "R/Enter 重试，Esc 返回题型选择，鼠标也可点击操作。",
                                    );
                                    self.screen = Screen::Generating;
                                }
                            }
                        }
                    }
                }
            }

            if let Some(task) = &self.mock_exam_generating_task {
                let mut pending_messages = Vec::new();
                while let Ok(message) = task.receiver.try_recv() {
                    pending_messages.push(message);
                }
                for message in pending_messages {
                    match message {
                        MockExamGenerationMessage::Progress { type_choice, message } => {
                            self.push_generation_log(format!(
                                "{}：{}",
                                type_choice.label(),
                                message
                            ));
                            if matches!(self.screen, Screen::MockExamWaiting) {
                                self.generation_message = format!(
                                    "后台正在准备 {}，{}",
                                    type_choice.label(),
                                    message
                                );
                            }
                        }
                        MockExamGenerationMessage::Retry {
                            type_choice,
                            attempt,
                            error,
                        } => {
                            self.push_generation_log(format!(
                                "{} 第 {} 次重试：{}",
                                type_choice.label(),
                                attempt,
                                error
                            ));
                        }
                        MockExamGenerationMessage::Ready {
                            type_choice,
                            question_set,
                        } => {
                            let mut should_resume = false;
                            let mut should_switch_translation = false;
                            if let Some(session) = &mut self.mock_exam_session {
                                session.register_ready_section(type_choice, question_set);
                                should_resume = session.is_waiting() && session.all_sections_ready();
                                should_switch_translation = should_resume
                                    && session.active_section == TypeChoice::Writing;
                            }
                            self.push_generation_log(format!(
                                "{} 已就绪，可进入作答。",
                                type_choice.label()
                            ));
                            if should_resume {
                                if let Some(session) = &mut self.mock_exam_session {
                                    session.resume_from_waiting();
                                    if should_switch_translation {
                                        session.switch_section(TypeChoice::Translation);
                                    }
                                }
                                self.status_line = String::from(
                                        "其余题目已全部准备完成，可继续整套模拟四六级考试。",
                                );
                                self.screen = Screen::Practice;
                            }
                        }
                        MockExamGenerationMessage::Finished => {
                            self.mock_exam_generating_task = None;
                            if let Some(session) = &self.mock_exam_session {
                                if session.all_sections_ready()
                                    && matches!(self.screen, Screen::MockExamWaiting)
                                {
                                    self.status_line = String::from(
                                        "模拟四六级考试题目已全部就绪，继续作答即可。",
                                    );
                                }
                            }
                        }
                    }
                }
            }

            if let Some(task) = &self.submitting_task {
                let active_job_id = task.job_id;
                let mut pending_messages = Vec::new();
                while let Ok(message) = task.receiver.try_recv() {
                    pending_messages.push(message);
                }
                for message in pending_messages {
                    match message {
                        SubmissionMessage::Progress {
                            job_id,
                            phase,
                            message,
                        } => {
                            if job_id != active_job_id {
                                continue;
                            }
                            self.submission_phase = phase;
                            self.submission_message = message.clone();
                            self.push_submission_log(message);
                        }
                        SubmissionMessage::Finished { job_id, result } => {
                            if job_id != active_job_id {
                                continue;
                            }
                            self.submitting_task = None;
                            match result {
                                Ok(result) => {
                                    self.submission_error_message = None;
                                    match self.submission_context {
                                        SubmissionContext::Practice => {
                                            if let Some(practice) = &self.practice {
                                                self.review = Some(ReviewBundle {
                                                    question_set: practice.question_set.clone(),
                                                    result: result.clone(),
                                                    answers: practice.answers.clone(),
                                                });
                                            }
                                            self.review_back_screen = Screen::Result;
                                            self.result = Some(result);
                                            self.result_detail_scroll = 0;
                                            self.result_action_index = 0;
                                            self.review_action_index = 0;
                                            self.status_line = String::from(
                                                "Enter/1 查看解析，2 下一题，3 重做，4 题型，5 首页，PageUp/PageDown 可浏览摘要。",
                                            );
                                            self.screen = Screen::Result;
                                        }
                                        SubmissionContext::MockExam => {}
                                    }
                                }
                                Err(message) => {
                                    if self.submission_retry_count < SUBMISSION_AUTO_RETRY_LIMIT {
                                        self.submission_retry_count += 1;
                                        self.submission_error_message = None;
                                        self.submission_phase = String::from("retry");
                                        self.submission_message = format!(
                                            "评分暂时失败，正在自动重试第 {}/{} 次。",
                                            self.submission_retry_count, SUBMISSION_AUTO_RETRY_LIMIT
                                        );
                                        self.push_submission_log(format!(
                                            "评分失败，正在自动重试第 {}/{} 次：{}",
                                            self.submission_retry_count,
                                            SUBMISSION_AUTO_RETRY_LIMIT,
                                            message
                                        ));
                                        self.status_line = String::from(
                                            "评分暂时失败，系统正在原地自动重试，请稍候。",
                                        );
                                        self.start_submission_internal(false)?;
                                    } else {
                                        self.submission_error_message = Some(message.clone());
                                        self.submission_phase = String::from("failed");
                                        self.submission_message = String::from(
                                            "本轮评分多次重试后仍未成功，请手动重试或返回作答界面。",
                                        );
                                        self.push_submission_log(format!(
                                            "自动重试已用尽：{}",
                                            message
                                        ));
                                        self.status_line = String::from(
                                            "自动重试已达上限，R/Enter 可继续重试，Esc 返回作答界面。",
                                        );
                                        self.screen = Screen::Submitting;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let Some(task) = &self.mock_exam_submitting_task {
                let active_job_id = task.job_id;
                let mut pending_messages = Vec::new();
                while let Ok(message) = task.receiver.try_recv() {
                    pending_messages.push(message);
                }
                for message in pending_messages {
                    match message {
                        MockExamSubmissionMessage::Progress {
                            job_id,
                            phase,
                            message,
                        } => {
                            if job_id != active_job_id {
                                continue;
                            }
                            self.submission_phase = phase;
                            self.submission_message = message.clone();
                            self.push_submission_log(message);
                        }
                        MockExamSubmissionMessage::Finished { job_id, result } => {
                            if job_id != active_job_id {
                                continue;
                            }
                            self.mock_exam_submitting_task = None;
                            match result {
                                Ok(record) => {
                                    self.submission_error_message = None;
                                    self.mock_exam_review = Some(record);
                                    self.history = self.backend.history()?.history;
                                    self.mock_exam_history = self.backend.mock_exam_history()?.history;
                                    self.history_tab = HistoryTab::MockExam;
                                    self.review_back_screen = Screen::History;
                                    self.review_detail_scroll = 0;
                                    self.status_line = String::from(
                                        "模拟四六级考试总评已生成，PageUp/PageDown 查看详情，Enter/Esc 返回历史。",
                                    );
                                    self.mock_exam_session = None;
                                    self.refresh_overview()?;
                                    self.screen = Screen::MockExamReview;
                                }
                                Err(message) => {
                                    if self.submission_retry_count < SUBMISSION_AUTO_RETRY_LIMIT {
                                        self.submission_retry_count += 1;
                                        self.submission_error_message = None;
                                        self.submission_phase = String::from("retry");
                                        self.submission_message = format!(
                                            "整套评分暂时失败，正在自动重试第 {}/{} 次。",
                                            self.submission_retry_count, SUBMISSION_AUTO_RETRY_LIMIT
                                        );
                                        self.push_submission_log(format!(
                                            "整套评分失败，正在自动重试第 {}/{} 次：{}",
                                            self.submission_retry_count,
                                            SUBMISSION_AUTO_RETRY_LIMIT,
                                            message
                                        ));
                                        self.status_line = String::from(
                                            "整套评分暂时失败，系统正在原地自动重试，请稍候。",
                                        );
                                        self.submit_mock_exam_internal(false)?;
                                    } else {
                                        if let Some(session) = &mut self.mock_exam_session {
                                            session.resume_from_waiting();
                                        }
                                        self.submission_error_message = Some(message.clone());
                                        self.submission_phase = String::from("failed");
                                        self.submission_message = String::from(
                                            "模拟四六级考试整套评分多次重试后仍未成功，请手动重试或返回作答界面。",
                                        );
                                        self.push_submission_log(format!(
                                            "自动重试已用尽：{}",
                                            message
                                        ));
                                        self.status_line = String::from(
                                            "自动重试已达上限，R/Enter 可继续重试，Esc 返回模拟四六级考试作答界面。",
                                        );
                                        self.screen = Screen::Submitting;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if event::poll(Duration::from_millis(120))? {
                match event::read()? {
                    Event::Key(key) => {
                        if self.handle_key(key)? {
                            self.should_quit = true;
                            break;
                        }
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse(mouse.kind, mouse.column, mouse.row)?;
                        if self.should_quit {
                            break;
                        }
                    }
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            } else if matches!(
                self.screen,
                Screen::Generating
                    | Screen::Submitting
                    | Screen::Result
                    | Screen::MockExamWaiting
            ) {
                self.generating_tick = self.generating_tick.wrapping_add(1);
            }

            self.tick_mock_exam_session()?;
        }
        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q'))
        {
            return Ok(true);
        }
        if key.code == KeyCode::Esc {
            self.handle_escape()?;
            return Ok(false);
        }

        match self.screen {
            Screen::Home => self.handle_home_key(key),
            Screen::LevelSelect => self.handle_level_key(key),
            Screen::TypeSelect => self.handle_type_key(key)?,
            Screen::Generating => self.handle_generating_key(key)?,
            Screen::Submitting => self.handle_submitting_key(key)?,
            Screen::Practice => self.handle_practice_key(key)?,
            Screen::Result => self.handle_result_key(key)?,
            Screen::History => self.handle_history_key(key)?,
            Screen::Review => self.handle_review_key(key)?,
            Screen::Weakness => self.handle_weakness_key(key),
            Screen::Vocabulary => self.handle_vocabulary_key(key),
            Screen::Insights => self.handle_insights_key(key),
            Screen::MockExamWaiting => self.handle_mock_exam_waiting_key(key)?,
            Screen::MockExamReview => self.handle_mock_exam_review_key(key)?,
            Screen::Settings => self.handle_settings_key(key)?,
        }
        Ok(false)
    }

    fn handle_generating_key(&mut self, key: KeyEvent) -> Result<()> {
        if self.generation_error_message.is_none() {
            return Ok(());
        }
        match key.code {
            KeyCode::Enter | KeyCode::Char('r') | KeyCode::Char('R') => {
                match self.generation_context {
                    GenerationContext::Practice => self.perform_action(Action::RetryGeneration)?,
                    GenerationContext::MockExamBootstrap => self.start_mock_exam()?,
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_submitting_key(&mut self, key: KeyEvent) -> Result<()> {
        if self.submission_error_message.is_none() {
            return Ok(());
        }
        match key.code {
            KeyCode::Enter | KeyCode::Char('r') | KeyCode::Char('R') => {
                match self.submission_context {
                    SubmissionContext::Practice => self.perform_action(Action::RetrySubmission)?,
                    SubmissionContext::MockExam => self.submit_mock_exam()?,
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_escape(&mut self) -> Result<()> {
        match self.screen {
            Screen::Home => {}
            Screen::LevelSelect => self.return_home()?,
            Screen::TypeSelect => {
                self.screen = Screen::LevelSelect;
                self.status_line = String::from("已返回等级选择。");
            }
            Screen::Generating => {
                if let Some(task) = &self.generating_task {
                    task.cancel_flag.store(true, Ordering::Relaxed);
                }
                self.generating_task = None;
                match self.generation_context {
                    GenerationContext::Practice => {
                        self.open_type_screen()?;
                        self.status_line = String::from("已取消本次生成。");
                    }
                    GenerationContext::MockExamBootstrap => {
                        self.mock_exam_session = None;
                        self.return_home()?;
                        self.status_line = String::from("已取消本次模拟四六级考试生成。");
                    }
                }
            }
            Screen::Submitting => {
                if self.submission_context == SubmissionContext::MockExam {
                    self.status_line = String::from(
                        "整套模拟四六级考试评分正在进行或等待重试，当前不会返回可编辑作答界面。",
                    );
                } else {
                    if let Some(task) = &self.submitting_task {
                        task.cancel_flag.store(true, Ordering::Relaxed);
                    }
                    self.submitting_task = None;
                    self.screen = Screen::Practice;
                    self.status_line = String::from("已取消本次评分，返回作答界面。");
                }
            }
            Screen::Practice => {
                if self.mock_exam_session.is_some() {
                    if let Some(task) = &self.mock_exam_generating_task {
                        task.cancel_flag.store(true, Ordering::Relaxed);
                    }
                    self.mock_exam_session = None;
                    self.mock_exam_generating_task = None;
                    self.practice = None;
                    self.return_home()?;
                    self.status_line = String::from("已退出模拟四六级考试界面并返回首页。");
                } else {
                    self.open_type_screen()?;
                    self.status_line = String::from("已返回题型选择。");
                }
            }
            Screen::Result => {
                self.open_type_screen()?;
                self.status_line = String::from("已返回题型选择。");
            }
            Screen::History => self.return_home()?,
            Screen::Review => {
                self.screen = Screen::History;
                self.status_line = String::from("已返回刷题历史。");
            }
            Screen::Weakness => self.return_home()?,
            Screen::Vocabulary => self.return_home()?,
            Screen::Insights => self.return_home()?,
            Screen::MockExamWaiting => {
                if let Some(session) = &mut self.mock_exam_session {
                    session.resume_from_waiting();
                }
                self.screen = Screen::Practice;
                self.status_line = String::from("已返回模拟四六级考试界面。");
            }
            Screen::MockExamReview => {
                self.screen = Screen::History;
                self.status_line = String::from("已返回刷题历史。");
            }
            Screen::Settings => self.return_home()?,
        };
        Ok(())
    }

    fn handle_home_key(&mut self, key: KeyEvent) {
        let max_index = 5usize;
        match key.code {
            KeyCode::Left => {
                if self.home_menu_index % 2 == 1 {
                    self.home_menu_index -= 1;
                }
            }
            KeyCode::Right => {
                if self.home_menu_index % 2 == 0 {
                    self.home_menu_index = (self.home_menu_index + 1).min(max_index);
                }
            }
            KeyCode::Up => {
                if self.home_menu_index >= 2 {
                    self.home_menu_index -= 2;
                }
            }
            KeyCode::Down => {
                if self.home_menu_index <= 3 {
                    self.home_menu_index = (self.home_menu_index + 2).min(max_index);
                }
            }
            KeyCode::Enter => {
                let _ = self.perform_action(Action::HomeMenu(self.home_menu_index));
            }
            _ => {}
        }
    }

    fn handle_level_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Left | KeyCode::Char('h') => self.selected_level = LevelChoice::Cet4,
            KeyCode::Right | KeyCode::Char('l') => self.selected_level = LevelChoice::Cet6,
            KeyCode::Enter => {
                let _ = self.perform_action(Action::Level(self.selected_level));
            }
            _ => {}
        }
    }

    fn handle_type_key(&mut self, key: KeyEvent) -> Result<()> {
        let max_index = TypeChoice::all().len().saturating_sub(1);
        match key.code {
            KeyCode::Left => {
                if self.type_index % 2 == 1 {
                    self.type_index -= 1;
                }
            }
            KeyCode::Right => {
                if self.type_index % 2 == 0 && self.type_index < max_index {
                    self.type_index += 1;
                }
            }
            KeyCode::Up => {
                if self.type_index >= 2 {
                    self.type_index -= 2;
                }
            }
            KeyCode::Down => {
                if self.type_index + 2 <= max_index {
                    self.type_index += 2;
                }
            }
            KeyCode::Enter => {
                self.selected_type = TypeChoice::all()[self.type_index];
                self.start_generation()?;
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_practice_key(&mut self, key: KeyEvent) -> Result<()> {
        if self.mock_exam_session.is_some() {
            return self.handle_mock_exam_practice_key(key);
        }
        match key.code {
            KeyCode::F(8) => {
                self.open_type_screen()?;
                self.status_line = String::from("已返回题型选择。");
                return Ok(());
            }
            KeyCode::F(9) => {
                self.submit_practice()?;
                return Ok(());
            }
            _ => {}
        }
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('s') | KeyCode::Char('S'))
        {
            self.submit_practice()?;
            return Ok(());
        }
        if let Some(practice) = &mut self.practice {
            handle_practice_input_core(practice, key, false)?;
        }
        Ok(())
    }

    fn handle_mock_exam_waiting_key(&mut self, _key: KeyEvent) -> Result<()> {
        Ok(())
    }

    fn handle_mock_exam_practice_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::F(8) => {
                if let Some(task) = &self.mock_exam_generating_task {
                    task.cancel_flag.store(true, Ordering::Relaxed);
                }
                self.mock_exam_session = None;
                self.mock_exam_generating_task = None;
                self.practice = None;
                self.return_home()?;
                self.status_line = String::from("已退出模拟四六级考试并返回首页。");
                return Ok(());
            }
            KeyCode::F(9) => {
                self.submit_current_mock_exam_section()?;
                return Ok(());
            }
            KeyCode::F(10) => {
                self.submit_mock_exam()?;
                return Ok(());
            }
            _ => {}
        }
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('s') | KeyCode::Char('S'))
        {
            self.submit_current_mock_exam_section()?;
            return Ok(());
        }
        if let Some(session) = &self.mock_exam_session {
            if session.is_waiting() {
                return Ok(());
            }
        }
        if let Some(session) = &mut self.mock_exam_session {
            let can_switch = session.writing_locked;
            if matches!(key.code, KeyCode::F(1)) {
                session.switch_section(TypeChoice::Writing);
                return Ok(());
            }
            if matches!(key.code, KeyCode::F(2)) && can_switch && session.is_ready(TypeChoice::Translation) {
                session.switch_section(TypeChoice::Translation);
                return Ok(());
            }
            if matches!(key.code, KeyCode::F(3)) && can_switch && session.is_ready(TypeChoice::BankedCloze) {
                session.switch_section(TypeChoice::BankedCloze);
                return Ok(());
            }
            if matches!(key.code, KeyCode::F(4)) && can_switch && session.is_ready(TypeChoice::LongReading) {
                session.switch_section(TypeChoice::LongReading);
                return Ok(());
            }
            if matches!(key.code, KeyCode::F(5)) && can_switch && session.is_ready(TypeChoice::Careful1) {
                session.switch_section(TypeChoice::Careful1);
                return Ok(());
            }
            if matches!(key.code, KeyCode::F(6)) && can_switch && session.is_ready(TypeChoice::Careful2) {
                session.switch_section(TypeChoice::Careful2);
                return Ok(());
            }
            if !can_switch
                && matches!(
                    key.code,
                    KeyCode::F(2) | KeyCode::F(3) | KeyCode::F(4) | KeyCode::F(5) | KeyCode::F(6)
                )
            {
                self.status_line = String::from("模拟四六级考试第一步只能写作文，提交作文后才能切换其他题型。");
                return Ok(());
            }

            let active_type = session.active_section;
            if session.is_section_locked(active_type) {
                self.status_line = String::from("作文已锁定，不能再修改；请切换到其他题型继续作答。");
                return Ok(());
            }
            if let Some(section) = session.active_section_mut() {
                return handle_practice_input_core(&mut section.practice, key, true);
            }
        }
        Ok(())
    }

    fn handle_result_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Left => self.result_action_index = self.result_action_index.saturating_sub(1),
            KeyCode::Right => self.result_action_index = (self.result_action_index + 1).min(4),
            KeyCode::Char('1') => self.perform_action(Action::ResultViewAnalysis)?,
            KeyCode::Char('2') => self.perform_action(Action::ResultContinue)?,
            KeyCode::Char('3') => self.perform_action(Action::ResultRedo)?,
            KeyCode::Char('4') => self.perform_action(Action::ResultBackTypes)?,
            KeyCode::Char('5') => self.perform_action(Action::ResultBackHome)?,
            KeyCode::Enter => match self.result_action_index {
                0 => self.perform_action(Action::ResultViewAnalysis)?,
                1 => self.perform_action(Action::ResultContinue)?,
                2 => self.perform_action(Action::ResultRedo)?,
                3 => self.perform_action(Action::ResultBackTypes)?,
                _ => self.perform_action(Action::ResultBackHome)?,
            },
            KeyCode::PageUp => {
                self.result_detail_scroll = self.result_detail_scroll.saturating_sub(3);
            }
            KeyCode::PageDown => {
                self.result_detail_scroll = self.result_detail_scroll.saturating_add(3);
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_history_key(&mut self, key: KeyEvent) -> Result<()> {
        let active_empty = match self.history_tab {
            HistoryTab::Practice => self.history.is_empty(),
            HistoryTab::MockExam => self.mock_exam_history.is_empty(),
        };
        match key.code {
            KeyCode::Tab | KeyCode::Char('1') | KeyCode::Char('2') => {
                self.toggle_history_tab(key.code);
                self.pending_history_delete_attempt_id = None;
            }
            KeyCode::Left => {
                self.history_action_index = self.history_action_index.saturating_sub(1);
                self.pending_history_delete_attempt_id = None;
            }
            KeyCode::Right => {
                let max_index = if self.history_tab == HistoryTab::Practice {
                    3
                } else {
                    2
                };
                self.history_action_index = (self.history_action_index + 1).min(max_index);
                self.pending_history_delete_attempt_id = None;
            }
            KeyCode::Up => {
                if self.history_tab == HistoryTab::Practice {
                    self.history_index = self.history_index.saturating_sub(1);
                } else {
                    self.mock_exam_history_index = self.mock_exam_history_index.saturating_sub(1);
                }
                self.pending_history_delete_attempt_id = None;
            }
            KeyCode::Down => {
                if self.history_tab == HistoryTab::Practice {
                    if !self.history.is_empty() {
                        self.history_index = (self.history_index + 1).min(self.history.len() - 1);
                    }
                } else if !self.mock_exam_history.is_empty() {
                    self.mock_exam_history_index =
                        (self.mock_exam_history_index + 1).min(self.mock_exam_history.len() - 1);
                }
                self.pending_history_delete_attempt_id = None;
            }
            KeyCode::Enter if !active_empty => match self.history_tab {
                HistoryTab::Practice => match self.history_action_index {
                    0 => self.perform_action(Action::HistoryReview(self.history_index))?,
                    1 => self.perform_action(Action::HistoryRedo(self.history_index))?,
                    2 => self.perform_action(Action::HistoryDelete(self.history_index))?,
                    _ => self.perform_action(Action::BackHistory)?,
                },
                HistoryTab::MockExam => match self.history_action_index {
                    0 => self.perform_action(Action::MockExamHistoryReview(
                        self.mock_exam_history_index,
                    ))?,
                    1 => self.perform_action(Action::MockExamHistoryDelete(
                        self.mock_exam_history_index,
                    ))?,
                    _ => self.perform_action(Action::BackHistory)?,
                },
            },
            KeyCode::Char('r') if self.history_tab == HistoryTab::Practice && !active_empty => {
                self.perform_action(Action::HistoryRedo(self.history_index))?
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                if !active_empty {
                    if self.history_tab == HistoryTab::Practice {
                        self.perform_action(Action::HistoryDelete(self.history_index))?
                    } else {
                        self.perform_action(Action::MockExamHistoryDelete(
                            self.mock_exam_history_index,
                        ))?
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_review_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Left => self.review_action_index = 0,
            KeyCode::Right => self.review_action_index = 1,
            KeyCode::Char('r') => self.perform_action(Action::ReviewRedo)?,
            KeyCode::Enter => {
                if self.review_action_index == 0 {
                    self.perform_action(Action::ReviewRedo)?
                } else {
                    self.perform_action(Action::BackReview)?
                }
            }
            KeyCode::PageUp => {
                self.review_detail_scroll = self.review_detail_scroll.saturating_sub(3);
            }
            KeyCode::PageDown => {
                self.review_detail_scroll = self.review_detail_scroll.saturating_add(3);
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_weakness_key(&mut self, key: KeyEvent) {
        if self.weakness.is_empty() {
            return;
        }
        match key.code {
            KeyCode::Up => self.weakness_index = self.weakness_index.saturating_sub(1),
            KeyCode::Down => {
                self.weakness_index = (self.weakness_index + 1).min(self.weakness.len() - 1)
            }
            _ => {}
        }
    }

    fn handle_vocabulary_key(&mut self, key: KeyEvent) {
        if self.vocabulary.is_empty() {
            return;
        }
        match key.code {
            KeyCode::Up => self.vocabulary_index = self.vocabulary_index.saturating_sub(1),
            KeyCode::Down => {
                self.vocabulary_index = (self.vocabulary_index + 1).min(self.vocabulary.len() - 1)
            }
            _ => {}
        }
    }

    fn handle_insights_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Left => self.insights_tab = previous_insights_tab(self.insights_tab),
            KeyCode::Right | KeyCode::Tab => {
                self.insights_tab = next_insights_tab(self.insights_tab)
            }
            KeyCode::Char('1') => self.insights_tab = InsightsTab::PracticeWeakness,
            KeyCode::Char('2') => self.insights_tab = InsightsTab::MockExamWeakness,
            KeyCode::Char('3') => self.insights_tab = InsightsTab::Vocabulary,
            KeyCode::Up => match self.insights_tab {
                InsightsTab::PracticeWeakness => {
                    self.weakness_index = self.weakness_index.saturating_sub(1)
                }
                InsightsTab::MockExamWeakness => {
                    self.mock_exam_weakness_index = self.mock_exam_weakness_index.saturating_sub(1)
                }
                InsightsTab::Vocabulary => {
                    self.vocabulary_index = self.vocabulary_index.saturating_sub(1)
                }
            },
            KeyCode::Down => match self.insights_tab {
                InsightsTab::PracticeWeakness => {
                    if !self.weakness.is_empty() {
                        self.weakness_index = (self.weakness_index + 1).min(self.weakness.len() - 1);
                    }
                }
                InsightsTab::MockExamWeakness => {
                    if !self.mock_exam_weakness.is_empty() {
                        self.mock_exam_weakness_index =
                            (self.mock_exam_weakness_index + 1).min(self.mock_exam_weakness.len() - 1);
                    }
                }
                InsightsTab::Vocabulary => {
                    if !self.vocabulary.is_empty() {
                        self.vocabulary_index = (self.vocabulary_index + 1).min(self.vocabulary.len() - 1);
                    }
                }
            },
            _ => {}
        }
    }

    fn handle_mock_exam_review_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Up => {
                self.review_detail_scroll = self.review_detail_scroll.saturating_sub(2);
            }
            KeyCode::Down => {
                self.review_detail_scroll = self.review_detail_scroll.saturating_add(2);
            }
            KeyCode::PageUp => {
                self.review_detail_scroll = self.review_detail_scroll.saturating_sub(3);
            }
            KeyCode::PageDown => {
                self.review_detail_scroll = self.review_detail_scroll.saturating_add(3);
            }
            KeyCode::Home => {
                self.review_detail_scroll = 0;
            }
            KeyCode::Enter => {
                self.perform_action(Action::BackReview)?;
            }
            KeyCode::Esc => {
                self.perform_action(Action::BackReview)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_settings_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Up => self.settings_focus = self.settings_focus.saturating_sub(1),
            KeyCode::Down => self.settings_focus = (self.settings_focus + 1).min(3),
            KeyCode::Left | KeyCode::Right | KeyCode::Enter => match self.settings_focus {
                0 => self.perform_action(Action::ToggleTheme)?,
                1 => self.perform_action(Action::ToggleBackground)?,
                2 => self.perform_action(Action::TogglePalette)?,
                _ => self.perform_action(Action::BackSettings)?,
            },
            _ => {}
        }
        Ok(())
    }

    fn handle_mouse(&mut self, kind: MouseEventKind, x: u16, y: u16) -> Result<()> {
        match kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some(action) = self
                    .click_areas
                    .iter()
                    .rev()
                    .find(|area| contains(area.rect, x, y))
                    .map(|area| area.action.clone())
                {
                    self.perform_action(action)?;
                }
            }
            MouseEventKind::ScrollUp => match self.screen {
                Screen::Practice => {
                    if let Some(session) = &mut self.mock_exam_session {
                        if let Some(section) = session.active_section_mut() {
                            section.practice.passage_scroll =
                                section.practice.passage_scroll.saturating_sub(2);
                        }
                    } else if let Some(practice) = &mut self.practice {
                        practice.passage_scroll = practice.passage_scroll.saturating_sub(2);
                    }
                }
                Screen::Result => {
                    self.result_detail_scroll = self.result_detail_scroll.saturating_sub(2);
                }
                Screen::Review => {
                    self.review_detail_scroll = self.review_detail_scroll.saturating_sub(2);
                }
                Screen::MockExamReview => {
                    self.review_detail_scroll = self.review_detail_scroll.saturating_sub(2);
                }
                Screen::History
                    if (self.history_tab == HistoryTab::Practice && !self.history.is_empty())
                        || (self.history_tab == HistoryTab::MockExam
                            && !self.mock_exam_history.is_empty()) =>
                {
                    if self.history_tab == HistoryTab::Practice {
                        self.history_index = self.history_index.saturating_sub(1);
                    } else {
                        self.mock_exam_history_index =
                            self.mock_exam_history_index.saturating_sub(1);
                    }
                    self.pending_history_delete_attempt_id = None;
                }
                Screen::Weakness if !self.weakness.is_empty() => {
                    self.weakness_index = self.weakness_index.saturating_sub(1);
                }
                Screen::Vocabulary if !self.vocabulary.is_empty() => {
                    self.vocabulary_index = self.vocabulary_index.saturating_sub(1);
                }
                _ => {}
            },
            MouseEventKind::ScrollDown => match self.screen {
                Screen::Practice => {
                    if let Some(session) = &mut self.mock_exam_session {
                        if let Some(section) = session.active_section_mut() {
                            section.practice.passage_scroll =
                                section.practice.passage_scroll.saturating_add(2);
                        }
                    } else if let Some(practice) = &mut self.practice {
                        practice.passage_scroll = practice.passage_scroll.saturating_add(2);
                    }
                }
                Screen::Result => {
                    self.result_detail_scroll = self.result_detail_scroll.saturating_add(2);
                }
                Screen::Review => {
                    self.review_detail_scroll = self.review_detail_scroll.saturating_add(2);
                }
                Screen::MockExamReview => {
                    self.review_detail_scroll = self.review_detail_scroll.saturating_add(2);
                }
                Screen::History
                    if (self.history_tab == HistoryTab::Practice && !self.history.is_empty())
                        || (self.history_tab == HistoryTab::MockExam
                            && !self.mock_exam_history.is_empty()) =>
                {
                    if self.history_tab == HistoryTab::Practice {
                        self.history_index = (self.history_index + 1).min(self.history.len() - 1);
                    } else if !self.mock_exam_history.is_empty() {
                        self.mock_exam_history_index = (self.mock_exam_history_index + 1)
                            .min(self.mock_exam_history.len() - 1);
                    }
                    self.pending_history_delete_attempt_id = None;
                }
                Screen::Weakness if !self.weakness.is_empty() => {
                    self.weakness_index = (self.weakness_index + 1).min(self.weakness.len() - 1);
                }
                Screen::Vocabulary if !self.vocabulary.is_empty() => {
                    self.vocabulary_index =
                        (self.vocabulary_index + 1).min(self.vocabulary.len() - 1);
                }
                _ => {}
            },
            _ => {}
        }
        Ok(())
    }

    fn perform_action(&mut self, action: Action) -> Result<()> {
        match action {
            Action::HomeMenu(index) => match index {
                0 => {
                    self.home_mode = HomeMode::Practice;
                    self.screen = Screen::LevelSelect;
                    self.status_line = String::from("请选择普通刷题的等级。");
                }
                1 => {
                    self.home_mode = HomeMode::MockExam;
                    self.screen = Screen::LevelSelect;
                    self.status_line = String::from("请选择模拟四六级考试等级，确认后将先生成作文。");
                }
                2 => self.open_history()?,
                3 => self.open_weakness()?,
                4 => self.open_settings()?,
                5 => self.should_quit = true,
                _ => {}
            },
            Action::OpenInsightsTab(index) => {
                self.insights_tab = match index {
                    0 => InsightsTab::PracticeWeakness,
                    1 => InsightsTab::MockExamWeakness,
                    _ => InsightsTab::Vocabulary,
                };
            }
            Action::OpenHistoryTab(index) => {
                self.history_tab = if index == 0 {
                    HistoryTab::Practice
                } else {
                    HistoryTab::MockExam
                };
                self.history_action_index = 0;
                self.pending_history_delete_attempt_id = None;
            }
            Action::Level(level) => {
                self.selected_level = level;
                if self.home_mode == HomeMode::MockExam {
                    self.start_mock_exam()?;
                } else {
                    self.open_type_screen()?;
                }
            }
            Action::Type(type_choice) => {
                self.selected_type = type_choice;
                self.type_index = TypeChoice::all()
                    .iter()
                    .position(|item| *item == type_choice)
                    .unwrap_or(0);
                self.start_generation()?;
            }
            Action::RetryGeneration => {
                self.start_generation()?;
            }
            Action::RetrySubmission => {
                match self.submission_context {
                    SubmissionContext::Practice => self.start_submission_internal(true)?,
                    SubmissionContext::MockExam => self.submit_mock_exam_internal(true)?,
                }
            }
            Action::BackHome => self.return_home()?,
            Action::BackType => self.open_type_screen()?,
            Action::BackHistory => self.return_home()?,
            Action::BackWeakness => self.return_home()?,
            Action::BackVocabulary => self.return_home()?,
            Action::BackSettings => self.return_home()?,
            Action::BackReview => {
                if matches!(self.screen, Screen::MockExamReview) {
                    self.screen = Screen::History;
                    self.status_line = String::from("已返回刷题历史。");
                } else {
                    self.screen = self.review_back_screen;
                    self.status_line = if self.review_back_screen == Screen::History {
                        String::from("已返回刷题历史。")
                    } else {
                        String::from("已返回结果总览。")
                    };
                }
            }
            Action::HistoryReview(index) => {
                if let Some(item) = self.history.get(index) {
                    self.pending_history_delete_attempt_id = None;
                    let review = self.backend.review(&item.attempt_id)?;
                    self.sync_selection_from_question_set(&review.question_set);
                    self.review_detail_scroll = 0;
                    self.review = Some(ReviewBundle {
                        question_set: review.question_set,
                        result: review.result,
                        answers: review.answers,
                    });
                    self.review_back_screen = Screen::History;
                    self.review_action_index = 0;
                    self.status_line = String::from(
                        "PageUp/PageDown 或滚轮查看复盘详情，Enter/R 可重新作答，Esc 返回历史。",
                    );
                    self.screen = Screen::Review;
                }
            }
            Action::HistoryRedo(index) => {
                if let Some(item) = self.history.get(index) {
                    self.pending_history_delete_attempt_id = None;
                    let review = self.backend.review(&item.attempt_id)?;
                    self.sync_selection_from_question_set(&review.question_set);
                    self.practice = Some(PracticeState::new(review.question_set, true));
                    self.status_line = String::from("已载入历史题目，可直接重新作答。");
                    self.screen = Screen::Practice;
                }
            }
            Action::HistoryDelete(index) => {
                if let Some(item) = self.history.get(index).cloned() {
                    self.history_index = index.min(self.history.len().saturating_sub(1));
                    if self.pending_history_delete_attempt_id.as_deref() != Some(&item.attempt_id) {
                        self.pending_history_delete_attempt_id = Some(item.attempt_id.clone());
                        self.status_line = String::from(
                            "再次按 D 或点击“删除记录”确认删除；对应词汇、正确率和薄弱项会一起重算。",
                        );
                        return Ok(());
                    }

                    let response = self.backend.delete_history(&item.attempt_id)?;
                    self.pending_history_delete_attempt_id = None;
                    self.refresh_overview()?;
                    self.history = self.backend.history()?.history;
                    self.history_index =
                        self.history_index.min(self.history.len().saturating_sub(1));
                    self.weakness.clear();
                    self.vocabulary.clear();
                    self.review = None;
                    self.review_detail_scroll = 0;
                    self.status_line = if self.history.is_empty() {
                        String::from("记录已删除，历史为空；相关统计、词汇和薄弱项已重算。")
                    } else if response.deleted.question_set_deleted {
                        format!(
                            "已删除记录「{}」，题集与关联词汇统计已移除并完成重算。",
                            response.deleted.title
                        )
                    } else {
                        format!(
                            "已删除记录「{}」，统计、词汇与薄弱项已完成重算。",
                            response.deleted.title
                        )
                    };
                }
            }
            Action::HistorySelect(index) => {
                self.history_index = index.min(self.history.len().saturating_sub(1));
                self.pending_history_delete_attempt_id = None;
                self.status_line =
                    String::from("已选中历史记录，Enter 查看解析，R 重新作答，D 删除记录。");
            }
            Action::MockExamHistoryReview(index) => {
                if let Some(item) = self.mock_exam_history.get(index) {
                    let review = self.backend.mock_exam_review(&item.exam_id)?;
                    self.mock_exam_review = Some(review.mock_exam);
                    self.review_back_screen = Screen::History;
                    self.review_detail_scroll = 0;
                    self.status_line = String::from(
                        "Up/Down、PageUp/PageDown 或滚轮可滚动查看模拟四六级考试总评与各部分表现，Enter/Esc 返回历史。",
                    );
                    self.screen = Screen::MockExamReview;
                }
            }
            Action::MockExamHistoryDelete(index) => {
                if let Some(item) = self.mock_exam_history.get(index).cloned() {
                    let response = self.backend.delete_mock_exam(&item.exam_id)?;
                    self.refresh_overview()?;
                    self.mock_exam_history = self.backend.mock_exam_history()?.history;
                    self.mock_exam_history_index = self
                        .mock_exam_history_index
                        .min(self.mock_exam_history.len().saturating_sub(1));
                    self.mock_exam_weakness.clear();
                    self.vocabulary.clear();
                    self.status_line = if self.mock_exam_history.is_empty() {
                        String::from("模拟四六级考试记录已删除，相关词汇与模拟四六级考试弱势已完成重算。")
                    } else {
                        format!(
                            "已删除一条 {} 模拟四六级考试记录（{:.1} 分），相关统计已重算。",
                            format_level_label(&response.deleted.level),
                            response.deleted.total_score
                        )
                    };
                }
            }
            Action::MockExamHistorySelect(index) => {
                self.mock_exam_history_index =
                    index.min(self.mock_exam_history.len().saturating_sub(1));
                self.status_line = String::from("已选中模拟四六级考试记录，Enter 查看总评，D 删除记录。");
            }
            Action::MockExamSelectSection(type_choice) => {
                if let Some(session) = &mut self.mock_exam_session {
                    if type_choice != TypeChoice::Writing && !session.writing_locked {
                        self.status_line =
                            String::from("模拟四六级考试第一步只能写作文，提交作文后才能切换其他题型。");
                    } else if session.is_ready(type_choice) {
                        session.switch_section(type_choice);
                        self.screen = Screen::Practice;
                        self.status_line =
                            format!("已切换到 {}，可继续作答或修改答案。", type_choice.label());
                    } else {
                        self.status_line =
                            format!("{} 仍在后台生成，请稍候。", type_choice.label());
                    }
                }
            }
            Action::MockExamSubmit => self.submit_mock_exam()?,
            Action::SubmitPractice => {
                if self.mock_exam_session.is_some() {
                    self.submit_current_mock_exam_section()?;
                } else {
                    self.submit_practice()?;
                }
            }
            Action::PracticeBack => {
                if self.mock_exam_session.is_some() {
                    if let Some(task) = &self.mock_exam_generating_task {
                        task.cancel_flag.store(true, Ordering::Relaxed);
                    }
                    self.mock_exam_session = None;
                    self.mock_exam_generating_task = None;
                    self.practice = None;
                    self.return_home()?;
                    self.status_line = String::from("已退出模拟四六级考试并返回首页。");
                } else if self.practice.is_some() {
                    self.screen = Screen::Practice;
                    self.status_line = String::from("已返回作答界面。");
                } else {
                    self.open_type_screen()?;
                    self.status_line = String::from("已返回题型选择。");
                }
            }
            Action::PracticeSelectBlank(index) => {
                if let Some(session) = &mut self.mock_exam_session {
                    if let Some(section) = session.active_section_mut() {
                        section.practice.selected_blank =
                            index.min(section.practice.question_set.questions.len() - 1);
                        section.practice.submit_confirm_pending = false;
                    }
                } else if let Some(practice) = &mut self.practice {
                    practice.selected_blank = index.min(practice.question_set.questions.len() - 1);
                    practice.submit_confirm_pending = false;
                }
            }
            Action::PracticeSelectQuestion(index) => {
                if let Some(session) = &mut self.mock_exam_session {
                    if let Some(section) = session.active_section_mut() {
                        section.practice.selected_question =
                            index.min(section.practice.question_set.questions.len() - 1);
                        section.practice.submit_confirm_pending = false;
                        section.practice.sync_choice_cursor_to_current_answer();
                    }
                } else if let Some(practice) = &mut self.practice {
                    practice.selected_question =
                        index.min(practice.question_set.questions.len() - 1);
                    practice.submit_confirm_pending = false;
                    practice.sync_choice_cursor_to_current_answer();
                }
            }
            Action::PracticeAssign(answer) => {
                if let Some(session) = &mut self.mock_exam_session {
                    if let Some(section) = session.active_section_mut() {
                        if !section.locked {
                            section.practice.assign_answer(answer);
                        }
                    }
                } else if let Some(practice) = &mut self.practice {
                    practice.assign_answer(answer);
                }
            }
            Action::ResultContinue => self.start_generation()?,
            Action::ResultViewAnalysis => {
                self.review_back_screen = Screen::Result;
                self.review_action_index = 0;
                self.status_line = String::from(
                    "PageUp/PageDown 或滚轮查看完整解析，R 可重做，Esc 返回结果总览。",
                );
                self.screen = Screen::Review;
            }
            Action::ResultRedo => {
                if let Some(practice) = &self.practice {
                    let question_set = practice.question_set.clone();
                    self.sync_selection_from_question_set(&question_set);
                    self.practice = Some(PracticeState::new(question_set, true));
                    self.result_detail_scroll = 0;
                    self.status_line = String::from("已重置答案，可以重新作答。");
                    self.screen = Screen::Practice;
                }
            }
            Action::ResultBackTypes => {
                self.open_type_screen()?;
                self.status_line = String::from("已返回题型选择。");
            }
            Action::ResultBackHome => {
                self.return_home()?;
            }
            Action::ReviewRedo => {
                if let Some(bundle) = &self.review {
                    let question_set = bundle.question_set.clone();
                    self.sync_selection_from_question_set(&question_set);
                    self.practice = Some(PracticeState::new(question_set, true));
                    self.review_detail_scroll = 0;
                    self.status_line = String::from("已从复盘页重新进入作答。");
                    self.screen = Screen::Practice;
                }
            }
            Action::ToggleTheme => {
                self.settings.theme_mode = if self.settings.theme_mode == "dark" {
                    "light".to_string()
                } else {
                    "dark".to_string()
                };
                if self.settings.theme_mode == "light" && self.settings.background_mode != "opaque"
                {
                    self.settings.background_mode = "opaque".to_string();
                    self.backend
                        .set_setting("background_mode", &self.settings.background_mode)?;
                }
                self.backend
                    .set_setting("theme_mode", &self.settings.theme_mode)?;
                self.status_line = format!(
                    "已切换主题：{}。",
                    if self.settings.theme_mode == "dark" {
                        "深色"
                    } else {
                        "浅色"
                    }
                );
            }
            Action::ToggleBackground => {
                if self.settings.theme_mode == "light" {
                    self.settings.background_mode = "opaque".to_string();
                    self.backend
                        .set_setting("background_mode", &self.settings.background_mode)?;
                    self.status_line = String::from("浅色模式仅支持不透明背景。");
                    return Ok(());
                }
                self.settings.background_mode = if self.settings.background_mode == "opaque" {
                    "transparent".to_string()
                } else {
                    "opaque".to_string()
                };
                self.backend
                    .set_setting("background_mode", &self.settings.background_mode)?;
                self.status_line = format!(
                    "已切换背景：{}。",
                    if self.settings.background_mode == "opaque" {
                        "不透明"
                    } else {
                        "透明"
                    }
                );
            }
            Action::TogglePalette => {
                self.settings.palette_mode =
                    next_palette_mode(&self.settings.palette_mode).to_string();
                self.backend
                    .set_setting("palette_mode", &self.settings.palette_mode)?;
                self.status_line = format!(
                    "已切换配色方案：{}。",
                    palette_mode_label(&self.settings.palette_mode)
                );
            }
            Action::WeaknessSelect(index) => {
                self.weakness_index = index.min(self.weakness.len().saturating_sub(1));
                self.status_line = String::from("已选中薄弱项，右侧查看当前分析。");
            }
            Action::VocabularySelect(index) => {
                self.vocabulary_index = index.min(self.vocabulary.len().saturating_sub(1));
                self.status_line = String::from("已选中词汇，右侧查看详细信息。");
            }
            Action::MockWeaknessSelect(index) => {
                self.mock_exam_weakness_index =
                    index.min(self.mock_exam_weakness.len().saturating_sub(1));
                self.status_line = String::from("已选中模拟四六级考试薄弱项，右侧查看当前分析。");
            }
        }
        Ok(())
    }

    fn open_type_screen(&mut self) -> Result<()> {
        self.home_mode = HomeMode::Practice;
        self.generation_context = GenerationContext::Practice;
        let response = self.backend.type_stats(self.selected_level)?;
        self.type_cards = response.cards;
        self.type_index = TypeChoice::all()
            .iter()
            .position(|item| *item == self.selected_type)
            .unwrap_or(0);
        self.screen = Screen::TypeSelect;
        Ok(())
    }

    fn return_home(&mut self) -> Result<()> {
        self.refresh_overview()?;
        self.screen = Screen::Home;
        self.status_line = String::from("已回到首页。");
        Ok(())
    }

    fn start_mock_exam(&mut self) -> Result<()> {
        self.home_mode = HomeMode::MockExam;
        self.selected_type = TypeChoice::Writing;
        self.generation_context = GenerationContext::MockExamBootstrap;
        self.mock_exam_session = None;
        self.practice = None;
        self.result = None;
        self.review = None;
        self.mock_exam_review = None;
        self.generation_log.clear();
        self.start_generation()
    }

    fn start_generation(&mut self) -> Result<()> {
        let backend = BackendBridge::new()?;
        let level = self.selected_level;
        let type_choice = self.selected_type;
        self.generation_error_message = None;
        self.generation_sequence = self.generation_sequence.wrapping_add(1);
        let job_id = self.generation_sequence;
        let (tx, rx) = mpsc::channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.generation_phase = String::from("boot");
        self.generation_message = format!(
            "已进入 {} {} 生成流程，正在启动 AI 出题引擎。",
            level.label(),
            type_choice.label()
        );
        self.generation_log = vec![self.generation_message.clone()];
        self.generating_task = Some(GeneratingTask {
            job_id,
            started_at: Instant::now(),
            receiver: rx,
            cancel_flag: cancel_flag.clone(),
        });
        self.generating_tick = 0;
        self.screen = Screen::Generating;
        std::thread::spawn(move || {
            if let Err(err) =
                backend.stream_generate(level, type_choice, job_id, tx.clone(), cancel_flag)
            {
                let _ = tx.send(GenerationMessage::Finished {
                    job_id,
                    result: Err(err.to_string()),
                });
            }
        });
        Ok(())
    }

    fn start_mock_exam_background_generation(&mut self) -> Result<()> {
        let backend = BackendBridge::new()?;
        let level = self.selected_level;
        let (tx, rx) = mpsc::channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.mock_exam_generating_task = Some(MockExamGenerationTask {
            started_at: Instant::now(),
            receiver: rx,
            cancel_flag: cancel_flag.clone(),
        });
        self.push_generation_log(String::from("后台开始准备翻译与阅读各部分题目。"));

        std::thread::spawn(move || {
            let queue = [
                TypeChoice::Translation,
                TypeChoice::BankedCloze,
                TypeChoice::LongReading,
                TypeChoice::Careful1,
                TypeChoice::Careful2,
            ];
            for type_choice in queue {
                let mut attempt = 0usize;
                loop {
                    if cancel_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    attempt += 1;
                    let _ = tx.send(MockExamGenerationMessage::Progress {
                        type_choice,
                        message: format!("正在生成 {}。", type_choice.label()),
                    });
                    match backend.generate(level, type_choice) {
                        Ok(response) => {
                            let _ = tx.send(MockExamGenerationMessage::Ready {
                                type_choice,
                                question_set: response.question_set,
                            });
                            break;
                        }
                        Err(error) => {
                            let _ = tx.send(MockExamGenerationMessage::Retry {
                                type_choice,
                                attempt,
                                error: error.to_string(),
                            });
                            std::thread::sleep(Duration::from_secs(2));
                        }
                    }
                }
            }
            let _ = tx.send(MockExamGenerationMessage::Finished);
        });
        Ok(())
    }

    fn push_generation_log(&mut self, message: String) {
        if is_generation_heartbeat(&message) {
            if let Some(last) = self.generation_log.last_mut() {
                if is_generation_heartbeat(last) {
                    *last = message;
                    return;
                }
            }
        }
        self.generation_log.push(message);
        if self.generation_log.len() > 8 {
            let overflow = self.generation_log.len() - 8;
            self.generation_log.drain(0..overflow);
        }
    }

    fn push_submission_log(&mut self, message: String) {
        if is_generation_heartbeat(&message) {
            if let Some(last) = self.submission_log.last_mut() {
                if is_generation_heartbeat(last) {
                    *last = message;
                    return;
                }
            }
        }
        self.submission_log.push(message);
        if self.submission_log.len() > 8 {
            let overflow = self.submission_log.len() - 8;
            self.submission_log.drain(0..overflow);
        }
    }

    fn start_submission(&mut self) -> Result<()> {
        self.start_submission_internal(true)
    }

    fn start_submission_internal(&mut self, reset_retry: bool) -> Result<()> {
        let Some(practice) = self.practice.clone() else {
            return Ok(());
        };
        let backend = BackendBridge::new()?;
        self.submission_context = SubmissionContext::Practice;
        if reset_retry {
            self.submission_retry_count = 0;
        }
        self.submission_error_message = None;
        self.submission_sequence = self.submission_sequence.wrapping_add(1);
        let job_id = self.submission_sequence;
        let (tx, rx) = mpsc::channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let response_word_count = practice.response_text().split_whitespace().count();
        self.submission_phase = String::from("boot");
        self.submission_message = format!(
            "已提交 {} {}，检测到约 {} 词内容，正在启动 AI 评分与批注流程。",
            format_level_label(&practice.question_set.level),
            format_question_label(
                &practice.question_set.question_type,
                practice.question_set.slot
            ),
            response_word_count,
        );
        if reset_retry || self.submission_log.is_empty() {
            self.submission_log = vec![self.submission_message.clone()];
        } else {
            self.push_submission_log(self.submission_message.clone());
        }
        self.submitting_task = Some(SubmittingTask {
            job_id,
            started_at: Instant::now(),
            receiver: rx,
            cancel_flag: cancel_flag.clone(),
        });
        self.generating_tick = 0;
        self.screen = Screen::Submitting;

        std::thread::spawn(move || {
            if let Err(err) = backend.stream_submit(
                &practice.question_set.id,
                &practice.started_at_iso,
                &practice.answers,
                practice.is_history_retry,
                job_id,
                tx.clone(),
                cancel_flag,
            ) {
                let _ = tx.send(SubmissionMessage::Finished {
                    job_id,
                    result: Err(err.to_string()),
                });
            }
        });
        Ok(())
    }

    fn submit_current_mock_exam_section(&mut self) -> Result<()> {
        let Some(session) = &mut self.mock_exam_session else {
            return Ok(());
        };
        let active_type = session.active_section;
        let Some(section) = session.active_section_mut() else {
            return Ok(());
        };
        if section.practice.question_set.questions.is_empty() {
            if section.practice.response_text().trim().is_empty()
                && !section.practice.submit_confirm_pending
            {
                section.practice.submit_confirm_pending = true;
                self.status_line = if active_type == TypeChoice::Writing {
                    String::from(
                        "当前作文为空白；再按一次 F9 / Ctrl+S / 提交按钮，将按空白作文提交。",
                    )
                } else {
                    String::from(
                        "当前翻译为空白；再按一次 F9 / Ctrl+S / 保存按钮，将按空白作答保存该部分。",
                    )
                };
                return Ok(());
            }
            if active_type == TypeChoice::Writing {
                section.practice.submit_confirm_pending = false;
                section.locked = true;
                session.lock_writing();
                if session.all_sections_ready() {
                    session.switch_section(TypeChoice::Translation);
                    self.status_line = String::from(
                        "作文已提交并锁定，可自由切换其余题型继续完成模拟四六级考试。",
                    );
                } else {
                    session.pause_for_waiting();
                    self.generation_message = String::from("作文已提交，正在等待其余题目生成完成。");
                    self.status_line = String::from(
                        "作文已提交，其余题目仍在后台生成；等待页期间不计时。",
                    );
                    self.screen = Screen::MockExamWaiting;
                }
                return Ok(());
            }
        } else if section.practice.unanswered_count() > 0 && !section.practice.submit_confirm_pending {
            section.practice.submit_confirm_pending = true;
            self.status_line = format!(
                "当前还有 {} 题未作答，再按一次 Ctrl+S / Enter 将保留现有答案。",
                section.practice.unanswered_count()
            );
            return Ok(());
        }
        section.practice.submit_confirm_pending = false;
        self.status_line = format!(
            "{} 已保存当前答案，你可以继续切换其他题型或最终交卷。",
            active_type.label()
        );
        Ok(())
    }

    fn build_mock_exam_sections_payload(session: &MockExamSession) -> Vec<Value> {
        let ordered = [
            TypeChoice::Writing,
            TypeChoice::Translation,
            TypeChoice::BankedCloze,
            TypeChoice::LongReading,
            TypeChoice::Careful1,
            TypeChoice::Careful2,
        ];
        ordered
            .iter()
            .filter_map(|type_choice| session.section(*type_choice))
            .map(|section| {
                json!({
                    "question_set_id": section.practice.question_set.id,
                    "answers": section.practice.answers,
                    "duration_seconds": session.current_section_elapsed_seconds(section.type_choice),
                })
            })
            .collect()
    }

    fn submit_mock_exam(&mut self) -> Result<()> {
        self.submit_mock_exam_internal(true)
    }

    fn submit_mock_exam_internal(&mut self, reset_retry: bool) -> Result<()> {
        if let Some(existing) = &mut self.mock_exam_session {
            existing.pause_for_waiting();
        }
        let Some(session) = self.mock_exam_session.clone() else {
            return Ok(());
        };
        if reset_retry {
            self.submission_retry_count = 0;
        }
        if !session.writing_locked {
            if let Some(existing) = &mut self.mock_exam_session {
                existing.resume_from_waiting();
            }
            self.status_line = String::from("模拟四六级考试第一步必须先提交作文，作文提交后才能整套交卷。");
            return Ok(());
        }
        if !session.all_sections_ready() {
            if let Some(existing) = &mut self.mock_exam_session {
                existing.pause_for_waiting();
            }
            self.generation_message = String::from("其余题目尚未全部准备完毕，正在继续自动重试生成。");
            self.status_line = String::from("其余题目尚未准备完毕，等待页期间不计时。");
            self.screen = Screen::MockExamWaiting;
            return Ok(());
        }

        let backend = BackendBridge::new()?;
        let sections = Self::build_mock_exam_sections_payload(&session);
        self.submission_context = SubmissionContext::MockExam;
        self.submission_error_message = None;
        self.submission_sequence = self.submission_sequence.wrapping_add(1);
        let job_id = self.submission_sequence;
        let (tx, rx) = mpsc::channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.submission_phase = String::from("boot");
        self.submission_message = String::from("正在启动整套模拟四六级考试评分与总评流程。");
        if reset_retry || self.submission_log.is_empty() {
            self.submission_log = vec![self.submission_message.clone()];
        } else {
            self.push_submission_log(self.submission_message.clone());
        }
        self.mock_exam_submitting_task = Some(MockExamSubmittingTask {
            job_id,
            started_at: Instant::now(),
            receiver: rx,
            cancel_flag: cancel_flag.clone(),
        });
        self.generating_tick = 0;
        self.screen = Screen::Submitting;

        std::thread::spawn(move || {
            if let Err(err) = backend.stream_submit_mock_exam(
                session.level,
                &session.started_at_iso,
                session.elapsed_exam_seconds(),
                &sections,
                job_id,
                tx.clone(),
                cancel_flag,
            ) {
                let _ = tx.send(MockExamSubmissionMessage::Finished {
                    job_id,
                    result: Err(err.to_string()),
                });
            }
        });
        Ok(())
    }

    fn tick_mock_exam_session(&mut self) -> Result<()> {
        let Some(session) = &mut self.mock_exam_session else {
            return Ok(());
        };
        if matches!(self.screen, Screen::Submitting | Screen::MockExamReview) {
            return Ok(());
        }
        if !session.writing_locked && session.writing_remaining_seconds() <= 0 {
            session.writing_force_submitted = true;
            session.lock_writing();
            if session.all_sections_ready() {
                session.switch_section(TypeChoice::Translation);
                self.screen = Screen::Practice;
                self.status_line = String::from("作文时间已到，系统已自动锁定作文并切换到下一部分。");
            } else {
                session.pause_for_waiting();
                self.screen = Screen::MockExamWaiting;
                self.status_line =
                    String::from("作文时间已到，系统已自动提交作文；其余题目正在后台生成。");
            }
        }
        if session.total_remaining_seconds() <= 0 && !session.final_force_submitted {
            session.final_force_submitted = true;
            self.status_line = String::from("整套模拟四六级考试时间已到，系统正在按当前答案自动交卷。");
            self.submit_mock_exam()?;
        }
        Ok(())
    }

    fn submit_practice(&mut self) -> Result<()> {
        if let Some(practice) = &mut self.practice {
            if practice.question_set.questions.is_empty()
                && practice.response_text().trim().is_empty()
                && !practice.submit_confirm_pending
            {
                practice.submit_confirm_pending = true;
                self.status_line = String::from(
                    "当前写作/翻译为空白；再按一次 F9 / Ctrl+S / 提交按钮，将按空白作答交卷。",
                );
                return Ok(());
            }
            let unanswered = practice.unanswered_count();
            if unanswered > 0 && !practice.submit_confirm_pending {
                practice.submit_confirm_pending = true;
                self.status_line = format!(
                    "还有 {} 题未作答，再按一次 F9 / Ctrl+S / Enter 将按当前答案交卷。",
                    unanswered
                );
                return Ok(());
            }
            practice.submit_confirm_pending = false;
            if practice.question_set.questions.is_empty() {
                return self.start_submission();
            }
            let response = self.backend.submit(
                &practice.question_set.id,
                &practice.started_at_iso,
                &practice.answers,
                practice.is_history_retry,
            )?;
            let result = response.result.clone();
            self.review = Some(ReviewBundle {
                question_set: practice.question_set.clone(),
                result: result.clone(),
                answers: practice.answers.clone(),
            });
            self.review_back_screen = Screen::Result;
            self.result = Some(result);
            self.result_detail_scroll = 0;
            self.status_line = String::from(
                "Enter/1 查看解析，2 下一题，3 重做，4 题型，5 首页，PageUp/PageDown 可浏览摘要。",
            );
            self.screen = Screen::Result;
        }
        Ok(())
    }

    fn open_history(&mut self) -> Result<()> {
        self.history = self.backend.history()?.history;
        self.mock_exam_history = self.backend.mock_exam_history()?.history;
        self.history_index = 0;
        self.mock_exam_history_index = 0;
        self.history_tab = if self.history.is_empty() && !self.mock_exam_history.is_empty() {
            HistoryTab::MockExam
        } else {
            HistoryTab::Practice
        };
        self.history_action_index = 0;
        self.pending_history_delete_attempt_id = None;
        self.screen = Screen::History;
        self.status_line = String::from(
            "Tab/1/2 切换普通刷题与模拟四六级考试历史；上下切换，Enter 查看详情。",
        );
        Ok(())
    }

    fn open_weakness(&mut self) -> Result<()> {
        self.weakness = self.backend.weakness()?.weakness;
        self.mock_exam_weakness = self.backend.mock_exam_weakness()?.weakness;
        self.vocabulary = self.backend.vocabulary()?.vocabulary;
        self.weakness_index = 0;
        self.mock_exam_weakness_index = 0;
        self.vocabulary_index = 0;
        self.insights_tab = if !self.weakness.is_empty() {
            InsightsTab::PracticeWeakness
        } else if !self.mock_exam_weakness.is_empty() {
            InsightsTab::MockExamWeakness
        } else {
            InsightsTab::Vocabulary
        };
        self.screen = Screen::Insights;
        self.status_line = String::from("左右/Tab 或 1/2/3 切换普通弱势、模拟四六级考试弱势与词汇表。");
        Ok(())
    }

    fn open_vocabulary(&mut self) -> Result<()> {
        self.open_weakness()
    }

    fn toggle_history_tab(&mut self, code: KeyCode) {
        self.history_tab = match code {
            KeyCode::Char('1') => HistoryTab::Practice,
            KeyCode::Char('2') => HistoryTab::MockExam,
            _ => match self.history_tab {
                HistoryTab::Practice => HistoryTab::MockExam,
                HistoryTab::MockExam => HistoryTab::Practice,
            },
        };
        self.history_action_index = 0;
    }

    fn open_settings(&mut self) -> Result<()> {
        self.settings = self.backend.settings()?.settings;
        if force_light_mode_opaque(&mut self.settings) {
            self.backend
                .set_setting("background_mode", &self.settings.background_mode)?;
        }
        self.settings_focus = 0;
        self.screen = Screen::Settings;
        self.status_line = String::from("上下切换项目，左右或回车切换主题、背景与配色。");
        Ok(())
    }

    fn refresh_overview(&mut self) -> Result<()> {
        let response = self.backend.overview()?;
        self.overview = response.overview;
        self.settings = response.settings;
        if force_light_mode_opaque(&mut self.settings) {
            self.backend
                .set_setting("background_mode", &self.settings.background_mode)?;
        }
        Ok(())
    }

    fn sync_selection_from_question_set(&mut self, question_set: &QuestionSet) {
        self.selected_level = match question_set.level.as_str() {
            "cet6" => LevelChoice::Cet6,
            _ => LevelChoice::Cet4,
        };
        self.selected_type = match (
            question_set.question_type.as_str(),
            question_set.slot.unwrap_or_default(),
        ) {
            ("writing", _) => TypeChoice::Writing,
            ("banked_cloze", _) => TypeChoice::BankedCloze,
            ("long_reading", _) => TypeChoice::LongReading,
            ("careful_reading", 2) => TypeChoice::Careful2,
            ("careful_reading", _) => TypeChoice::Careful1,
            ("translation", _) => TypeChoice::Translation,
            _ => TypeChoice::BankedCloze,
        };
        self.type_index = TypeChoice::all()
            .iter()
            .position(|item| *item == self.selected_type)
            .unwrap_or(0);
    }

    fn draw(&mut self, frame: &mut Frame) {
        self.click_areas.clear();
        let palette = Palette::new(
            &self.settings.theme_mode,
            &self.settings.background_mode,
            &self.settings.palette_mode,
        );
        let area = frame.area();
        frame.render_widget(
            Block::default().style(Style::default().bg(palette.background)),
            area,
        );
        if area.width < MIN_TERMINAL_WIDTH || area.height < MIN_TERMINAL_HEIGHT {
            self.draw_terminal_size_hint(frame, area, palette);
            return;
        }
        match self.screen {
            Screen::Home => self.draw_home(frame, area, palette),
            Screen::LevelSelect => self.draw_level_select(frame, area, palette),
            Screen::TypeSelect => self.draw_type_select(frame, area, palette),
            Screen::Generating => self.draw_generating(frame, area, palette),
            Screen::Submitting => self.draw_submitting(frame, area, palette),
            Screen::Practice => self.draw_practice(frame, area, palette),
            Screen::Result => self.draw_result(frame, area, palette),
            Screen::History => self.draw_history(frame, area, palette),
            Screen::Review => self.draw_review(frame, area, palette),
            Screen::Weakness => self.draw_weakness(frame, area, palette),
            Screen::Vocabulary => self.draw_vocabulary(frame, area, palette),
            Screen::Insights => self.draw_insights(frame, area, palette),
            Screen::MockExamWaiting => self.draw_mock_exam_waiting(frame, area, palette),
            Screen::MockExamReview => self.draw_mock_exam_review(frame, area, palette),
            Screen::Settings => self.draw_settings(frame, area, palette),
        }
    }

    fn draw_home(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let total_attempts = self.overview.total_attempts;
        let total_mock_exams = self.overview.total_mock_exams;
        let total_cet4 = self.overview.total_cet4;
        let total_cet6 = self.overview.total_cet6;
        let recent_performance_percent = self.overview.recent_performance_percent;
        let recent_pace_percent = self.overview.recent_pace_percent;
        let raw_recent_accuracy_percent = self.overview.raw_recent_accuracy_percent;
        let raw_recent_duration_text = self.overview.raw_recent_duration_text.clone();
        let raw_accuracy_ratio = (raw_recent_accuracy_percent / 100.0).min(1.0);
        let recent_performance_ratio = (recent_performance_percent / 100.0).min(1.0);
        let recent_pace_ratio = (recent_pace_percent / 100.0).min(1.0);
        let most_common_type_label = self.overview.most_common_type_label.clone();
        let latest_weakness_text = self
            .overview
            .latest_weakness_updated_at
            .clone()
            .map(|value| format_iso_brief(&value))
            .unwrap_or_else(|| "薄弱项暂无更新".to_string());
        let performance_series = self.overview.recent_performance_series.clone();
        let pace_series = self.overview.recent_pace_series.clone();
        let mock_exam_score_series = self.overview.recent_mock_exam_score_series.clone();
        let mock_exam_pace_series = self.overview.recent_mock_exam_pace_series.clone();
        let outer = centered_rect(95, 94, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(9),
                Constraint::Length(6),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(outer);

        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("阅阶 CET", title_style(palette))),
                Line::from(Span::styled(
                    format!(
                        "AI 四六级阅读、写作与翻译训练 · {} · {} · 背景{}",
                        if self.settings.theme_mode == "dark" {
                            "深色"
                        } else {
                            "浅色"
                        },
                        palette_mode_label(&self.settings.palette_mode),
                        effective_background_mode_label(
                            &self.settings.theme_mode,
                            &self.settings.background_mode,
                        )
                    ),
                    Style::default().fg(palette.muted),
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[0],
        );

        let hero = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(chunks[1]);
        let hero_left = vec![
            Line::from(Span::styled("欢迎回来", title_style(palette))),
            Line::from(""),
            Line::from(format!(
                "最近 5 次表现指数 {:.1}，节奏匹配 {:.1}，当前状态：{}。",
                recent_performance_percent,
                recent_pace_percent,
                accuracy_band(recent_performance_ratio)
            )),
            Line::from(format!(
                "训练建议：{}",
                if total_attempts == 0 {
                    "先从四级选词填空开始，熟悉界面和出题风格。"
                } else if recent_performance_percent < 70.0 {
                    "先继续当前题型，优先看解析和词汇，拉稳基础正确率。"
                } else if recent_performance_percent < 85.0 {
                    "可以保持当前节奏，适度加入长篇阅读强化定位速度。"
                } else {
                    "状态不错，可以切到更高难度或继续做同题型巩固稳定性。"
                }
            )),
            Line::from(format!("薄弱项更新：{}", latest_weakness_text)),
        ];
        frame.render_widget(
            Paragraph::new(Text::from(hero_left))
                .wrap(Wrap { trim: false })
                .block(simple_block("训练概览", palette)),
            hero[0],
        );

        let hero_right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Length(5)])
            .split(hero[1]);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!("累计刷题：{} 题", total_attempts)),
                Line::from(format!("累计模拟四六级考试：{} 套", total_mock_exams)),
                Line::from(format!("最常练题型：{}", most_common_type_label)),
                Line::from(format!(
                    "原始均值：{} / {}",
                    format!("{:.1}%", raw_recent_accuracy_percent),
                    raw_recent_duration_text
                )),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("速览", palette)),
            hero_right[0],
        );
        frame.render_widget(
            Gauge::default()
                .block(simple_block(
                    &format!("四级 / 六级占比  {} / {}", total_cet4, total_cet6),
                    palette,
                ))
                .gauge_style(Style::default().fg(palette.accent).bg(palette.panel_alt))
                .ratio(self.overview.cet4_ratio.clamp(0.0, 1.0)),
            hero_right[1],
        );

        let stat_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 4),
                Constraint::Ratio(1, 4),
                Constraint::Ratio(1, 4),
                Constraint::Ratio(1, 4),
            ])
            .split(chunks[2]);
        self.draw_metric_box(
            frame,
            stat_chunks[0],
            palette,
            "总刷题数",
            &format!("{}", total_attempts),
            &format!("四级 {} / 六级 {} · 模考 {}", total_cet4, total_cet6, total_mock_exams),
            None,
        );
        self.draw_metric_box(
            frame,
            stat_chunks[1],
            palette,
            "表现指数",
            &format!("{:.1}", recent_performance_percent),
            &format!(
                "原始正确率 {:.1}% · 已按题型难度归一",
                raw_recent_accuracy_percent
            ),
            Some(raw_accuracy_ratio),
        );
        self.draw_metric_box(
            frame,
            stat_chunks[2],
            palette,
            "节奏匹配",
            &format!("{:.1}", recent_pace_percent),
            &format!("原始均时 {} · 已按题型用时归一", raw_recent_duration_text),
            Some(recent_pace_ratio),
        );
        self.draw_metric_box(
            frame,
            stat_chunks[3],
            palette,
            "推荐动作",
            if total_attempts == 0 {
                "开始刷题"
            } else {
                "继续训练"
            },
            "从右侧菜单进入",
            None,
        );

        let bottom = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Percentage(34),
                Constraint::Percentage(26),
            ])
            .split(chunks[3]);
        let trend_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Min(4),
            ])
            .split(bottom[0]);
        self.draw_line_trend_card(
            frame,
            trend_rows[0],
            palette,
            "表现指数走势",
            &performance_series,
            Some(&mock_exam_score_series),
            palette.success,
            palette.accent,
            TrendMetric::Index,
            Some(100.0),
        );
        self.draw_line_trend_card(
            frame,
            trend_rows[1],
            palette,
            "节奏匹配走势",
            &pace_series,
            Some(&mock_exam_pace_series),
            palette.warning,
            palette.accent,
            TrendMetric::Index,
            Some(100.0),
        );
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("结构分布", title_style(palette))),
                Line::from(""),
                Line::from(format!(
                    "四级 {} {}",
                    mini_ratio_bar(self.overview.cet4_ratio, 10),
                    format!("{:.0}%", self.overview.cet4_ratio * 100.0)
                )),
                Line::from(format!(
                    "六级 {} {}",
                    mini_ratio_bar(self.overview.cet6_ratio, 10),
                    format!("{:.0}%", self.overview.cet6_ratio * 100.0)
                )),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("训练分布", palette)),
            trend_rows[2],
        );

        let menu_items = [
            ("开始刷题", "等级与题型", Action::HomeMenu(0)),
            ("模拟四六级考试", "作文开局整套练", Action::HomeMenu(1)),
            ("刷题历史", "历史与复盘", Action::HomeMenu(2)),
            ("能力与词汇", "弱势与词汇", Action::HomeMenu(3)),
            ("设置", "主题与配色", Action::HomeMenu(4)),
            ("退出", "安全退出程序", Action::HomeMenu(5)),
        ];
        frame.render_widget(simple_block("快速入口", palette), bottom[1]);
        let menu_inner = simple_block("快速入口", palette).inner(bottom[1]);
        let menu_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
            ])
            .split(menu_inner);
        for row in 0..3 {
            let row_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(menu_rows[row]);
            for col in 0..2 {
                let index = row * 2 + col;
                let (title, subtitle, action) = &menu_items[index];
                let selected = self.home_menu_index == index;
                self.draw_home_menu_card(
                    frame,
                    row_chunks[col],
                    palette,
                    title,
                    subtitle,
                    selected,
                    action.clone(),
                );
            }
        }

        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("今日提示", title_style(palette))),
                Line::from(""),
                Line::from("1. 选词填空先判词性，再看固定搭配和上下文逻辑。"),
                Line::from("2. 长篇阅读先扫题干特征，再抓段落功能与同义改写。"),
                Line::from("3. 仔细阅读 1 多抓研究/结果，仔细阅读 2 多看态度/推断。"),
                Line::from("4. 写作翻译先保信息准确，再优化句式、衔接与表达。"),
                Line::from(""),
                Line::from(format!(
                    "当前配色：{}",
                    palette_mode_label(&self.settings.palette_mode)
                )),
                Line::from(format!(
                    "当前状态：{}",
                    accuracy_band(recent_performance_ratio)
                )),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("做题策略", palette)),
            bottom[2],
        );

        self.draw_status_line(frame, chunks[4], palette);
    }

    fn draw_level_select(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(70, 60, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(5),
                Constraint::Length(1),
            ])
            .split(outer);
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                if self.home_mode == HomeMode::MockExam {
                    "选择模拟四六级考试等级"
                } else {
                    "选择等级"
                },
                title_style(palette),
            )),
            Line::from(Span::styled(
                if self.home_mode == HomeMode::MockExam {
                    "左右方向键或鼠标点击切换，Enter 后将先生成作文。"
                } else {
                    "左右方向键或鼠标点击切换，Enter 确认。"
                },
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        let buttons = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
            .split(centered_rect(80, 80, chunks[1]));
        self.draw_action_button(
            frame,
            buttons[0],
            palette,
            "四级",
            self.selected_level == LevelChoice::Cet4,
            Action::Level(LevelChoice::Cet4),
        );
        self.draw_action_button(
            frame,
            buttons[1],
            palette,
            "六级",
            self.selected_level == LevelChoice::Cet6,
            Action::Level(LevelChoice::Cet6),
        );
        self.draw_status_line(frame, chunks[2], palette);
    }

    fn draw_type_select(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(94, 92, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(1),
            ])
            .split(outer);

        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                format!("{}题型选择", self.selected_level.label()),
                title_style(palette),
            )),
            Line::from(Span::styled(
                "方向键切换卡片，Enter 进入，整张卡片可点击。",
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        let rows = [chunks[1], chunks[2], chunks[3]];
        for row_index in 0..3 {
            let row_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
                .split(rows[row_index]);
            for col_index in 0..2 {
                let idx = row_index * 2 + col_index;
                if idx >= TypeChoice::all().len() {
                    continue;
                }
                let type_choice = TypeChoice::all()[idx];
                let selected = self.type_index == idx;
                let card = self
                    .type_cards
                    .iter()
                    .find(|item| item.key == type_choice.key())
                    .cloned()
                    .unwrap_or(TypeCardData {
                        key: type_choice.key().to_string(),
                        short_label: type_choice.label().to_string(),
                        label: type_choice.label().to_string(),
                        attempt_count: 0,
                        recent_accuracy_percent: 0.0,
                        recent_duration_text: "00:00".into(),
                        best_recent_accuracy_percent: 0.0,
                        recent_accuracy_series: vec![],
                        recent_duration_series: vec![],
                    });
                self.draw_type_card(
                    frame,
                    row_chunks[col_index],
                    palette,
                    &card,
                    selected,
                    type_choice,
                );
            }
        }

        self.draw_status_line(frame, chunks[4], palette);
    }

    fn draw_generating(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(86, 72, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Min(12),
                Constraint::Length(1),
            ])
            .split(outer);
        let elapsed = self
            .generating_task
            .as_ref()
            .map(|task| task.started_at.elapsed().as_secs())
            .unwrap_or(0);
        let active_index = generation_step_index(&self.generation_phase);
        let steps = [
            ("准备环境", "加载配置、数据库和薄弱项"),
            ("锁定规范", "整理四六级词数、题量与题型约束"),
            ("正式出题", "向 DeepSeek 发送真实出题请求"),
            ("校验修复", "按 JSON 结构和 CET 要求复检"),
            ("保存结果", "入库题目、解析与词汇"),
        ];
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                if matches!(
                    self.selected_type,
                    TypeChoice::Writing | TypeChoice::Translation
                ) {
                    "AI 正在生成真题级主观题"
                } else {
                    "AI 正在生成真题级阅读题"
                },
                title_style(palette),
            )),
            Line::from(Span::styled(
                format!(
                    "{} · {}{} · {} · {} · 已等待 {}",
                    self.selected_level.label(),
                    self.selected_type.section_label(),
                    self.selected_type
                        .passage_label()
                        .map(|label| format!(" · {}", label))
                        .unwrap_or_default(),
                    self.selected_type.label(),
                    self.selected_type
                        .estimated_generation_time(self.selected_level),
                    seconds_to_text(elapsed as i64)
                ),
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        let scan = generation_scan_frame(self.generating_tick);
        let phase_marker = generation_phase_marker(self.generating_tick);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("生成脉冲", title_style(palette))),
                Line::from(format!(
                    "{}  {}  {}",
                    phase_marker,
                    format_generation_phase(&self.generation_phase),
                    scan
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[1],
        );

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(44), Constraint::Percentage(56)])
            .split(chunks[2]);
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(7), Constraint::Min(6)])
            .split(body[0]);
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(7), Constraint::Min(7)])
            .split(body[1]);

        let mut status_lines = vec![
            Line::from(Span::styled("当前状态", title_style(palette))),
            Line::from(self.generation_message.clone()),
            Line::from(format!(
                "题型特征：{}",
                self.selected_type.brief(self.selected_level)
            )),
            Line::from(self.selected_type.recommended_time()),
            Line::from(self.selected_type.estimated_generation_time(self.selected_level)),
        ];
        if self.generation_error_message.is_some() {
            status_lines.push(Line::from("本页可直接重试，无需重新选择题型。"));
        }
        frame.render_widget(
            Paragraph::new(Text::from(status_lines))
                .wrap(Wrap { trim: false })
                .block(simple_block("", palette)),
            left[0],
        );

        let mut step_lines = vec![
            Line::from(Span::styled("阶段看板", title_style(palette))),
            Line::from(""),
        ];
        let active_marker = generation_phase_marker(self.generating_tick);
        for (index, (title, detail)) in steps.iter().enumerate() {
            let marker = if index < active_index {
                "●"
            } else if index == active_index {
                active_marker
            } else {
                "○"
            };
            step_lines.push(Line::from(format!("{} {}  {}", marker, title, detail)));
        }
        step_lines.push(Line::from(""));
        step_lines.push(Line::from("Esc 可取消返回；取消后不会保存半成品。"));
        frame.render_widget(
            Paragraph::new(Text::from(step_lines))
                .wrap(Wrap { trim: false })
                .block(simple_block("流程", palette)),
            left[1],
        );

        let orbit = generation_orbit_frame(self.generating_tick);
        let wave = generation_wave_frame(self.generating_tick);
        let wait_hint = if elapsed >= 90 {
            "等待较久：当前更可能是远端模型仍在生成，不是前端失去响应。"
        } else if elapsed >= 45 {
            "若是长篇阅读或修复阶段，等待时间拉长属于正常现象。"
        } else {
            "Rust 前端保持响应，DeepSeek 与题库校验仍在后台继续执行。"
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("AI 推理中", title_style(palette))),
                Line::from(format!(
                    "{}  {}",
                    orbit,
                    format_generation_phase(&self.generation_phase)
                )),
                Line::from(format!("{}  正在组合题面、选项、解析与词汇整理", wave)),
                Line::from(wait_hint),
                Line::from("实时心跳会持续刷新状态，长时间等待通常发生在远端生成与校验阶段。"),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("", palette)),
            right[0],
        );

        let error_mode = self.generation_error_message.is_some();
        let log_block = simple_block(if error_mode { "失败详情" } else { "最近日志" }, palette);
        let log_inner = log_block.inner(right[1]);
        frame.render_widget(log_block, right[1]);
        if error_mode && log_inner.height >= 4 {
            let sections = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(4), Constraint::Length(3)])
                .split(log_inner);
            let mut log_lines = vec![
                Line::from(Span::styled("本轮未成功完成", title_style(palette))),
                Line::from(""),
            ];
            if let Some(error) = &self.generation_error_message {
                log_lines.push(Line::from(error.clone()));
                log_lines.push(Line::from(""));
            }
            log_lines.push(Line::from("建议先按 R / Enter 原地重试；若多次失败，再按 Esc 返回。"));
            frame.render_widget(
                Paragraph::new(Text::from(log_lines))
                    .wrap(Wrap { trim: false }),
                sections[0],
            );
            let buttons = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(sections[1]);
            self.draw_action_button(
                frame,
                buttons[0],
                palette,
                "重试生成",
                true,
                Action::RetryGeneration,
            );
            self.draw_action_button(
                frame,
                buttons[1],
                palette,
                "返回题型",
                false,
                Action::BackType,
            );
        } else {
            let mut log_lines = vec![
                Line::from(Span::styled("最近日志", title_style(palette))),
                Line::from(""),
            ];
            for item in &self.generation_log {
                log_lines.push(Line::from(format!("- {}", item)));
            }
            if self.generation_log.is_empty() {
                log_lines.push(Line::from("正在等待第一条进度消息。"));
            }
            frame.render_widget(
                Paragraph::new(Text::from(log_lines))
                    .wrap(Wrap { trim: false }),
                log_inner,
            );
        }
        self.draw_status_line(frame, chunks[3], palette);
    }

    fn draw_submitting(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(86, 72, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Min(12),
                Constraint::Length(1),
            ])
            .split(outer);
        let is_mock_exam = self.submission_context == SubmissionContext::MockExam;
        let elapsed = if is_mock_exam {
            self.mock_exam_submitting_task
                .as_ref()
                .map(|task| task.started_at.elapsed().as_secs())
                .unwrap_or(0)
        } else {
            self.submitting_task
                .as_ref()
                .map(|task| task.started_at.elapsed().as_secs())
                .unwrap_or(0)
        };
        let active_index = submission_step_index(&self.submission_phase);
        let steps = if is_mock_exam {
            [
                ("准备整卷", "读取六个部分答案、分值与用时"),
                ("请求评分", "向 DeepSeek 发送整套模拟四六级考试评阅请求"),
                ("整理总评", "汇总分项得分、弱势点与整套建议"),
                ("保存结果", "写入历史、词汇与模拟四六级考试弱势"),
            ]
        } else {
            [
                ("准备评分", "读取答案、题型和评分维度"),
                ("请求评分", "向 DeepSeek 发送主观题评阅请求"),
                ("整理批注", "抽取错词、语病与逐句反馈"),
                ("保存结果", "写入记录、词汇与薄弱项"),
            ]
        };

        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                if is_mock_exam {
                    "AI 正在评分整套模拟四六级考试"
                } else {
                    "AI 正在评分与批注"
                },
                title_style(palette),
            )),
            Line::from(Span::styled(
                format!("已等待 {}", seconds_to_text(elapsed as i64)),
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        let scan = generation_scan_frame(self.generating_tick);
        let phase_marker = generation_phase_marker(self.generating_tick);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("评分脉冲", title_style(palette))),
                Line::from(format!(
                    "{}  {}  {}",
                    phase_marker,
                    format_submission_phase(&self.submission_phase),
                    scan
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[1],
        );

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(44), Constraint::Percentage(56)])
            .split(chunks[2]);
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(6), Constraint::Min(6)])
            .split(body[0]);
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(7), Constraint::Min(7)])
            .split(body[1]);

        let mut status_lines = vec![
            Line::from(Span::styled("当前状态", title_style(palette))),
            Line::from(self.submission_message.clone()),
            Line::from(""),
            Line::from(if is_mock_exam {
                "评分将生成：整套总评、分项得分、弱势分析与提升建议。".to_string()
            } else {
                "评分将生成：总评、错词、病句改写、逐句批注与高分版本。".to_string()
            }),
        ];
        if self.submission_error_message.is_some() {
            status_lines.push(Line::from(if is_mock_exam {
                "已交卷答案仍保留在内存中，系统不会返回可编辑作答界面。".to_string()
            } else {
                "你的作答内容仍保留在内存中，可直接重试评分。".to_string()
            }));
        }
        frame.render_widget(
            Paragraph::new(Text::from(status_lines))
                .wrap(Wrap { trim: false })
                .block(simple_block("", palette)),
            left[0],
        );

        let active_marker = generation_phase_marker(self.generating_tick);
        let mut step_lines = vec![
            Line::from(Span::styled("阶段看板", title_style(palette))),
            Line::from(""),
        ];
        for (index, (title, detail)) in steps.iter().enumerate() {
            let marker = if index < active_index {
                "●"
            } else if index == active_index {
                active_marker
            } else {
                "○"
            };
            step_lines.push(Line::from(format!("{} {}  {}", marker, title, detail)));
        }
        step_lines.push(Line::from(""));
        step_lines.push(Line::from(if is_mock_exam {
            "系统会在本页原地自动重试；最终交卷后不会再返回可编辑作答界面。".to_string()
        } else {
            "Esc 可取消返回；已取消时不会写入半成品。".to_string()
        }));
        frame.render_widget(
            Paragraph::new(Text::from(step_lines))
                .wrap(Wrap { trim: false })
                .block(simple_block("流程", palette)),
            left[1],
        );

        let orbit = generation_orbit_frame(self.generating_tick);
        let wave = generation_wave_frame(self.generating_tick);
        let wait_hint = if elapsed >= 90 {
            if is_mock_exam {
                "等待较久时，多数情况是远端模型仍在生成整套总评与分项分析。"
            } else {
                "等待较久时，多数情况是远端模型仍在生成评分与逐句批注。"
            }
        } else if elapsed >= 45 {
            if is_mock_exam {
                "整套模拟四六级考试会同时整合多个部分的结果，等待略长属于正常现象。"
            } else {
                "主观题评分通常比客观题更慢，因为还要整理错词、语病和高分版本。"
            }
        } else {
            if is_mock_exam {
                "Rust 前端保持响应，AI 整卷评分与总评整理仍在后台继续执行。"
            } else {
                "Rust 前端保持响应，AI 评分与批注仍在后台继续执行。"
            }
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    if is_mock_exam { "整卷评阅引擎" } else { "批注引擎" },
                    title_style(palette),
                )),
                Line::from(format!(
                    "{}  {}",
                    orbit,
                    format_submission_phase(&self.submission_phase)
                )),
                Line::from(format!(
                    "{}  {}",
                    wave,
                    if is_mock_exam {
                        "正在生成整套总评、分项得分与弱势分析"
                    } else {
                        "正在生成总评、逐句批注与改写参考"
                    }
                )),
                Line::from(wait_hint),
                Line::from(if is_mock_exam {
                    "实时心跳会持续刷新状态，自动重试会在本页原地完成。"
                } else {
                    "实时心跳会持续刷新状态。"
                }),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("", palette)),
            right[0],
        );

        let error_mode = self.submission_error_message.is_some();
        let log_block = simple_block(if error_mode { "失败详情" } else { "最近日志" }, palette);
        let log_inner = log_block.inner(right[1]);
        frame.render_widget(log_block, right[1]);
        if error_mode && log_inner.height >= 4 {
            let sections = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(4), Constraint::Length(3)])
                .split(log_inner);
            let mut log_lines = vec![
                Line::from(Span::styled("本轮评分未成功", title_style(palette))),
                Line::from(""),
            ];
            if let Some(error) = &self.submission_error_message {
                log_lines.push(Line::from(error.clone()));
                log_lines.push(Line::from(""));
            }
            log_lines.push(Line::from(if is_mock_exam {
                "建议先按 R / Enter 继续重试；整套交卷后不会再返回可编辑作答界面。"
                    .to_string()
            } else {
                "建议先按 R / Enter 重试；按 Esc 可返回继续修改答案。".to_string()
            }));
            frame.render_widget(
                Paragraph::new(Text::from(log_lines))
                    .wrap(Wrap { trim: false }),
                sections[0],
            );
            let buttons = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(sections[1]);
            self.draw_action_button(
                frame,
                buttons[0],
                palette,
                "重试评分",
                true,
                Action::RetrySubmission,
            );
            self.draw_action_button(
                frame,
                buttons[1],
                palette,
                if is_mock_exam { "继续重试" } else { "返回作答" },
                false,
                if is_mock_exam {
                    Action::RetrySubmission
                } else {
                    Action::PracticeBack
                },
            );
        } else {
            let mut log_lines = vec![
                Line::from(Span::styled("最近日志", title_style(palette))),
                Line::from(""),
            ];
            for item in &self.submission_log {
                log_lines.push(Line::from(format!("- {}", item)));
            }
            if self.submission_log.is_empty() {
                log_lines.push(Line::from("正在等待第一条评分进度消息。"));
            }
            frame.render_widget(
                Paragraph::new(Text::from(log_lines))
                    .wrap(Wrap { trim: false }),
                log_inner,
            );
        }
        self.draw_status_line(frame, chunks[3], palette);
    }

    fn draw_practice(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let practice = if let Some(session) = &self.mock_exam_session {
            let Some(section) = session.active_section() else {
                return;
            };
            section.practice.clone()
        } else {
            let Some(practice) = self.practice.clone() else {
                return;
            };
            practice
        };
        let root = centered_rect(98, 94, area);
        let header_height = if self.mock_exam_session.is_some() { 10 } else { 5 };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(header_height), Constraint::Min(20)])
            .split(root);
        if let Some(session) = self.mock_exam_session.clone() {
            self.draw_mock_exam_header(frame, chunks[0], palette, &session, &practice);
        } else {
            self.draw_practice_header(frame, chunks[0], palette, &practice);
        }
        if practice.question_set.questions.is_empty() {
            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(24), Constraint::Min(60)])
                .split(chunks[1]);
            if let Some(session) = self.mock_exam_session.clone() {
                self.draw_mock_exam_sidebar(frame, columns[0], palette, &session, &practice);
            } else {
                self.draw_practice_sidebar(frame, columns[0], palette, &practice);
            }
            self.draw_subjective_panel(frame, columns[1], palette, &practice);
        } else {
            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(24),
                    Constraint::Percentage(
                        if practice.question_set.question_type == "long_reading" {
                            52
                        } else {
                            46
                        },
                    ),
                    Constraint::Min(34),
                ])
                .split(chunks[1]);

            if let Some(session) = self.mock_exam_session.clone() {
                self.draw_mock_exam_sidebar(frame, columns[0], palette, &session, &practice);
            } else {
                self.draw_practice_sidebar(frame, columns[0], palette, &practice);
            }
            self.draw_passage_panel(frame, columns[1], palette, &practice);
            self.draw_answer_panel(frame, columns[2], palette, &practice);
        }
    }

    fn draw_mock_exam_header(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        session: &MockExamSession,
        practice: &PracticeState,
    ) {
        let parts = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(area);
        let stage_text = if !session.writing_locked {
            format!("作文阶段 · 剩余 {}", seconds_to_text(session.writing_remaining_seconds()))
        } else if session.is_waiting() {
            String::from("作文已锁定 · 其余题目后台生成中 · 当前等待不计时")
        } else {
            format!("整套剩余 {}", seconds_to_text(session.total_remaining_seconds()))
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    format!(
                        "{} 模拟四六级考试 · {}",
                        session.level.label(),
                        format_question_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot
                        )
                    ),
                    title_style(palette),
                )),
                Line::from(stage_text),
                Line::from(format!(
                    "已准备 {}/6 部分 · 当前可用 {}",
                    session.sections.len(),
                    session.active_section.label()
                )),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("模拟四六级考试进行中", palette)),
            parts[0],
        );

        frame.render_widget(simple_block("题型切换", palette), parts[1]);
        let inner = simple_block("题型切换", palette).inner(parts[1]);
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Length(4)])
            .split(inner);
        let row1 = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
            ])
            .split(rows[0]);
        let row2 = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
            ])
            .split(rows[1]);
        let buttons = [
            (TypeChoice::Writing, row1[0]),
            (TypeChoice::Translation, row1[1]),
            (TypeChoice::BankedCloze, row1[2]),
            (TypeChoice::LongReading, row2[0]),
            (TypeChoice::Careful1, row2[1]),
            (TypeChoice::Careful2, row2[2]),
        ];
        for (type_choice, rect) in buttons {
            let ready = session.is_ready(type_choice);
            let locked = session.is_section_locked(type_choice);
            let label = short_type_label(type_choice);
            let status = if session.active_section == type_choice {
                "当前作答"
            } else if locked {
                "已锁定"
            } else if ready {
                "可作答"
            } else {
                "准备中"
            };
            self.draw_action_button_multiline(
                frame,
                rect,
                palette,
                &[label, status],
                session.active_section == type_choice,
                Action::MockExamSelectSection(type_choice),
            );
        }
    }

    fn draw_mock_exam_sidebar(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        session: &MockExamSession,
        practice: &PracticeState,
    ) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(3),
                Constraint::Length(5),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(3),
            ])
            .split(area);

        frame.render_widget(
            Paragraph::new(Text::from(big_timer_lines(&seconds_to_text(
                session.total_remaining_seconds(),
            ))))
            .alignment(Alignment::Center)
            .style(
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            )
            .block(simple_block("EXAM", palette)),
            chunks[0],
        );

        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!(
                    "作文剩余 {}",
                    if session.writing_locked {
                        "已锁定".to_string()
                    } else {
                        seconds_to_text(session.writing_remaining_seconds())
                    }
                )),
                Line::from(format!(
                    "完成 {} / {}",
                    session.completed_sections(),
                    session.total_sections()
                )),
            ]))
            .block(simple_block("倒计时", palette)),
            chunks[1],
        );

        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!("当前：{}", practice.question_set.topic)),
                Line::from(format!(
                    "题型：{}",
                    format_question_label(
                        &practice.question_set.question_type,
                        practice.question_set.slot
                    )
                )),
                Line::from(format!("已就绪：{}", session.sections.len())),
                Line::from(format!("待生成：{}", session.pending_types.len())),
                Line::from(format!("重试中：{}", session.failed_types.len())),
                Line::from(if session.writing_locked {
                    "F2-F6 可快速切换题型".to_string()
                } else {
                    "先完成作文，其余题型后台生成".to_string()
                }),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("状态", palette)),
            chunks[2],
        );

        if session.writing_locked {
            self.draw_action_button(
                frame,
                chunks[3],
                palette,
                "保存当前部分",
                false,
                Action::SubmitPractice,
            );
            self.draw_action_button(
                frame,
                chunks[4],
                palette,
                "最终交卷",
                false,
                Action::MockExamSubmit,
            );
        } else {
            self.draw_action_button(
                frame,
                chunks[3],
                palette,
                "提交作文",
                false,
                Action::SubmitPractice,
            );
            frame.render_widget(
                Paragraph::new("作文提交后将锁定，随后可切换其余题型。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .block(simple_block("", palette)),
                chunks[4],
            );
        }

        frame.render_widget(
            Paragraph::new(if session.writing_locked {
                "F9 保存当前部分 | F10 最终交卷 | F1-F6 切换题型 | F8 退出"
            } else {
                "F9 提交作文 | Ctrl+S 同样可提交 | 作文 30 分钟后自动锁定"
            })
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: false })
            .block(simple_block("", palette)),
            chunks[5],
        );
    }

    fn draw_mock_exam_waiting(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(86, 74, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Min(14),
                Constraint::Length(1),
            ])
            .split(outer);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("模拟四六级考试等待页", title_style(palette))),
                Line::from(Span::styled(
                    "作文已提交，其余题目正在后台自动重试生成；本页等待时间不计入模拟四六级考试总时长。",
                    Style::default().fg(palette.muted),
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[0],
        );
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("AI 仍在准备", title_style(palette))),
                Line::from(format!(
                    "{}  {}",
                    generation_orbit_frame(self.generating_tick),
                    generation_wave_frame(self.generating_tick)
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[1],
        );
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(chunks[2]);
        let mut left_lines = vec![
            Line::from(Span::styled("后台状态", title_style(palette))),
            Line::from(""),
        ];
        if let Some(session) = &self.mock_exam_session {
            for type_choice in [
                TypeChoice::Translation,
                TypeChoice::BankedCloze,
                TypeChoice::LongReading,
                TypeChoice::Careful1,
                TypeChoice::Careful2,
            ] {
                let label = if session.is_ready(type_choice) {
                    format!("{}：已就绪", type_choice.label())
                } else if session.failed_types.contains(&type_choice) {
                    format!("{}：正在重试", type_choice.label())
                } else {
                    format!("{}：生成中", type_choice.label())
                };
                left_lines.push(Line::from(label));
            }
        }
        frame.render_widget(
            Paragraph::new(Text::from(left_lines))
                .wrap(Wrap { trim: false })
                .block(simple_block("部分进度", palette)),
            body[0],
        );
        let mut log_lines = vec![
            Line::from(Span::styled("最近日志", title_style(palette))),
            Line::from(""),
        ];
        for item in &self.generation_log {
            log_lines.push(Line::from(format!("- {}", item)));
        }
        frame.render_widget(
            Paragraph::new(Text::from(log_lines))
                .wrap(Wrap { trim: false })
                .block(simple_block("生成日志", palette)),
            body[1],
        );
        self.draw_status_line(frame, chunks[3], palette);
    }

    fn draw_practice_header(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let total_items = practice.question_set.questions.len().max(1);
        let hide_topic_in_practice = practice.question_set.question_type == "careful_reading";
        let parts = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(64), Constraint::Percentage(36)])
            .split(area);
        let meta_line = if hide_topic_in_practice {
            format!(
                "建议用时：{} | 题目数：{} | 主题将在复盘中查看",
                recommended_time_text(&practice.question_set.question_type),
                if practice.question_set.questions.is_empty() {
                    1
                } else {
                    practice.question_set.questions.len()
                }
            )
        } else {
            format!(
                "主题：{} | 建议用时：{} | 题目数：{}",
                practice.question_set.topic,
                recommended_time_text(&practice.question_set.question_type),
                if practice.question_set.questions.is_empty() {
                    1
                } else {
                    practice.question_set.questions.len()
                }
            )
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    format!(
                        "{} · {}{} · {}",
                        format_level_label(&practice.question_set.level),
                        format_question_group_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot,
                        ),
                        format_passage_slot_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot,
                        )
                        .map(|label| format!(" · {}", label))
                        .unwrap_or_default(),
                        format_question_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot,
                        )
                    ),
                    title_style(palette),
                )),
                Line::from(meta_line),
                Line::from(if practice.submit_confirm_pending {
                    "当前仍有未作答题目，再次提交将按现有答案交卷。".to_string()
                } else {
                    "支持键盘与鼠标作答；文章和解析区域均可滚动浏览。".to_string()
                }),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("当前训练", palette)),
            parts[0],
        );
        frame.render_widget(
            Gauge::default()
                .block(simple_block(
                    &format!(
                        "进度  {}/{}  · 未答 {}",
                        practice.answered_count(),
                        total_items,
                        practice.unanswered_count()
                    ),
                    palette,
                ))
                .gauge_style(Style::default().fg(palette.success).bg(palette.panel_alt))
                .ratio(practice.answered_count() as f64 / total_items as f64),
            parts[1],
        );
    }

    fn draw_result(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let Some(result) = self.result.clone() else {
            return;
        };
        let Some(practice) = self.practice.clone() else {
            return;
        };
        let outer = centered_rect(82, 96, area);
        let score_card_height = if result.subjective_evaluation.is_some() {
            10
        } else {
            9
        };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(score_card_height),
                Constraint::Length(5),
                Constraint::Min(12),
                Constraint::Length(4),
                Constraint::Length(1),
            ])
            .split(outer);

        let subjective = result.subjective_evaluation.clone();
        let headline_value = if let Some(evaluation) = &subjective {
            evaluation.score_15.round().clamp(0.0, 99.0) as i64
        } else {
            (result.accuracy * 100.0).round().clamp(0.0, 99.0) as i64
        };
        let badge_label = if let Some(evaluation) = &subjective {
            format!(
                "{:.1}/15  · 估算 {:.1} 分",
                evaluation.score_15, evaluation.estimated_reported_score
            )
        } else {
            format!("{}%  正确率", headline_value)
        };
        let score_main = if let Some(evaluation) = &subjective {
            format!("{:.1}/15", evaluation.score_15)
        } else {
            format!("{:.1}%", result.accuracy * 100.0)
        };
        let score_sub = if subjective.is_some() {
            "按 CET 主观题标准估算".to_string()
        } else {
            format!("{}/{}", result.correct_count, result.total_count)
        };
        let status_main = if let Some(evaluation) = &subjective {
            evaluation.grade_band.clone()
        } else {
            accuracy_band(result.accuracy).to_string()
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("本次作答已完成", title_style(palette))),
                Line::from(Span::styled(
                    format!(
                        "{} · {}{} · {}",
                        format_level_label(&practice.question_set.level),
                        format_question_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot
                        ),
                        format_passage_slot_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot,
                        )
                        .map(|label| format!(" · {}", label))
                        .unwrap_or_default(),
                        if let Some(evaluation) = &subjective {
                            evaluation.grade_band.clone()
                        } else {
                            accuracy_band(result.accuracy).to_string()
                        }
                    ),
                    Style::default().fg(palette.muted),
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[0],
        );

        frame.render_widget(
            Paragraph::new(Text::from({
                let mut lines = big_timer_lines(&format!("{:02}", headline_value));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    badge_label.clone(),
                    Style::default()
                        .fg(palette.accent)
                        .add_modifier(Modifier::BOLD),
                )));
                lines
            }))
            .alignment(Alignment::Center)
            .block(simple_block("结果速览", palette)),
            chunks[1],
        );

        let stats = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
            ])
            .split(chunks[2]);
        self.draw_metric_box(
            frame,
            stats[0],
            palette,
            if subjective.is_some() {
                "得分"
            } else {
                "正确率"
            },
            &score_main,
            &score_sub,
            Some(result.accuracy),
        );
        self.draw_metric_box(
            frame,
            stats[1],
            palette,
            "用时",
            &seconds_to_text(result.duration_seconds),
            "本次作答",
            None,
        );
        self.draw_metric_box(
            frame,
            stats[2],
            palette,
            "状态",
            &status_main,
            "查看完整解析",
            Some(0.62 + ((self.generating_tick % 8) as f64 / 20.0)),
        );

        let recommendation_lines = if result.recommendations.is_empty() {
            vec![Line::from("继续保持当前节奏。")]
        } else {
            result
                .recommendations
                .iter()
                .map(|item| Line::from(format!("• {}", item)))
                .collect::<Vec<_>>()
        };
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(chunks[3]);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("结果摘要", title_style(palette))),
                Line::from(""),
                Line::from(
                    subjective
                        .as_ref()
                        .map(|item| item.overall_feedback_zh.clone())
                        .unwrap_or_else(|| result.summary.clone()),
                ),
                Line::from(""),
                Line::from(format!(
                    "来源：{} / 模型：{}",
                    display_source_type(&practice.question_set.source_type),
                    practice.question_set.generator_model
                )),
                Line::from(format!(
                    "主题：{} / 建议用时：{}",
                    practice.question_set.topic,
                    recommended_time_text(&practice.question_set.question_type)
                )),
                Line::from(""),
                Line::from("PageUp/PageDown 或滚轮可继续浏览本区。"),
            ]))
            .wrap(Wrap { trim: false })
            .scroll((self.result_detail_scroll, 0))
            .block(simple_block("结果摘要", palette)),
            body[0],
        );

        frame.render_widget(
            Paragraph::new(Text::from({
                let mut lines = vec![
                    Line::from(Span::styled("下一步建议", title_style(palette))),
                    Line::from(""),
                ];
                lines.extend(recommendation_lines);
                lines.push(Line::from(""));
                lines.push(Line::from(format!(
                    "当前状态：{}",
                    if let Some(evaluation) = &subjective {
                        evaluation.grade_band.clone()
                    } else {
                        accuracy_band(result.accuracy).to_string()
                    }
                )));
                lines.push(Line::from(format!(
                    "作答用时：{}",
                    seconds_to_text(result.duration_seconds)
                )));
                lines.push(Line::from(""));
                lines.push(Line::from("Enter/1 查看完整解析"));
                lines.push(Line::from("2 下一题  3 重做本题"));
                lines.push(Line::from("4 返回题型  5 回首页"));
                lines
            }))
            .wrap(Wrap { trim: false })
            .block(simple_block("行动面板", palette)),
            body[1],
        );

        let buttons = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 5),
                Constraint::Ratio(1, 5),
                Constraint::Ratio(1, 5),
                Constraint::Ratio(1, 5),
                Constraint::Ratio(1, 5),
            ])
            .split(chunks[4]);
        self.draw_action_button(
            frame,
            buttons[0],
            palette,
            "查看解析",
            self.result_action_index == 0,
            Action::ResultViewAnalysis,
        );
        self.draw_action_button(
            frame,
            buttons[1],
            palette,
            "下一题",
            self.result_action_index == 1,
            Action::ResultContinue,
        );
        self.draw_action_button(
            frame,
            buttons[2],
            palette,
            "重做本题",
            self.result_action_index == 2,
            Action::ResultRedo,
        );
        self.draw_action_button(
            frame,
            buttons[3],
            palette,
            "题型选择",
            self.result_action_index == 3,
            Action::ResultBackTypes,
        );
        self.draw_action_button(
            frame,
            buttons[4],
            palette,
            "回到首页",
            self.result_action_index == 4,
            Action::ResultBackHome,
        );
        self.draw_status_line(frame, chunks[5], palette);
    }

    fn draw_history(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(94, 92, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(3),
                Constraint::Length(5),
                Constraint::Min(10),
                Constraint::Length(5),
                Constraint::Length(1),
            ])
            .split(outer);
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled("刷题历史", title_style(palette))),
            Line::from(Span::styled(
                "Tab/1/2 切换普通刷题与模拟四六级考试；上下切换记录，Enter 进入详情。",
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        let tab_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);
        self.draw_action_button(
            frame,
            tab_chunks[0],
            palette,
            "普通刷题",
            self.history_tab == HistoryTab::Practice,
            Action::OpenHistoryTab(0),
        );
        self.draw_action_button(
            frame,
            tab_chunks[1],
            palette,
            "模拟四六级考试",
            self.history_tab == HistoryTab::MockExam,
            Action::OpenHistoryTab(1),
        );

        let practice_tab = self.history_tab == HistoryTab::Practice;
        let metric_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
            ])
            .split(chunks[2]);
        if practice_tab {
            let recent_items = self.history.iter().take(5).collect::<Vec<_>>();
            let recent_count = recent_items.len().max(1) as f64;
            let recent_accuracy =
                recent_items.iter().map(|item| item.accuracy).sum::<f64>() / recent_count;
            let recent_duration = recent_items
                .iter()
                .map(|item| item.duration_seconds)
                .sum::<i64>()
                / recent_items.len().max(1) as i64;
            self.draw_metric_box(
                frame,
                metric_chunks[0],
                palette,
                "历史总数",
                &format!("{}", self.history.len()),
                "普通刷题记录",
                None,
            );
            self.draw_metric_box(
                frame,
                metric_chunks[1],
                palette,
                "近 5 次正确率",
                &format!("{:.1}%", recent_accuracy * 100.0),
                accuracy_band(recent_accuracy),
                Some(recent_accuracy),
            );
            self.draw_metric_box(
                frame,
                metric_chunks[2],
                palette,
                "近 5 次用时",
                &seconds_to_text(recent_duration),
                "平均作答时长",
                Some((recent_duration as f64 / 2400.0).clamp(0.0, 1.0)),
            );
        } else {
            let recent_items = self.mock_exam_history.iter().take(5).collect::<Vec<_>>();
            let recent_count = recent_items.len().max(1) as f64;
            let recent_score =
                recent_items.iter().map(|item| item.total_score).sum::<f64>() / recent_count;
            let recent_duration = recent_items
                .iter()
                .map(|item| item.duration_seconds)
                .sum::<i64>()
                / recent_items.len().max(1) as i64;
            self.draw_metric_box(
                frame,
                metric_chunks[0],
                palette,
                "历史总数",
                &format!("{}", self.mock_exam_history.len()),
                "模拟四六级考试记录",
                None,
            );
            self.draw_metric_box(
                frame,
                metric_chunks[1],
                palette,
                "近 5 次总分",
                &format!("{:.1}/100", recent_score),
                "模拟四六级考试综合分",
                Some((recent_score / 100.0).clamp(0.0, 1.0)),
            );
            self.draw_metric_box(
                frame,
                metric_chunks[2],
                palette,
                "近 5 次用时",
                &seconds_to_text(recent_duration),
                "整套作答时长",
                Some((recent_duration as f64 / 6000.0).clamp(0.0, 1.0)),
            );
        }

        if practice_tab && self.history.is_empty() {
            frame.render_widget(
                Paragraph::new("还没有历史记录。\n先去开始刷题，完成一次作答后这里会自动出现复盘档案。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .block(simple_block("历史为空", palette)),
                chunks[3],
            );
        } else if !practice_tab && self.mock_exam_history.is_empty() {
            frame.render_widget(
                Paragraph::new("还没有模拟四六级考试记录。\n完成一次模拟四六级考试后，这里会展示整套得分、用时与总评。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .block(simple_block("历史为空", palette)),
                chunks[3],
            );
        } else if practice_tab {
            let body = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[3]);
            let list_inner = simple_block("历史记录", palette).inner(body[0]);
            let items: Vec<ListItem> = self
                .history
                .iter()
                .map(|item| {
                    ListItem::new(vec![
                        Line::from(Span::styled(
                            item.title.clone(),
                            Style::default()
                                .fg(palette.text)
                                .add_modifier(Modifier::BOLD),
                        )),
                        Line::from(format!(
                            "{} | {} | {:.1}% | {}",
                            format_iso_brief(&item.submitted_at),
                            format_question_label(&item.question_type, item.slot),
                            item.accuracy * 100.0,
                            seconds_to_text(item.duration_seconds)
                        )),
                        Line::from(Span::styled(
                            truncate_text(&item.topic, 44),
                            Style::default().fg(palette.muted),
                        )),
                    ])
                })
                .collect();
            let mut state = ListState::default();
            state.select(Some(self.history_index));
            let list = List::new(items)
                .block(simple_block("历史记录", palette))
                .highlight_style(interactive_selected_style(palette));
            frame.render_stateful_widget(list, body[0], &mut state);
            let row_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Length(3); self.history.len()])
                .split(list_inner);
            for (index, rect) in row_chunks.iter().enumerate() {
                self.click_areas.push(ClickArea {
                    rect: *rect,
                    action: Action::HistorySelect(index),
                });
            }

            let selected = &self.history[self.history_index];
            let detail_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(6), Constraint::Min(9)])
                .split(body[1]);
            frame.render_widget(
                Paragraph::new(Text::from(vec![
                    Line::from(format!(
                        "正确率  {}  {:.1}%",
                        mini_ratio_bar(selected.accuracy, 10),
                        selected.accuracy * 100.0
                    )),
                    Line::from(format!(
                        "用时    {}  {}",
                        mini_ratio_bar(
                            (selected.duration_seconds as f64 / 2400.0).clamp(0.0, 1.0),
                            10
                        ),
                        seconds_to_text(selected.duration_seconds)
                    )),
                    Line::from(format!(
                        "重做    {}",
                        if selected.is_history_retry == 1 {
                            "是"
                        } else {
                            "否"
                        }
                    )),
                ]))
                .wrap(Wrap { trim: false })
                .block(simple_block("表现概览", palette)),
                detail_chunks[0],
            );
            frame.render_widget(
                Paragraph::new(Text::from(vec![
                    Line::from(format!("标题：{}", selected.title)),
                    Line::from(format!("主题：{}", selected.topic)),
                    Line::from(format!(
                        "题型：{} / 等级：{}",
                        format_question_label(&selected.question_type, selected.slot),
                        format_level_label(&selected.level)
                    )),
                    Line::from(format!("时间：{}", format_iso_brief(&selected.submitted_at))),
                    Line::from(format!(
                        "正确率：{:.1}% | 用时 {}",
                        selected.accuracy * 100.0,
                        seconds_to_text(selected.duration_seconds)
                    )),
                    Line::from(format!(
                        "得分：{}/{} | 重做：{}",
                        selected.correct_count,
                        selected.total_count,
                        if selected.is_history_retry == 1 {
                            "是"
                        } else {
                            "否"
                        }
                    )),
                    Line::from(""),
                    Line::from(format!("表现评估：{}", accuracy_band(selected.accuracy))),
                    Line::from(format!("题集 ID：{}", selected.question_set_id)),
                    Line::from(""),
                    Line::from("Enter 查看完整复盘，R 重新作答，D 删除该记录。"),
                ]))
                .wrap(Wrap { trim: false })
                .block(simple_block("当前记录", palette)),
                detail_chunks[1],
            );
        } else {
            let body = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[3]);
            let list_inner = simple_block("模拟四六级考试历史", palette).inner(body[0]);
            let items: Vec<ListItem> = self
                .mock_exam_history
                .iter()
                .map(|item| {
                    ListItem::new(vec![
                        Line::from(Span::styled(
                            format!("{} 模拟四六级考试", format_level_label(&item.level)),
                            Style::default()
                                .fg(palette.text)
                                .add_modifier(Modifier::BOLD),
                        )),
                        Line::from(format!(
                            "{} | 总分 {:.1}/100 | {}",
                            format_iso_brief(&item.submitted_at),
                            item.total_score,
                            seconds_to_text(item.duration_seconds)
                        )),
                        Line::from(Span::styled(
                            format!("考试 ID：{}", item.exam_id),
                            Style::default().fg(palette.muted),
                        )),
                    ])
                })
                .collect();
            let mut state = ListState::default();
            state.select(Some(self.mock_exam_history_index));
            let list = List::new(items)
                .block(simple_block("模拟四六级考试历史", palette))
                .highlight_style(interactive_selected_style(palette));
            frame.render_stateful_widget(list, body[0], &mut state);
            let row_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Length(3); self.mock_exam_history.len()])
                .split(list_inner);
            for (index, rect) in row_chunks.iter().enumerate() {
                self.click_areas.push(ClickArea {
                    rect: *rect,
                    action: Action::MockExamHistorySelect(index),
                });
            }

            let selected = &self.mock_exam_history[self.mock_exam_history_index];
            let detail_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(6), Constraint::Min(9)])
                .split(body[1]);
            frame.render_widget(
                Paragraph::new(Text::from(vec![
                    Line::from(format!(
                        "总分    {}  {:.1}/100",
                        mini_ratio_bar((selected.total_score / 100.0).clamp(0.0, 1.0), 10),
                        selected.total_score
                    )),
                    Line::from(format!(
                        "用时    {}  {}",
                        mini_ratio_bar(
                            (selected.duration_seconds as f64 / 6000.0).clamp(0.0, 1.0),
                            10
                        ),
                        seconds_to_text(selected.duration_seconds)
                    )),
                    Line::from(format!("等级    {}", format_level_label(&selected.level))),
                ]))
                .wrap(Wrap { trim: false })
                .block(simple_block("表现概览", palette)),
                detail_chunks[0],
            );
            frame.render_widget(
                Paragraph::new(Text::from(vec![
                    Line::from(format!("考试 ID：{}", selected.exam_id)),
                    Line::from(format!("等级：{}", format_level_label(&selected.level))),
                    Line::from(format!("开始：{}", format_iso_brief(&selected.started_at))),
                    Line::from(format!("结束：{}", format_iso_brief(&selected.submitted_at))),
                    Line::from(format!(
                        "总分：{:.1}/100 | 用时 {}",
                        selected.total_score,
                        seconds_to_text(selected.duration_seconds)
                    )),
                    Line::from(""),
                    Line::from("Enter 查看整套总评与分项表现，D 删除该记录。"),
                ]))
                .wrap(Wrap { trim: false })
                .block(simple_block("当前记录", palette)),
                detail_chunks[1],
            );
        }

        if practice_tab {
            let buttons = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Ratio(1, 4),
                    Constraint::Ratio(1, 4),
                    Constraint::Ratio(1, 4),
                    Constraint::Ratio(1, 4),
                ])
                .split(chunks[4]);
            self.draw_action_button(
                frame,
                buttons[0],
                palette,
                "查看解析",
                self.history_action_index == 0,
                Action::HistoryReview(self.history_index),
            );
            self.draw_action_button(
                frame,
                buttons[1],
                palette,
                "重新作答",
                self.history_action_index == 1,
                Action::HistoryRedo(self.history_index),
            );
            self.draw_action_button(
                frame,
                buttons[2],
                palette,
                "删除记录",
                self.history_action_index == 2,
                Action::HistoryDelete(self.history_index),
            );
            self.draw_action_button(
                frame,
                buttons[3],
                palette,
                "返回",
                self.history_action_index == 3,
                Action::BackHistory,
            );
        } else {
            let buttons = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Ratio(1, 3),
                    Constraint::Ratio(1, 3),
                    Constraint::Ratio(1, 3),
                ])
                .split(chunks[4]);
            self.draw_action_button(
                frame,
                buttons[0],
                palette,
                "查看总评",
                self.history_action_index == 0,
                Action::MockExamHistoryReview(self.mock_exam_history_index),
            );
            self.draw_action_button(
                frame,
                buttons[1],
                palette,
                "删除记录",
                self.history_action_index == 1,
                Action::MockExamHistoryDelete(self.mock_exam_history_index),
            );
            self.draw_action_button(
                frame,
                buttons[2],
                palette,
                "返回",
                self.history_action_index == 2,
                Action::BackHistory,
            );
        }
        self.draw_status_line(frame, chunks[5], palette);
    }

    fn draw_review(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let Some(bundle) = self.review.clone() else {
            return;
        };
        let from_history = self.review_back_screen == Screen::History;
        let subjective = bundle.result.subjective_evaluation.clone();
        let review_headline = if let Some(evaluation) = &subjective {
            format!("{:.1}/15", evaluation.score_15)
        } else {
            format!("{:.1}%", bundle.result.accuracy * 100.0)
        };
        let metric_main = if let Some(evaluation) = &subjective {
            format!("{:.1}/15", evaluation.score_15)
        } else {
            format!("{:.1}%", bundle.result.accuracy * 100.0)
        };
        let metric_sub = if let Some(evaluation) = &subjective {
            format!("估算 {:.1} 分", evaluation.estimated_reported_score)
        } else {
            format!(
                "{}/{}",
                bundle.result.correct_count, bundle.result.total_count
            )
        };
        let metric_status = if let Some(evaluation) = &subjective {
            evaluation.grade_band.clone()
        } else {
            accuracy_band(bundle.result.accuracy).to_string()
        };
        let outer = centered_rect(98, 98, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(5),
                Constraint::Min(18),
                Constraint::Length(4),
                Constraint::Length(1),
            ])
            .split(outer);
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                if from_history {
                    "历史复盘"
                } else {
                    "本次详解"
                },
                title_style(palette),
            )),
            Line::from(format!(
                "{}{} · {} · {}",
                format_question_label(&bundle.question_set.question_type, bundle.question_set.slot),
                format_passage_slot_label(
                    &bundle.question_set.question_type,
                    bundle.question_set.slot,
                )
                .map(|label| format!(" · {}", label))
                .unwrap_or_default(),
                review_headline,
                seconds_to_text(bundle.result.duration_seconds)
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        let metrics = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
            ])
            .split(chunks[1]);
        self.draw_metric_box(
            frame,
            metrics[0],
            palette,
            if subjective.is_some() {
                "得分"
            } else {
                "正确率"
            },
            &metric_main,
            &metric_sub,
            Some(bundle.result.accuracy),
        );
        self.draw_metric_box(
            frame,
            metrics[1],
            palette,
            "用时",
            &seconds_to_text(bundle.result.duration_seconds),
            if from_history {
                "历史作答时长"
            } else {
                "本次作答时长"
            },
            None,
        );
        self.draw_metric_box(
            frame,
            metrics[2],
            palette,
            "表现",
            &metric_status,
            "支持重新作答",
            None,
        );

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(chunks[2]);
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(9), Constraint::Min(9)])
            .split(body[0]);
        let overview_preview = if let Some(evaluation) = &subjective {
            format!(
                "{}\n\n{}",
                truncate_text(&evaluation.overall_feedback_zh, 180),
                truncate_text(&bundle.question_set.analysis.overall_summary, 120)
            )
        } else {
            format!(
                "{}\n\n{}",
                truncate_text(&bundle.question_set.analysis.overall_strategy, 180),
                truncate_text(&bundle.question_set.analysis.overall_summary, 120)
            )
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("复盘概览", title_style(palette))),
                Line::from(""),
                Line::from(overview_preview),
                Line::from(""),
                Line::from(format!(
                    "来源：{} / 模型：{}",
                    display_source_type(&bundle.question_set.source_type),
                    bundle.question_set.generator_model
                )),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("概览预览", palette)),
            left[0],
        );
        let vocab_lines = if bundle.question_set.vocabulary.is_empty() {
            vec![Line::from("暂无词汇数据。")]
        } else {
            bundle
                .question_set
                .vocabulary
                .iter()
                .take(6)
                .map(|item| {
                    Line::from(format!(
                        "{} ({}) - {}",
                        item.surface_form, item.level_hint, item.meaning_zh
                    ))
                })
                .collect::<Vec<_>>()
        };
        let mut snapshot_lines = vec![
            Line::from(Span::styled("能力与词汇", title_style(palette))),
            Line::from(""),
        ];
        if let Some(evaluation) = &subjective {
            for item in &evaluation.score_dimensions {
                snapshot_lines.push(Line::from(format!(
                    "{}  {}/{}",
                    format_skill_label(&item.name),
                    item.score,
                    item.max_score
                )));
            }
            snapshot_lines.push(Line::from(""));
            if evaluation.wrong_words.is_empty() {
                snapshot_lines.push(Line::from("暂无明显错词。"));
            } else {
                for item in evaluation.wrong_words.iter().take(4) {
                    snapshot_lines.push(Line::from(format!(
                        "{} -> {}",
                        item.original, item.corrected
                    )));
                }
            }
        } else {
            snapshot_lines.extend(build_skill_summary_lines(&bundle.result.question_results));
            snapshot_lines.push(Line::from(""));
            snapshot_lines.extend(vocab_lines.iter().take(4).cloned());
        }
        frame.render_widget(
            Paragraph::new(Text::from(snapshot_lines))
                .wrap(Wrap { trim: false })
                .block(simple_block("能力速览", palette)),
            left[1],
        );

        let mut lines = vec![
            Line::from(Span::styled("完整复盘档案", title_style(palette))),
            Line::from(""),
        ];
        lines.push(Line::from("复盘摘要："));
        lines.push(Line::from(""));
        lines.push(Line::from(if let Some(evaluation) = &subjective {
            evaluation.overall_feedback_zh.clone()
        } else {
            bundle.question_set.analysis.overall_strategy.clone()
        }));
        lines.push(Line::from(""));
        lines.push(Line::from(
            bundle.question_set.analysis.overall_summary.clone(),
        ));
        if !bundle.question_set.analysis.test_tips.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from("解题提示："));
            for tip in &bundle.question_set.analysis.test_tips {
                lines.push(Line::from(format!("• {}", tip)));
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::from(if subjective.is_some() {
            "评分维度："
        } else {
            "能力分布："
        }));
        lines.push(Line::from(""));
        if let Some(evaluation) = &subjective {
            for item in &evaluation.score_dimensions {
                lines.push(Line::from(format!(
                    "{}  {}/{}",
                    format_skill_label(&item.name),
                    item.score,
                    item.max_score
                )));
                lines.push(Line::from(format!("  {}", item.feedback_zh)));
            }
            lines.push(Line::from(""));
            lines.push(Line::from("错词纠正："));
            lines.push(Line::from(""));
            if evaluation.wrong_words.is_empty() {
                lines.push(Line::from("暂无明显错词。"));
            } else {
                for item in &evaluation.wrong_words {
                    lines.push(Line::from(format!(
                        "{} -> {} ({})",
                        item.original, item.corrected, item.meaning_zh
                    )));
                    lines.push(Line::from(format!("  {}", item.reason_zh)));
                }
            }
            lines.push(Line::from(""));
            lines.push(Line::from("逐句批注："));
            lines.push(Line::from(""));
            if evaluation.sentence_annotations.is_empty() {
                lines.push(Line::from("暂无逐句批注。"));
            } else {
                for (index, item) in evaluation.sentence_annotations.iter().enumerate() {
                    lines.push(Line::from(format!(
                        "句子 {}：{}",
                        index + 1,
                        item.original_sentence
                    )));
                    if !item.strengths_zh.trim().is_empty() {
                        lines.push(Line::from(format!("亮点：{}", item.strengths_zh)));
                    }
                    if !item.issues_zh.trim().is_empty() {
                        lines.push(Line::from(format!("问题：{}", item.issues_zh)));
                    }
                    if !item.revised_sentence.trim().is_empty() {
                        lines.push(Line::from(format!("参考改写：{}", item.revised_sentence)));
                    }
                    if !item.skill_tag.trim().is_empty() && item.skill_tag != "general" {
                        lines.push(Line::from(format!(
                            "关注点：{}",
                            format_skill_label(&item.skill_tag)
                        )));
                    }
                    lines.push(Line::from(""));
                }
            }
            lines.push(Line::from(""));
            if evaluation.sentence_rewrites.is_empty() {
                lines.push(Line::from("暂无病句改写。"));
            } else {
                lines.push(Line::from("病句改写："));
                lines.push(Line::from(""));
                for item in &evaluation.sentence_rewrites {
                    lines.push(Line::from(format!("原句：{}", item.original_sentence)));
                    lines.push(Line::from(format!("改写：{}", item.revised_sentence)));
                    lines.push(Line::from(format!("说明：{}", item.reason_zh)));
                    lines.push(Line::from(""));
                }
            }
            if !evaluation.high_score_version.trim().is_empty() {
                lines.push(Line::from(""));
                lines.push(Line::from("高分版本："));
                lines.push(Line::from(""));
                for line in evaluation.high_score_version.lines() {
                    lines.push(Line::from(line.to_string()));
                }
            }
        } else {
            lines.extend(build_skill_summary_lines(&bundle.result.question_results));
            lines.push(Line::from(""));
            lines.push(Line::from("复盘词汇："));
            lines.push(Line::from(""));
            lines.extend(vocab_lines);
            lines.push(Line::from(""));
            lines.push(Line::from("逐题复盘："));
            lines.push(Line::from(""));
            for (index, item) in bundle.result.question_results.iter().enumerate() {
                lines.push(Line::from(format!(
                    "{}. 你的答案 {} / 正确答案 {}",
                    index + 1,
                    blank_or_value(&item.user_answer),
                    item.correct_answer
                )));
                lines.push(Line::from(format!("   {}", item.explanation)));
            }
        }
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .wrap(Wrap { trim: false })
                .scroll((self.review_detail_scroll, 0))
                .block(simple_block("复盘详情（可滚动）", palette)),
            body[1],
        );
        let buttons = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
            .split(chunks[3]);
        self.draw_action_button(
            frame,
            buttons[0],
            palette,
            "重新作答",
            self.review_action_index == 0,
            Action::ReviewRedo,
        );
        self.draw_action_button(
            frame,
            buttons[1],
            palette,
            if from_history {
                "返回历史"
            } else {
                "返回结果"
            },
            self.review_action_index == 1,
            Action::BackReview,
        );
        self.draw_status_line(frame, chunks[4], palette);
    }

    fn draw_weakness(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(94, 92, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(outer);
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled("我的薄弱项", title_style(palette))),
            Line::from(Span::styled(
                "同题型训练达到足够样本后会自动更新；上下切换总结，右侧查看详情。",
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        if self.weakness.is_empty() {
            frame.render_widget(
                Paragraph::new("还没有足够数据来生成薄弱项。\n继续做同题型训练，样本累积后会自动生成能力总结。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .block(simple_block("暂无数据", palette)),
                chunks[1],
            );
        } else {
            let body = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
                .split(chunks[1]);
            let list_inner = simple_block("总结列表", palette).inner(body[0]);
            let items: Vec<ListItem> = self
                .weakness
                .iter()
                .map(|item| {
                    let dimension_preview = parse_dimension_preview(&item.dimensions_json);
                    ListItem::new(vec![
                        Line::from(format!(
                            "{} | {}",
                            format_iso_brief(&item.updated_at),
                            format_question_label(&item.question_type, None)
                        )),
                        Line::from(item.summary.clone()),
                        Line::from(dimension_preview),
                    ])
                })
                .collect();
            let mut state = ListState::default();
            state.select(Some(self.weakness_index));
            frame.render_stateful_widget(
                List::new(items)
                    .block(simple_block("总结列表", palette))
                    .highlight_style(interactive_selected_style(palette)),
                body[0],
                &mut state,
            );
            let row_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Length(3); self.weakness.len()])
                .split(list_inner);
            for (index, rect) in row_chunks.iter().enumerate() {
                self.click_areas.push(ClickArea {
                    rect: *rect,
                    action: Action::WeaknessSelect(index),
                });
            }

            let selected = &self.weakness[self.weakness_index];
            frame.render_widget(
                Paragraph::new(Text::from(vec![
                    Line::from(format!("更新时间：{}", format_iso_brief(&selected.updated_at))),
                    Line::from(format!("等级：{}", format_level_label(&selected.level))),
                    Line::from(format!(
                        "题型：{}",
                        format_question_label(&selected.question_type, None)
                    )),
                    Line::from(format!("样本数：{}", selected.based_on_attempt_count)),
                    Line::from(""),
                    Line::from(selected.summary.clone()),
                    Line::from(""),
                    Line::from(parse_dimension_preview(&selected.dimensions_json)),
                ]))
                .wrap(Wrap { trim: false })
                .block(simple_block("当前分析", palette)),
                body[1],
            );
        }
        self.draw_action_button(
            frame,
            chunks[2],
            palette,
            "返回",
            false,
            Action::BackWeakness,
        );
        self.draw_status_line(frame, chunks[3], palette);
    }

    fn draw_vocabulary(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(94, 92, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(outer);
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled("词汇表", title_style(palette))),
            Line::from(Span::styled(
                "按刷题历史中的高频重点词汇累计；上下切换词条，右侧查看详情。",
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        if self.vocabulary.is_empty() {
            frame.render_widget(
                Paragraph::new("还没有词汇数据。\n先完成练习和复盘，系统会自动累计高频重点词汇。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .block(simple_block("暂无数据", palette)),
                chunks[1],
            );
        } else {
            let body = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
                .split(chunks[1]);
            let list_inner = simple_block("词汇累计", palette).inner(body[0]);
            let items: Vec<ListItem> = self
                .vocabulary
                .iter()
                .map(|item| {
                    ListItem::new(vec![
                        Line::from(format!(
                            "{} ({}) | 频次 {} | 错题关联 {}",
                            item.surface_form,
                            item.level_hint,
                            item.frequency_score,
                            item.error_related_score
                        )),
                        Line::from(item.meaning_zh.clone()),
                    ])
                })
                .collect();
            let mut state = ListState::default();
            state.select(Some(self.vocabulary_index));
            frame.render_stateful_widget(
                List::new(items)
                    .block(simple_block("词汇累计", palette))
                    .highlight_style(interactive_selected_style(palette)),
                body[0],
                &mut state,
            );
            let row_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Length(2); self.vocabulary.len()])
                .split(list_inner);
            for (index, rect) in row_chunks.iter().enumerate() {
                self.click_areas.push(ClickArea {
                    rect: *rect,
                    action: Action::VocabularySelect(index),
                });
            }

            let selected = &self.vocabulary[self.vocabulary_index];
            let example = if selected.example_en.trim().is_empty() {
                "暂无例句".to_string()
            } else {
                selected.example_en.clone()
            };
            frame.render_widget(
                Paragraph::new(Text::from(vec![
                    Line::from(format!("词元：{}", selected.lemma)),
                    Line::from(format!("词形：{}", selected.surface_form)),
                    Line::from(format!("等级：{}", selected.level_hint)),
                    Line::from(format!("释义：{}", selected.meaning_zh)),
                    Line::from(format!("频次：{}", selected.frequency_score)),
                    Line::from(format!("错题关联：{}", selected.error_related_score)),
                    Line::from(format!("最近出现：{}", format_iso_brief(&selected.last_seen_at))),
                    Line::from(""),
                    Line::from(example),
                ]))
                .wrap(Wrap { trim: false })
                .block(simple_block("词汇详情", palette)),
                body[1],
            );
        }
        self.draw_action_button(
            frame,
            chunks[2],
            palette,
            "返回",
            false,
            Action::BackVocabulary,
        );
        self.draw_status_line(frame, chunks[3], palette);
    }

    fn draw_insights(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(94, 92, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(outer);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("能力与词汇", title_style(palette))),
                Line::from(Span::styled(
                    "左右/Tab 或 1/2/3 切换普通薄弱项、模拟四六级考试薄弱项与词汇表。",
                    Style::default().fg(palette.muted),
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[0],
        );
        let tabs = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(34),
                Constraint::Percentage(33),
                Constraint::Percentage(33),
            ])
            .split(chunks[1]);
        self.draw_action_button(
            frame,
            tabs[0],
            palette,
            "普通弱势",
            self.insights_tab == InsightsTab::PracticeWeakness,
            Action::OpenInsightsTab(0),
        );
        self.draw_action_button(
            frame,
            tabs[1],
            palette,
            "模拟四六级考试弱势",
            self.insights_tab == InsightsTab::MockExamWeakness,
            Action::OpenInsightsTab(1),
        );
        self.draw_action_button(
            frame,
            tabs[2],
            palette,
            "词汇表",
            self.insights_tab == InsightsTab::Vocabulary,
            Action::OpenInsightsTab(2),
        );

        match self.insights_tab {
            InsightsTab::PracticeWeakness => self.draw_weakness_content(frame, chunks[2], palette),
            InsightsTab::MockExamWeakness => {
                self.draw_mock_exam_weakness_content(frame, chunks[2], palette)
            }
            InsightsTab::Vocabulary => self.draw_vocabulary_content(frame, chunks[2], palette),
        }
        self.draw_action_button(
            frame,
            chunks[3],
            palette,
            "返回",
            false,
            Action::BackWeakness,
        );
        self.draw_status_line(frame, chunks[4], palette);
    }

    fn draw_weakness_content(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        if self.weakness.is_empty() {
            frame.render_widget(
                Paragraph::new("普通刷题的薄弱项还没有足够样本。\n继续做同题型训练后，这里会自动生成能力总结。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .block(simple_block("暂无数据", palette)),
                area,
            );
            return;
        }
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
            .split(area);
        let list_inner = simple_block("普通刷题薄弱项", palette).inner(body[0]);
        let items: Vec<ListItem> = self
            .weakness
            .iter()
            .map(|item| {
                let dimension_preview = parse_dimension_preview(&item.dimensions_json);
                ListItem::new(vec![
                    Line::from(format!(
                        "{} | {}",
                        format_iso_brief(&item.updated_at),
                        format_question_label(&item.question_type, None)
                    )),
                    Line::from(item.summary.clone()),
                    Line::from(dimension_preview),
                ])
            })
            .collect();
        let mut state = ListState::default();
        state.select(Some(self.weakness_index));
        frame.render_stateful_widget(
            List::new(items)
                .block(simple_block("总结列表", palette))
                .highlight_style(interactive_selected_style(palette)),
            body[0],
            &mut state,
        );
        let row_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(3); self.weakness.len()])
            .split(list_inner);
        for (index, rect) in row_chunks.iter().enumerate() {
            self.click_areas.push(ClickArea {
                rect: *rect,
                action: Action::WeaknessSelect(index),
            });
        }

        let selected = &self.weakness[self.weakness_index];
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!("更新时间：{}", format_iso_brief(&selected.updated_at))),
                Line::from(format!("等级：{}", format_level_label(&selected.level))),
                Line::from(format!(
                    "题型：{}",
                    format_question_label(&selected.question_type, None)
                )),
                Line::from(format!("样本数：{}", selected.based_on_attempt_count)),
                Line::from(""),
                Line::from(selected.summary.clone()),
                Line::from(""),
                Line::from(parse_dimension_preview(&selected.dimensions_json)),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("当前分析", palette)),
            body[1],
        );
    }

    fn draw_mock_exam_weakness_content(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
    ) {
        if self.mock_exam_weakness.is_empty() {
            frame.render_widget(
                Paragraph::new("模拟四六级考试薄弱项还没有足够样本。\n完成至少几次模拟四六级考试后，这里会生成整套能力总结。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .block(simple_block("暂无数据", palette)),
                area,
            );
            return;
        }
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
            .split(area);
        let list_inner = simple_block("模拟四六级考试薄弱项", palette).inner(body[0]);
        let items: Vec<ListItem> = self
            .mock_exam_weakness
            .iter()
            .map(|item| {
                ListItem::new(vec![
                    Line::from(format!(
                        "{} | {}",
                        format_iso_brief(&item.updated_at),
                        format_level_label(&item.level)
                    )),
                    Line::from(item.summary.clone()),
                    Line::from(parse_dimension_preview(&item.dimensions_json)),
                ])
            })
            .collect();
        let mut state = ListState::default();
        state.select(Some(self.mock_exam_weakness_index));
        frame.render_stateful_widget(
            List::new(items)
                .block(simple_block("总结列表", palette))
                .highlight_style(interactive_selected_style(palette)),
            body[0],
            &mut state,
        );
        let row_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(3); self.mock_exam_weakness.len()])
            .split(list_inner);
        for (index, rect) in row_chunks.iter().enumerate() {
            self.click_areas.push(ClickArea {
                rect: *rect,
                action: Action::MockWeaknessSelect(index),
            });
        }

        let selected = &self.mock_exam_weakness[self.mock_exam_weakness_index];
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!("更新时间：{}", format_iso_brief(&selected.updated_at))),
                Line::from(format!("等级：{}", format_level_label(&selected.level))),
                Line::from(format!("样本数：{}", selected.based_on_exam_count)),
                Line::from(""),
                Line::from(selected.summary.clone()),
                Line::from(""),
                Line::from(parse_dimension_preview(&selected.dimensions_json)),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("当前分析", palette)),
            body[1],
        );
    }

    fn draw_vocabulary_content(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        if self.vocabulary.is_empty() {
            frame.render_widget(
                Paragraph::new("还没有词汇数据。\n先完成练习和复盘，系统会自动累计高频重点词汇。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .block(simple_block("暂无数据", palette)),
                area,
            );
            return;
        }
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
            .split(area);
        let list_inner = simple_block("词汇累计", palette).inner(body[0]);
        let items: Vec<ListItem> = self
            .vocabulary
            .iter()
            .map(|item| {
                ListItem::new(vec![
                    Line::from(format!(
                        "{} ({}) | 频次 {} | 错题关联 {}",
                        item.surface_form,
                        item.level_hint,
                        item.frequency_score,
                        item.error_related_score
                    )),
                    Line::from(item.meaning_zh.clone()),
                ])
            })
            .collect();
        let mut state = ListState::default();
        state.select(Some(self.vocabulary_index));
        frame.render_stateful_widget(
            List::new(items)
                .block(simple_block("词汇累计", palette))
                .highlight_style(interactive_selected_style(palette)),
            body[0],
            &mut state,
        );
        let row_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(2); self.vocabulary.len()])
            .split(list_inner);
        for (index, rect) in row_chunks.iter().enumerate() {
            self.click_areas.push(ClickArea {
                rect: *rect,
                action: Action::VocabularySelect(index),
            });
        }

        let selected = &self.vocabulary[self.vocabulary_index];
        let example = if selected.example_en.trim().is_empty() {
            "暂无例句".to_string()
        } else {
            selected.example_en.clone()
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!("词元：{}", selected.lemma)),
                Line::from(format!("词形：{}", selected.surface_form)),
                Line::from(format!("等级：{}", selected.level_hint)),
                Line::from(format!("释义：{}", selected.meaning_zh)),
                Line::from(format!("频次：{}", selected.frequency_score)),
                Line::from(format!("错题关联：{}", selected.error_related_score)),
                Line::from(format!("最近出现：{}", format_iso_brief(&selected.last_seen_at))),
                Line::from(""),
                Line::from(example),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("词汇详情", palette)),
            body[1],
        );
    }

    fn draw_mock_exam_review(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let Some(review) = self.mock_exam_review.clone() else {
            return;
        };
        let outer = centered_rect(96, 96, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(5),
                Constraint::Min(16),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(outer);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("模拟四六级考试总评", title_style(palette))),
                Line::from(format!(
                    "{} · 总分 {:.1}/100 · 用时 {}",
                    format_level_label(&review.level),
                    review.total_score,
                    seconds_to_text(review.duration_seconds)
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[0],
        );

        let metric_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
                Constraint::Ratio(1, 3),
            ])
            .split(chunks[1]);
        self.draw_metric_box(
            frame,
            metric_chunks[0],
            palette,
            "总分",
            &format!("{:.1}/100", review.total_score),
            "整套模拟四六级考试",
            Some((review.total_score / 100.0).clamp(0.0, 1.0)),
        );
        self.draw_metric_box(
            frame,
            metric_chunks[1],
            palette,
            "用时",
            &seconds_to_text(review.duration_seconds),
            "总时长",
            Some((review.duration_seconds as f64 / 6000.0).clamp(0.0, 1.0)),
        );
        self.draw_metric_box(
            frame,
            metric_chunks[2],
            palette,
            "弱势点",
            &format!("{}", review.weakness_tags.len()),
            "见下方总评",
            None,
        );

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(chunks[2]);
        let mut left_lines = vec![
            Line::from(Span::styled("分项得分", title_style(palette))),
            Line::from(""),
        ];
        for (key, value) in &review.score_breakdown {
            left_lines.push(Line::from(format!(
                "{}：{:.1}",
                format_mock_exam_section_key(key),
                value
            )));
        }
        left_lines.push(Line::from(""));
        if review.recommendations.is_empty() {
            left_lines.push(Line::from("暂无建议。"));
        } else {
            for item in &review.recommendations {
                left_lines.push(Line::from(format!("• {}", item)));
            }
        }
        frame.render_widget(
            Paragraph::new(Text::from(left_lines))
                .wrap(Wrap { trim: false })
                .scroll((self.review_detail_scroll, 0))
                .block(simple_block("分数与建议", palette)),
            body[0],
        );

        let mut right_lines = vec![
            Line::from(Span::styled("整套复盘", title_style(palette))),
            Line::from(""),
            Line::from(review.summary.clone()),
            Line::from(""),
        ];
        if review.weakness_tags.is_empty() {
            right_lines.push(Line::from("暂无单独弱势标签。"));
        } else {
            right_lines.push(Line::from(format!(
                "薄弱点：{}",
                review
                    .weakness_tags
                    .iter()
                    .map(|item| format_skill_label(item))
                    .collect::<Vec<_>>()
                    .join("、")
            )));
            right_lines.push(Line::from(""));
        }
        right_lines.push(Line::from("各部分概览："));
        right_lines.push(Line::from(""));
        for section in &review.sections {
            let section_label = format_question_label(&section.question_type, section.slot);
            let section_score = if let Some(subjective) = &section.result.subjective_evaluation {
                format!(
                    "{:.1}/15 · 估算 {:.1} 分",
                    subjective.score_15, subjective.estimated_reported_score
                )
            } else {
                format!(
                    "{}/{} · {:.1}%",
                    section.result.correct_count,
                    section.result.total_count,
                    section.result.accuracy * 100.0
                )
            };
            right_lines.push(Line::from(format!(
                "{}  {}  ·  用时 {}",
                section_label,
                section_score,
                seconds_to_text(section.result.duration_seconds),
            )));
            right_lines.push(Line::from(format!(
                "  {}",
                truncate_text(&section.result.summary, 88)
            )));
            if let Some(subjective) = &section.result.subjective_evaluation {
                if !subjective.weakness_tags.is_empty() {
                    right_lines.push(Line::from(format!(
                        "  薄弱点：{}",
                        subjective
                            .weakness_tags
                            .iter()
                            .map(|item| format_skill_label(item))
                            .collect::<Vec<_>>()
                            .join("、")
                    )));
                }
                if !subjective.wrong_words.is_empty() {
                    let sample = subjective
                        .wrong_words
                        .iter()
                        .take(3)
                        .map(|item| format!("{}→{}", item.original, item.corrected))
                        .collect::<Vec<_>>()
                        .join("，");
                    right_lines.push(Line::from(format!("  错词：{}", sample)));
                }
                if !subjective.sentence_rewrites.is_empty() {
                    right_lines.push(Line::from(format!(
                        "  语病改写：{} 处，建议进入普通刷题继续针对性重写。",
                        subjective.sentence_rewrites.len()
                    )));
                }
            } else {
                let wrong_items = section
                    .result
                    .question_results
                    .iter()
                    .filter(|item| !item.is_correct)
                    .take(2)
                    .collect::<Vec<_>>();
                if wrong_items.is_empty() {
                    right_lines.push(Line::from("  该部分当前未见明显失分点。"));
                } else {
                    for item in wrong_items {
                        right_lines.push(Line::from(format!(
                            "  失分：{} -> {}",
                            blank_or_value(&item.user_answer),
                            item.correct_answer
                        )));
                        right_lines.push(Line::from(format!(
                            "  解析：{}",
                            truncate_text(&item.explanation, 76)
                        )));
                    }
                }
            }
            right_lines.push(Line::from(""));
        }
        frame.render_widget(
            Paragraph::new(Text::from(right_lines))
                .wrap(Wrap { trim: false })
                .scroll((self.review_detail_scroll, 0))
                .block(simple_block("复盘详情（可滚动）", palette)),
            body[1],
        );

        self.draw_action_button(
            frame,
            chunks[3],
            palette,
            "返回历史",
            true,
            Action::BackReview,
        );
        frame.render_widget(
            Paragraph::new("Up/Down、PgUp/PgDn 或滚轮滚动 | Enter/Esc 返回历史")
                .alignment(Alignment::Center)
                .style(Style::default().fg(palette.muted)),
            chunks[3],
        );
        self.draw_status_line(frame, chunks[4], palette);
    }

    fn draw_settings(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let outer = centered_rect(86, 78, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(6),
                Constraint::Length(9),
                Constraint::Length(6),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(outer);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("设置", title_style(palette))),
                Line::from(Span::styled(
                    "上下切换项目，左右或回车切换值；配色会立即预览，高亮对比也会同步更新。",
                    Style::default().fg(palette.muted),
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[0],
        );
        let toggles = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(33),
                Constraint::Percentage(34),
            ])
            .split(chunks[1]);
        self.draw_toggle_box(
            frame,
            toggles[0],
            palette,
            "主题",
            if self.settings.theme_mode == "dark" {
                "深色"
            } else {
                "浅色"
            },
            self.settings_focus == 0,
            Action::ToggleTheme,
        );
        self.draw_toggle_box(
            frame,
            toggles[1],
            palette,
            "背景",
            if self.settings.theme_mode == "light" {
                "浅色模式已锁定不透明"
            } else {
                effective_background_mode_label(
                    &self.settings.theme_mode,
                    &self.settings.background_mode,
                )
            },
            self.settings_focus == 1,
            Action::ToggleBackground,
        );
        self.draw_toggle_box(
            frame,
            toggles[2],
            palette,
            "配色方案",
            palette_mode_label(&self.settings.palette_mode),
            self.settings_focus == 2,
            Action::TogglePalette,
        );

        let middle = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
            .split(chunks[2]);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("配色预览", title_style(palette))),
                Line::from(""),
                Line::from(vec![
                    Span::styled("      ", Style::default().bg(palette.background)),
                    Span::raw(" 背景   "),
                    Span::styled("      ", Style::default().bg(palette.panel)),
                    Span::raw(" 面板"),
                ]),
                Line::from(vec![
                    Span::styled("      ", Style::default().bg(palette.panel_alt)),
                    Span::raw(" 次级   "),
                    Span::styled("  高亮  ", interactive_selected_style(palette)),
                ]),
                Line::from(vec![
                    Span::styled("      ", Style::default().bg(palette.success)),
                    Span::raw(" 成功   "),
                    Span::styled("      ", Style::default().bg(palette.warning)),
                    Span::raw(" 警示"),
                ]),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("视觉预览", palette)),
            middle[0],
        );
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!("模型：{}", self.settings.deepseek_model)),
                Line::from(format!(
                    "主题：{} / 背景：{}",
                    if self.settings.theme_mode == "dark" {
                        "深色"
                    } else {
                        "浅色"
                    },
                    effective_background_mode_label(
                        &self.settings.theme_mode,
                        &self.settings.background_mode,
                    )
                )),
                Line::from(format!(
                    "配色：{}",
                    palette_mode_label(&self.settings.palette_mode)
                )),
                Line::from(format!("数据库：{}", self.settings.db_path)),
                Line::from(format!(
                    "API Key：{}",
                    if self.settings.api_key_configured {
                        "已配置"
                    } else {
                        "未配置"
                    }
                )),
                Line::from(""),
                Line::from("建议：深色适合长时间刷题，浅色更接近纸面阅读。"),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("运行状态", palette)),
            middle[1],
        );

        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from("操作提示："),
                Line::from("1. 主题控制整体明暗。"),
                Line::from("2. 深色模式可切换透明背景，浅色模式固定不透明。"),
                Line::from("3. 配色方案控制高亮、边框与图表观感。"),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("说明", palette)),
            chunks[3],
        );
        self.draw_action_button(
            frame,
            chunks[4],
            palette,
            "返回",
            self.settings_focus == 3,
            Action::BackSettings,
        );
        self.draw_status_line(frame, chunks[5], palette);
    }

    fn draw_practice_sidebar(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let recommended_seconds =
            recommended_seconds_for_question_type(&practice.question_set.question_type);
        let pace_ratio =
            (practice.elapsed_seconds() as f64 / recommended_seconds.max(1) as f64).clamp(0.0, 1.0);
        let total_items = practice.question_set.questions.len().max(1);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(4),
                Constraint::Length(5),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(3),
            ])
            .split(area);

        let timer = Paragraph::new(Text::from(big_timer_lines(&seconds_to_text(
            practice.elapsed_seconds(),
        ))))
        .alignment(Alignment::Center)
        .style(
            Style::default()
                .fg(palette.accent)
                .add_modifier(Modifier::BOLD),
        )
        .block(simple_block("TIME", palette));
        frame.render_widget(timer, chunks[0]);

        let progress_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Length(3)])
            .split(chunks[1]);
        frame.render_widget(
            Gauge::default()
                .block(simple_block("答题进度", palette))
                .ratio(practice.answered_count() as f64 / total_items as f64)
                .gauge_style(Style::default().fg(palette.success).bg(palette.panel_alt)),
            progress_chunks[0],
        );
        frame.render_widget(
            Gauge::default()
                .block(simple_block(
                    &format!("节奏  {}", pace_band(pace_ratio)),
                    palette,
                ))
                .ratio(pace_ratio)
                .gauge_style(Style::default().fg(palette.warning).bg(palette.panel_alt)),
            progress_chunks[1],
        );

        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!(
                    "等级：{}",
                    format_level_label(&practice.question_set.level)
                )),
                Line::from(format!(
                    "题型：{}",
                    format_question_label(
                        &practice.question_set.question_type,
                        practice.question_set.slot
                    )
                )),
                Line::from(format!(
                    "词数：{} | 已答：{} / {}",
                    practice.question_set.word_count,
                    practice.answered_count(),
                    total_items
                )),
                Line::from(format!(
                    "未答：{} | 建议：{}",
                    practice.unanswered_count(),
                    recommended_time_text(&practice.question_set.question_type),
                )),
                Line::from(format!(
                    "来源：{} | 模型：{}",
                    display_source_type(&practice.question_set.source_type),
                    truncate_text(&practice.question_set.generator_model, 10)
                )),
                Line::from(if practice.submit_confirm_pending {
                    "再次提交将按当前答案交卷。".to_string()
                } else {
                    "先做会做的题，卡住可先跳过。".to_string()
                }),
            ]))
            .block(simple_block("信息", palette)),
            chunks[2],
        );

        self.draw_action_button(
            frame,
            chunks[3],
            palette,
            "提交答案",
            false,
            Action::SubmitPractice,
        );
        self.draw_action_button(
            frame,
            chunks[4],
            palette,
            "返回题型",
            false,
            Action::BackType,
        );
        frame.render_widget(
            Paragraph::new(if practice.question_set.questions.is_empty() {
                "F9 提交 | F8 返回题型 | Ctrl+S 也可提交 | Enter 换行 | 左右移动光标"
            } else {
                "F9 提交 | F8 返回题型 | Ctrl+S 也可提交 | Space/Enter 选中项 | PageUp/PageDown 滚动文章"
            })
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[5],
        );
    }

    fn draw_subjective_panel(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let prompt_height = ((area.height.saturating_mul(42)) / 100).clamp(10, 16);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(prompt_height), Constraint::Min(12)])
            .split(area);

        let mut prompt_lines = vec![
            Line::from(Span::styled(
                format_question_group_label(
                    &practice.question_set.question_type,
                    practice.question_set.slot,
                ),
                title_style(palette),
            )),
            Line::from(Span::styled(
                subjective_exam_heading(&practice.question_set.question_type),
                Style::default()
                    .fg(palette.text)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(practice.question_set.task_prompt.clone()),
            Line::from(""),
        ];
        for (index, line) in practice.question_set.passage.paragraphs.iter().enumerate() {
            if index == 0 {
                let title_normalized = practice.question_set.title.trim().to_lowercase();
                let line_normalized = line.trim().to_lowercase();
                if !title_normalized.is_empty()
                    && (line_normalized == title_normalized
                        || line_normalized.contains(&title_normalized)
                        || title_normalized.contains(&line_normalized))
                {
                    continue;
                }
            }
            prompt_lines.push(Line::from(line.clone()));
        }
        prompt_lines.push(Line::from(""));
        prompt_lines.push(Line::from(format!(
            "建议字数：{}-{} | 当前已输入约 {} 词",
            practice.question_set.min_response_words,
            practice.question_set.max_response_words,
            practice.response_text().split_whitespace().count()
        )));
        frame.render_widget(
            Paragraph::new(Text::from(prompt_lines))
                .wrap(Wrap { trim: false })
                .scroll((practice.passage_scroll, 0))
                .block(simple_block("试卷题面", palette)),
            chunks[0],
        );

        let answer_block = simple_block("答题卡", palette);
        let answer_inner = answer_block.inner(chunks[1]);
        frame.render_widget(answer_block, chunks[1]);

        let answer_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(1)])
            .split(answer_inner);
        let editor_width = answer_chunks[1].width.max(1) as usize;
        let editor_rows = subjective_editor_rows(
            practice.response_text(),
            practice.subjective_cursor,
            editor_width.saturating_sub(1).max(1),
        );
        let (cursor_line, cursor_col) = subjective_cursor_line_col(&editor_rows);
        let visible_height = answer_chunks[1].height.max(1) as usize;
        let max_scroll = editor_rows.len().saturating_sub(visible_height) as u16;
        let scroll = practice.subjective_scroll.min(max_scroll) as usize;

        frame.render_widget(
            Paragraph::new(Text::from(vec![Line::from(vec![
                Span::styled(
                    format!("Ln {:02}  Col {:02}", cursor_line, cursor_col),
                    title_style(palette),
                ),
                Span::raw("  ·  "),
                Span::styled(
                    format!(
                        "答题横线已对齐，当前约 {} 词",
                        practice.response_text().split_whitespace().count()
                    ),
                    Style::default().fg(palette.muted),
                ),
            ])]))
            .alignment(Alignment::Left),
            answer_chunks[0],
        );

        let mut answer_lines = Vec::with_capacity(visible_height);
        for offset in 0..visible_height {
            let row_index = scroll + offset;
            if let Some(row) = editor_rows.get(row_index) {
                let mut spans = subjective_editor_line_spans(
                    row,
                    editor_width,
                    palette,
                    practice.response_text().trim().is_empty() && row_index == 0,
                );
                if spans.is_empty() {
                    spans.push(Span::styled(
                        "─".repeat(editor_width),
                        Style::default().fg(palette.border),
                    ));
                }
                answer_lines.push(Line::from(spans));
            } else {
                answer_lines.push(Line::from(Span::styled(
                    "─".repeat(editor_width),
                    Style::default().fg(palette.border),
                )));
            }
        }
        frame.render_widget(
            Paragraph::new(Text::from(answer_lines)).alignment(Alignment::Left),
            answer_chunks[1],
        );
    }

    fn draw_passage_panel(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let hide_title_in_practice = practice.question_set.question_type == "careful_reading";
        let mut lines = vec![
            Line::from(Span::styled(
                "Part III  Reading Comprehension",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!(
                    "{}  {}",
                    format_question_group_label(
                        &practice.question_set.question_type,
                        practice.question_set.slot,
                    ),
                    format_question_label(
                        &practice.question_set.question_type,
                        practice.question_set.slot,
                    )
                ),
                Style::default()
                    .fg(palette.text)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format_directions(&practice.question_set.question_type),
                Style::default().fg(palette.muted),
            )),
            Line::from(""),
        ];
        if !hide_title_in_practice && !practice.question_set.passage.title.trim().is_empty() {
            lines.insert(
                3,
                Line::from(Span::styled(
                    practice.question_set.passage.title.clone(),
                    Style::default().fg(palette.text),
                )),
            );
        }
        for paragraph in &practice.question_set.passage.paragraphs {
            lines.push(Line::from(paragraph.clone()));
            lines.push(Line::from(""));
        }
        let paragraph = Paragraph::new(Text::from(lines))
            .wrap(Wrap { trim: false })
            .scroll((practice.passage_scroll, 0))
            .block(simple_block("试卷正文", palette));
        frame.render_widget(paragraph, area);
    }

    fn draw_answer_panel(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        match practice.question_set.question_type.as_str() {
            "banked_cloze" => self.draw_banked_cloze_panel(frame, area, palette, practice),
            "long_reading" => self.draw_long_reading_panel(frame, area, palette, practice),
            "careful_reading" => self.draw_careful_reading_panel(frame, area, palette, practice),
            _ => self.draw_generic_question_panel(frame, area, palette, practice),
        }
    }

    fn draw_banked_cloze_panel(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Length(8),
                Constraint::Min(10),
            ])
            .split(area);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    format!(
                        "{}  Answer Bank",
                        format_question_group_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot
                        )
                    ),
                    title_style(palette),
                )),
                Line::from(format!(
                    "当前空位 {:02} | 当前答案 {}",
                    practice.selected_blank + 1,
                    blank_or_value(
                        practice
                            .answers
                            .get(&practice.question_set.questions[practice.selected_blank].id)
                            .map(String::as_str)
                            .unwrap_or("")
                    )
                )),
                Line::from("先选空位，再点选项或直接键入 A-O；作答后自动跳到下一空。"),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[0],
        );

        let blank_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Length(4)])
            .split(chunks[1]);
        for row in 0..2 {
            let row_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Ratio(1, 5),
                    Constraint::Ratio(1, 5),
                    Constraint::Ratio(1, 5),
                    Constraint::Ratio(1, 5),
                    Constraint::Ratio(1, 5),
                ])
                .split(blank_rows[row]);
            for col in 0..5 {
                let index = row * 5 + col;
                let question = &practice.question_set.questions[index];
                let answer = practice
                    .answers
                    .get(&question.id)
                    .map_or(" ".to_string(), |value| blank_or_value(value));
                let selected = practice.selected_blank == index;
                let text = if answer.trim() == "__" {
                    format!("{:02}\n[__]", index + 1)
                } else {
                    format!("{:02}\n[{}]", index + 1, answer)
                };
                self.draw_action_button(
                    frame,
                    row_chunks[col],
                    palette,
                    &text,
                    selected,
                    Action::PracticeSelectBlank(index),
                );
            }
        }

        let option_columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
            .split(chunks[2]);
        let labels = practice.available_labels();
        let left_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(1); 8])
            .split(option_columns[0]);
        let right_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(1); 8])
            .split(option_columns[1]);
        for index in 0..practice.question_set.shared_options.len() {
            let target = if index < 8 {
                left_rows[index]
            } else {
                right_rows[index - 8]
            };
            let option = &practice.question_set.shared_options[index];
            let label = labels.get(index).cloned().unwrap_or_default();
            let selected = practice
                .answers
                .get(&practice.question_set.questions[practice.selected_blank].id)
                .map(|value| value == &label)
                .unwrap_or(false);
            let assigned_to = assigned_blank_index(practice, &label);
            let option_label = match assigned_to {
                Some(blank_index) if selected => {
                    format!("{}   -> 当前第 {:02} 空", option, blank_index + 1)
                }
                Some(blank_index) => format!("{}   -> 已填第 {:02} 空", option, blank_index + 1),
                None => format!("{}   -> 未使用", option),
            };
            self.draw_list_like_button(
                frame,
                target,
                palette,
                &option_label,
                selected,
                Action::PracticeAssign(label),
            );
        }
    }

    fn draw_generic_question_panel(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Min(7),
            ])
            .split(area);

        let current = &practice.question_set.questions[practice.selected_question];
        let current_answer = practice
            .answers
            .get(&current.id)
            .cloned()
            .unwrap_or_default();
        let current_block = vec![
            Line::from(Span::styled("Answer Sheet", title_style(palette))),
            Line::from(format!(
                "第 {} 题 / 共 {} 题",
                practice.selected_question + 1,
                practice.question_set.questions.len()
            )),
            Line::from(current.prompt.clone()),
            Line::from(format!("当前答案：{}", blank_or_value(&current_answer))),
            Line::from("Enter / Space 选中，Backspace 清空，作答后自动跳下一题。"),
        ];
        frame.render_widget(
            Paragraph::new(Text::from(current_block))
                .wrap(Wrap { trim: false })
                .block(simple_block("", palette)),
            chunks[0],
        );

        if practice.question_set.question_type == "careful_reading" {
            let option_heights = current
                .options
                .iter()
                .map(|option| wrapped_block_height(option, chunks[1].width.saturating_sub(4), 2, 4))
                .collect::<Vec<_>>();
            let option_rows = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(100)])
                .split(chunks[1]);
            let option_rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints(
                    option_heights
                        .iter()
                        .map(|height| Constraint::Length(*height))
                        .collect::<Vec<_>>(),
                )
                .split(option_rows[0]);
            let labels = practice.available_labels();
            for (index, option) in current.options.iter().enumerate() {
                let label = labels.get(index).cloned().unwrap_or_default();
                let selected = practice
                    .answers
                    .get(&current.id)
                    .map(|value| value == &label)
                    .unwrap_or(false);
                self.draw_list_like_button(
                    frame,
                    option_rows[index],
                    palette,
                    option,
                    selected,
                    Action::PracticeAssign(label),
                );
            }
        } else {
            let labels = practice.available_labels();
            let choice_area = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
                .split(chunks[1]);
            let left_rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Length(1); 6])
                .split(choice_area[0]);
            let right_rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Length(1); 6])
                .split(choice_area[1]);
            for (index, label) in labels.iter().enumerate() {
                let target = if index < 6 {
                    left_rows[index]
                } else {
                    right_rows[index - 6]
                };
                let selected = practice.choice_cursor == index
                    || practice
                        .answers
                        .get(&current.id)
                        .map(|value| value == label)
                        .unwrap_or(false);
                self.draw_list_like_button(
                    frame,
                    target,
                    palette,
                    &format!("段落 {}", label),
                    selected,
                    Action::PracticeAssign(label.clone()),
                );
            }
        }

        let items: Vec<ListItem> = practice
            .question_set
            .questions
            .iter()
            .enumerate()
            .map(|(index, question)| {
                let answer = practice
                    .answers
                    .get(&question.id)
                    .cloned()
                    .unwrap_or_default();
                ListItem::new(Line::from(format!(
                    "{}. {} [{}]",
                    index + 1,
                    truncate_text(&question.prompt, 26),
                    blank_or_value(&answer)
                )))
            })
            .collect();
        let mut state = ListState::default();
        state.select(Some(practice.selected_question));
        frame.render_stateful_widget(
            List::new(items)
                .block(simple_block("题目列表", palette))
                .highlight_style(interactive_selected_style(palette)),
            chunks[2],
            &mut state,
        );
        let question_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                std::iter::repeat(Constraint::Length(1))
                    .take(practice.question_set.questions.len())
                    .collect::<Vec<_>>(),
            )
            .split(chunks[2]);
        for (index, rect) in question_chunks.iter().enumerate() {
            self.click_areas.push(ClickArea {
                rect: *rect,
                action: Action::PracticeSelectQuestion(index),
            });
        }
    }

    fn draw_careful_reading_panel(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Min(6),
            ])
            .split(area);

        let current = &practice.question_set.questions[practice.selected_question];
        let current_answer = practice
            .answers
            .get(&current.id)
            .cloned()
            .unwrap_or_default();
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    format!(
                        "{}  {}",
                        format_question_group_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot
                        ),
                        "Answer Sheet",
                    ),
                    title_style(palette),
                )),
                Line::from(format!(
                    "仔细阅读第 {} 题 / 共 {} 题",
                    practice.selected_question + 1,
                    practice.question_set.questions.len()
                )),
                Line::from(current.prompt.clone()),
                Line::from(format!("当前答案：{}", blank_or_value(&current_answer))),
                Line::from("Enter / Space 选中，Backspace 清空，作答后自动跳下一题。"),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("", palette)),
            chunks[0],
        );

        let labels = practice.available_labels();
        let option_heights = current
            .options
            .iter()
            .map(|option| wrapped_block_height(option, chunks[1].width.saturating_sub(4), 2, 4))
            .collect::<Vec<_>>();
        let option_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                option_heights
                    .iter()
                    .map(|height| Constraint::Length(*height))
                    .collect::<Vec<_>>(),
            )
            .split(chunks[1]);
        for (index, option) in current.options.iter().enumerate() {
            let label = labels.get(index).cloned().unwrap_or_default();
            let selected = practice
                .answers
                .get(&current.id)
                .map(|value| value == &label)
                .unwrap_or(false);
            self.draw_list_like_button(
                frame,
                option_rows[index],
                palette,
                option,
                selected,
                Action::PracticeAssign(label),
            );
        }

        self.draw_question_navigator(frame, chunks[2], palette, practice, "题目导航");
    }

    fn draw_long_reading_panel(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let root = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Min(7),
            ])
            .split(area);

        let current = &practice.question_set.questions[practice.selected_question];
        let current_answer = practice
            .answers
            .get(&current.id)
            .cloned()
            .unwrap_or_default();
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    format!(
                        "{}  Matching Grid",
                        format_question_group_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot
                        )
                    ),
                    title_style(palette),
                )),
                Line::from(format!(
                    "长篇阅读第 {} 题 / 共 {} 题",
                    practice.selected_question + 1,
                    practice.question_set.questions.len()
                )),
                Line::from(current.prompt.clone()),
                Line::from(format!("当前段落：{}", blank_or_value(&current_answer))),
                Line::from("Enter / Space 选中，Backspace 清空，作答后自动跳下一题。"),
            ]))
            .wrap(Wrap { trim: false })
            .block(simple_block("", palette)),
            root[0],
        );

        let choice_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
            .split(root[1]);
        let labels = practice.available_labels();
        let left_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(1); 6])
            .split(choice_area[0]);
        let right_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(1); 6])
            .split(choice_area[1]);
        for (index, label) in labels.iter().enumerate() {
            let target = if index < 6 {
                left_rows[index]
            } else {
                right_rows[index - 6]
            };
            let selected = practice
                .answers
                .get(&current.id)
                .map(|value| value == label)
                .unwrap_or(false);
            self.draw_list_like_button(
                frame,
                target,
                palette,
                &format!("段落 {}", label),
                selected,
                Action::PracticeAssign(label.clone()),
            );
        }

        self.draw_question_navigator(frame, root[2], palette, practice, "匹配题列表");
    }

    fn draw_question_navigator(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
        title: &str,
    ) {
        let items: Vec<ListItem> = practice
            .question_set
            .questions
            .iter()
            .enumerate()
            .map(|(index, question)| {
                let answer = practice
                    .answers
                    .get(&question.id)
                    .cloned()
                    .unwrap_or_default();
                ListItem::new(Line::from(format!(
                    "{}. {} [{}]",
                    index + 1,
                    truncate_text(&question.prompt, 30),
                    blank_or_value(&answer)
                )))
            })
            .collect();
        let mut state = ListState::default();
        state.select(Some(practice.selected_question));
        frame.render_stateful_widget(
            List::new(items)
                .block(simple_block(title, palette))
                .highlight_style(interactive_selected_style(palette)),
            area,
            &mut state,
        );
        let question_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                std::iter::repeat(Constraint::Length(1))
                    .take(practice.question_set.questions.len())
                    .collect::<Vec<_>>(),
            )
            .split(area);
        for (index, rect) in question_chunks.iter().enumerate() {
            self.click_areas.push(ClickArea {
                rect: *rect,
                action: Action::PracticeSelectQuestion(index),
            });
        }
    }

    fn draw_list_like_button(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        label: &str,
        selected: bool,
        action: Action,
    ) {
        let style = if selected {
            interactive_selected_style(palette)
        } else {
            interactive_idle_style(palette)
        };
        frame.render_widget(
            Paragraph::new(label.to_string())
                .wrap(Wrap { trim: false })
                .style(style)
                .block(
                    Block::default()
                        .borders(Borders::LEFT | Borders::RIGHT)
                        .border_style(interactive_border_style(palette, selected))
                        .style(style),
                ),
            area,
        );
        self.click_areas.push(ClickArea { rect: area, action });
    }

    fn draw_metric_box(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        title: &str,
        main: &str,
        sub: &str,
        ratio: Option<f64>,
    ) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .split(area);
        frame.render_widget(simple_block(title, palette), area);
        frame.render_widget(
            Paragraph::new(main.to_string())
                .alignment(Alignment::Center)
                .style(
                    Style::default()
                        .fg(palette.accent)
                        .add_modifier(Modifier::BOLD),
                ),
            chunks[1],
        );
        frame.render_widget(
            Paragraph::new(sub.to_string())
                .alignment(Alignment::Center)
                .style(Style::default().fg(palette.muted)),
            chunks[2],
        );
        if let Some(ratio) = ratio {
            frame.render_widget(
                Gauge::default()
                    .gauge_style(Style::default().fg(palette.success).bg(palette.panel_alt))
                    .ratio(ratio.clamp(0.0, 1.0)),
                chunks[3],
            );
        }
    }

    fn draw_line_trend_card(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        title: &str,
        series: &[f64],
        overlay_series: Option<&[f64]>,
        color: Color,
        overlay_color: Color,
        metric: TrendMetric,
        fixed_upper_bound: Option<f64>,
    ) {
        let block = simple_block(title, palette);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.width < 3 || inner.height < 2 {
            return;
        }

        let max_points = inner.width.saturating_sub(1) as usize;
        let mut sampled = sample_series_for_width(series, max_points.max(1));
        let mut sampled_overlay = overlay_series
            .map(|items| sample_series_for_width(items, max_points.max(1)))
            .unwrap_or_default();
        if sampled.is_empty() && !sampled_overlay.is_empty() {
            sampled = sampled_overlay.clone();
            sampled_overlay.clear();
        }
        if sampled.is_empty() && sampled_overlay.is_empty() {
            frame.render_widget(
                Paragraph::new("暂无走势数据\n先开始训练，图表会自动累积最近记录。")
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: false })
                    .style(Style::default().fg(palette.muted)),
                inner,
            );
            return;
        }

        let min_sample = sampled
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::INFINITY, f64::min);
        let max_sample = sampled
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::NEG_INFINITY, f64::max);
        let overlay_min = sampled_overlay
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::INFINITY, f64::min);
        let overlay_max = sampled_overlay
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::NEG_INFINITY, f64::max);
        let all_min = if overlay_min.is_finite() {
            min_sample.min(overlay_min)
        } else {
            min_sample
        };
        let all_max = if overlay_max.is_finite() {
            max_sample.max(overlay_max)
        } else {
            max_sample
        };
        let (lower, upper) = if let Some(bound) = fixed_upper_bound {
            (0.0, bound.max(1.0))
        } else {
            let spread = (all_max - all_min).abs();
            let padding = (spread * 0.2).max(1.0);
            let lower = (all_min - padding).max(0.0);
            let upper = (all_max + padding).max(lower + 1.0);
            (lower, upper)
        };
        let last_value = sampled.last().map(|(_, value)| *value).unwrap_or(0.0);
        let first_value = sampled.first().map(|(_, value)| *value).unwrap_or(last_value);
        let delta = last_value - first_value;
        let chart_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(inner);
        let chart_style = Style::default().bg(palette.panel_alt);
        frame.render_widget(
            Paragraph::new(String::new()).style(chart_style),
            chart_area[0],
        );

        let dataset = Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(
                Style::default()
                    .fg(color)
                    .bg(palette.panel_alt)
                    .add_modifier(Modifier::BOLD),
            )
            .data(&sampled);
        let overlay_dataset = Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(
                Style::default()
                    .fg(overlay_color)
                    .bg(palette.panel_alt)
                    .add_modifier(Modifier::DIM),
            )
            .data(&sampled_overlay);
        let last_point = sampled
            .last()
            .copied()
            .map(|point| vec![point])
            .unwrap_or_default();
        let focus_dataset = Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(
                Style::default()
                    .fg(palette.accent)
                    .bg(palette.panel_alt)
                    .add_modifier(Modifier::BOLD),
            )
            .data(&last_point);
        let mut datasets = vec![dataset];
        if !sampled_overlay.is_empty() {
            datasets.push(overlay_dataset);
        }
        datasets.push(focus_dataset);
        frame.render_widget(
            Chart::new(datasets)
                .style(chart_style)
                .x_axis(
                    Axis::default()
                        .bounds([0.0, (sampled.len().saturating_sub(1)) as f64])
                        .style(Style::default().fg(palette.muted).bg(palette.panel_alt))
                        .labels(Vec::<Span>::new()),
                )
                .y_axis(
                    Axis::default()
                        .bounds([lower, upper])
                        .style(Style::default().fg(palette.muted).bg(palette.panel_alt))
                        .labels(Vec::<Span>::new()),
                ),
            chart_area[0],
        );

        let footer_lead = if series.len() > sampled.len() {
            format!("近 {} / {} 次", sampled.len(), series.len())
        } else {
            format!("近 {} 次", sampled.len())
        };
        let (delta_symbol, delta_color) = trend_delta_visual(metric, delta, palette);
        let footer = if sampled.len() >= 2 {
            Line::from(vec![
                Span::styled(footer_lead, Style::default().fg(palette.muted)),
                Span::raw("  "),
                Span::styled(
                    format!("最新 {}", trend_value_text(metric, last_value)),
                    Style::default().fg(palette.text),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("区间 {}", trend_range_text(metric, min_sample, max_sample)),
                    Style::default().fg(palette.muted),
                ),
                Span::raw("  "),
                Span::styled(delta_symbol, Style::default().fg(delta_color)),
                Span::styled(
                    format!(" {}", trend_delta_text(metric, delta)),
                    Style::default().fg(delta_color),
                ),
                if sampled_overlay.is_empty() {
                    Span::raw("")
                } else {
                    Span::styled(
                        "  亮线练习 / 暗线模考",
                        Style::default().fg(palette.muted),
                    )
                },
            ])
        } else {
            Line::from(vec![
                Span::styled(footer_lead, Style::default().fg(palette.muted)),
                Span::raw("  "),
                Span::styled(
                    format!("最新 {}", trend_value_text(metric, last_value)),
                    Style::default().fg(palette.text),
                ),
                if sampled_overlay.is_empty() {
                    Span::raw("")
                } else {
                    Span::styled(
                        "  亮线练习 / 暗线模考",
                        Style::default().fg(palette.muted),
                    )
                },
            ])
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![footer]))
                .alignment(Alignment::Center)
                .style(Style::default().bg(palette.panel_alt)),
            chart_area[1],
        );
    }

    fn draw_type_card(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        card: &TypeCardData,
        selected: bool,
        action_type: TypeChoice,
    ) {
        let block = Block::default()
            .title(card.label.clone())
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(if selected {
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(palette.border)
            })
            .style(Style::default().bg(palette.panel));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(2),
                Constraint::Length(1),
            ])
            .split(inner);
        frame.render_widget(
            Paragraph::new(format!(
                "{}{}  {}",
                action_type.section_label(),
                action_type
                    .passage_label()
                    .map(|label| format!(" · {}", label))
                    .unwrap_or_default(),
                action_type.recommended_time()
            ))
            .alignment(Alignment::Center)
            .style(Style::default().fg(palette.muted)),
            chunks[0],
        );
        frame.render_widget(
            Paragraph::new(format!("{:.1}%", card.recent_accuracy_percent))
                .alignment(Alignment::Center)
                .style(
                    Style::default()
                        .fg(palette.accent)
                        .add_modifier(Modifier::BOLD),
                ),
            chunks[1],
        );
        frame.render_widget(
            Paragraph::new(format!(
                "{} 次 | 平均 {}",
                card.attempt_count, card.recent_duration_text
            ))
            .alignment(Alignment::Center)
            .style(Style::default().fg(palette.muted)),
            chunks[2],
        );
        frame.render_widget(
            Paragraph::new(action_type.estimated_generation_time(self.selected_level))
                .alignment(Alignment::Center)
                .style(Style::default().fg(palette.accent)),
            chunks[3],
        );
        frame.render_widget(
            Gauge::default()
                .ratio((card.recent_accuracy_percent / 100.0).clamp(0.0, 1.0))
                .gauge_style(Style::default().fg(palette.success).bg(palette.panel_alt)),
            chunks[4],
        );
        frame.render_widget(
            Paragraph::new(format!(
                "峰值 {:.1}% {}",
                card.best_recent_accuracy_percent,
                mini_ratio_bar(card.best_recent_accuracy_percent / 100.0, 8)
            ))
            .alignment(Alignment::Center)
            .style(Style::default().fg(palette.warning)),
            chunks[5],
        );
        let accuracy_series = if card.recent_accuracy_series.is_empty() {
            vec![0.0]
        } else {
            card.recent_accuracy_series.clone()
        };
        let spark_values: Vec<u64> = accuracy_series
            .iter()
            .map(|value| value.max(0.0).round() as u64)
            .collect();
        frame.render_widget(
            Sparkline::default()
                .data(&spark_values)
                .bar_set(symbols::bar::NINE_LEVELS)
                .style(Style::default().fg(palette.warning)),
            chunks[6],
        );
        frame.render_widget(
            Paragraph::new(action_type.brief(self.selected_level))
                .alignment(Alignment::Center)
                .style(Style::default().fg(palette.text)),
            chunks[7],
        );
        self.click_areas.push(ClickArea {
            rect: area,
            action: Action::Type(action_type),
        });
    }

    fn draw_home_menu_card(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        title: &str,
        subtitle: &str,
        selected: bool,
        action: Action,
    ) {
        let style = if selected {
            interactive_selected_style(palette)
        } else {
            interactive_idle_style(palette)
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    title.to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(subtitle.to_string()),
            ]))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: false })
            .style(style)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(interactive_border_style(palette, selected))
                    .style(style),
            ),
            area,
        );
        self.click_areas.push(ClickArea { rect: area, action });
    }

    fn draw_action_button(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        label: &str,
        selected: bool,
        action: Action,
    ) {
        let style = if selected {
            interactive_selected_style(palette)
        } else {
            interactive_idle_style(palette)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(interactive_border_style(palette, selected))
            .style(style);
        let text = Paragraph::new(label.to_string())
            .alignment(Alignment::Center)
            .style(style)
            .block(block);
        frame.render_widget(text, area);
        self.click_areas.push(ClickArea { rect: area, action });
    }

    fn draw_action_button_multiline(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        lines: &[&str],
        selected: bool,
        action: Action,
    ) {
        let style = if selected {
            interactive_selected_style(palette)
        } else {
            interactive_idle_style(palette)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(interactive_border_style(palette, selected))
            .style(style);
        let text = Paragraph::new(
            Text::from(
                lines
                    .iter()
                    .map(|line| Line::from((*line).to_string()))
                    .collect::<Vec<_>>(),
            ),
        )
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: false })
        .style(style)
        .block(block);
        frame.render_widget(text, area);
        self.click_areas.push(ClickArea { rect: area, action });
    }

    fn draw_toggle_box(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        title: &str,
        value: &str,
        selected: bool,
        action: Action,
    ) {
        let style = if selected {
            interactive_selected_style(palette)
        } else {
            interactive_idle_style(palette)
        };
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    title.to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(value.to_string()),
            ]))
            .alignment(Alignment::Center)
            .style(style)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(interactive_border_style(palette, selected))
                    .style(style),
            ),
            area,
        );
        self.click_areas.push(ClickArea { rect: area, action });
    }

    fn draw_terminal_size_hint(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let box_area = centered_rect(76, 56, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(6),
                Constraint::Length(3),
            ])
            .split(box_area);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("窗口尺寸偏小", title_style(palette))),
                Line::from(Span::styled(
                    "为了保证考卷式布局、电子表计时器和多栏作答区完整显示，请放大终端。",
                    Style::default().fg(palette.muted),
                )),
            ]))
            .alignment(Alignment::Center)
            .block(simple_block("", palette)),
            chunks[0],
        );
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(format!("当前尺寸：{} x {}", area.width, area.height)),
                Line::from(format!(
                    "推荐至少：{} x {}",
                    MIN_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT
                )),
                Line::from(""),
                Line::from("可继续使用 Q 退出，或调整终端后继续练习。"),
            ]))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: false })
            .block(simple_block("显示建议", palette)),
            chunks[1],
        );
        frame.render_widget(
            Paragraph::new("当前窗口下已启用安全提示页，避免界面错位。")
                .alignment(Alignment::Center)
                .block(simple_block("", palette)),
            chunks[2],
        );
    }

    fn draw_status_line(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        frame.render_widget(
            Paragraph::new(self.status_line.clone())
                .alignment(Alignment::Center)
                .style(Style::default().fg(palette.muted).bg(palette.panel)),
            area,
        );
    }
}

#[derive(Clone, Copy)]
struct Palette {
    background: Color,
    panel: Color,
    panel_alt: Color,
    accent: Color,
    highlight_text: Color,
    text: Color,
    muted: Color,
    success: Color,
    warning: Color,
    border: Color,
}

impl Palette {
    fn new(theme_mode: &str, background_mode: &str, palette_mode: &str) -> Self {
        let transparent = effective_background_mode(theme_mode, background_mode) == "transparent";
        let transparent_panels = transparent && theme_mode == "dark";
        match (theme_mode, palette_mode) {
            ("light", "ink") => Self {
                background: if transparent {
                    Color::Reset
                } else {
                    Color::Rgb(243, 246, 248)
                },
                panel: Color::Rgb(250, 252, 253),
                panel_alt: Color::Rgb(235, 241, 245),
                accent: Color::Rgb(53, 108, 150),
                highlight_text: Color::Rgb(255, 255, 255),
                text: Color::Rgb(27, 39, 48),
                muted: Color::Rgb(95, 111, 122),
                success: Color::Rgb(68, 141, 114),
                warning: Color::Rgb(158, 129, 72),
                border: Color::Rgb(191, 204, 214),
            },
            ("light", "amber") => Self {
                background: if transparent {
                    Color::Reset
                } else {
                    Color::Rgb(247, 243, 236)
                },
                panel: Color::Rgb(252, 249, 244),
                panel_alt: Color::Rgb(242, 235, 225),
                accent: Color::Rgb(138, 98, 34),
                highlight_text: Color::Rgb(255, 255, 255),
                text: Color::Rgb(49, 40, 28),
                muted: Color::Rgb(118, 103, 84),
                success: Color::Rgb(84, 135, 92),
                warning: Color::Rgb(170, 126, 56),
                border: Color::Rgb(210, 196, 176),
            },
            ("light", "rose") => Self {
                background: if transparent {
                    Color::Reset
                } else {
                    Color::Rgb(247, 241, 241)
                },
                panel: Color::Rgb(252, 248, 248),
                panel_alt: Color::Rgb(241, 232, 233),
                accent: Color::Rgb(144, 85, 96),
                highlight_text: Color::Rgb(255, 255, 255),
                text: Color::Rgb(52, 35, 39),
                muted: Color::Rgb(121, 97, 103),
                success: Color::Rgb(87, 138, 106),
                warning: Color::Rgb(168, 118, 92),
                border: Color::Rgb(214, 193, 197),
            },
            ("light", _) => Self {
                background: if transparent {
                    Color::Reset
                } else {
                    Color::Rgb(243, 246, 242)
                },
                panel: Color::Rgb(251, 252, 249),
                panel_alt: Color::Rgb(235, 241, 236),
                accent: Color::Rgb(49, 122, 104),
                highlight_text: Color::Rgb(255, 255, 255),
                text: Color::Rgb(28, 39, 35),
                muted: Color::Rgb(94, 110, 104),
                success: Color::Rgb(55, 141, 105),
                warning: Color::Rgb(139, 127, 71),
                border: Color::Rgb(190, 203, 195),
            },
            ("dark", "ink") => Self {
                background: if transparent {
                    Color::Reset
                } else {
                    Color::Rgb(12, 18, 24)
                },
                panel: if transparent_panels {
                    Color::Reset
                } else {
                    Color::Rgb(21, 31, 39)
                },
                panel_alt: if transparent_panels {
                    Color::Reset
                } else {
                    Color::Rgb(28, 40, 49)
                },
                accent: Color::Rgb(122, 184, 228),
                highlight_text: Color::Rgb(10, 18, 24),
                text: Color::Rgb(232, 239, 243),
                muted: Color::Rgb(152, 166, 176),
                success: Color::Rgb(112, 196, 170),
                warning: Color::Rgb(208, 180, 114),
                border: Color::Rgb(68, 88, 101),
            },
            ("dark", "amber") => Self {
                background: if transparent {
                    Color::Reset
                } else {
                    Color::Rgb(20, 16, 12)
                },
                panel: if transparent_panels {
                    Color::Reset
                } else {
                    Color::Rgb(31, 25, 20)
                },
                panel_alt: if transparent_panels {
                    Color::Reset
                } else {
                    Color::Rgb(40, 32, 26)
                },
                accent: Color::Rgb(224, 182, 96),
                highlight_text: Color::Rgb(24, 16, 10),
                text: Color::Rgb(242, 236, 226),
                muted: Color::Rgb(177, 162, 141),
                success: Color::Rgb(136, 196, 145),
                warning: Color::Rgb(224, 182, 96),
                border: Color::Rgb(95, 80, 64),
            },
            ("dark", "rose") => Self {
                background: if transparent {
                    Color::Reset
                } else {
                    Color::Rgb(21, 14, 17)
                },
                panel: if transparent_panels {
                    Color::Reset
                } else {
                    Color::Rgb(33, 23, 27)
                },
                panel_alt: if transparent_panels {
                    Color::Reset
                } else {
                    Color::Rgb(43, 30, 35)
                },
                accent: Color::Rgb(220, 156, 166),
                highlight_text: Color::Rgb(27, 16, 20),
                text: Color::Rgb(243, 234, 236),
                muted: Color::Rgb(179, 157, 162),
                success: Color::Rgb(121, 196, 151),
                warning: Color::Rgb(213, 170, 121),
                border: Color::Rgb(96, 73, 79),
            },
            _ => Self {
                background: if transparent {
                    Color::Reset
                } else {
                    Color::Rgb(14, 20, 19)
                },
                panel: if transparent_panels {
                    Color::Reset
                } else {
                    Color::Rgb(24, 33, 31)
                },
                panel_alt: if transparent_panels {
                    Color::Reset
                } else {
                    Color::Rgb(31, 42, 39)
                },
                accent: Color::Rgb(134, 208, 182),
                highlight_text: Color::Rgb(10, 18, 16),
                text: Color::Rgb(234, 240, 236),
                muted: Color::Rgb(154, 168, 162),
                success: Color::Rgb(116, 203, 162),
                warning: Color::Rgb(208, 190, 124),
                border: Color::Rgb(71, 91, 85),
            },
        }
    }
}

fn simple_block<'a>(title: &'a str, palette: Palette) -> Block<'a> {
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette.border))
        .style(Style::default().bg(palette.panel).fg(palette.text));
    if !title.is_empty() {
        block = block.title(Span::styled(
            title.to_string(),
            Style::default()
                .fg(palette.accent)
                .add_modifier(Modifier::BOLD),
        ));
    }
    block
}

fn title_style(palette: Palette) -> Style {
    Style::default()
        .fg(palette.accent)
        .add_modifier(Modifier::BOLD)
}

fn glass_mode(palette: Palette) -> bool {
    matches!(palette.panel, Color::Reset) && matches!(palette.background, Color::Reset)
}

fn interactive_idle_style(palette: Palette) -> Style {
    Style::default().fg(palette.text).bg(palette.panel)
}

fn interactive_selected_style(palette: Palette) -> Style {
    if glass_mode(palette) {
        Style::default()
            .fg(palette.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(palette.highlight_text)
            .bg(palette.accent)
            .add_modifier(Modifier::BOLD)
    }
}

fn interactive_border_style(palette: Palette, selected: bool) -> Style {
    if selected {
        Style::default()
            .fg(palette.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(palette.border)
    }
}

fn effective_background_mode(theme_mode: &str, background_mode: &str) -> &'static str {
    if theme_mode == "light" {
        "opaque"
    } else if background_mode == "transparent" {
        "transparent"
    } else {
        "opaque"
    }
}

fn effective_background_mode_label(theme_mode: &str, background_mode: &str) -> &'static str {
    if effective_background_mode(theme_mode, background_mode) == "transparent" {
        "透明"
    } else {
        "不透明"
    }
}

fn force_light_mode_opaque(settings: &mut BridgeSettings) -> bool {
    if settings.theme_mode == "light" && settings.background_mode != "opaque" {
        settings.background_mode = "opaque".to_string();
        true
    } else {
        false
    }
}

fn palette_mode_label(mode: &str) -> &'static str {
    match mode {
        "ink" => "墨青蓝",
        "amber" => "琥珀米",
        "rose" => "灰玫棕",
        _ => "蕨影绿",
    }
}

fn next_palette_mode(mode: &str) -> &'static str {
    match mode {
        "fern" => "ink",
        "ink" => "amber",
        "amber" => "rose",
        _ => "fern",
    }
}

fn accuracy_band(accuracy: f64) -> &'static str {
    if accuracy >= 0.85 {
        "状态优秀"
    } else if accuracy >= 0.7 {
        "状态稳定"
    } else if accuracy >= 0.55 {
        "仍可提升"
    } else {
        "建议继续巩固"
    }
}

fn next_insights_tab(tab: InsightsTab) -> InsightsTab {
    match tab {
        InsightsTab::PracticeWeakness => InsightsTab::MockExamWeakness,
        InsightsTab::MockExamWeakness => InsightsTab::Vocabulary,
        InsightsTab::Vocabulary => InsightsTab::PracticeWeakness,
    }
}

fn previous_insights_tab(tab: InsightsTab) -> InsightsTab {
    match tab {
        InsightsTab::PracticeWeakness => InsightsTab::Vocabulary,
        InsightsTab::MockExamWeakness => InsightsTab::PracticeWeakness,
        InsightsTab::Vocabulary => InsightsTab::MockExamWeakness,
    }
}

fn format_mock_exam_section_key(key: &str) -> &'static str {
    match key {
        "writing" => "写作",
        "translation" => "翻译",
        "banked_cloze" => "选词填空",
        "long_reading" => "长篇阅读",
        "careful_reading_1" => "仔细阅读 1",
        "careful_reading_2" => "仔细阅读 2",
        _ => "题目部分",
    }
}

fn short_type_label(type_choice: TypeChoice) -> &'static str {
    match type_choice {
        TypeChoice::Writing => "作文",
        TypeChoice::Translation => "翻译",
        TypeChoice::BankedCloze => "选填",
        TypeChoice::LongReading => "长阅",
        TypeChoice::Careful1 => "细读1",
        TypeChoice::Careful2 => "细读2",
    }
}

fn generation_step_index(phase: &str) -> usize {
    match phase {
        "boot" | "prepare" => 0,
        "blueprint" => 1,
        "generate_request" => 2,
        "validate" | "repair" | "validated" => 3,
        "failed" => 3,
        "save" | "done" => 4,
        _ => 0,
    }
}

fn submission_step_index(phase: &str) -> usize {
    match phase {
        "boot" | "prepare" => 0,
        "score_request" => 1,
        "analysis" => 2,
        "save" | "done" => 3,
        "grade" => 2,
        "failed" => 2,
        _ => 0,
    }
}

fn format_generation_phase(phase: &str) -> &'static str {
    match phase {
        "boot" => "启动环境",
        "prepare" => "读取数据",
        "blueprint" => "锁定蓝图",
        "generate_request" => "正式出题",
        "retry_generation" => "重新生成",
        "validate" => "规范校验",
        "repair" => "结构修复",
        "validated" => "校验通过",
        "failed" => "生成失败",
        "save" => "保存结果",
        "done" => "生成完成",
        _ => "处理中",
    }
}

fn format_submission_phase(phase: &str) -> &'static str {
    match phase {
        "boot" => "启动评分",
        "prepare" => "整理答案",
        "score_request" => "AI 评分",
        "retry" => "自动重试",
        "analysis" => "整理批注",
        "save" => "保存结果",
        "done" => "评分完成",
        "grade" => "整理判分",
        "failed" => "评分失败",
        _ => "处理中",
    }
}

fn is_generation_heartbeat(message: &str) -> bool {
    message.contains("已等待") || message.contains("仍在")
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn contains(rect: Rect, x: u16, y: u16) -> bool {
    x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height
}

fn find_project_root() -> Option<PathBuf> {
    if let Ok(cwd) = env::current_dir() {
        if let Some(root) = search_ancestors_for_marker(&cwd, "pyproject.toml") {
            return Some(root);
        }
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(root) = search_ancestors_for_marker(&exe, "pyproject.toml") {
            return Some(root);
        }
    }
    None
}

fn search_ancestors_for_marker(start: &Path, marker: &str) -> Option<PathBuf> {
    let mut current = if start.is_dir() {
        start.to_path_buf()
    } else {
        start.parent()?.to_path_buf()
    };
    loop {
        if current.join(marker).exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn format_level_label(level: &str) -> &'static str {
    match level {
        "cet4" => "四级",
        "cet6" => "六级",
        _ => "未知",
    }
}

fn format_question_label(question_type: &str, slot: Option<i32>) -> String {
    match (question_type, slot) {
        ("writing", _) => "写作".to_string(),
        ("banked_cloze", _) => "选词填空".to_string(),
        ("long_reading", _) => "长篇阅读".to_string(),
        ("careful_reading", Some(1)) => "仔细阅读 1".to_string(),
        ("careful_reading", Some(2)) => "仔细阅读 2".to_string(),
        ("careful_reading", _) => "仔细阅读".to_string(),
        ("translation", _) => "翻译".to_string(),
        _ => question_type.to_string(),
    }
}

fn format_question_group_label(question_type: &str, slot: Option<i32>) -> String {
    match (question_type, slot) {
        ("writing", _) => "Part I".to_string(),
        ("banked_cloze", _) => "Section A".to_string(),
        ("long_reading", _) => "Section B".to_string(),
        ("careful_reading", Some(1)) => "Section C".to_string(),
        ("careful_reading", Some(2)) => "Section C".to_string(),
        ("careful_reading", _) => "Section C".to_string(),
        ("translation", _) => "Part IV".to_string(),
        _ => "Reading Set".to_string(),
    }
}

fn format_passage_slot_label(question_type: &str, slot: Option<i32>) -> Option<&'static str> {
    match (question_type, slot) {
        ("careful_reading", Some(1)) => Some("Passage One"),
        ("careful_reading", Some(2)) => Some("Passage Two"),
        _ => None,
    }
}

fn format_directions(question_type: &str) -> &'static str {
    match question_type {
        "writing" => {
            "Directions: For this part, you are allowed 30 minutes to write an essay according to the task given below."
        }
        "banked_cloze" => {
            "Directions: In this section, there is a passage with ten blanks. You are required to select one word for each blank from a list of choices given in a word bank following the passage. Read the passage through carefully before making your choices."
        }
        "long_reading" => {
            "Directions: In this section, you are going to read a passage with ten statements attached to it. Each statement contains information given in one of the paragraphs. Identify the paragraph from which the information is derived. You may choose a paragraph more than once."
        }
        "careful_reading" => {
            "Directions: There are 2 passages in this section. Each passage is followed by some questions or unfinished statements. For each of them there are four choices marked A, B, C and D. You should decide on the best choice."
        }
        "translation" => {
            "Directions: For this part, you are allowed 30 minutes to translate the following Chinese passage into English."
        }
        _ => "Directions: Read the passage and complete the questions below.",
    }
}

fn subjective_exam_heading(question_type: &str) -> &'static str {
    match question_type {
        "writing" => "Writing (30 minutes)",
        "translation" => "Translation (30 minutes)",
        _ => "",
    }
}

fn recommended_time_text(question_type: &str) -> &'static str {
    match question_type {
        "writing" => "30 分钟",
        "banked_cloze" => "8 分钟",
        "long_reading" => "15 分钟",
        "careful_reading" => "12 分钟",
        "translation" => "30 分钟",
        _ => "10 分钟",
    }
}

fn recommended_seconds_for_question_type(question_type: &str) -> i64 {
    match question_type {
        "writing" => 30 * 60,
        "banked_cloze" => 8 * 60,
        "long_reading" => 15 * 60,
        "careful_reading" => 12 * 60,
        "translation" => 30 * 60,
        _ => 10 * 60,
    }
}

fn pace_band(ratio: f64) -> &'static str {
    if ratio <= 0.65 {
        "节奏从容"
    } else if ratio <= 0.9 {
        "节奏正常"
    } else if ratio <= 1.0 {
        "接近建议时长"
    } else {
        "节奏偏慢"
    }
}

fn blank_or_value(value: &str) -> String {
    if value.trim().is_empty() {
        "__".to_string()
    } else {
        value.to_string()
    }
}

fn display_source_type(value: &str) -> &'static str {
    match value.trim() {
        "" => "AI 生成",
        "ai" => "AI 生成",
        "ai_repaired" => "AI 修复生成",
        "mock" => "本地样题",
        _ => "AI 生成",
    }
}

fn assigned_blank_index(practice: &PracticeState, label: &str) -> Option<usize> {
    practice
        .question_set
        .questions
        .iter()
        .enumerate()
        .find_map(|(index, question)| {
            practice
                .answers
                .get(&question.id)
                .filter(|value| value.as_str() == label)
                .map(|_| index)
        })
}

fn truncate_text(value: &str, max: usize) -> String {
    let mut truncated = value.chars().take(max).collect::<String>();
    if value.chars().count() > max {
        truncated.push_str("...");
    }
    truncated
}

fn wrapped_block_height(text: &str, inner_width: u16, min_height: u16, max_height: u16) -> u16 {
    let width = inner_width.max(1) as usize;
    let line_count = text
        .lines()
        .map(|line| {
            let chars = line.chars().count().max(1);
            chars.div_ceil(width)
        })
        .sum::<usize>()
        .max(1) as u16;
    line_count.clamp(min_height, max_height)
}

fn seconds_to_text(seconds: i64) -> String {
    let minutes = seconds / 60;
    let remain = seconds % 60;
    format!("{:02}:{:02}", minutes, remain)
}

fn format_iso_brief(value: &str) -> String {
    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(value) {
        return parsed
            .with_timezone(&Local)
            .format("%Y-%m-%d %H:%M")
            .to_string();
    }
    value.to_string()
}

fn parse_dimension_preview(raw: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return "维度数据解析失败".to_string();
    };
    let Some(map) = value.as_object() else {
        return "维度数据解析失败".to_string();
    };
    let mut pairs = map
        .iter()
        .filter_map(|(key, value)| value.as_f64().map(|score| (key.as_str(), score)))
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let preview = pairs
        .into_iter()
        .take(3)
        .map(|(key, score)| format!("{} {:.0}", format_skill_label(key), score))
        .collect::<Vec<_>>();
    if preview.is_empty() {
        "维度数据为空".to_string()
    } else {
        format!("维度: {}", preview.join(" | "))
    }
}

fn format_skill_label(key: &str) -> &str {
    match key {
        "main_idea" => "主旨",
        "detail" => "细节",
        "detail_location" => "细节",
        "inference" => "推断",
        "attitude" => "态度",
        "attitude_tone" => "态度",
        "vocabulary" => "词汇",
        "vocabulary_in_context" => "词义",
        "vocab_in_context" => "词义",
        "matching" => "匹配",
        "matching_scan" => "匹配",
        "logic" => "逻辑",
        "collocation" => "搭配",
        "speed" => "速度",
        "speed_control" => "速度",
        "distractor_resistance" => "干扰项",
        _ => key,
    }
}

fn mini_ratio_bar(ratio: f64, width: usize) -> String {
    let clamped = ratio.clamp(0.0, 1.0);
    let filled = (clamped * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn trend_value_text(metric: TrendMetric, value: f64) -> String {
    match metric {
        TrendMetric::Index => format!("{value:.0}"),
        TrendMetric::Percentage => format!("{value:.1}%"),
        TrendMetric::Duration => seconds_to_text(value.round() as i64),
    }
}

fn trend_range_text(metric: TrendMetric, min_value: f64, max_value: f64) -> String {
    format!(
        "{}~{}",
        trend_value_text(metric, min_value),
        trend_value_text(metric, max_value)
    )
}

fn trend_delta_text(metric: TrendMetric, delta: f64) -> String {
    match metric {
        TrendMetric::Index => format!("{:+.0}", delta),
        TrendMetric::Percentage => format!("{:+.1}%", delta),
        TrendMetric::Duration => {
            let sign = if delta >= 0.0 { "+" } else { "-" };
            format!("{sign}{}", seconds_to_text(delta.abs().round() as i64))
        }
    }
}

fn trend_delta_visual(metric: TrendMetric, delta: f64, palette: Palette) -> (&'static str, Color) {
    let epsilon = 0.05;
    if delta.abs() < epsilon {
        return ("→", palette.muted);
    }
    match metric {
        TrendMetric::Index if delta > 0.0 => ("↑", palette.success),
        TrendMetric::Index => ("↓", palette.warning),
        TrendMetric::Percentage if delta > 0.0 => ("↑", palette.success),
        TrendMetric::Percentage => ("↓", palette.warning),
        TrendMetric::Duration if delta < 0.0 => ("↓", palette.success),
        TrendMetric::Duration => ("↑", palette.warning),
    }
}

fn sample_series_for_width(series: &[f64], max_points: usize) -> Vec<(f64, f64)> {
    if series.is_empty() || max_points == 0 {
        return Vec::new();
    }
    if series.len() <= max_points {
        return series
            .iter()
            .enumerate()
            .map(|(index, value)| (index as f64, *value))
            .collect();
    }

    let mut sampled = Vec::with_capacity(max_points);
    let last_index = series.len() - 1;
    let denom = max_points.saturating_sub(1).max(1) as f64;
    let mut previous_index: Option<usize> = None;
    for index in 0..max_points {
        let position = (index as f64 / denom) * last_index as f64;
        let source_index = position.round() as usize;
        let clamped_index = source_index.min(last_index);
        if previous_index == Some(clamped_index) {
            continue;
        }
        previous_index = Some(clamped_index);
        sampled.push((sampled.len() as f64, series[clamped_index]));
    }
    if sampled.last().map(|(_, value)| *value) != Some(series[last_index]) {
        sampled.push((sampled.len() as f64, series[last_index]));
    }
    sampled
}

fn submit_timeout_for_answers(answers: &HashMap<String, String>) -> Duration {
    if answers
        .get("response_text")
        .is_some_and(|value| !value.trim().is_empty())
    {
        // Subjective scoring calls the real AI evaluator and may take much longer
        // than objective grading, especially when the upstream model is busy.
        Duration::from_secs(420)
    } else {
        Duration::from_secs(15)
    }
}

fn generation_phase_marker(tick: usize) -> &'static str {
    const MARKERS: [&str; 4] = ["◐", "◓", "◑", "◒"];
    MARKERS[tick % MARKERS.len()]
}

fn generation_orbit_frame(tick: usize) -> &'static str {
    const FRAMES: [&str; 6] = ["◜◎◝", "◠◎◡", "◝◎◜", "◡◎◠", "◟◎◞", "◠◎◡"];
    FRAMES[tick % FRAMES.len()]
}

fn generation_wave_frame(tick: usize) -> &'static str {
    const FRAMES: [&str; 6] = [
        "▁▃▅▇▅▃▁",
        "▃▅▇▅▃▁▃",
        "▅▇▅▃▁▃▅",
        "▇▅▃▁▃▅▇",
        "▅▃▁▃▅▇▅",
        "▃▁▃▅▇▅▃",
    ];
    FRAMES[tick % FRAMES.len()]
}

fn generation_scan_frame(tick: usize) -> &'static str {
    const FRAMES: [&str; 6] = [
        "· · ◆ · ·",
        "· ◆ · ◆ ·",
        "◆ · · · ◆",
        "· ◆ · ◆ ·",
        "· · ◆ · ·",
        "· ◇ ◆ ◇ ·",
    ];
    FRAMES[tick % FRAMES.len()]
}

#[derive(Clone, Debug)]
struct SubjectiveEditorRow {
    text: String,
    cursor_col: Option<usize>,
}

fn subjective_editor_rows(
    text: &str,
    cursor_byte: usize,
    wrap_width: usize,
) -> Vec<SubjectiveEditorRow> {
    let width = wrap_width.max(1);
    let mut rows = Vec::new();
    let mut line_start = 0usize;

    for logical_line in text.split('\n') {
        let line_end = line_start + logical_line.len();
        let local_cursor = if cursor_byte >= line_start && cursor_byte <= line_end {
            Some(text[line_start..cursor_byte].chars().count())
        } else {
            None
        };
        let chars = logical_line.chars().collect::<Vec<_>>();

        if chars.is_empty() {
            rows.push(SubjectiveEditorRow {
                text: String::new(),
                cursor_col: local_cursor.or(if cursor_byte == line_start {
                    Some(0)
                } else {
                    None
                }),
            });
        } else {
            let mut chunk_start = 0usize;
            while chunk_start < chars.len() {
                let chunk_end = (chunk_start + width).min(chars.len());
                let chunk_text = chars[chunk_start..chunk_end].iter().collect::<String>();
                let cursor_col = local_cursor.and_then(|cursor_chars| {
                    if cursor_chars < chunk_end && cursor_chars >= chunk_start {
                        Some(cursor_chars - chunk_start)
                    } else if cursor_chars == chunk_end && chunk_end == chars.len() {
                        Some(cursor_chars - chunk_start)
                    } else {
                        None
                    }
                });
                rows.push(SubjectiveEditorRow {
                    text: chunk_text,
                    cursor_col,
                });
                chunk_start = chunk_end;
            }
        }

        line_start = line_end + 1;
    }

    if rows.is_empty() {
        rows.push(SubjectiveEditorRow {
            text: String::new(),
            cursor_col: Some(0),
        });
    }

    rows
}

fn subjective_cursor_line_col(rows: &[SubjectiveEditorRow]) -> (usize, usize) {
    for (index, row) in rows.iter().enumerate() {
        if let Some(col) = row.cursor_col {
            return (index + 1, col + 1);
        }
    }
    (1, 1)
}

fn subjective_editor_line_spans(
    row: &SubjectiveEditorRow,
    width: usize,
    palette: Palette,
    show_placeholder: bool,
) -> Vec<Span<'static>> {
    let underline_style = Style::default()
        .fg(palette.text)
        .add_modifier(Modifier::UNDERLINED);
    let filler_style = Style::default()
        .fg(palette.border)
        .add_modifier(Modifier::UNDERLINED);
    let text_style = underline_style;
    let muted_underlined = Style::default()
        .fg(palette.muted)
        .add_modifier(Modifier::UNDERLINED);
    let caret_style = if glass_mode(palette) {
        Style::default()
            .fg(palette.accent)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
    } else {
        Style::default()
            .fg(palette.highlight_text)
            .bg(palette.accent)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
    };

    let mut spans = Vec::new();
    if let Some(cursor_col) = row.cursor_col {
        if show_placeholder && row.text.is_empty() {
            spans.push(Span::styled("▏", caret_style));
            spans.push(Span::styled(" 在这里输入你的作文或译文", muted_underlined));
            let used = 1 + " 在这里输入你的作文或译文".chars().count();
            let remaining = width.saturating_sub(used);
            if remaining > 0 {
                spans.push(Span::styled(" ".repeat(remaining), filler_style));
            }
            return spans;
        }

        let chars = row.text.chars().collect::<Vec<_>>();
        let split_at = cursor_col.min(chars.len());
        let left = chars[..split_at].iter().collect::<String>();
        let right = chars[split_at..].iter().collect::<String>();
        if !left.is_empty() {
            spans.push(Span::styled(left.clone(), text_style));
        }
        spans.push(Span::styled("▏", caret_style));
        if !right.is_empty() {
            spans.push(Span::styled(right.clone(), text_style));
        }
        let used = left.chars().count() + right.chars().count() + 1;
        let remaining = width.saturating_sub(used);
        if remaining > 0 {
            spans.push(Span::styled(" ".repeat(remaining), filler_style));
        }
        return spans;
    }

    spans.push(Span::styled(row.text.clone(), text_style));
    let remaining = width.saturating_sub(row.text.chars().count());
    if remaining > 0 {
        spans.push(Span::styled(" ".repeat(remaining), filler_style));
    }
    spans
}

fn handle_practice_input_core(
    practice: &mut PracticeState,
    key: KeyEvent,
    _suppress_enter_submit: bool,
) -> Result<()> {
    if practice.question_set.questions.is_empty() {
        match key.code {
            KeyCode::Left => practice.move_response_cursor_left(),
            KeyCode::Right => practice.move_response_cursor_right(),
            KeyCode::Home => practice.move_response_cursor_home(),
            KeyCode::End => practice.move_response_cursor_end(),
            KeyCode::Up => {
                practice.subjective_scroll = practice.subjective_scroll.saturating_sub(1);
            }
            KeyCode::Down => {
                practice.subjective_scroll = practice.subjective_scroll.saturating_add(1);
            }
            KeyCode::Enter => practice.insert_response_newline(),
            KeyCode::Backspace => practice.backspace_response_char(),
            KeyCode::Delete => practice.clear_current_answer(),
            KeyCode::Tab => practice.insert_response_char(' '),
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                practice.insert_response_char(c);
            }
            KeyCode::PageUp => {
                practice.passage_scroll = practice.passage_scroll.saturating_sub(2);
            }
            KeyCode::PageDown => {
                practice.passage_scroll = practice.passage_scroll.saturating_add(2);
            }
            _ => {}
        }
        return Ok(());
    }

    match practice.question_set.question_type.as_str() {
        "banked_cloze" => match key.code {
            KeyCode::Left => {
                practice.selected_blank = practice.selected_blank.saturating_sub(1);
            }
            KeyCode::Right => {
                practice.selected_blank =
                    (practice.selected_blank + 1).min(practice.question_set.questions.len() - 1);
            }
            KeyCode::Up => {
                practice.selected_blank = practice.selected_blank.saturating_sub(5);
            }
            KeyCode::Down => {
                practice.selected_blank =
                    (practice.selected_blank + 5).min(practice.question_set.questions.len() - 1);
            }
            KeyCode::Char(c) => {
                let upper = c.to_ascii_uppercase().to_string();
                if practice.available_labels().contains(&upper) {
                    practice.assign_answer(upper);
                }
            }
            KeyCode::Backspace | KeyCode::Delete => practice.clear_current_answer(),
            KeyCode::PageUp => {
                practice.passage_scroll = practice.passage_scroll.saturating_sub(3);
            }
            KeyCode::PageDown => {
                practice.passage_scroll = practice.passage_scroll.saturating_add(3);
            }
            _ => {}
        },
        _ => match key.code {
            KeyCode::Up => {
                practice.selected_question = practice.selected_question.saturating_sub(1);
                practice.sync_choice_cursor_to_current_answer();
            }
            KeyCode::Down => {
                practice.selected_question =
                    (practice.selected_question + 1).min(practice.question_set.questions.len() - 1);
                practice.sync_choice_cursor_to_current_answer();
            }
            KeyCode::Left => {
                practice.choice_cursor = practice.choice_cursor.saturating_sub(1);
            }
            KeyCode::Right => {
                let labels = practice.available_labels();
                if !labels.is_empty() {
                    practice.choice_cursor =
                        (practice.choice_cursor + 1).min(labels.len().saturating_sub(1));
                }
            }
            KeyCode::Char(' ') => {
                let labels = practice.available_labels();
                if !labels.is_empty() {
                    practice.assign_answer(labels[practice.choice_cursor].clone());
                }
            }
            KeyCode::Char(c) => {
                let upper = c.to_ascii_uppercase().to_string();
                if practice.available_labels().contains(&upper) {
                    practice.assign_answer(upper);
                }
            }
            KeyCode::Backspace | KeyCode::Delete => practice.clear_current_answer(),
            KeyCode::Enter => {
                let labels = practice.available_labels();
                if !labels.is_empty() {
                    practice.assign_answer(labels[practice.choice_cursor].clone());
                }
            }
            KeyCode::PageUp => {
                practice.passage_scroll = practice.passage_scroll.saturating_sub(3);
            }
            KeyCode::PageDown => {
                practice.passage_scroll = practice.passage_scroll.saturating_add(3);
            }
            KeyCode::Tab => {
                let labels = practice.available_labels();
                if !labels.is_empty() {
                    practice.choice_cursor = (practice.choice_cursor + 1) % labels.len();
                }
            }
            _ => {}
        },
    }
    Ok(())
}

fn build_skill_summary_lines(results: &[AttemptQuestionResult]) -> Vec<Line<'static>> {
    let mut stats: HashMap<String, (usize, usize)> = HashMap::new();
    for item in results {
        let entry = stats.entry(item.skill_tag.clone()).or_insert((0, 0));
        entry.0 += 1;
        if item.is_correct {
            entry.1 += 1;
        }
    }
    let mut rows = stats.into_iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    if rows.is_empty() {
        return vec![Line::from("暂无技能数据。")];
    }
    rows.into_iter()
        .map(|(skill, (total, correct))| {
            let ratio = if total == 0 {
                0.0
            } else {
                correct as f64 / total as f64
            };
            Line::from(format!(
                "{} {} {}/{}",
                format_skill_label(&skill),
                mini_ratio_bar(ratio, 6),
                correct,
                total
            ))
        })
        .collect()
}

fn big_timer_lines(value: &str) -> Vec<Line<'static>> {
    let glyphs = timer_glyphs();
    let mut rows = vec![
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    ];
    for ch in value.chars() {
        let pattern = glyphs
            .get(&ch)
            .cloned()
            .unwrap_or_else(|| glyphs.get(&' ').unwrap().clone());
        for (index, item) in pattern.iter().enumerate() {
            rows[index].push_str(item);
            rows[index].push(' ');
        }
    }
    rows.into_iter()
        .map(|row| Line::from(Span::raw(row)))
        .collect()
}

fn timer_glyphs() -> HashMap<char, [&'static str; 5]> {
    HashMap::from([
        ('0', ["┏━┓", "┃ ┃", "┃ ┃", "┃ ┃", "┗━┛"]),
        ('1', [" ╻ ", " ┃ ", " ┃ ", " ┃ ", " ╹ "]),
        ('2', ["┏━┓", "  ┃", "┏━┛", "┃  ", "┗━┛"]),
        ('3', ["┏━┓", "  ┃", " ━┫", "  ┃", "┗━┛"]),
        ('4', ["╻ ╻", "┃ ┃", "┗━┫", "  ┃", "  ╹"]),
        ('5', ["┏━┓", "┃  ", "┗━┓", "  ┃", "┗━┛"]),
        ('6', ["┏━┓", "┃  ", "┣━┓", "┃ ┃", "┗━┛"]),
        ('7', ["┏━┓", "  ┃", "  ┃", "  ┃", "  ╹"]),
        ('8', ["┏━┓", "┃ ┃", "┣━┫", "┃ ┃", "┗━┛"]),
        ('9', ["┏━┓", "┃ ┃", "┗━┫", "  ┃", "┗━┛"]),
        (':', ["   ", " • ", "   ", " • ", "   "]),
        (' ', ["   ", "   ", "   ", "   ", "   "]),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_question_label_handles_slots() {
        assert_eq!(format_question_label("banked_cloze", None), "选词填空");
        assert_eq!(format_question_label("long_reading", None), "长篇阅读");
        assert_eq!(
            format_question_label("careful_reading", Some(1)),
            "仔细阅读 1"
        );
        assert_eq!(
            format_question_label("careful_reading", Some(2)),
            "仔细阅读 2"
        );
    }

    #[test]
    fn blank_or_value_maps_empty_to_placeholder() {
        assert_eq!(blank_or_value(""), "__");
        assert_eq!(blank_or_value("  "), "__");
        assert_eq!(blank_or_value("A"), "A");
    }

    #[test]
    fn timer_ascii_art_has_five_rows() {
        let lines = big_timer_lines("12:34");
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn timer_lines_fit_sidebar_width() {
        let lines = big_timer_lines("12:34");
        for line in lines {
            let width: usize = line
                .spans
                .iter()
                .map(|span| span.content.chars().count())
                .sum();
            assert!(width <= 20, "timer line too wide: {}", width);
        }
    }

    #[test]
    fn question_group_labels_match_expected_sections() {
        assert_eq!(
            format_question_group_label("banked_cloze", None),
            "Section A"
        );
        assert_eq!(
            format_question_group_label("long_reading", None),
            "Section B"
        );
        assert_eq!(
            format_question_group_label("careful_reading", Some(2)),
            "Section C"
        );
    }

    #[test]
    fn timer_digit_four_uses_lowered_seven_segment_shape() {
        let glyphs = timer_glyphs();
        assert_eq!(glyphs.get(&'4').unwrap()[0], "╻ ╻");
        assert_eq!(glyphs.get(&'4').unwrap()[2], "┗━┫");
    }

    #[test]
    fn submit_timeout_is_extended_for_subjective_answers() {
        let mut answers = HashMap::new();
        answers.insert(
            "response_text".to_string(),
            "This is a subjective response.".to_string(),
        );
        assert_eq!(
            submit_timeout_for_answers(&answers),
            Duration::from_secs(420)
        );
    }

    #[test]
    fn submit_timeout_stays_short_for_objective_answers() {
        let mut answers = HashMap::new();
        answers.insert("q1".to_string(), "A".to_string());
        assert_eq!(
            submit_timeout_for_answers(&answers),
            Duration::from_secs(15)
        );
    }

    #[test]
    fn directions_cover_exam_instructions() {
        assert!(format_directions("banked_cloze").contains("ten blanks"));
        assert!(format_directions("long_reading").contains("ten statements attached"));
        assert!(format_directions("careful_reading").contains("questions or unfinished statements"));
    }

    #[test]
    fn pace_band_tracks_elapsed_ratio() {
        assert_eq!(pace_band(0.4), "节奏从容");
        assert_eq!(pace_band(0.8), "节奏正常");
        assert_eq!(pace_band(0.98), "接近建议时长");
        assert_eq!(pace_band(1.2), "节奏偏慢");
    }

    #[test]
    fn wrapped_block_height_clamps_expected_range() {
        assert_eq!(wrapped_block_height("short", 20, 2, 4), 2);
        assert_eq!(
            wrapped_block_height("a very long option that should wrap", 8, 2, 4),
            4
        );
    }

    #[test]
    fn generation_phase_labels_are_human_readable() {
        assert_eq!(format_generation_phase("generate_request"), "正式出题");
        assert_eq!(format_generation_phase("repair"), "结构修复");
    }

    #[test]
    fn estimated_generation_time_labels_match_major_types() {
        assert_eq!(
            TypeChoice::BankedCloze.estimated_generation_time(LevelChoice::Cet4),
            "预计生成 20-35 秒"
        );
        assert_eq!(
            TypeChoice::LongReading.estimated_generation_time(LevelChoice::Cet6),
            "预计生成 60-120 秒"
        );
        assert_eq!(
            TypeChoice::Careful2.estimated_generation_time(LevelChoice::Cet6),
            "预计生成 45-90 秒"
        );
    }

    #[test]
    fn trend_helpers_format_percentage_and_duration_as_expected() {
        assert_eq!(trend_value_text(TrendMetric::Index, 92.4), "92");
        assert_eq!(trend_value_text(TrendMetric::Percentage, 83.25), "83.2%");
        assert_eq!(trend_value_text(TrendMetric::Duration, 125.0), "02:05");
        assert_eq!(trend_delta_text(TrendMetric::Index, -7.0), "-7");
        assert_eq!(trend_delta_text(TrendMetric::Percentage, 4.2), "+4.2%");
        assert_eq!(trend_delta_text(TrendMetric::Duration, -18.0), "-00:18");
    }

    #[test]
    fn generation_heartbeat_detection_catches_wait_messages() {
        assert!(is_generation_heartbeat(
            "DeepSeek 正在生成题目，已等待 36 秒。"
        ));
        assert!(!is_generation_heartbeat(
            "题目已通过校验，正在保存题集与词汇。"
        ));
    }

    fn test_mock_question_set() -> QuestionSet {
        QuestionSet {
            id: "mock-writing".to_string(),
            level: "cet4".to_string(),
            question_type: "writing".to_string(),
            title: "Mock Writing".to_string(),
            topic: "campus learning".to_string(),
            passage: Passage {
                title: "Mock Writing".to_string(),
                paragraphs: vec!["Write an essay about campus learning.".to_string()],
            },
            questions: vec![],
            answer_key: vec![],
            analysis: AnalysisReport {
                overall_strategy: String::new(),
                overall_summary: String::new(),
                item_explanations: vec![],
                test_tips: vec![],
            },
            vocabulary: vec![],
            shared_options: vec![],
            task_prompt: "Write at least 120 words.".to_string(),
            reference_answer: String::new(),
            rubric_focus: vec![],
            min_response_words: 120,
            max_response_words: 180,
            slot: None,
            word_count: 0,
            created_at: String::new(),
            generator_model: "test".to_string(),
            source_type: "ai".to_string(),
        }
    }

    #[test]
    fn mock_exam_total_remaining_uses_writing_remaining_before_lock() {
        let mut session = MockExamSession::new(LevelChoice::Cet4, test_mock_question_set());
        session.started_instant = Instant::now() - Duration::from_secs(15 * 60);
        if let Some(section) = session.section_mut(TypeChoice::Writing) {
            section.practice.started_instant = Instant::now() - Duration::from_secs(15 * 60);
        }
        assert_eq!(session.writing_remaining_seconds(), 15 * 60);
        assert_eq!(session.total_remaining_seconds(), 85 * 60);
    }

    #[test]
    fn mock_exam_total_remaining_resets_to_shared_seventy_minutes_after_writing_lock() {
        let mut session = MockExamSession::new(LevelChoice::Cet4, test_mock_question_set());
        session.started_instant = Instant::now() - Duration::from_secs(12 * 60);
        session.lock_writing();
        assert_eq!(session.total_remaining_seconds(), 70 * 60);
    }
}
