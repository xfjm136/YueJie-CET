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
    Block, BorderType, Borders, Gauge, List, ListItem, ListState, Paragraph, Sparkline, Wrap,
};
use ratatui::{Frame, Terminal};
use serde::Deserialize;
use serde_json::{Value, json};

const MIN_TERMINAL_WIDTH: u16 = 120;
const MIN_TERMINAL_HEIGHT: u16 = 34;

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
struct SettingsResponse {
    ok: bool,
    settings: BridgeSettings,
}

#[derive(Clone, Debug, Deserialize)]
struct OverviewData {
    total_attempts: usize,
    total_cet4: usize,
    total_cet6: usize,
    recent_accuracy_percent: f64,
    recent_duration_text: String,
    recent_accuracy_series: Vec<f64>,
    recent_duration_series: Vec<i64>,
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
    Practice,
    Result,
    History,
    Review,
    Weakness,
    Vocabulary,
    Settings,
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
            Self::Careful2 => "Section D",
            Self::Translation => "Part IV",
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

    fn brief(self) -> &'static str {
        match self {
            Self::BankedCloze => "词性、搭配、上下文逻辑同步考查",
            Self::LongReading => "信息定位、同义改写、段落匹配",
            Self::Careful1 => "标准仔细阅读，覆盖主旨到细节",
            Self::Careful2 => "更强调推断、态度与语境词义",
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

struct GeneratingTask {
    job_id: u64,
    started_at: Instant,
    receiver: Receiver<GenerationMessage>,
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
        let value = self.run_bridge(&["submit"], Some(payload))?;
        Ok(serde_json::from_value(value)?)
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

    fn weakness(&self) -> Result<WeaknessResponse> {
        let value = self.run_bridge(&["weakness", "--limit", "20"], None)?;
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
    generating_tick: usize,
    generation_sequence: u64,
    generation_phase: String,
    generation_message: String,
    generation_log: Vec<String>,
    practice: Option<PracticeState>,
    result: Option<AttemptResult>,
    result_detail_scroll: u16,
    history: Vec<HistoryEntry>,
    history_index: usize,
    review: Option<ReviewBundle>,
    review_back_screen: Screen,
    review_detail_scroll: u16,
    weakness: Vec<WeaknessEntry>,
    weakness_index: usize,
    vocabulary: Vec<VocabularyEntry>,
    vocabulary_index: usize,
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
            status_line: String::from("方向键、Enter、Esc、Q 和鼠标都可使用。"),
            generating_task: None,
            generating_tick: 0,
            generation_sequence: 0,
            generation_phase: String::from("idle"),
            generation_message: String::from("尚未开始生成。"),
            generation_log: Vec::new(),
            practice: None,
            result: None,
            result_detail_scroll: 0,
            history: Vec::new(),
            history_index: 0,
            review: None,
            review_back_screen: Screen::Home,
            review_detail_scroll: 0,
            weakness: Vec::new(),
            weakness_index: 0,
            vocabulary: Vec::new(),
            vocabulary_index: 0,
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
                                    self.sync_selection_from_question_set(&question_set);
                                    self.status_line = match question_set.source_type.as_str() {
                                        "ai" => String::from("题目已生成，开始作答。"),
                                        "ai_repaired" => {
                                            String::from("题目已生成，并已自动修复结构后进入作答。")
                                        }
                                        other => format!("题目来源：{}，现在开始作答。", other),
                                    };
                                    self.practice = Some(PracticeState::new(question_set, false));
                                    self.screen = Screen::Practice;
                                }
                                Err(message) => {
                                    self.screen = Screen::TypeSelect;
                                    self.status_line = message;
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
            } else if matches!(self.screen, Screen::Generating | Screen::Result) {
                self.generating_tick = self.generating_tick.wrapping_add(1);
            }
        }
        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => return Ok(true),
            KeyCode::Esc => {
                self.handle_escape()?;
                return Ok(false);
            }
            _ => {}
        }

        match self.screen {
            Screen::Home => self.handle_home_key(key),
            Screen::LevelSelect => self.handle_level_key(key),
            Screen::TypeSelect => self.handle_type_key(key)?,
            Screen::Generating => {}
            Screen::Practice => self.handle_practice_key(key)?,
            Screen::Result => self.handle_result_key(key)?,
            Screen::History => self.handle_history_key(key)?,
            Screen::Review => self.handle_review_key(key)?,
            Screen::Weakness => self.handle_weakness_key(key),
            Screen::Vocabulary => self.handle_vocabulary_key(key),
            Screen::Settings => self.handle_settings_key(key)?,
        }
        Ok(false)
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
                self.open_type_screen()?;
                self.status_line = String::from("已取消本次生成。");
            }
            Screen::Practice => {
                self.open_type_screen()?;
                self.status_line = String::from("已返回题型选择。");
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
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('s') | KeyCode::Char('S'))
        {
            self.submit_practice()?;
            return Ok(());
        }
        if let Some(practice) = &mut self.practice {
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
                        practice.selected_blank = (practice.selected_blank + 1)
                            .min(practice.question_set.questions.len() - 1);
                    }
                    KeyCode::Up => {
                        practice.selected_blank = practice.selected_blank.saturating_sub(5);
                    }
                    KeyCode::Down => {
                        practice.selected_blank = (practice.selected_blank + 5)
                            .min(practice.question_set.questions.len() - 1);
                    }
                    KeyCode::Char(c) => {
                        let upper = c.to_ascii_uppercase().to_string();
                        if practice.available_labels().contains(&upper) {
                            practice.assign_answer(upper);
                        }
                    }
                    KeyCode::Backspace | KeyCode::Delete => practice.clear_current_answer(),
                    KeyCode::Enter => self.submit_practice()?,
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
                        practice.selected_question = (practice.selected_question + 1)
                            .min(practice.question_set.questions.len() - 1);
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
                        } else {
                            self.submit_practice()?;
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
        }
        Ok(())
    }

    fn handle_result_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Char('1') | KeyCode::Enter => {
                self.perform_action(Action::ResultViewAnalysis)?
            }
            KeyCode::Char('2') => self.perform_action(Action::ResultContinue)?,
            KeyCode::Char('3') => self.perform_action(Action::ResultRedo)?,
            KeyCode::Char('4') => self.perform_action(Action::ResultBackTypes)?,
            KeyCode::Char('5') => self.perform_action(Action::ResultBackHome)?,
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
        if self.history.is_empty() {
            return Ok(());
        }
        match key.code {
            KeyCode::Up => {
                self.history_index = self.history_index.saturating_sub(1);
                self.pending_history_delete_attempt_id = None;
            }
            KeyCode::Down => {
                self.history_index = (self.history_index + 1).min(self.history.len() - 1);
                self.pending_history_delete_attempt_id = None;
            }
            KeyCode::Enter => self.perform_action(Action::HistoryReview(self.history_index))?,
            KeyCode::Char('r') => self.perform_action(Action::HistoryRedo(self.history_index))?,
            KeyCode::Char('d') | KeyCode::Char('D') => {
                self.perform_action(Action::HistoryDelete(self.history_index))?
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_review_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Char('r') | KeyCode::Enter => self.perform_action(Action::ReviewRedo)?,
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
                    .find(|area| contains(area.rect, x, y))
                    .map(|area| area.action.clone())
                {
                    self.perform_action(action)?;
                }
            }
            MouseEventKind::ScrollUp => match self.screen {
                Screen::Practice => {
                    if let Some(practice) = &mut self.practice {
                        practice.passage_scroll = practice.passage_scroll.saturating_sub(2);
                    }
                }
                Screen::Result => {
                    self.result_detail_scroll = self.result_detail_scroll.saturating_sub(2);
                }
                Screen::Review => {
                    self.review_detail_scroll = self.review_detail_scroll.saturating_sub(2);
                }
                Screen::History if !self.history.is_empty() => {
                    self.history_index = self.history_index.saturating_sub(1);
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
                    if let Some(practice) = &mut self.practice {
                        practice.passage_scroll = practice.passage_scroll.saturating_add(2);
                    }
                }
                Screen::Result => {
                    self.result_detail_scroll = self.result_detail_scroll.saturating_add(2);
                }
                Screen::Review => {
                    self.review_detail_scroll = self.review_detail_scroll.saturating_add(2);
                }
                Screen::History if !self.history.is_empty() => {
                    self.history_index = (self.history_index + 1).min(self.history.len() - 1);
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
                0 => self.screen = Screen::LevelSelect,
                1 => self.open_history()?,
                2 => self.open_weakness()?,
                3 => self.open_vocabulary()?,
                4 => self.open_settings()?,
                5 => self.should_quit = true,
                _ => {}
            },
            Action::Level(level) => {
                self.selected_level = level;
                self.open_type_screen()?;
            }
            Action::Type(type_choice) => {
                self.selected_type = type_choice;
                self.type_index = TypeChoice::all()
                    .iter()
                    .position(|item| *item == type_choice)
                    .unwrap_or(0);
                self.start_generation()?;
            }
            Action::BackHome => self.return_home()?,
            Action::BackType => self.open_type_screen()?,
            Action::BackHistory => self.return_home()?,
            Action::BackWeakness => self.return_home()?,
            Action::BackVocabulary => self.return_home()?,
            Action::BackSettings => self.return_home()?,
            Action::BackReview => {
                self.screen = self.review_back_screen;
                self.status_line = if self.review_back_screen == Screen::History {
                    String::from("已返回刷题历史。")
                } else {
                    String::from("已返回结果总览。")
                };
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
            Action::SubmitPractice => self.submit_practice()?,
            Action::PracticeBack => {
                self.open_type_screen()?;
                self.status_line = String::from("已返回题型选择。");
            }
            Action::PracticeSelectBlank(index) => {
                if let Some(practice) = &mut self.practice {
                    practice.selected_blank = index.min(practice.question_set.questions.len() - 1);
                    practice.submit_confirm_pending = false;
                }
            }
            Action::PracticeSelectQuestion(index) => {
                if let Some(practice) = &mut self.practice {
                    practice.selected_question =
                        index.min(practice.question_set.questions.len() - 1);
                    practice.submit_confirm_pending = false;
                    practice.sync_choice_cursor_to_current_answer();
                }
            }
            Action::PracticeAssign(answer) => {
                if let Some(practice) = &mut self.practice {
                    practice.assign_answer(answer);
                }
            }
            Action::ResultContinue => self.start_generation()?,
            Action::ResultViewAnalysis => {
                self.review_back_screen = Screen::Result;
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
        }
        Ok(())
    }

    fn open_type_screen(&mut self) -> Result<()> {
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

    fn start_generation(&mut self) -> Result<()> {
        let backend = BackendBridge::new()?;
        let level = self.selected_level;
        let type_choice = self.selected_type;
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

    fn submit_practice(&mut self) -> Result<()> {
        if let Some(practice) = &mut self.practice {
            if practice.question_set.questions.is_empty()
                && practice.response_text().trim().is_empty()
            {
                self.status_line = String::from("写作/翻译作答区为空，请先输入内容后再提交。");
                return Ok(());
            }
            let unanswered = practice.unanswered_count();
            if unanswered > 0 && !practice.submit_confirm_pending {
                practice.submit_confirm_pending = true;
                self.status_line = format!(
                    "还有 {} 题未作答，再按一次 Ctrl+S / Enter 将按当前答案交卷。",
                    unanswered
                );
                return Ok(());
            }
            practice.submit_confirm_pending = false;
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
        self.history_index = 0;
        self.pending_history_delete_attempt_id = None;
        self.screen = Screen::History;
        self.status_line = String::from(
            "上下方向键切换，Enter 查看解析，R 重新作答，D 删除记录，滚轮可快速浏览。",
        );
        Ok(())
    }

    fn open_weakness(&mut self) -> Result<()> {
        self.weakness = self.backend.weakness()?.weakness;
        self.weakness_index = 0;
        self.screen = Screen::Weakness;
        self.status_line = String::from("上下方向键或鼠标选择条目，右侧查看详情。");
        Ok(())
    }

    fn open_vocabulary(&mut self) -> Result<()> {
        self.vocabulary = self.backend.vocabulary()?.vocabulary;
        self.vocabulary_index = 0;
        self.screen = Screen::Vocabulary;
        self.status_line = String::from("上下方向键或鼠标选择词汇，右侧查看详情。");
        Ok(())
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
            Screen::Practice => self.draw_practice(frame, area, palette),
            Screen::Result => self.draw_result(frame, area, palette),
            Screen::History => self.draw_history(frame, area, palette),
            Screen::Review => self.draw_review(frame, area, palette),
            Screen::Weakness => self.draw_weakness(frame, area, palette),
            Screen::Vocabulary => self.draw_vocabulary(frame, area, palette),
            Screen::Settings => self.draw_settings(frame, area, palette),
        }
    }

    fn draw_home(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let total_attempts = self.overview.total_attempts;
        let total_cet4 = self.overview.total_cet4;
        let total_cet6 = self.overview.total_cet6;
        let recent_accuracy_percent = self.overview.recent_accuracy_percent;
        let recent_duration_text = self.overview.recent_duration_text.clone();
        let recent_duration_ratio = (self
            .overview
            .recent_duration_series
            .last()
            .copied()
            .unwrap_or(0) as f64
            / 2400.0)
            .min(1.0);
        let most_common_type_label = self.overview.most_common_type_label.clone();
        let latest_weakness_text = self
            .overview
            .latest_weakness_updated_at
            .clone()
            .unwrap_or_else(|| "薄弱项暂无更新".to_string());
        let accuracy_series: Vec<u64> = self
            .overview
            .recent_accuracy_series
            .iter()
            .map(|value| value.max(0.0).round() as u64)
            .collect();
        let duration_series: Vec<u64> = self
            .overview
            .recent_duration_series
            .iter()
            .map(|value| (*value).max(0) as u64)
            .collect();
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
                        "AI 四六级阅读专项训练 · {} · {} · 背景{}",
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
                "最近 5 次正确率 {}，当前状态：{}。",
                format!("{:.1}%", recent_accuracy_percent),
                accuracy_band(recent_accuracy_percent / 100.0)
            )),
            Line::from(format!(
                "训练建议：{}",
                if total_attempts == 0 {
                    "先从四级选词填空开始，熟悉界面和出题风格。"
                } else if recent_accuracy_percent < 60.0 {
                    "先继续当前题型，优先看解析和词汇，拉稳基础正确率。"
                } else if recent_accuracy_percent < 80.0 {
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
                Line::from(format!("最常练题型：{}", most_common_type_label)),
                Line::from(format!("最近平均用时：{}", recent_duration_text)),
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
            &format!("四级 {} / 六级 {}", total_cet4, total_cet6),
            None,
        );
        self.draw_metric_box(
            frame,
            stat_chunks[1],
            palette,
            "最近正确率",
            &format!("{:.1}%", recent_accuracy_percent),
            accuracy_band(recent_accuracy_percent / 100.0),
            Some(recent_accuracy_percent / 100.0),
        );
        self.draw_metric_box(
            frame,
            stat_chunks[2],
            palette,
            "平均用时",
            &recent_duration_text,
            "近 5 次均值",
            Some(recent_duration_ratio),
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
                Constraint::Length(5),
                Constraint::Length(5),
                Constraint::Min(4),
            ])
            .split(bottom[0]);
        frame.render_widget(
            Sparkline::default()
                .block(simple_block("正确率走势", palette))
                .data(&accuracy_series)
                .style(Style::default().fg(palette.success))
                .absent_value_style(Style::default().fg(palette.muted))
                .bar_set(symbols::bar::NINE_LEVELS),
            trend_rows[0],
        );
        frame.render_widget(
            Sparkline::default()
                .block(simple_block("用时走势", palette))
                .data(&duration_series)
                .style(Style::default().fg(palette.warning))
                .absent_value_style(Style::default().fg(palette.muted))
                .bar_set(symbols::bar::NINE_LEVELS),
            trend_rows[1],
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
            ("开始刷题", "进入等级与题型选择", Action::HomeMenu(0)),
            ("刷题历史", "查看过往记录与复盘", Action::HomeMenu(1)),
            ("我的薄弱项", "查看动态能力总结", Action::HomeMenu(2)),
            ("词汇表", "浏览高频重点词汇", Action::HomeMenu(3)),
            ("设置", "主题、背景与配色", Action::HomeMenu(4)),
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
                Line::from("1. 选词填空先看词性与固定搭配。"),
                Line::from("2. 长篇阅读先扫题干，再回文定位。"),
                Line::from("3. 仔细阅读优先抓首段和转折句。"),
                Line::from(""),
                Line::from(format!(
                    "当前配色：{}",
                    palette_mode_label(&self.settings.palette_mode)
                )),
                Line::from(format!(
                    "当前状态：{}",
                    accuracy_band(recent_accuracy_percent / 100.0)
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
            Line::from(Span::styled("选择等级", title_style(palette))),
            Line::from(Span::styled(
                "左右方向键或鼠标点击切换，Enter 确认。",
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
                Constraint::Length(11),
                Constraint::Length(11),
                Constraint::Length(11),
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
                "AI 正在生成真题级阅读题",
                title_style(palette),
            )),
            Line::from(Span::styled(
                format!(
                    "{} · {} · {} · 已等待 {}",
                    self.selected_level.label(),
                    self.selected_type.section_label(),
                    self.selected_type.label(),
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
            .constraints([Constraint::Length(6), Constraint::Min(6)])
            .split(body[0]);
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(7), Constraint::Min(7)])
            .split(body[1]);

        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled("当前状态", title_style(palette))),
                Line::from(self.generation_message.clone()),
                Line::from(format!("题型特征：{}", self.selected_type.brief())),
                Line::from(self.selected_type.recommended_time()),
            ]))
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
                .wrap(Wrap { trim: false })
                .block(simple_block("", palette)),
            right[1],
        );
        self.draw_status_line(frame, chunks[3], palette);
    }

    fn draw_practice(&mut self, frame: &mut Frame, area: Rect, palette: Palette) {
        let Some(practice) = self.practice.clone() else {
            return;
        };
        let root = centered_rect(98, 94, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(20)])
            .split(root);
        self.draw_practice_header(frame, chunks[0], palette, &practice);
        if practice.question_set.questions.is_empty() {
            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(24), Constraint::Min(60)])
                .split(chunks[1]);
            self.draw_practice_sidebar(frame, columns[0], palette, &practice);
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

            self.draw_practice_sidebar(frame, columns[0], palette, &practice);
            self.draw_passage_panel(frame, columns[1], palette, &practice);
            self.draw_answer_panel(frame, columns[2], palette, &practice);
        }
    }

    fn draw_practice_header(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        palette: Palette,
        practice: &PracticeState,
    ) {
        let total_items = practice.question_set.questions.len().max(1);
        let parts = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(64), Constraint::Percentage(36)])
            .split(area);
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    format!(
                        "{} · {} · {}",
                        format_level_label(&practice.question_set.level),
                        format_question_group_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot,
                        ),
                        format_question_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot,
                        )
                    ),
                    title_style(palette),
                )),
                Line::from(format!(
                    "主题：{} | 建议用时：{} | 题目数：{}",
                    practice.question_set.topic,
                    recommended_time_text(&practice.question_set.question_type),
                    if practice.question_set.questions.is_empty() {
                        1
                    } else {
                        practice.question_set.questions.len()
                    }
                )),
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
                        "{} · {} · {}",
                        format_level_label(&practice.question_set.level),
                        format_question_label(
                            &practice.question_set.question_type,
                            practice.question_set.slot
                        ),
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
                    practice.question_set.source_type, practice.question_set.generator_model
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
            true,
            Action::ResultViewAnalysis,
        );
        self.draw_action_button(
            frame,
            buttons[1],
            palette,
            "下一题",
            false,
            Action::ResultContinue,
        );
        self.draw_action_button(
            frame,
            buttons[2],
            palette,
            "重做本题",
            false,
            Action::ResultRedo,
        );
        self.draw_action_button(
            frame,
            buttons[3],
            palette,
            "题型选择",
            false,
            Action::ResultBackTypes,
        );
        self.draw_action_button(
            frame,
            buttons[4],
            palette,
            "回到首页",
            false,
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
                Constraint::Length(5),
                Constraint::Min(10),
                Constraint::Length(5),
                Constraint::Length(1),
            ])
            .split(outer);
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled("刷题历史", title_style(palette))),
            Line::from(Span::styled(
                "这里保留每次训练记录，方便回看解析、词汇和再次作答。",
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        let recent_items = self.history.iter().take(5).collect::<Vec<_>>();
        let recent_count = recent_items.len().max(1) as f64;
        let recent_accuracy =
            recent_items.iter().map(|item| item.accuracy).sum::<f64>() / recent_count;
        let recent_duration = recent_items
            .iter()
            .map(|item| item.duration_seconds)
            .sum::<i64>()
            / recent_items.len().max(1) as i64;
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
            "历史总数",
            &format!("{}", self.history.len()),
            "已保存训练记录",
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

        if self.history.is_empty() {
            frame.render_widget(
                Paragraph::new("还没有历史记录。")
                    .alignment(Alignment::Center)
                    .block(simple_block("历史为空", palette)),
                chunks[2],
            );
        } else {
            let body = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[2]);
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
                            item.submitted_at,
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
                    Line::from(format!("时间：{}", selected.submitted_at)),
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
        }

        let buttons = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Ratio(1, 4),
                Constraint::Ratio(1, 4),
                Constraint::Ratio(1, 4),
                Constraint::Ratio(1, 4),
            ])
            .split(chunks[3]);
        self.draw_action_button(
            frame,
            buttons[0],
            palette,
            "查看解析",
            true,
            Action::HistoryReview(self.history_index),
        );
        self.draw_action_button(
            frame,
            buttons[1],
            palette,
            "重新作答",
            false,
            Action::HistoryRedo(self.history_index),
        );
        self.draw_action_button(
            frame,
            buttons[2],
            palette,
            "删除记录",
            false,
            Action::HistoryDelete(self.history_index),
        );
        self.draw_action_button(
            frame,
            buttons[3],
            palette,
            "返回",
            false,
            Action::BackHistory,
        );
        self.draw_status_line(frame, chunks[4], palette);
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
                "{} · {} · {}",
                format_question_label(&bundle.question_set.question_type, bundle.question_set.slot),
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
                    bundle.question_set.source_type, bundle.question_set.generator_model
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
            true,
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
            false,
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
                "这里会显示同题型累计训练后的能力总结。",
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        if self.weakness.is_empty() {
            frame.render_widget(
                Paragraph::new("还没有足够数据来生成薄弱项。")
                    .alignment(Alignment::Center)
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
                            item.updated_at,
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
                    Line::from(format!("更新时间：{}", selected.updated_at)),
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
                "按刷题历史中的高频重点词汇累计。",
                Style::default().fg(palette.muted),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(simple_block("", palette));
        frame.render_widget(header, chunks[0]);

        if self.vocabulary.is_empty() {
            frame.render_widget(
                Paragraph::new("还没有词汇数据。")
                    .alignment(Alignment::Center)
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
                    Line::from(format!("最近出现：{}", selected.last_seen_at)),
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
                Constraint::Length(8),
                Constraint::Length(6),
                Constraint::Length(6),
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
                    "词数：{} | 未答：{}",
                    practice.question_set.word_count,
                    practice.unanswered_count()
                )),
                Line::from(format!(
                    "建议：{} | 来源：{}",
                    recommended_time_text(&practice.question_set.question_type),
                    practice.question_set.source_type
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
            Action::PracticeBack,
        );
        frame.render_widget(
            Paragraph::new(if practice.question_set.questions.is_empty() {
                "Ctrl+S 提交 | Enter 换行 | 左右移动光标 | 上下滚动作答区 | PageUp/PageDown 滚动题面"
            } else {
                "Ctrl+S 提交 | Backspace 清空当前答案 | Space/Enter 选中项 | PageUp/PageDown 滚动文章"
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
                title_style(palette),
            )),
            Line::from(practice.question_set.task_prompt.clone()),
            Line::from(""),
        ];
        for line in &practice.question_set.passage.paragraphs {
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
            Line::from(Span::styled(
                practice.question_set.passage.title.clone(),
                Style::default().fg(palette.text),
            )),
            Line::from(""),
        ];
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
                Constraint::Length(5),
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
            .constraints([Constraint::Length(2), Constraint::Length(2)])
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
                let text = format!("{:02} [{}]", index + 1, answer);
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
            self.draw_list_like_button(
                frame,
                target,
                palette,
                option,
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
                Constraint::Length(2),
                Constraint::Length(1),
            ])
            .split(inner);
        frame.render_widget(
            Paragraph::new(format!(
                "{}  {}",
                action_type.section_label(),
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
            Gauge::default()
                .ratio((card.recent_accuracy_percent / 100.0).clamp(0.0, 1.0))
                .gauge_style(Style::default().fg(palette.success).bg(palette.panel_alt)),
            chunks[3],
        );
        frame.render_widget(
            Paragraph::new(format!(
                "峰值 {:.1}% {}",
                card.best_recent_accuracy_percent,
                mini_ratio_bar(card.best_recent_accuracy_percent / 100.0, 8)
            ))
            .alignment(Alignment::Center)
            .style(Style::default().fg(palette.warning)),
            chunks[4],
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
            chunks[5],
        );
        frame.render_widget(
            Paragraph::new(action_type.brief())
                .alignment(Alignment::Center)
                .style(Style::default().fg(palette.text)),
            chunks[6],
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

fn generation_step_index(phase: &str) -> usize {
    match phase {
        "boot" | "prepare" => 0,
        "blueprint" => 1,
        "generate_request" => 2,
        "validate" | "repair" | "validated" => 3,
        "save" | "done" => 4,
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
        "save" => "保存结果",
        "done" => "生成完成",
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
        ("careful_reading", Some(2)) => "Section D".to_string(),
        ("careful_reading", _) => "Section C".to_string(),
        ("translation", _) => "Part IV".to_string(),
        _ => "Reading Set".to_string(),
    }
}

fn format_directions(question_type: &str) -> &'static str {
    match question_type {
        "writing" => {
            "Directions: For this part, you are allowed 30 minutes to write an essay according to the task given below."
        }
        "banked_cloze" => {
            "Directions: In this section, there is a passage with ten blanks. Choose one word for each blank from the fifteen choices in the answer bank."
        }
        "long_reading" => {
            "Directions: In this section, you will read a long passage with paragraphs labeled A to K. Match each statement with the paragraph from which the information is derived."
        }
        "careful_reading" => {
            "Directions: Read the passage carefully and choose the best answer to each question according to the information given in the passage."
        }
        "translation" => {
            "Directions: For this part, you are allowed 30 minutes to translate the following Chinese passage into English."
        }
        _ => "Directions: Read the passage and complete the questions below.",
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
        ('7', ["┏━┓", "  ┃", "  ┃", "  ┃", "  ┃"]),
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
            "Section D"
        );
    }

    #[test]
    fn timer_digit_four_uses_lowered_seven_segment_shape() {
        let glyphs = timer_glyphs();
        assert_eq!(glyphs.get(&'4').unwrap()[0], "╻ ╻");
        assert_eq!(glyphs.get(&'4').unwrap()[2], "┗━┫");
    }

    #[test]
    fn directions_cover_exam_instructions() {
        assert!(format_directions("banked_cloze").contains("ten blanks"));
        assert!(format_directions("long_reading").contains("Match each statement"));
        assert!(format_directions("careful_reading").contains("choose the best answer"));
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
    fn generation_heartbeat_detection_catches_wait_messages() {
        assert!(is_generation_heartbeat(
            "DeepSeek 正在生成题目，已等待 36 秒。"
        ));
        assert!(!is_generation_heartbeat(
            "题目已通过校验，正在保存题集与词汇。"
        ));
    }
}
