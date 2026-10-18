//! P2 orchestrator: layered composition, structural dictionary queries, typed actions.
mod edit_line;
mod lattice;
mod romaji;

pub use edit_line::{EditLine, TextLayer, TextRange};
pub use lattice::{
    DictionaryQuery, InputEdge, InputLattice, MemoryDictionary, OutputEdge, OutputLattice,
    OutputPath, QueryWord,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InputMode {
    Direct,
    #[default]
    Hiragana,
    Katakana,
    FullWidthAscii,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Phase {
    #[default]
    Direct,
    Composing,
    Converting,
    Predicting,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Key {
    Character(char),
    Space,
    Enter,
    Tab,
    Escape,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // Independent physical modifier keys.
pub struct Modifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    pub meta: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaretInfo {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
    pub dpi: u32,
    pub monitor_id: String,
}
impl Default for CaretInfo {
    fn default() -> Self {
        Self {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
            dpi: 96,
            monitor_id: String::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateId {
    pub session_id: u64,
    pub generation: u64,
    pub index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateView {
    pub id: CandidateId,
    pub text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    KeyDown {
        key: Key,
        modifiers: Modifiers,
    },
    TextInput {
        text: String,
    },
    FocusIn {
        context_id: u64,
        caret: CaretInfo,
        committed_context: String,
    },
    FocusOut,
    CaretMoved {
        caret: CaretInfo,
    },
    SetSelection {
        position: usize,
        length: usize,
    },
    SetMode {
        mode: InputMode,
    },
    UpdateContext {
        text: String,
    },
    CandidateSelected {
        id: CandidateId,
    },
    PredictionReady {
        session_id: u64,
        generation: u64,
        result: Result<String, String>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    UpdateComposition {
        text: String,
        cursor: usize,
        selection_len: usize,
    },
    ShowCandidates {
        candidates: Vec<CandidateView>,
        selected: usize,
        caret: CaretInfo,
        generation: u64,
    },
    HideCandidates,
    Commit {
        text: String,
    },
    SetMode {
        mode: InputMode,
    },
    RequestPrediction {
        session_id: u64,
        generation: u64,
        context: String,
        prefix: String,
    },
    CancelPrediction {
        session_id: u64,
        generation: u64,
    },
    ShowGhost {
        generation: u64,
        text: String,
    },
    HideGhost,
    /// A P4 write queue consumes this; event processing performs no persistence.
    LearnSelection {
        words: Vec<QueryWord>,
    },
    Error {
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventResult {
    pub consumed: bool,
    pub actions: Vec<Action>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControllerConfig {
    pub candidate_limit: usize,
    pub prediction_enabled: bool,
    pub initial_mode: InputMode,
}
impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            candidate_limit: 5,
            prediction_enabled: false,
            initial_mode: InputMode::Hiragana,
        }
    }
}

/// The host assigns unique session IDs and serializes events for each session.
/// The dictionary reader is injected and owns no frontend or OS state.
#[derive(Debug)]
pub struct ImeSession<D> {
    session_id: u64,
    context_id: Option<u64>,
    mode: InputMode,
    phase: Phase,
    generation: u64,
    line: EditLine,
    dictionary: D,
    config: ControllerConfig,
    context: String,
    caret: CaretInfo,
    output: Option<OutputLattice>,
    paths: Vec<OutputPath>,
    selected: usize,
    prediction_origin: Phase,
    ghost: Option<String>,
}

impl<D: DictionaryQuery> ImeSession<D> {
    #[must_use]
    pub fn new(session_id: u64, dictionary: D, mut config: ControllerConfig) -> Self {
        config.candidate_limit = config.candidate_limit.clamp(1, 100);
        Self {
            session_id,
            context_id: None,
            mode: config.initial_mode,
            phase: Phase::Direct,
            generation: 0,
            line: EditLine::default(),
            dictionary,
            config,
            context: String::new(),
            caret: CaretInfo::default(),
            output: None,
            paths: Vec::new(),
            selected: 0,
            prediction_origin: Phase::Direct,
            ghost: None,
        }
    }
    #[must_use]
    pub fn edit_line(&self) -> &EditLine {
        &self.line
    }
    #[must_use]
    pub fn phase(&self) -> Phase {
        self.phase
    }
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
    #[must_use]
    pub fn committed_context(&self) -> &str {
        &self.context
    }
    #[must_use]
    pub fn candidates(&self) -> Vec<CandidateView> {
        self.paths
            .iter()
            .enumerate()
            .map(|(index, path)| CandidateView {
                id: CandidateId {
                    session_id: self.session_id,
                    generation: self.generation,
                    index,
                },
                text: path.text.clone(),
            })
            .collect()
    }

    /// No model inference or file I/O runs here. The host executes actions in order.
    #[must_use]
    pub fn process_event(&mut self, event: Event) -> EventResult {
        let mut actions = Vec::new();
        let consumed = match event {
            Event::FocusIn {
                context_id,
                caret,
                committed_context,
            } => {
                if self.context_id != Some(context_id) {
                    self.clear(&mut actions);
                }
                self.context_id = Some(context_id);
                self.caret = caret;
                self.set_context(committed_context, &mut actions);
                false
            }
            Event::FocusOut => {
                self.clear(&mut actions);
                self.context_id = None;
                self.context.clear();
                false
            }
            Event::CaretMoved { caret } => {
                self.caret = caret;
                if self.phase == Phase::Converting {
                    self.show_candidates(&mut actions);
                }
                false
            }
            Event::UpdateContext { text } => {
                self.set_context(text, &mut actions);
                false
            }
            Event::SetMode { mode } => {
                if self.mode != mode {
                    self.commit(&mut actions);
                    self.mode = mode;
                    actions.push(Action::SetMode { mode });
                }
                true
            }
            Event::SetSelection { position, length } => {
                if self.context_id.is_none() {
                    false
                } else if let Err(message) = self.line.set_caret(position, length) {
                    actions.push(Action::Error { message });
                    false
                } else {
                    self.cancel_prediction(&mut actions);
                    self.invalidate(&mut actions);
                    self.composition(&mut actions);
                    true
                }
            }
            Event::TextInput { text } => {
                if self.context_id.is_some() && self.mode != InputMode::Direct && !text.is_empty() {
                    self.insert(&text, &mut actions);
                    true
                } else {
                    false
                }
            }
            Event::KeyDown { key, modifiers } => {
                if self.context_id.is_some()
                    && !modifiers.control
                    && !modifiers.alt
                    && !modifiers.meta
                {
                    self.key(key, &mut actions)
                } else {
                    false
                }
            }
            Event::CandidateSelected { id } => {
                if self.phase == Phase::Converting
                    && self.context_id.is_some()
                    && id.session_id == self.session_id
                    && id.generation == self.generation
                    && id.index < self.paths.len()
                {
                    self.select(id.index, &mut actions);
                    self.commit(&mut actions);
                    true
                } else {
                    false
                }
            }
            Event::PredictionReady {
                session_id,
                generation,
                result,
            } => {
                self.prediction_ready(session_id, generation, result, &mut actions);
                false
            }
        };
        EventResult { consumed, actions }
    }

    fn prediction_ready(
        &mut self,
        session_id: u64,
        generation: u64,
        result: Result<String, String>,
        actions: &mut Vec<Action>,
    ) {
        if self.context_id.is_none()
            || self.phase != Phase::Predicting
            || self.ghost.is_some()
            || session_id != self.session_id
            || generation != self.generation
        {
            return;
        }
        match result {
            Ok(text) if !text.is_empty() => {
                self.ghost = Some(text.clone());
                actions.push(Action::ShowGhost { generation, text });
            }
            Ok(_) => self.cancel_prediction(actions),
            Err(message) => {
                self.cancel_prediction(actions);
                actions.push(Action::Error { message });
            }
        }
    }

    fn key(&mut self, key: Key, actions: &mut Vec<Action>) -> bool {
        if self.mode == InputMode::Direct {
            return false;
        }
        match key {
            Key::Character(ch) if !ch.is_control() => {
                self.insert(&ch.to_string(), actions);
                true
            }
            Key::Enter if self.phase != Phase::Direct => {
                self.commit(actions);
                true
            }
            Key::Escape if self.phase != Phase::Direct => {
                match self.phase {
                    Phase::Predicting => self.cancel_prediction(actions),
                    Phase::Converting => self.return_to_composing(actions),
                    _ => self.clear(actions),
                }
                true
            }
            Key::Backspace | Key::Delete if !self.line.is_empty() => {
                self.cancel_prediction(actions);
                if self.phase == Phase::Converting {
                    self.return_to_composing(actions);
                } else {
                    self.invalidate(actions);
                    if key == Key::Backspace {
                        self.line.backspace();
                    } else {
                        self.line.delete();
                    }
                    self.phase = if self.line.is_empty() {
                        Phase::Direct
                    } else {
                        Phase::Composing
                    };
                    self.composition(actions);
                }
                true
            }
            Key::Space if self.phase != Phase::Direct => {
                match self.phase {
                    Phase::Predicting => self.cancel_prediction(actions),
                    Phase::Converting => {
                        self.select((self.selected + 1) % self.paths.len(), actions);
                    }
                    Phase::Composing if self.mode == InputMode::Hiragana => self.convert(actions),
                    Phase::Composing => self.insert(" ", actions),
                    Phase::Direct => {}
                }
                true
            }
            Key::Tab if self.config.prediction_enabled => {
                if self.phase == Phase::Predicting {
                    if self.ghost.is_some() {
                        self.commit(actions);
                    }
                } else {
                    self.invalidate(actions);
                    self.prediction_origin = self.phase;
                    self.phase = Phase::Predicting;
                    actions.push(Action::RequestPrediction {
                        session_id: self.session_id,
                        generation: self.generation,
                        context: self.context.clone(),
                        prefix: self.line.composition(),
                    });
                }
                true
            }
            Key::Up | Key::Down if self.phase == Phase::Converting => {
                let count = self.paths.len();
                let index = if key == Key::Up {
                    (self.selected + count - 1) % count
                } else {
                    (self.selected + 1) % count
                };
                self.select(index, actions);
                true
            }
            Key::Left | Key::Right | Key::Home | Key::End if !self.line.is_empty() => {
                self.cancel_prediction(actions);
                if self.phase == Phase::Converting {
                    self.return_to_composing(actions);
                }
                self.invalidate(actions);
                let (caret, _) = self.line.caret();
                let next = match key {
                    Key::Left => caret.saturating_sub(1),
                    Key::Right => (caret + 1).min(self.line.len()),
                    Key::Home => 0,
                    _ => self.line.len(),
                };
                self.line.set_caret(next, 0).expect("bounded cursor");
                self.composition(actions);
                true
            }
            _ => false,
        }
    }

    fn insert(&mut self, text: &str, actions: &mut Vec<Action>) {
        self.cancel_prediction(actions);
        if self.phase == Phase::Converting {
            self.commit(actions);
        }
        self.invalidate(actions);
        for ch in text.chars() {
            self.line.insert_raw(&ch.to_string());
            match self.mode {
                InputMode::Hiragana => self.line.normalize_before_caret(false),
                InputMode::Katakana => {
                    self.line.normalize_before_caret(false);
                    self.line.katakana_display();
                }
                InputMode::FullWidthAscii => {
                    let display = if ch == ' ' {
                        '　'
                    } else if ch.is_ascii_graphic() {
                        char::from_u32(u32::from(ch) + 0xfee0).expect("fullwidth range")
                    } else {
                        ch
                    };
                    let (caret, _) = self.line.caret();
                    self.line
                        .set_caret(caret - 1, 1)
                        .expect("inserted character");
                    self.line.replace(&display.to_string(), TextLayer::Phonetic);
                }
                InputMode::Direct => {}
            }
        }
        self.phase = Phase::Composing;
        self.composition(actions);
    }

    fn convert(&mut self, actions: &mut Vec<Action>) {
        self.invalidate(actions);
        self.line.normalize_before_caret(true);
        let input = self.line.input_lattice();
        match self
            .dictionary
            .query(&self.line, &self.context)
            .and_then(|output| {
                let paths = output.nbest(&input, self.config.candidate_limit)?;
                if paths.is_empty() {
                    return Err("dictionary returned no complete path".into());
                }
                Ok((output, paths))
            }) {
            Ok((output, paths)) => {
                self.output = Some(output);
                self.paths = paths;
                self.phase = Phase::Converting;
                self.select(0, actions);
            }
            Err(message) => {
                self.phase = Phase::Composing;
                self.composition(actions);
                actions.push(Action::Error { message });
            }
        }
    }

    fn select(&mut self, index: usize, actions: &mut Vec<Action>) {
        if let (Some(output), Some(path)) = (&self.output, self.paths.get(index)) {
            let words: Vec<_> = path
                .edges
                .iter()
                .map(|&id| {
                    let edge = &output.edges[id];
                    (
                        edge.start..edge.end,
                        edge.word.surface.clone(),
                        edge.word.reading.clone(),
                    )
                })
                .collect();
            match self.line.apply_words(&words) {
                Ok(()) => {
                    self.selected = index;
                    self.composition(actions);
                    self.show_candidates(actions);
                }
                Err(message) => actions.push(Action::Error { message }),
            }
        }
    }

    fn commit(&mut self, actions: &mut Vec<Action>) {
        if self.mode == InputMode::Hiragana || self.mode == InputMode::Katakana {
            self.line.normalize_before_caret(true);
            if self.mode == InputMode::Katakana {
                self.line.katakana_display();
            }
        }
        let mut text = self.line.composition();
        if let Some(ghost) = &self.ghost {
            text.push_str(ghost);
        }
        if !text.is_empty() {
            actions.push(Action::Commit { text: text.clone() });
            self.context.push_str(&text);
            if self.phase == Phase::Converting
                || (self.phase == Phase::Predicting && self.prediction_origin == Phase::Converting)
            {
                if let (Some(output), Some(path)) = (&self.output, self.paths.get(self.selected)) {
                    let words = path
                        .edges
                        .iter()
                        .filter_map(|&id| {
                            let edge = &output.edges[id];
                            (!edge.fallback).then(|| edge.word.clone())
                        })
                        .collect();
                    actions.push(Action::LearnSelection { words });
                }
            }
        }
        self.clear(actions);
    }

    fn composition(&self, actions: &mut Vec<Action>) {
        let (cursor, selection_len) = self.line.caret();
        actions.push(Action::UpdateComposition {
            text: self.line.composition(),
            cursor,
            selection_len,
        });
    }
    fn show_candidates(&self, actions: &mut Vec<Action>) {
        actions.push(Action::ShowCandidates {
            candidates: self.candidates(),
            selected: self.selected,
            caret: self.caret.clone(),
            generation: self.generation,
        });
    }
    fn return_to_composing(&mut self, actions: &mut Vec<Action>) {
        self.invalidate(actions);
        self.line.cancel_conversion();
        self.phase = Phase::Composing;
        self.paths.clear();
        self.output = None;
        self.selected = 0;
        actions.push(Action::HideCandidates);
        self.composition(actions);
    }
    fn set_context(&mut self, text: String, actions: &mut Vec<Action>) {
        if self.context != text {
            self.cancel_prediction(actions);
            self.context = text;
        }
    }
    fn cancel_prediction(&mut self, actions: &mut Vec<Action>) {
        if self.phase == Phase::Predicting {
            self.invalidate(actions);
            self.phase = self.prediction_origin;
            if self.phase == Phase::Converting {
                self.show_candidates(actions);
            }
        }
    }
    fn invalidate(&mut self, actions: &mut Vec<Action>) {
        if self.phase == Phase::Predicting {
            actions.push(Action::CancelPrediction {
                session_id: self.session_id,
                generation: self.generation,
            });
            actions.push(Action::HideGhost);
        }
        self.ghost = None;
        self.generation = self
            .generation
            .checked_add(1)
            .expect("IME generation exhausted");
    }
    fn clear(&mut self, actions: &mut Vec<Action>) {
        self.invalidate(actions);
        self.line.clear();
        self.phase = Phase::Direct;
        self.paths.clear();
        self.output = None;
        self.selected = 0;
        self.composition(actions);
        actions.push(Action::HideCandidates);
    }
}

#[cfg(test)]
mod tests;
