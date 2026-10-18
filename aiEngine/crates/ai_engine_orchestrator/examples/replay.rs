use ai_engine_orchestrator::{
    Action, CaretInfo, ControllerConfig, Event, ImeSession, Key, MemoryDictionary, Modifiers,
    QueryWord,
};

fn main() {
    let dictionary = MemoryDictionary::new([
        QueryWord {
            reading: "にほんご".into(),
            surface: "日本語".into(),
            score: 30.0,
        },
        QueryWord {
            reading: "へんかん".into(),
            surface: "変換".into(),
            score: 20.0,
        },
    ])
    .expect("fixture dictionary");
    let mut session = ImeSession::new(1, dictionary, ControllerConfig::default());
    let _ = session.process_event(Event::FocusIn {
        context_id: 1,
        caret: CaretInfo::default(),
        committed_context: String::new(),
    });
    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "nihongohenkan".into());
    let keys = input
        .chars()
        .map(Key::Character)
        .chain([Key::Space, Key::Enter]);
    for key in keys {
        let result = session.process_event(Event::KeyDown {
            key,
            modifiers: Modifiers::default(),
        });
        for action in result.actions {
            match action {
                Action::UpdateComposition { text, cursor, .. } => {
                    println!("composition: {text} (cursor={cursor})");
                }
                other => println!("{other:?}"),
            }
        }
    }
}
