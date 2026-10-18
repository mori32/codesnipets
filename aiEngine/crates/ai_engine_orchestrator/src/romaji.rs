//! Incremental P2 romaji parsing. The core batch normalizer is intentionally separate.

const ROMAJI: &[(&str, &str)] = &[
    ("ja", "じゃ"),
    ("ju", "じゅ"),
    ("jo", "じょ"),
    ("fa", "ふぁ"),
    ("fi", "ふぃ"),
    ("fe", "ふぇ"),
    ("fo", "ふぉ"),
    ("xa", "ぁ"),
    ("xi", "ぃ"),
    ("xu", "ぅ"),
    ("xe", "ぇ"),
    ("xo", "ぉ"),
    ("xtu", "っ"),
    ("xtsu", "っ"),
    ("kya", "きゃ"),
    ("kyu", "きゅ"),
    ("kyo", "きょ"),
    ("sha", "しゃ"),
    ("shu", "しゅ"),
    ("sho", "しょ"),
    ("sya", "しゃ"),
    ("syu", "しゅ"),
    ("syo", "しょ"),
    ("cha", "ちゃ"),
    ("chu", "ちゅ"),
    ("cho", "ちょ"),
    ("tya", "ちゃ"),
    ("tyu", "ちゅ"),
    ("tyo", "ちょ"),
    ("nya", "にゃ"),
    ("nyu", "にゅ"),
    ("nyo", "にょ"),
    ("hya", "ひゃ"),
    ("hyu", "ひゅ"),
    ("hyo", "ひょ"),
    ("mya", "みゃ"),
    ("myu", "みゅ"),
    ("myo", "みょ"),
    ("rya", "りゃ"),
    ("ryu", "りゅ"),
    ("ryo", "りょ"),
    ("gya", "ぎゃ"),
    ("gyu", "ぎゅ"),
    ("gyo", "ぎょ"),
    ("jya", "じゃ"),
    ("jyu", "じゅ"),
    ("jyo", "じょ"),
    ("bya", "びゃ"),
    ("byu", "びゅ"),
    ("byo", "びょ"),
    ("pya", "ぴゃ"),
    ("pyu", "ぴゅ"),
    ("pyo", "ぴょ"),
    ("shi", "し"),
    ("chi", "ち"),
    ("tsu", "つ"),
    ("fu", "ふ"),
    ("ji", "じ"),
    ("ka", "か"),
    ("ki", "き"),
    ("ku", "く"),
    ("ke", "け"),
    ("ko", "こ"),
    ("sa", "さ"),
    ("si", "し"),
    ("su", "す"),
    ("se", "せ"),
    ("so", "そ"),
    ("ta", "た"),
    ("ti", "ち"),
    ("tu", "つ"),
    ("te", "て"),
    ("to", "と"),
    ("na", "な"),
    ("ni", "に"),
    ("nu", "ぬ"),
    ("ne", "ね"),
    ("no", "の"),
    ("ha", "は"),
    ("hi", "ひ"),
    ("hu", "ふ"),
    ("he", "へ"),
    ("ho", "ほ"),
    ("ma", "ま"),
    ("mi", "み"),
    ("mu", "む"),
    ("me", "め"),
    ("mo", "も"),
    ("ya", "や"),
    ("yu", "ゆ"),
    ("yo", "よ"),
    ("ra", "ら"),
    ("ri", "り"),
    ("ru", "る"),
    ("re", "れ"),
    ("ro", "ろ"),
    ("wa", "わ"),
    ("wo", "を"),
    ("ga", "が"),
    ("gi", "ぎ"),
    ("gu", "ぐ"),
    ("ge", "げ"),
    ("go", "ご"),
    ("za", "ざ"),
    ("zi", "じ"),
    ("zu", "ず"),
    ("ze", "ぜ"),
    ("zo", "ぞ"),
    ("da", "だ"),
    ("di", "ぢ"),
    ("du", "づ"),
    ("de", "で"),
    ("do", "ど"),
    ("ba", "ば"),
    ("bi", "び"),
    ("bu", "ぶ"),
    ("be", "べ"),
    ("bo", "ぼ"),
    ("pa", "ぱ"),
    ("pi", "ぴ"),
    ("pu", "ぷ"),
    ("pe", "ぺ"),
    ("po", "ぽ"),
    ("a", "あ"),
    ("i", "い"),
    ("u", "う"),
    ("e", "え"),
    ("o", "お"),
];

/// Parses a raw input buffer. `finish` resolves a trailing n on explicit commit.
/// Incomplete consonants stay visible; they are never silently dropped.
pub(crate) fn parse(input: &str, finish: bool) -> String {
    let source = input.to_ascii_lowercase();
    let mut rest = source.as_str();
    let mut output = String::new();
    while let Some(current) = rest.chars().next() {
        if rest.starts_with("n'") {
            output.push('ん');
            rest = &rest[2..];
            continue;
        }
        if current == 'n' {
            let next = rest.chars().nth(1);
            if next == Some('n') {
                output.push('ん');
                rest = &rest[if rest.len() == 2 { 2 } else { 1 }..];
                continue;
            }
            if next.is_none() {
                output.push(if finish { 'ん' } else { 'n' });
                break;
            }
            if next.is_some_and(|ch| !matches!(ch, 'a' | 'i' | 'u' | 'e' | 'o' | 'y')) {
                output.push('ん');
                rest = &rest[1..];
                continue;
            }
        }
        if current.is_ascii_alphabetic()
            && !matches!(current, 'a' | 'i' | 'u' | 'e' | 'o' | 'n')
            && rest.chars().nth(1) == Some(current)
        {
            output.push('っ');
            rest = &rest[1..];
            continue;
        }
        if let Some((key, kana)) = ROMAJI
            .iter()
            .filter(|(key, _)| rest.starts_with(key))
            .max_by_key(|(key, _)| key.len())
        {
            output.push_str(kana);
            rest = &rest[key.len()..];
        } else {
            let code = u32::from(current);
            output.push(if (0x30a1..=0x30f6).contains(&code) {
                char::from_u32(code - 0x60).unwrap_or(current)
            } else if current == '-' {
                'ー'
            } else {
                current
            });
            rest = &rest[current.len_utf8()..];
        }
    }
    output
}

/// Returns source character counts and completed phonetic replacements.
pub(crate) fn steps(input: &str, finish: bool) -> Vec<(usize, Option<String>)> {
    let source = input.to_ascii_lowercase();
    let mut rest = source.as_str();
    let mut result = Vec::new();
    while let Some(ch) = rest.chars().next() {
        let (count, converted) = if rest.starts_with("n'") {
            (2, Some("ん".into()))
        } else if ch == 'n' && (rest == "n" || rest == "nn") {
            if finish {
                (rest.len(), Some("ん".into()))
            } else {
                (1, None)
            }
        } else if ch == 'n'
            && rest
                .chars()
                .nth(1)
                .is_some_and(|next| !matches!(next, 'a' | 'i' | 'u' | 'e' | 'o' | 'y'))
        {
            (
                if rest.starts_with("nn") && !rest[2..].starts_with(['a', 'i', 'u', 'e', 'o', 'y'])
                {
                    2
                } else {
                    1
                },
                Some("ん".into()),
            )
        } else if ch.is_ascii_alphabetic()
            && !matches!(ch, 'a' | 'i' | 'u' | 'e' | 'o' | 'n')
            && rest.chars().nth(1) == Some(ch)
            && ROMAJI.iter().any(|(key, _)| rest[1..].starts_with(key))
        {
            (1, Some("っ".into()))
        } else if let Some((key, kana)) = ROMAJI
            .iter()
            .filter(|(key, _)| rest.starts_with(key))
            .max_by_key(|(key, _)| key.len())
        {
            (key.len(), Some((*kana).into()))
        } else if !ch.is_ascii_alphabetic() && ch != '\'' {
            (1, Some(parse(&ch.to_string(), false)))
        } else {
            (1, None)
        };
        let bytes = rest
            .char_indices()
            .nth(count)
            .map_or(rest.len(), |(offset, _)| offset);
        result.push((count, converted));
        rest = &rest[bytes..];
    }
    result
}
