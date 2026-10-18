//! Input normalization for the first CLI prototype.

const ROMAJI: &[(&str, &str)] = &[
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
    ("n", "ん"),
];

/// Converts romaji and katakana to the hiragana form used by the dictionary.
/// Non-romaji characters are preserved, so unknown input remains convertible.
///
/// This function does not panic for any UTF-8 input.
#[must_use]
pub fn to_hiragana(input: &str) -> String {
    let source = input.to_lowercase();
    let mut output = String::new();
    let mut index = 0;

    while index < source.len() {
        let Some(current) = source[index..].chars().next() else {
            break;
        };
        if !current.is_ascii() {
            output.push(to_hiragana_char(current));
            index += current.len_utf8();
            continue;
        }

        let remaining = &source[index..];
        if current == '-' {
            output.push('ー');
            index += 1;
            continue;
        }
        if remaining.starts_with("n'") {
            output.push('ん');
            index += 2;
            continue;
        }
        if remaining.len() >= 2 {
            let next = remaining.as_bytes()[1] as char;
            if current == next && !matches!(current, 'a' | 'i' | 'u' | 'e' | 'o' | 'n') {
                output.push('っ');
                index += 1;
                continue;
            }
            if current == 'n' && !matches!(next, 'a' | 'i' | 'u' | 'e' | 'o' | 'y' | 'n') {
                output.push('ん');
                index += 1;
                continue;
            }
        }

        if let Some((romaji, kana)) = ROMAJI
            .iter()
            .find(|(romaji, _)| remaining.starts_with(romaji))
        {
            output.push_str(kana);
            index += romaji.len();
        } else {
            output.push(current);
            index += current.len_utf8();
        }
    }

    output
}

fn to_hiragana_char(character: char) -> char {
    let code = character as u32;
    if (0x30a1..=0x30f6).contains(&code) {
        char::from_u32(code - 0x60).unwrap_or(character)
    } else {
        character
    }
}
