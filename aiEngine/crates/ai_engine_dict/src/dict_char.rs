//! Compact internal character encoding used by dictionary index keys.
//!
//! This mirrors the established dictionary codec: hiragana occupies one byte,
//! while other BMP code units use dedicated one-, two-, or three-byte forms.
//! A zero byte is reserved for the key terminator and is not valid in input.

use std::io::{Error, ErrorKind};

const MAX_KEY_BYTES: usize = 255;

pub(crate) fn encode(text: &str) -> Result<Vec<u8>, Error> {
    let mut encoded = Vec::with_capacity(text.len());
    for code_unit in text.encode_utf16() {
        if code_unit == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "U+0000 is reserved as the dictionary-key terminator",
            ));
        }
        encode_code_unit(code_unit, &mut encoded);
        if encoded.len() > MAX_KEY_BYTES {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("encoded dictionary key exceeds {MAX_KEY_BYTES} bytes"),
            ));
        }
    }
    Ok(encoded)
}

pub(crate) fn decode(encoded: &[u8]) -> Result<String, Error> {
    let mut code_units = Vec::with_capacity(encoded.len());
    let mut offset = 0;
    while offset < encoded.len() {
        let first = encoded[offset];
        if first == 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "unexpected dictionary-key terminator inside encoded text",
            ));
        }
        let (code_unit, width) = match first {
            0x01..=0x56 => (0x3040 + u16::from(first), 1),
            0x57..=0x60 => {
                const PUNCTUATION: [u16; 10] = [
                    0x3001, 0x3002, 0x30fb, 0x30fc, 0x301c, 0x300c, 0x300d, 0x007b, 0x007c, 0x007d,
                ];
                (PUNCTUATION[usize::from(first - 0x57)], 1)
            }
            0x61..=0xa0 => (u16::from(first - 0x40), 1),
            0xaa => (0x3000 + u16::from(next_byte(encoded, offset)?), 2),
            0xab => (u16::from(next_byte(encoded, offset)?), 2),
            0xac => (u16::from(next_byte(encoded, offset)?) << 8, 2),
            0xad => {
                let high = u16::from(next_byte(encoded, offset)?);
                let low = u16::from(*encoded.get(offset + 2).ok_or_else(truncated)?);
                ((high << 8) | low, 3)
            }
            0xae..=0xff => {
                let high = u16::from(first - 0x60);
                let low = u16::from(next_byte(encoded, offset)?);
                ((high << 8) | low, 2)
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("reserved dictionary byte 0x{first:02X}"),
                ));
            }
        };
        code_units.push(code_unit);
        offset += width;
    }
    let decoded = String::from_utf16(&code_units)
        .map_err(|error| Error::new(ErrorKind::InvalidData, error.to_string()))?;
    let reencoded =
        encode(&decoded).map_err(|error| Error::new(ErrorKind::InvalidData, error.to_string()))?;
    if reencoded.as_slice() != encoded {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "non-canonical dictionary character encoding",
        ));
    }
    Ok(decoded)
}

fn encode_code_unit(code_unit: u16, output: &mut Vec<u8>) {
    let [low, high] = code_unit.to_le_bytes();
    if (0x3041..=0x3096).contains(&code_unit) {
        output.push(low - 0x40);
        return;
    }
    let punctuation = match code_unit {
        0x3001 => Some(0x57),
        0x3002 => Some(0x58),
        0x30fb => Some(0x59),
        0x30fc => Some(0x5a),
        0x301c => Some(0x5b),
        0x300c => Some(0x5c),
        0x300d => Some(0x5d),
        0x007b => Some(0x5e),
        0x007c => Some(0x5f),
        0x007d => Some(0x60),
        _ => None,
    };
    if let Some(byte) = punctuation {
        output.push(byte);
        return;
    }
    if (0x0021..=0x0060).contains(&code_unit) {
        output.push(low + 0x40);
        return;
    }
    if (0x3001..=0x30ff).contains(&code_unit) {
        output.extend_from_slice(&[0xaa, low]);
    } else if code_unit < 0x0100 {
        output.extend_from_slice(&[0xab, low]);
    } else if code_unit.trailing_zeros() >= 8 {
        output.extend_from_slice(&[0xac, high]);
    } else if (0x4e00..=0x9fff).contains(&code_unit) {
        let high = (code_unit - 0x4e00 + 0xae00) >> 8;
        output.extend_from_slice(&[high.to_le_bytes()[0], low]);
    } else {
        output.extend_from_slice(&[0xad, high, low]);
    }
}

fn next_byte(encoded: &[u8], offset: usize) -> Result<u8, Error> {
    encoded.get(offset + 1).copied().ok_or_else(truncated)
}

fn truncated() -> Error {
    Error::new(ErrorKind::InvalidData, "truncated dictionary character")
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};

    #[test]
    fn hiragana_is_one_byte_per_code_unit() {
        assert_eq!(
            encode("あかんじ").expect("encode"),
            [0x02, 0x0b, 0x53, 0x18]
        );
    }

    #[test]
    fn punctuation_ascii_and_katakana_use_the_defined_ranges() {
        assert_eq!(
            encode("、。・ー〜「」{|}Aカ").expect("encode"),
            [
                0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60, 0x81, 0xaa, 0xab
            ]
        );
    }

    #[test]
    fn kanji_and_other_code_units_follow_the_two_and_three_byte_forms() {
        assert_eq!(encode("一日").expect("encode"), [0xac, 0x4e, 0xc5, 0xe5]);
        assert_eq!(
            encode("Āā").expect("encode"),
            [0xac, 0x01, 0xad, 0x01, 0x01]
        );
    }

    #[test]
    fn representative_strings_round_trip() {
        for text in [
            "にわ",
            "犬",
            "日本語",
            "カタカナ",
            "ABC",
            "括弧「」",
            "𠮷野家",
        ] {
            assert_eq!(
                decode(&encode(text).expect("encode")).expect("decode"),
                text
            );
        }
    }

    #[test]
    fn terminator_and_index_length_are_checked() {
        assert!(encode("\0").is_err());
        assert_eq!(encode(&"あ".repeat(255)).expect("255 bytes").len(), 255);
        assert!(encode(&"あ".repeat(256)).is_err());
        assert!(decode(&[0xab, 0x00]).is_err());
        assert!(decode(&[0xaa, 0x01]).is_err());
    }
}
