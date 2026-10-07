//! Split a recording into speech segments at pauses.
//!
//! Parakeet TDT silently drops speech from long recordings: text after a
//! multi-second pause goes missing, and so does the tail of a long
//! stretch of continuous speech. Transcribing each segment on its own
//! avoids both, because no segment contains a long silence or runs long
//! enough to lose its end.
//!
//! The split is energy-based on 20 ms frames:
//! 1. Cut at the middle of every pause of at least `min_pause_secs`.
//! 2. Cut any piece still longer than `max_segment_secs` at its longest
//!    internal gap (or at the limit if it has none), repeating as needed.
//! 3. Join neighbouring pieces again while they fit `max_segment_secs`
//!    and the pause between them is at most `max_merged_pause_secs`, so
//!    slow speech with many short pauses is not cut into fragments too
//!    small for Parakeet to transcribe well.
//! 4. Trim each piece's leading and trailing silence down to
//!    `padding_secs`, and drop pieces that contain no speech.
//!
//! Cuts always fall in silence when one exists, so words are not split,
//! and the returned ranges never overlap, so words are not duplicated.

use std::ops::Range;

const SAMPLE_RATE: usize = 16000;
const FRAME_SIZE: usize = SAMPLE_RATE / 50; // 20 ms
const FRAME_SECS: f32 = 0.02;
/// Frames needed for a segment to count as speech (100 ms)
const MIN_SPEECH_FRAMES: usize = 5;

#[derive(Debug, Clone)]
pub struct PauseSplitConfig {
    /// Frames quieter than this RMS level (dBFS) count as silence
    pub silence_threshold_db: f32,
    /// Pauses at least this long always become cut points
    pub min_pause_secs: f32,
    /// Segments longer than this are split further
    pub max_segment_secs: f32,
    /// Silence kept around the speech in each segment. Keep this small:
    /// with 0.4 s or more, a breath or click before the speech can make
    /// Parakeet return nothing for the whole segment.
    pub padding_secs: f32,
    /// Neighbouring segments separated by a pause no longer than this are
    /// joined while they fit `max_segment_secs`. Very short segments lose
    /// words at their edges, which hurts slow speech with many pauses.
    pub max_merged_pause_secs: f32,
}

impl Default for PauseSplitConfig {
    fn default() -> Self {
        Self {
            silence_threshold_db: -45.0,
            min_pause_secs: 1.0,
            max_segment_secs: 15.0,
            padding_secs: 0.1,
            max_merged_pause_secs: 3.0,
        }
    }
}

/// Plan speech segments for 16 kHz mono `samples`, as sample ranges in order.
pub fn plan_segments(samples: &[f32], config: &PauseSplitConfig) -> Vec<Range<usize>> {
    let frame_count = samples.len() / FRAME_SIZE;
    if frame_count == 0 {
        return vec![Range {
            start: 0,
            end: samples.len(),
        }];
    }

    let threshold = 10f32.powf(config.silence_threshold_db / 20.0);
    let speech: Vec<bool> = samples
        .as_chunks::<FRAME_SIZE>()
        .0
        .iter()
        .map(|frame| {
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / FRAME_SIZE as f32).sqrt();
            rms >= threshold
        })
        .collect();
    let gaps = silent_runs(&speech);

    let min_pause = secs_to_frames(config.min_pause_secs);
    let max_segment = secs_to_frames(config.max_segment_secs).max(1);

    let mut cuts = vec![0];
    cuts.extend(
        gaps.iter()
            .filter(|gap| gap.start > 0 && gap.end < frame_count && gap.len() >= min_pause)
            .map(|gap| (gap.start + gap.end) / 2),
    );
    cuts.push(frame_count);

    let mut pieces = Vec::new();
    for window in cuts.windows(2) {
        split_long(window[0]..window[1], &gaps, max_segment, &mut pieces);
    }
    let max_merged_pause = secs_to_frames(config.max_merged_pause_secs);
    let pieces = merge_short(pieces, &speech, max_segment, max_merged_pause);

    let padding = secs_to_frames(config.padding_secs);
    pieces
        .into_iter()
        .filter_map(|piece| {
            let first = piece.clone().find(|&i| speech[i])?;
            let last = piece.clone().rev().find(|&i| speech[i])?;
            let speech_frames = speech[piece.clone()].iter().filter(|&&s| s).count();
            if speech_frames < MIN_SPEECH_FRAMES {
                return None;
            }
            let start = first.saturating_sub(padding).max(piece.start);
            let end = (last + 1 + padding).min(piece.end);
            // The last piece keeps any samples past the final whole frame
            let end_sample = if end == frame_count {
                samples.len()
            } else {
                end * FRAME_SIZE
            };
            Some(start * FRAME_SIZE..end_sample)
        })
        .collect()
}

fn secs_to_frames(secs: f32) -> usize {
    (secs / FRAME_SECS).round() as usize
}

/// Runs of consecutive silent frames
fn silent_runs(speech: &[bool]) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut start = None;
    for (i, &is_speech) in speech.iter().enumerate() {
        match (is_speech, start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                runs.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push(s..speech.len());
    }
    runs
}

/// Join neighbouring pieces while the result still fits `max_len`, so slow
/// speech with many pauses keeps enough context for Parakeet.
/// Pauses longer than `max_pause` stay cuts, since Parakeet drops speech
/// that follows a long silence.
fn merge_short(
    pieces: Vec<Range<usize>>,
    speech: &[bool],
    max_len: usize,
    max_pause: usize,
) -> Vec<Range<usize>> {
    let mut merged: Vec<Range<usize>> = Vec::new();
    for piece in pieces {
        let pause = |last: &Range<usize>| {
            let before = last.clone().rev().take_while(|&i| !speech[i]).count();
            let after = piece.clone().take_while(|&i| !speech[i]).count();
            before + after
        };
        match merged.last_mut() {
            Some(last) if piece.end - last.start <= max_len && pause(last) <= max_pause => {
                last.end = piece.end
            }
            _ => merged.push(piece),
        }
    }
    merged
}

/// Split `piece` at its longest internal gap until every part fits `max_len`.
fn split_long(
    piece: Range<usize>,
    gaps: &[Range<usize>],
    max_len: usize,
    out: &mut Vec<Range<usize>>,
) {
    if piece.len() <= max_len {
        out.push(piece);
        return;
    }
    let cut = gaps
        .iter()
        .filter(|gap| gap.start > piece.start && gap.end < piece.end)
        .max_by_key(|gap| gap.len())
        .map(|gap| (gap.start + gap.end) / 2)
        .unwrap_or(piece.start + max_len);
    split_long(piece.start..cut, gaps, max_len, out);
    split_long(cut..piece.end, gaps, max_len, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build audio from (seconds, is_speech) spans: speech is a loud tone,
    /// silence is digital zero.
    fn audio(spans: &[(f32, bool)]) -> Vec<f32> {
        spans
            .iter()
            .flat_map(|&(secs, is_speech)| {
                let n = (secs * SAMPLE_RATE as f32) as usize;
                (0..n).map(move |i| {
                    if is_speech {
                        0.3 * (i as f32 * 0.1).sin()
                    } else {
                        0.0
                    }
                })
            })
            .collect()
    }

    fn secs(ranges: &[Range<usize>]) -> Vec<(f32, f32)> {
        ranges
            .iter()
            .map(|r| {
                (
                    r.start as f32 / SAMPLE_RATE as f32,
                    r.end as f32 / SAMPLE_RATE as f32,
                )
            })
            .collect()
    }

    #[test]
    fn continuous_speech_is_one_segment() {
        let samples = audio(&[(5.0, true)]);
        assert_eq!(
            plan_segments(&samples, &PauseSplitConfig::default()),
            vec![0..samples.len()]
        );
    }

    #[test]
    fn long_pause_splits_and_is_trimmed() {
        let samples = audio(&[(3.0, true), (6.0, false), (2.0, true)]);
        let segments = secs(&plan_segments(&samples, &PauseSplitConfig::default()));
        assert_eq!(segments, vec![(0.0, 3.1), (8.9, 11.0)]);
    }

    #[test]
    fn short_pieces_across_moderate_pauses_are_joined() {
        // Slow speech: 1.5 s pauses between short phrases, 9 s in all
        let samples = audio(&[
            (2.0, true),
            (1.5, false),
            (1.0, true),
            (1.5, false),
            (3.0, true),
        ]);
        let segments = plan_segments(&samples, &PauseSplitConfig::default());
        assert_eq!(segments, vec![0..samples.len()]);
    }

    #[test]
    fn joining_stops_at_the_segment_limit() {
        // Four 5 s phrases with 1.5 s pauses: only two fit in 15 s
        let samples = audio(&[
            (5.0, true),
            (1.5, false),
            (5.0, true),
            (1.5, false),
            (5.0, true),
            (1.5, false),
            (5.0, true),
        ]);
        let segments = secs(&plan_segments(&samples, &PauseSplitConfig::default()));
        assert_eq!(segments.len(), 2, "{segments:?}");
    }

    #[test]
    fn short_pause_does_not_split() {
        let samples = audio(&[(3.0, true), (0.5, false), (2.0, true)]);
        assert_eq!(
            plan_segments(&samples, &PauseSplitConfig::default()).len(),
            1
        );
    }

    #[test]
    fn long_speech_splits_at_longest_gap() {
        // 20 s of speech with short gaps at 6 s (0.2 s) and 12 s (0.6 s)
        let samples = audio(&[
            (6.0, true),
            (0.2, false),
            (5.8, true),
            (0.6, false),
            (7.4, true),
        ]);
        let segments = secs(&plan_segments(&samples, &PauseSplitConfig::default()));
        assert_eq!(segments.len(), 2);
        // Cut inside the 0.6 s gap at 12.0-12.6 s, each side trimmed to 0.1 s
        assert!((segments[0].1 - 12.1).abs() < 0.05, "{segments:?}");
        assert!((segments[1].0 - 12.5).abs() < 0.05, "{segments:?}");
    }

    #[test]
    fn long_speech_without_gaps_splits_at_limit() {
        let samples = audio(&[(40.0, true)]);
        let segments = secs(&plan_segments(&samples, &PauseSplitConfig::default()));
        assert_eq!(segments, vec![(0.0, 15.0), (15.0, 30.0), (30.0, 40.0)]);
    }

    #[test]
    fn segments_never_overlap() {
        let samples = audio(&[
            (14.0, true),
            (0.1, false),
            (14.0, true),
            (1.2, false),
            (0.3, true),
        ]);
        let segments = plan_segments(&samples, &PauseSplitConfig::default());
        for pair in segments.windows(2) {
            assert!(pair[0].end <= pair[1].start, "{segments:?}");
        }
    }

    #[test]
    fn silence_only_yields_no_segments() {
        let samples = audio(&[(5.0, false)]);
        assert!(plan_segments(&samples, &PauseSplitConfig::default()).is_empty());
    }

    #[test]
    fn audio_shorter_than_a_frame_is_one_segment() {
        let samples = vec![0.1; 100];
        assert_eq!(
            plan_segments(&samples, &PauseSplitConfig::default()),
            vec![0..100]
        );
    }
}
