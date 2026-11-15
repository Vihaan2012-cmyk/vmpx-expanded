//! Minimal standard LZMA1 decoder, used to replace the emulated runtime depack.
//!
//! VMProtect's runtime decompressor is a *standard* LZMA1 range coder (verified:
//! lc=3, lp=0, pb=2, EOS-terminated; the fold's own literal decode is the same
//! algorithm). `lzma-rs` cannot be used directly because it decodes the full
//! `.lzma` container and errors on the data that follows an EOS marker (the next
//! stream), discarding its output. This decoder takes a raw range-coder stream
//! and decodes one LZMA1 stream to its EOS marker (or an explicit byte cap),
//! returning the decompressed bytes and the number of input bytes consumed.

const K_NUM_BIT_MODEL_TOTAL_BITS: u32 = 11;
const K_BIT_MODEL_TOTAL: u16 = 1 << K_NUM_BIT_MODEL_TOTAL_BITS;
const K_NUM_MOVE_BITS: u32 = 5;
const K_TOP_VALUE: u32 = 1 << 24;

const K_NUM_POS_BITS_MAX: usize = 4;
const K_NUM_STATES: usize = 12;
const K_NUM_LEN_TO_POS_STATES: usize = 4;
const K_NUM_ALIGN_BITS: u32 = 4;
const K_END_POS_MODEL_INDEX: usize = 14;
const K_NUM_FULL_DISTANCES: usize = 1 << (K_END_POS_MODEL_INDEX >> 1);
const K_MATCH_MIN_LEN: usize = 2;

type Prob = u16;

struct RangeDecoder<'a> {
    input: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDecoder<'a> {
    fn new(input: &'a [u8]) -> Option<Self> {
        if input.len() < 5 {
            return None;
        }
        // Byte 0 is ignored; the next four are the initial code (big-endian).
        let code = (u32::from(input[1]) << 24)
            | (u32::from(input[2]) << 16)
            | (u32::from(input[3]) << 8)
            | u32::from(input[4]);
        Some(RangeDecoder { input, pos: 5, range: 0xFFFF_FFFF, code })
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        let b = self.input.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    #[inline]
    fn normalize(&mut self) {
        if self.range < K_TOP_VALUE {
            self.range <<= 8;
            self.code = (self.code << 8) | u32::from(self.next_byte());
        }
    }

    #[inline]
    fn decode_bit(&mut self, prob: &mut Prob) -> u32 {
        let bound = (self.range >> K_NUM_BIT_MODEL_TOTAL_BITS) * u32::from(*prob);
        if self.code < bound {
            self.range = bound;
            *prob += (K_BIT_MODEL_TOTAL - *prob) >> K_NUM_MOVE_BITS;
            self.normalize();
            0
        } else {
            self.range -= bound;
            self.code -= bound;
            *prob -= *prob >> K_NUM_MOVE_BITS;
            self.normalize();
            1
        }
    }

    #[inline]
    fn decode_direct_bits(&mut self, num: u32) -> u32 {
        let mut result = 0u32;
        for _ in 0..num {
            self.range >>= 1;
            self.code = self.code.wrapping_sub(self.range);
            let t = 0u32.wrapping_sub(self.code >> 31);
            self.code = self.code.wrapping_add(self.range & t);
            self.normalize();
            result = (result << 1).wrapping_add(t.wrapping_add(1));
        }
        result
    }
}

#[inline]
fn bit_tree_decode(dec: &mut RangeDecoder, probs: &mut [Prob], num_bits: u32) -> usize {
    let mut m = 1usize;
    for _ in 0..num_bits {
        m = (m << 1) + dec.decode_bit(&mut probs[m]) as usize;
    }
    m - (1usize << num_bits)
}

#[inline]
fn bit_tree_reverse_decode(dec: &mut RangeDecoder, probs: &mut [Prob], num_bits: u32) -> usize {
    let mut m = 1usize;
    let mut symbol = 0usize;
    for i in 0..num_bits {
        let bit = dec.decode_bit(&mut probs[m]) as usize;
        m = (m << 1) + bit;
        symbol |= bit << i;
    }
    symbol
}

struct LenDecoder {
    choice: Prob,
    choice2: Prob,
    low: Vec<Prob>,  // [pos_state][8]
    mid: Vec<Prob>,  // [pos_state][8]
    high: Vec<Prob>, // [256]
}

impl LenDecoder {
    fn new() -> Self {
        LenDecoder {
            choice: K_BIT_MODEL_TOTAL / 2,
            choice2: K_BIT_MODEL_TOTAL / 2,
            low: vec![K_BIT_MODEL_TOTAL / 2; (1 << K_NUM_POS_BITS_MAX) * 8],
            mid: vec![K_BIT_MODEL_TOTAL / 2; (1 << K_NUM_POS_BITS_MAX) * 8],
            high: vec![K_BIT_MODEL_TOTAL / 2; 256],
        }
    }

    fn decode(&mut self, dec: &mut RangeDecoder, pos_state: usize) -> usize {
        if dec.decode_bit(&mut self.choice) == 0 {
            bit_tree_decode(dec, &mut self.low[pos_state * 8..pos_state * 8 + 8], 3)
        } else if dec.decode_bit(&mut self.choice2) == 0 {
            8 + bit_tree_decode(dec, &mut self.mid[pos_state * 8..pos_state * 8 + 8], 3)
        } else {
            16 + bit_tree_decode(dec, &mut self.high, 8)
        }
    }
}

/// Decode one raw LZMA1 stream. Stops at the EOS marker or when `cap` output
/// bytes are produced (whichever first). `(output, input_bytes_consumed)`.
pub fn decode_lzma1(
    input: &[u8],
    cap: usize,
    lc: u32,
    lp: u32,
    pb: u32,
) -> Result<(Vec<u8>, usize), &'static str> {
    let Some(mut dec) = RangeDecoder::new(input) else {
        return Err("stream too short");
    };
    let pos_mask = (1usize << pb) - 1;
    let lit_pos_mask = (1usize << lp) - 1;

    let mut is_match = vec![K_BIT_MODEL_TOTAL / 2; K_NUM_STATES << K_NUM_POS_BITS_MAX];
    let mut is_rep = vec![K_BIT_MODEL_TOTAL / 2; K_NUM_STATES];
    let mut is_rep_g0 = vec![K_BIT_MODEL_TOTAL / 2; K_NUM_STATES];
    let mut is_rep_g1 = vec![K_BIT_MODEL_TOTAL / 2; K_NUM_STATES];
    let mut is_rep_g2 = vec![K_BIT_MODEL_TOTAL / 2; K_NUM_STATES];
    let mut is_rep0_long = vec![K_BIT_MODEL_TOTAL / 2; K_NUM_STATES << K_NUM_POS_BITS_MAX];
    let mut pos_slot = vec![K_BIT_MODEL_TOTAL / 2; K_NUM_LEN_TO_POS_STATES * 64];
    let mut spec_pos = vec![K_BIT_MODEL_TOTAL / 2; 1 + K_NUM_FULL_DISTANCES - K_END_POS_MODEL_INDEX];
    let mut align = vec![K_BIT_MODEL_TOTAL / 2; 1 << K_NUM_ALIGN_BITS];
    let mut literal = vec![K_BIT_MODEL_TOTAL / 2; 0x300usize << (lc + lp)];
    let mut len_coder = LenDecoder::new();
    let mut rep_len_coder = LenDecoder::new();

    let mut out: Vec<u8> = Vec::new();
    let (mut state, mut rep0, mut rep1, mut rep2, mut rep3) = (0usize, 0u32, 0u32, 0u32, 0u32);

    while out.len() < cap {
        let pos_state = out.len() & pos_mask;
        if dec.decode_bit(&mut is_match[(state << K_NUM_POS_BITS_MAX) + pos_state]) == 0 {
            // literal
            let prev = out.last().copied().unwrap_or(0);
            let lit_state = (((out.len() & lit_pos_mask) << lc) + (usize::from(prev) >> (8 - lc)))
                * 0x300;
            let probs = &mut literal[lit_state..lit_state + 0x300];
            let symbol = if state < 7 {
                let mut m = 1usize;
                while m < 0x100 {
                    m = (m << 1) + dec.decode_bit(&mut probs[m]) as usize;
                }
                (m & 0xff) as u8
            } else {
                let mut match_byte = u32::from(out[out.len() - rep0 as usize - 1]);
                let mut m = 1usize;
                loop {
                    let match_bit = (match_byte >> 7) & 1;
                    match_byte <<= 1;
                    let bit = dec.decode_bit(
                        &mut probs[((1 + match_bit as usize) << 8) + m],
                    ) as u32;
                    m = (m << 1) + bit as usize;
                    if match_bit != bit {
                        while m < 0x100 {
                            m = (m << 1) + dec.decode_bit(&mut probs[m]) as usize;
                        }
                        break;
                    }
                    if m >= 0x100 {
                        break;
                    }
                }
                (m & 0xff) as u8
            };
            out.push(symbol);
            state = if state < 4 {
                0
            } else if state < 10 {
                state - 3
            } else {
                state - 6
            };
            continue;
        }
        // match
        let len;
        if dec.decode_bit(&mut is_rep[state]) == 1 {
            // rep match
            if dec.decode_bit(&mut is_rep_g0[state]) == 0 {
                if dec.decode_bit(&mut is_rep0_long[(state << K_NUM_POS_BITS_MAX) + pos_state]) == 0
                {
                    state = if state < 7 { 9 } else { 11 };
                    if rep0 as usize + 1 > out.len() {
                        return Err("rep0 distance out of range");
                    }
                    let b = out[out.len() - rep0 as usize - 1];
                    out.push(b);
                    continue;
                }
            } else {
                let dist;
                if dec.decode_bit(&mut is_rep_g1[state]) == 0 {
                    dist = rep1;
                } else if dec.decode_bit(&mut is_rep_g2[state]) == 0 {
                    dist = rep2;
                    rep2 = rep1;
                } else {
                    dist = rep3;
                    rep3 = rep2;
                    rep2 = rep1;
                }
                rep1 = rep0;
                rep0 = dist;
            }
            len = rep_len_coder.decode(&mut dec, pos_state) + K_MATCH_MIN_LEN;
            state = if state < 7 { 8 } else { 11 };
        } else {
            // new match
            rep3 = rep2;
            rep2 = rep1;
            rep1 = rep0;
            len = len_coder.decode(&mut dec, pos_state) + K_MATCH_MIN_LEN;
            state = if state < 7 { 7 } else { 10 };
            // distance
            let len_state = (len - K_MATCH_MIN_LEN).min(K_NUM_LEN_TO_POS_STATES - 1);
            let slot = bit_tree_decode(&mut dec, &mut pos_slot[len_state * 64..len_state * 64 + 64], 6);
            if slot < 4 {
                rep0 = slot as u32;
            } else {
                let num_direct = (slot >> 1) as u32 - 1;
                let mut dist = ((2 | (slot & 1)) as u32) << num_direct;
                if slot < K_END_POS_MODEL_INDEX {
                    let base = dist as usize - slot;
                    dist += bit_tree_reverse_decode(
                        &mut dec,
                        &mut spec_pos[base..base + (1usize << num_direct)],
                        num_direct,
                    ) as u32;
                } else {
                    dist += dec.decode_direct_bits(num_direct - K_NUM_ALIGN_BITS) << K_NUM_ALIGN_BITS;
                    dist += bit_tree_reverse_decode(&mut dec, &mut align, K_NUM_ALIGN_BITS) as u32;
                }
                rep0 = dist;
            }
            if rep0 == 0xFFFF_FFFF {
                // EOS marker.
                return Ok((out, dec.pos));
            }
        }
        if rep0 as usize + 1 > out.len() {
            return Err("match distance out of range");
        }
        for _ in 0..len {
            let b = out[out.len() - rep0 as usize - 1];
            out.push(b);
            if out.len() >= cap {
                break;
            }
        }
    }
    Ok((out, dec.pos))
}
