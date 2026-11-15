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

// --- Stock-layout state export / resume ----------------------------------
//
// VMProtect's depacker keeps its probabilities in one contiguous block in the
// stock 7-zip `LzmaDec.c` order (proven: the guest's literal-context base sat at
// prob index 1846 = the stock `Literal` offset). To hand a native decode back to
// the guest mid-stream, the probabilities must be serialized into that exact
// block. These offsets (u16 prob indices) are the stock layout:
const O_IS_MATCH: usize = 0;
const O_IS_REP: usize = 192;
const O_IS_REP_G0: usize = 204;
const O_IS_REP_G1: usize = 216;
const O_IS_REP_G2: usize = 228;
const O_IS_REP0_LONG: usize = 240;
const O_POS_SLOT: usize = 432;
const O_SPEC_POS: usize = 688; // stock block size 114 (= kNumFullDistances - kEndPosModelIndex)
const O_ALIGN: usize = 802;
const O_LEN_CODER: usize = 818; // 514 probs
const O_REP_LEN_CODER: usize = 1332; // 514 probs
const O_LITERAL: usize = 1846;
const LEN_PROBS: usize = 514; // choice, choice2, low[128], mid[128], high[256]

/// A full LZMA1 decoder state, with the probability block in stock 7-zip layout.
/// This is exactly the state the guest holds at a stream position: writing
/// `probs` into the guest's prob block and restoring the scalars reconstructs
/// the guest's decoder without it having executed a single bit.
#[derive(Clone)]
pub struct LzmaState {
    /// Probabilities in stock `LzmaDec.c` order; `len = O_LITERAL + (0x300 << (lc+lp))`.
    pub probs: Vec<u16>,
    pub range: u32,
    pub code: u32,
    /// Byte offset into the raw stream (what the guest's input cursor holds).
    pub input_pos: usize,
    /// Output length so far (the guest's `dicPos`); `pos_state`/`lit_state`
    /// derive from this.
    pub out_pos: usize,
    pub state: u32,
    pub reps: [u32; 4],
    pub lc: u32,
    pub lp: u32,
    pub pb: u32,
}

/// The decoder, factored into a struct so a decode can be paused, its state
/// serialized to stock layout, and resumed. Mirrors `decode_lzma1` exactly.
struct StreamDecoder {
    pos_mask: usize,
    lit_pos_mask: usize,
    lc: u32,
    lp: u32,
    pb: u32,
    is_match: Vec<Prob>,
    is_rep: Vec<Prob>,
    is_rep_g0: Vec<Prob>,
    is_rep_g1: Vec<Prob>,
    is_rep_g2: Vec<Prob>,
    is_rep0_long: Vec<Prob>,
    pos_slot: Vec<Prob>,
    spec_pos: Vec<Prob>,
    align: Vec<Prob>,
    literal: Vec<Prob>,
    len_coder: LenDecoder,
    rep_len_coder: LenDecoder,
    state: usize,
    rep0: u32,
    rep1: u32,
    rep2: u32,
    rep3: u32,
}

/// One decode step produced this.
enum Step {
    Byte,
    Eos,
}

impl StreamDecoder {
    fn new(lc: u32, lp: u32, pb: u32) -> Self {
        let half = K_BIT_MODEL_TOTAL / 2;
        StreamDecoder {
            pos_mask: (1usize << pb) - 1,
            lit_pos_mask: (1usize << lp) - 1,
            lc,
            lp,
            pb,
            is_match: vec![half; K_NUM_STATES << K_NUM_POS_BITS_MAX],
            is_rep: vec![half; K_NUM_STATES],
            is_rep_g0: vec![half; K_NUM_STATES],
            is_rep_g1: vec![half; K_NUM_STATES],
            is_rep_g2: vec![half; K_NUM_STATES],
            is_rep0_long: vec![half; K_NUM_STATES << K_NUM_POS_BITS_MAX],
            pos_slot: vec![half; K_NUM_LEN_TO_POS_STATES * 64],
            spec_pos: vec![half; 1 + K_NUM_FULL_DISTANCES - K_END_POS_MODEL_INDEX],
            align: vec![half; 1 << K_NUM_ALIGN_BITS],
            literal: vec![half; 0x300usize << (lc + lp)],
            len_coder: LenDecoder::new(),
            rep_len_coder: LenDecoder::new(),
            state: 0,
            rep0: 0,
            rep1: 0,
            rep2: 0,
            rep3: 0,
        }
    }

    /// Decode a single symbol into `out`. Returns `Err` on a malformed stream.
    fn step(&mut self, dec: &mut RangeDecoder, out: &mut Vec<u8>) -> Result<Step, &'static str> {
        let pos_state = out.len() & self.pos_mask;
        if dec.decode_bit(&mut self.is_match[(self.state << K_NUM_POS_BITS_MAX) + pos_state]) == 0 {
            let prev = out.last().copied().unwrap_or(0);
            let lit_state = (((out.len() & self.lit_pos_mask) << self.lc)
                + (usize::from(prev) >> (8 - self.lc)))
                * 0x300;
            let probs = &mut self.literal[lit_state..lit_state + 0x300];
            let symbol = if self.state < 7 {
                let mut m = 1usize;
                while m < 0x100 {
                    m = (m << 1) + dec.decode_bit(&mut probs[m]) as usize;
                }
                (m & 0xff) as u8
            } else {
                let mut match_byte = u32::from(out[out.len() - self.rep0 as usize - 1]);
                let mut m = 1usize;
                loop {
                    let match_bit = (match_byte >> 7) & 1;
                    match_byte <<= 1;
                    let bit = dec.decode_bit(&mut probs[((1 + match_bit as usize) << 8) + m]) as u32;
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
            self.state = if self.state < 4 {
                0
            } else if self.state < 10 {
                self.state - 3
            } else {
                self.state - 6
            };
            return Ok(Step::Byte);
        }
        let len;
        if dec.decode_bit(&mut self.is_rep[self.state]) == 1 {
            if dec.decode_bit(&mut self.is_rep_g0[self.state]) == 0 {
                if dec.decode_bit(
                    &mut self.is_rep0_long[(self.state << K_NUM_POS_BITS_MAX) + pos_state],
                ) == 0
                {
                    self.state = if self.state < 7 { 9 } else { 11 };
                    if self.rep0 as usize + 1 > out.len() {
                        return Err("rep0 distance out of range");
                    }
                    let b = out[out.len() - self.rep0 as usize - 1];
                    out.push(b);
                    return Ok(Step::Byte);
                }
            } else {
                let dist;
                if dec.decode_bit(&mut self.is_rep_g1[self.state]) == 0 {
                    dist = self.rep1;
                } else if dec.decode_bit(&mut self.is_rep_g2[self.state]) == 0 {
                    dist = self.rep2;
                    self.rep2 = self.rep1;
                } else {
                    dist = self.rep3;
                    self.rep3 = self.rep2;
                    self.rep2 = self.rep1;
                }
                self.rep1 = self.rep0;
                self.rep0 = dist;
            }
            len = self.rep_len_coder.decode(dec, pos_state) + K_MATCH_MIN_LEN;
            self.state = if self.state < 7 { 8 } else { 11 };
        } else {
            self.rep3 = self.rep2;
            self.rep2 = self.rep1;
            self.rep1 = self.rep0;
            len = self.len_coder.decode(dec, pos_state) + K_MATCH_MIN_LEN;
            self.state = if self.state < 7 { 7 } else { 10 };
            let len_state = (len - K_MATCH_MIN_LEN).min(K_NUM_LEN_TO_POS_STATES - 1);
            let slot =
                bit_tree_decode(dec, &mut self.pos_slot[len_state * 64..len_state * 64 + 64], 6);
            if slot < 4 {
                self.rep0 = slot as u32;
            } else {
                let num_direct = (slot >> 1) as u32 - 1;
                let mut dist = ((2 | (slot & 1)) as u32) << num_direct;
                if slot < K_END_POS_MODEL_INDEX {
                    let base = dist as usize - slot;
                    dist += bit_tree_reverse_decode(
                        dec,
                        &mut self.spec_pos[base..base + (1usize << num_direct)],
                        num_direct,
                    ) as u32;
                } else {
                    dist += dec.decode_direct_bits(num_direct - K_NUM_ALIGN_BITS) << K_NUM_ALIGN_BITS;
                    dist += bit_tree_reverse_decode(dec, &mut self.align, K_NUM_ALIGN_BITS) as u32;
                }
                self.rep0 = dist;
            }
            if self.rep0 == 0xFFFF_FFFF {
                return Ok(Step::Eos);
            }
        }
        if self.rep0 as usize + 1 > out.len() {
            return Err("match distance out of range");
        }
        for _ in 0..len {
            let b = out[out.len() - self.rep0 as usize - 1];
            out.push(b);
        }
        Ok(Step::Byte)
    }

    /// Serialize the probability arrays into one stock-layout block.
    fn export(&self, dec: &RangeDecoder, out_pos: usize) -> LzmaState {
        let lit = 0x300usize << (self.lc + self.lp);
        let mut probs = vec![0u16; O_LITERAL + lit];
        probs[O_IS_MATCH..O_IS_MATCH + self.is_match.len()].copy_from_slice(&self.is_match);
        probs[O_IS_REP..O_IS_REP + self.is_rep.len()].copy_from_slice(&self.is_rep);
        probs[O_IS_REP_G0..O_IS_REP_G0 + self.is_rep_g0.len()].copy_from_slice(&self.is_rep_g0);
        probs[O_IS_REP_G1..O_IS_REP_G1 + self.is_rep_g1.len()].copy_from_slice(&self.is_rep_g1);
        probs[O_IS_REP_G2..O_IS_REP_G2 + self.is_rep_g2.len()].copy_from_slice(&self.is_rep_g2);
        probs[O_IS_REP0_LONG..O_IS_REP0_LONG + self.is_rep0_long.len()]
            .copy_from_slice(&self.is_rep0_long);
        probs[O_POS_SLOT..O_POS_SLOT + self.pos_slot.len()].copy_from_slice(&self.pos_slot);
        // spec_pos: local array carries one dead leading slot; stock block omits it.
        probs[O_SPEC_POS..O_SPEC_POS + 114].copy_from_slice(&self.spec_pos[1..115]);
        probs[O_ALIGN..O_ALIGN + self.align.len()].copy_from_slice(&self.align);
        self.len_coder.export(&mut probs[O_LEN_CODER..O_LEN_CODER + LEN_PROBS]);
        self.rep_len_coder
            .export(&mut probs[O_REP_LEN_CODER..O_REP_LEN_CODER + LEN_PROBS]);
        probs[O_LITERAL..O_LITERAL + lit].copy_from_slice(&self.literal);
        LzmaState {
            probs,
            range: dec.range,
            code: dec.code,
            input_pos: dec.pos,
            out_pos,
            state: self.state as u32,
            reps: [self.rep0, self.rep1, self.rep2, self.rep3],
            lc: self.lc,
            lp: self.lp,
            pb: self.pb,
        }
    }

    /// Reconstruct a decoder from a stock-layout block (inverse of `export`).
    fn import(st: &LzmaState) -> Self {
        let mut d = StreamDecoder::new(st.lc, st.lp, st.pb);
        let p = &st.probs;
        // Source slices are bounded by the next sub-array's stock offset, so the
        // destination's own length is never borrowed during the copy.
        d.is_match.copy_from_slice(&p[O_IS_MATCH..O_IS_REP]);
        d.is_rep.copy_from_slice(&p[O_IS_REP..O_IS_REP_G0]);
        d.is_rep_g0.copy_from_slice(&p[O_IS_REP_G0..O_IS_REP_G1]);
        d.is_rep_g1.copy_from_slice(&p[O_IS_REP_G1..O_IS_REP_G2]);
        d.is_rep_g2.copy_from_slice(&p[O_IS_REP_G2..O_IS_REP0_LONG]);
        d.is_rep0_long.copy_from_slice(&p[O_IS_REP0_LONG..O_POS_SLOT]);
        d.pos_slot.copy_from_slice(&p[O_POS_SLOT..O_SPEC_POS]);
        d.spec_pos[1..115].copy_from_slice(&p[O_SPEC_POS..O_SPEC_POS + 114]);
        d.align.copy_from_slice(&p[O_ALIGN..O_LEN_CODER]);
        d.len_coder.import(&p[O_LEN_CODER..O_LEN_CODER + LEN_PROBS]);
        d.rep_len_coder.import(&p[O_REP_LEN_CODER..O_REP_LEN_CODER + LEN_PROBS]);
        let lit = 0x300usize << (st.lc + st.lp);
        d.literal.copy_from_slice(&p[O_LITERAL..O_LITERAL + lit]);
        d.state = st.state as usize;
        d.rep0 = st.reps[0];
        d.rep1 = st.reps[1];
        d.rep2 = st.reps[2];
        d.rep3 = st.reps[3];
        d
    }
}

impl LenDecoder {
    /// Serialize into a stock LenCoder block: choice, choice2, low[128], mid[128], high[256].
    fn export(&self, dst: &mut [Prob]) {
        dst[0] = self.choice;
        dst[1] = self.choice2;
        dst[2..130].copy_from_slice(&self.low);
        dst[130..258].copy_from_slice(&self.mid);
        dst[258..514].copy_from_slice(&self.high);
    }
    fn import(&mut self, src: &[Prob]) {
        self.choice = src[0];
        self.choice2 = src[1];
        self.low.copy_from_slice(&src[2..130]);
        self.mid.copy_from_slice(&src[130..258]);
        self.high.copy_from_slice(&src[258..514]);
    }
}

/// Decode one stream to EOS, capturing the full decoder state at the top of the
/// `snap_at`-th symbol (0-based) along the way. Returns the complete output, the
/// input bytes consumed, and the snapshot (stock-layout state at that symbol
/// boundary). `None` snapshot means the stream ended before `snap_at`.
pub fn decode_lzma1_snapshot(
    input: &[u8],
    cap: usize,
    lc: u32,
    lp: u32,
    pb: u32,
    snap_at: u64,
) -> Result<(Vec<u8>, usize, Option<LzmaState>), &'static str> {
    let Some(mut dec) = RangeDecoder::new(input) else {
        return Err("stream too short");
    };
    let mut d = StreamDecoder::new(lc, lp, pb);
    let mut out: Vec<u8> = Vec::new();
    let mut snap = None;
    let mut sym = 0u64;
    while out.len() < cap {
        if sym == snap_at {
            snap = Some(d.export(&dec, out.len()));
        }
        match d.step(&mut dec, &mut out)? {
            Step::Byte => {}
            Step::Eos => return Ok((out, dec.pos, snap)),
        }
        sym += 1;
    }
    Ok((out, dec.pos, snap))
}

/// Like `decode_lzma1_snapshot`, but the snapshot is taken at the top of the
/// first symbol whose range-coder input cursor has reached `input_off` bytes.
/// (The guest's state dumps are keyed by input offset, not symbol index.)
pub fn decode_lzma1_snapshot_at_input(
    input: &[u8],
    cap: usize,
    lc: u32,
    lp: u32,
    pb: u32,
    input_off: usize,
) -> Result<(Vec<u8>, usize, Option<LzmaState>), &'static str> {
    let Some(mut dec) = RangeDecoder::new(input) else {
        return Err("stream too short");
    };
    let mut d = StreamDecoder::new(lc, lp, pb);
    let mut out: Vec<u8> = Vec::new();
    let mut snap = None;
    while out.len() < cap {
        if snap.is_none() && dec.pos >= input_off {
            snap = Some(d.export(&dec, out.len()));
        }
        match d.step(&mut dec, &mut out)? {
            Step::Byte => {}
            Step::Eos => return Ok((out, dec.pos, snap)),
        }
    }
    Ok((out, dec.pos, snap))
}

/// Resume a decode from an exported state, appending to `out` (which must hold
/// the already-decoded prefix, since matches reference it). Returns the input
/// bytes consumed, relative to the resume position.
pub fn resume_lzma1(
    st: &LzmaState,
    input_from_resume: &[u8],
    cap: usize,
    out: &mut Vec<u8>,
) -> Result<usize, &'static str> {
    let mut dec = RangeDecoder {
        input: input_from_resume,
        pos: 0,
        range: st.range,
        code: st.code,
    };
    let mut d = StreamDecoder::import(st);
    while out.len() < cap {
        match d.step(&mut dec, out)? {
            Step::Byte => {}
            Step::Eos => return Ok(dec.pos),
        }
    }
    Ok(dec.pos)
}
