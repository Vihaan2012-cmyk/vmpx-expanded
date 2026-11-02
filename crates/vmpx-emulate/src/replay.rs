//! Replay of a recorded X-Plane `sim/` dataref stream (`VMPXREC1`, produced by
//! the vmpx sim recorder). The contained detonation serves these values on
//! `XPLMGetData*` reads so virtualized flight-loop functions execute on real
//! sim state instead of zeros. Observed-tier only: a single captured flight.

use std::collections::HashMap;
use std::io::{self, Read};

const T_INT: u8 = 1;
const T_FLOAT: u8 = 2;
const T_DOUBLE: u8 = 4;
const T_FARR: u8 = 8;
const T_IARR: u8 = 16;

struct Dref {
    kind: u8,
    arrlen: usize,
    size: usize,
}

struct FrameDelta {
    changes: Vec<(u32, Vec<u8>)>,
}

/// A loaded recording with a frame cursor and the reconstructed current state.
pub struct Replay {
    drefs: Vec<Dref>,
    by_name: HashMap<String, usize>,
    frames: Vec<FrameDelta>,
    cursor: usize,
    state: Vec<Vec<u8>>,
}

fn rd_u16(b: &[u8], p: &mut usize) -> Option<u16> {
    let v = u16::from_le_bytes(b.get(*p..*p + 2)?.try_into().ok()?);
    *p += 2;
    Some(v)
}
fn rd_u32(b: &[u8], p: &mut usize) -> Option<u32> {
    let v = u32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?);
    *p += 4;
    Some(v)
}

fn payload_size(kind: u8, arrlen: usize) -> usize {
    match kind {
        T_INT | T_FLOAT => 4,
        T_DOUBLE => 8,
        T_FARR | T_IARR => arrlen * 4,
        _ => 0,
    }
}

impl Replay {
    /// # Errors
    /// Fails when the file is missing, truncated, or not a `VMPXREC1` stream.
    pub fn load(path: &std::path::Path) -> io::Result<Replay> {
        let mut b = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut b)?;
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "not a VMPXREC1 recording");
        if b.len() < 12 || &b[0..8] != b"VMPXREC1" {
            return Err(bad());
        }
        let mut p = 8usize;
        let count = rd_u32(&b, &mut p).ok_or_else(bad)? as usize;
        let mut drefs = Vec::with_capacity(count);
        let mut by_name = HashMap::with_capacity(count);
        for i in 0..count {
            let nlen = rd_u16(&b, &mut p).ok_or_else(bad)? as usize;
            let name = String::from_utf8_lossy(b.get(p..p + nlen).ok_or_else(bad)?).into_owned();
            p += nlen;
            let kind = *b.get(p).ok_or_else(bad)?;
            p += 1;
            let arrlen = rd_u16(&b, &mut p).ok_or_else(bad)? as usize;
            by_name.insert(name, i);
            drefs.push(Dref {
                kind,
                arrlen,
                size: payload_size(kind, arrlen),
            });
        }
        // Frames: { f32 time, u32 nchanged, nchanged * (u32 idx, payload) }.
        let mut frames = Vec::new();
        while p + 8 <= b.len() {
            p += 4; // time (unused by the current driver; state advances per call)
            let n = rd_u32(&b, &mut p).ok_or_else(bad)? as usize;
            let mut changes = Vec::with_capacity(n);
            let mut ok = true;
            for _ in 0..n {
                let Some(idx) = rd_u32(&b, &mut p) else {
                    ok = false;
                    break;
                };
                let d = match drefs.get(idx as usize) {
                    Some(d) => d,
                    None => {
                        ok = false;
                        break;
                    }
                };
                let Some(bytes) = b.get(p..p + d.size) else {
                    ok = false;
                    break;
                };
                p += d.size;
                changes.push((idx, bytes.to_vec()));
            }
            if !ok {
                break; // tolerate a truncated tail (crash mid-flush)
            }
            frames.push(FrameDelta { changes });
        }
        let state = drefs.iter().map(|d| vec![0u8; d.size]).collect();
        Ok(Replay {
            drefs,
            by_name,
            frames,
            cursor: 0,
            state,
        })
    }

    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    #[must_use]
    pub fn n_datarefs(&self) -> usize {
        self.drefs.len()
    }

    /// Applies the next frame's deltas to the current state. Returns false when
    /// the stream is exhausted (the state then holds the last frame's values).
    pub fn advance(&mut self) -> bool {
        let Some(frame) = self.frames.get(self.cursor) else {
            return false;
        };
        for (idx, bytes) in &frame.changes {
            if let Some(slot) = self.state.get_mut(*idx as usize) {
                slot.clear();
                slot.extend_from_slice(bytes);
            }
        }
        self.cursor += 1;
        true
    }

    fn slot(&self, name: &str) -> Option<(&Dref, &[u8])> {
        let i = *self.by_name.get(name)?;
        Some((&self.drefs[i], self.state[i].as_slice()))
    }

    #[must_use]
    pub fn get_f32(&self, name: &str) -> Option<f32> {
        let (d, s) = self.slot(name)?;
        match d.kind {
            T_FLOAT if s.len() == 4 => Some(f32::from_le_bytes(s.try_into().ok()?)),
            T_DOUBLE if s.len() == 8 => Some(f64::from_le_bytes(s.try_into().ok()?) as f32),
            T_INT if s.len() == 4 => Some(i32::from_le_bytes(s.try_into().ok()?) as f32),
            T_FARR if s.len() >= 4 => Some(f32::from_le_bytes(s[0..4].try_into().ok()?)),
            _ => None,
        }
    }

    #[must_use]
    pub fn get_f64(&self, name: &str) -> Option<f64> {
        let (d, s) = self.slot(name)?;
        match d.kind {
            T_DOUBLE if s.len() == 8 => Some(f64::from_le_bytes(s.try_into().ok()?)),
            T_FLOAT if s.len() == 4 => Some(f64::from(f32::from_le_bytes(s.try_into().ok()?))),
            T_INT if s.len() == 4 => Some(f64::from(i32::from_le_bytes(s.try_into().ok()?))),
            _ => None,
        }
    }

    #[must_use]
    pub fn get_i32(&self, name: &str) -> Option<i32> {
        let (d, s) = self.slot(name)?;
        match d.kind {
            T_INT if s.len() == 4 => Some(i32::from_le_bytes(s.try_into().ok()?)),
            T_FLOAT if s.len() == 4 => Some(f32::from_le_bytes(s.try_into().ok()?) as i32),
            T_DOUBLE if s.len() == 8 => Some(f64::from_le_bytes(s.try_into().ok()?) as i32),
            T_IARR if s.len() >= 4 => Some(i32::from_le_bytes(s[0..4].try_into().ok()?)),
            _ => None,
        }
    }

    /// Fills `out` with the recorded float array (or int array as floats), up to
    /// the recorded length; returns how many elements were written.
    pub fn get_vf(&self, name: &str, out: &mut [f32]) -> usize {
        let Some((d, s)) = self.slot(name) else {
            return 0;
        };
        if d.kind != T_FARR && d.kind != T_IARR {
            return 0;
        }
        let n = d.arrlen.min(out.len());
        for i in 0..n {
            let Some(chunk) = s.get(i * 4..i * 4 + 4) else {
                return i;
            };
            let arr: [u8; 4] = chunk.try_into().unwrap();
            out[i] = if d.kind == T_FARR {
                f32::from_le_bytes(arr)
            } else {
                i32::from_le_bytes(arr) as f32
            };
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn roundtrip_delta_stream() {
        // header: 2 drefs (a float, an int), then 2 frames.
        let mut b = Vec::new();
        b.extend_from_slice(b"VMPXREC1");
        b.extend_from_slice(&2u32.to_le_bytes());
        for (name, kind) in [("sim/a", T_FLOAT), ("sim/b", T_INT)] {
            b.extend_from_slice(&(name.len() as u16).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.push(kind);
            b.extend_from_slice(&0u16.to_le_bytes());
        }
        // frame 0: both change
        b.extend_from_slice(&0.0f32.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&1.5f32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&7i32.to_le_bytes());
        // frame 1: only the float changes
        b.extend_from_slice(&0.03f32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&2.5f32.to_le_bytes());

        let dir = std::env::temp_dir();
        let path = dir.join("vmpx_replay_test.bin");
        std::fs::File::create(&path).unwrap().write_all(&b).unwrap();
        let mut r = Replay::load(&path).unwrap();
        assert_eq!(r.n_datarefs(), 2);
        assert_eq!(r.n_frames(), 2);
        assert!(r.advance());
        assert_eq!(r.get_f32("sim/a"), Some(1.5));
        assert_eq!(r.get_i32("sim/b"), Some(7));
        assert!(r.advance());
        assert_eq!(r.get_f32("sim/a"), Some(2.5));
        assert_eq!(r.get_i32("sim/b"), Some(7)); // carried
        assert!(!r.advance());
        let _ = std::fs::remove_file(&path);
    }
}
