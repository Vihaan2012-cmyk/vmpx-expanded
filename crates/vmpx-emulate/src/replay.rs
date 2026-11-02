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

    /// Exercises a full recording against known values, including the array and
    /// coercion paths the roundtrip test above does not reach. Set
    /// `VMPX_REPLAY_FIXTURE` to a generated stream; skipped when unset.
    #[test]
    fn fixture_serves_every_payload_type() {
        let Some(path) = std::env::var_os("VMPX_REPLAY_FIXTURE") else {
            return;
        };
        let mut r = Replay::load(std::path::Path::new(&path)).expect("fixture loads");
        assert_eq!(r.n_datarefs(), 33, "fixture dataref count");
        assert_eq!(r.n_frames(), 900, "fixture frame count");

        // Before the first advance every slot is still zero-initialised.
        assert_eq!(r.get_f32("sim/flightmodel/position/indicated_airspeed"), Some(0.0));
        assert!(r.advance(), "frame 0 applies");

        let near = |got: Option<f32>, want: f32| {
            let g = got.expect("value present");
            assert!((g - want).abs() < 1e-3, "got {g}, want {want}");
        };
        // f64 slot read as f64 and coerced down to f32.
        let elev = r.get_f64("sim/flightmodel/position/elevation").expect("f64");
        assert!((elev - 300.0).abs() < 1e-9, "elevation {elev}");
        near(r.get_f32("sim/flightmodel/position/elevation"), 300.0);
        // f32 slot read as f32 and coerced up to f64.
        near(r.get_f32("sim/flightmodel/position/indicated_airspeed"), 140.0);
        assert!(
            r.get_f64("sim/flightmodel/position/indicated_airspeed")
                .is_some_and(|v| (v - 140.0).abs() < 1e-3)
        );
        // int slot, plus coercion to float.
        assert_eq!(r.get_i32("sim/cockpit2/controls/gear_handle_down"), Some(1));
        near(r.get_f32("sim/cockpit2/controls/gear_handle_down"), 1.0);

        // float array: whole vector, and element 0 via the scalar accessor.
        let mut buf = [0f32; 4];
        let n = r.get_vf("sim/flightmodel/engine/ENGN_N1_", &mut buf);
        assert_eq!(n, 2, "ENGN_N1_ is 2 elements");
        assert!((buf[0] - 88.0).abs() < 1e-3, "N1[0] {}", buf[0]);
        assert!((buf[1] - 88.5).abs() < 1e-3, "N1[1] {}", buf[1]);
        near(r.get_f32("sim/flightmodel/engine/ENGN_N1_"), 88.0);

        // int array read as floats.
        let mut ib = [0f32; 4];
        assert_eq!(r.get_vf("sim/cockpit2/electrical/battery_on", &mut ib), 2);
        near(Some(ib[0]), 1.0);
        assert_eq!(r.get_i32("sim/cockpit2/electrical/battery_on"), Some(1));

        // A 4-element float array fills exactly 4 slots.
        let mut vb = [0f32; 4];
        assert_eq!(r.get_vf("sim/cockpit2/electrical/bus_volts", &mut vb), 4);
        near(Some(vb[3]), 27.8);

        // An absent name must stay None rather than reading a neighbour's bytes.
        assert_eq!(r.get_f32("sim/does/not/exist"), None);
        assert_eq!(r.get_vf("sim/does/not/exist", &mut buf), 0);

        // Drive the whole stream: values must move and the cursor must exhaust.
        let start = r.get_f32("sim/flightmodel/position/elevation").unwrap();
        let mut steps = 1;
        while r.advance() {
            steps += 1;
        }
        assert_eq!(steps, 900, "every frame steps once");
        let end = r.get_f32("sim/flightmodel/position/elevation").unwrap();
        assert!(end > start + 100.0, "climb should be recorded: {start} -> {end}");
        // Exhausted, but the last frame's state is retained.
        assert!(!r.advance());
        assert!(r.get_f32("sim/flightmodel/position/elevation").is_some());
    }
}
