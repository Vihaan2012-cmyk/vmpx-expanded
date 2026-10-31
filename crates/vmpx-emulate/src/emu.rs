use iced_x86::{
    Code, ConditionCode, Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind,
    Register,
};

use crate::alu::{self, Alu, Flags, Rotate, Shift};
use crate::cpu::{AF, CF, Cpu, OF, PF, RSP, SF, ZF, mask};
use crate::memory::{Access, Memory};
use crate::predecode::{self, K, Opnd, Pre, T_MEM, T_REG};

#[path = "sse.rs"]
mod sse;

// GPR indices into `Cpu::regs` (RSP lives in cpu.rs).
const RAX: usize = 0;
const RCX: usize = 1;
const RDX: usize = 2;
const RBP: usize = 5;
const RSI: usize = 6;
const RDI: usize = 7;
/// Direction flag (rflags bit 10); controls string-op pointer stride.
const DF: u64 = 1 << 10;
/// The six status flags mul/div/string ops read or leave undefined.
const STATUS: u64 = CF | PF | AF | ZF | SF | OF;

/// Anti-analysis-relevant instructions executed: the `cpuid` (leaf, subleaf)
/// queries and `rdtsc`/`rdtscp` reads. Pure bookkeeping; it never alters
/// execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InsnAudit {
    pub cpuid: std::collections::BTreeMap<(u64, u64), u64>,
    pub rdtsc: u64,
    pub rdtscp: u64,
    /// `(rip, retired-step)` of the first 64 `rdtsc`/`rdtscp` executions.
    pub tsc_sites: Vec<(u64, u64)>,
}

/// x86 Trap Flag (rflags bit 8).
const TF: u64 = 1 << 8;

/// A [`Stop`] on the hot paths: boxed so `Result<u64, Trap>` is two registers.
type Trap = Box<Stop>;

impl From<Trap> for Stop {
    fn from(t: Trap) -> Stop {
        *t
    }
}

/// Why a `step`/`run` did not retire an instruction. On any `Stop` the
/// architectural state is exactly as before the offending instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    Unsupported { rip: u64, mnemonic: Mnemonic },
    /// An SSE/XMM instruction outside the modeled subset (see `sse.rs`).
    UnmodeledSse { rip: u64, mnemonic: Mnemonic },
    DecodeError { rip: u64 },
    Fault { rip: u64, addr: u64, access: Access },
    UndefinedFlags { rip: u64, mask: u64 },
    /// `#DE`: `div`/`idiv` by zero or a quotient that does not fit. Modeled as a
    /// clean stop — never a host divide trap.
    DivideError { rip: u64 },
    Hook,
    Reached,
    StepLimit,
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stop::UnmodeledSse { rip, mnemonic } => {
                write!(f, "unmodeled SSE: {} at {rip:#x}", format!("{mnemonic:?}").to_lowercase())
            }
            _ => write!(f, "{self:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceEntry {
    pub rip: u64,
    pub mnemonic: Mnemonic,
    /// Effective address of the step's last memory access (detail mode only).
    pub ea: Option<u64>,
    /// Value read/written by that access (detail mode only).
    pub value: Option<u64>,
    /// Post-step value of the configured VM stack pointer (detail mode only).
    pub vsp: Option<u64>,
}

/// Opt-in richer trace capture. All-false default keeps the hot path cheap:
/// nothing extra is computed unless `detail` is set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObserveConfig {
    /// Record per-step effective address + accessed value.
    pub detail: bool,
    /// GPR treated as the VM stack pointer; its value is recorded each step.
    pub vsp: Option<Register>,
}

/// A guest memory access, reported before it commits (`value` is set for writes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemEvent {
    pub addr: u64,
    pub size: u32,
    pub access: Access,
    pub value: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowKind {
    Jmp,
    Jcc { taken: bool },
    Call,
    Ret,
}

/// A control-flow transfer, reported before it commits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlowEvent {
    pub kind: FlowKind,
    pub from: u64,
    pub to: u64,
    /// Fall-through address: the instruction after the branch. For a `Call`
    /// this is the return address a stub hook redirects to.
    pub next: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookAction {
    /// Let the instruction commit as decoded.
    Continue,
    /// Abort the run with [`Stop::Hook`] (state rolled back as for any stop).
    Stop,
    /// The hook has set `cpu.rip` (and any registers/memory it wants); resume
    /// there instead of this instruction's own transfer. Honoured for
    /// control-flow events; on a memory event it behaves like `Continue`
    /// (there is no safe mid-instruction redirect point).
    Redirect,
}

/// Observation/interception seam; all callbacks default to no-ops.
///
/// A hook may mutate `cpu`/`mem` in place (e.g. set RAX and redirect `rip` to
/// fake an API return) and then resume — see [`HookAction::Redirect`]. Because
/// a memory callback fires *before* the access commits, writing `mem` there
/// changes the value the pending load/store will see.
pub trait Hook {
    fn on_mem(&mut self, _ev: &MemEvent, _cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        HookAction::Continue
    }
    fn on_flow(&mut self, _ev: &FlowEvent, _cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        HookAction::Continue
    }
    /// Which events this hook can be affected by; `run`/`run_until` skip
    /// building and delivering the rest. Re-queried after every delivered
    /// callback and at the start of each `run`/`run_until`, so it may depend on
    /// state the callbacks change. The default (every event) is always safe.
    fn interest(&self) -> Interest {
        Interest::ALL
    }
}

/// Half-open address span `[lo, hi)`; empty when `lo >= hi`.
pub type Span = (u64, u64);

/// The events a [`Hook`] needs delivered. An event outside it must be a no-op
/// for the hook (no state change, `Continue`).
///
/// Spans select by the access's start address (`MemEvent::addr`) or a
/// transfer's target (`FlowEvent::to`). Two further promises make state-
/// dependent interest cheap:
/// - `write_pages`: writes outside `write` are interesting only until a write
///   covering their page(s) has been delivered; after that, later writes
///   wholly inside such pages are no-ops.
/// - `flow_pages`: only a transfer into a different page than it came from is
///   interesting, and only while no `Continue`-answered transfer into that
///   target page has been delivered (a `Continue`-answered one leaves the hook
///   unchanged; any other delivered callback restarts this).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interest {
    pub read: [Span; 2],
    pub write: [Span; 2],
    pub flow: [Span; 2],
    pub read_all: bool,
    pub write_all: bool,
    pub flow_all: bool,
    pub write_pages: bool,
    pub flow_pages: bool,
}

const NO_SPAN: [Span; 2] = [(0, 0); 2];

impl Interest {
    /// Nothing is interesting.
    pub const NONE: Interest = Interest {
        read: NO_SPAN,
        write: NO_SPAN,
        flow: NO_SPAN,
        read_all: false,
        write_all: false,
        flow_all: false,
        write_pages: false,
        flow_pages: false,
    };
    /// Every event is interesting.
    pub const ALL: Interest =
        Interest { read_all: true, write_all: true, flow_all: true, ..Interest::NONE };

    /// Interest of two hooks run back to back (spans merged conservatively).
    #[must_use]
    pub fn union(self, o: Interest) -> Interest {
        Interest {
            read: merge_spans(self.read, o.read),
            write: merge_spans(self.write, o.write),
            flow: merge_spans(self.flow, o.flow),
            read_all: self.read_all | o.read_all,
            write_all: self.write_all | o.write_all,
            flow_all: self.flow_all | o.flow_all,
            write_pages: self.write_pages | o.write_pages,
            flow_pages: self.flow_pages | o.flow_pages,
        }
    }
}

/// Cover up to four spans with two, merging the closest pair until it fits.
fn merge_spans(a: [Span; 2], b: [Span; 2]) -> [Span; 2] {
    let mut v = [(0u64, 0u64); 4];
    let mut n = 0;
    for s in a.into_iter().chain(b) {
        if s.0 < s.1 {
            v[n] = s;
            n += 1;
        }
    }
    v[..n].sort_unstable();
    while n > 2 {
        let gap = |i: usize| v[i + 1].0.saturating_sub(v[i].1);
        let i = (0..n - 1).min_by_key(|&i| gap(i)).unwrap_or(0);
        v[i] = (v[i].0.min(v[i + 1].0), v[i].1.max(v[i + 1].1));
        v.copy_within(i + 2..n, i + 1);
        n -= 1;
    }
    [v[0], v[1]]
}

const GATE_SLOTS: usize = 32;
const PAGE_SHIFT: u32 = 12;
const PAGE_BYTES: u64 = 1 << PAGE_SHIFT;
const _: () = assert!(crate::memory::PAGE_SIZE == 1 << PAGE_SHIFT);

/// [`Interest`] in the form the hot loop tests: spans as `(lo, len)`.
#[derive(Clone, Copy)]
struct Gate {
    read: [Span; 2],
    write: [Span; 2],
    flow: [Span; 2],
    read_all: bool,
    write_all: bool,
    flow_all: bool,
    write_pages: bool,
    flow_pages: bool,
    /// Whether `read_all` / any read span is set (else reads are never wanted).
    rd_any: bool,
    /// Likewise for `write_all` / the write spans.
    wr_any: bool,
    /// Set while a `run`/`run_until` is eliding; outside one every event is
    /// delivered and nothing is memoized.
    active: bool,
    /// Pages already vouched quiet (direct-mapped by page number).
    wq: [u64; GATE_SLOTS],
    fq: [u64; GATE_SLOTS],
}

const EMPTY_PAGE: u64 = u64::MAX;

impl Gate {
    const ALL: Gate = Gate {
        read: NO_SPAN,
        write: NO_SPAN,
        flow: NO_SPAN,
        read_all: true,
        write_all: true,
        flow_all: true,
        write_pages: false,
        flow_pages: false,
        rd_any: true,
        wr_any: true,
        active: false,
        wq: [EMPTY_PAGE; GATE_SLOTS],
        fq: [EMPTY_PAGE; GATE_SLOTS],
    };

    fn set(&mut self, i: &Interest) {
        let len = |s: [Span; 2]| s.map(|(lo, hi)| (lo, hi.saturating_sub(lo)));
        self.read = len(i.read);
        self.write = len(i.write);
        self.flow = len(i.flow);
        self.read_all = i.read_all;
        self.write_all = i.write_all;
        self.flow_all = i.flow_all;
        self.write_pages = i.write_pages;
        self.flow_pages = i.flow_pages;
        let any = |s: &[Span; 2]| s.iter().any(|&(_, len)| len != 0);
        self.rd_any = i.read_all || any(&self.read);
        self.wr_any = i.write_all || any(&self.write);
    }

    #[inline(always)]
    fn hit(s: &[Span; 2], a: u64) -> bool {
        (a.wrapping_sub(s[0].0) < s[0].1) | (a.wrapping_sub(s[1].0) < s[1].1)
    }

    #[inline(always)]
    fn wants_read(&self, addr: u64) -> bool {
        self.rd_any && (self.read_all | Self::hit(&self.read, addr))
    }

    #[inline(always)]
    fn wants_write(&self, addr: u64, size: u32) -> bool {
        if self.wr_any && (self.write_all | Self::hit(&self.write, addr)) {
            return true;
        }
        if !self.write_pages {
            return false;
        }
        let p0 = addr >> PAGE_SHIFT;
        if (addr & (PAGE_BYTES - 1)) + u64::from(size) <= PAGE_BYTES {
            return self.wq[p0 as usize % GATE_SLOTS] != p0;
        }
        match Self::pages(addr, size) {
            Some((p0, p1)) => {
                self.wq[p0 as usize % GATE_SLOTS] != p0 || self.wq[p1 as usize % GATE_SLOTS] != p1
            }
            None => true,
        }
    }

    /// First and last page number touched; `None` if the access wraps the address space.
    #[inline(always)]
    fn pages(addr: u64, size: u32) -> Option<(u64, u64)> {
        let last = addr.checked_add(u64::from(size.max(1)) - 1)?;
        Some((addr >> PAGE_SHIFT, last >> PAGE_SHIFT))
    }

    #[inline(always)]
    fn wants_flow(&self, from: u64, to: u64) -> bool {
        if self.flow_all | Self::hit(&self.flow, to) {
            return true;
        }
        let p = to >> PAGE_SHIFT;
        self.flow_pages && (from >> PAGE_SHIFT) != p && self.fq[p as usize % GATE_SLOTS] != p
    }

    fn note_write(&mut self, addr: u64, size: u32) {
        if let Some((p0, p1)) = Self::pages(addr, size) {
            self.wq[p0 as usize % GATE_SLOTS] = p0;
            self.wq[p1 as usize % GATE_SLOTS] = p1;
        }
    }
}

/// Internal resolution of a hook callback (a `Stop` is surfaced as `Err`).
enum HookOutcome {
    Continue,
    Redirect,
}

pub struct Emulator {
    /// CPU-fingerprint / timing instructions the guest has executed (audit only).
    pub audit: InsnAudit,
    pub cpu: Cpu,
    pub mem: Memory,
    pub trace: Vec<TraceEntry>,
    /// Record a [`TraceEntry`] per retired instruction (default). Long runs
    /// that only need the count turn this off and read [`Emulator::steps`].
    pub record_trace: bool,
    pub hook: Option<Box<dyn Hook>>,
    pub observe: ObserveConfig,
    /// DETONATION-only concretization. When `false` (the default), a read of an
    /// architecturally-undefined flag stops the run with [`Stop::UndefinedFlags`]
    /// — the sound, fail-closed behaviour the static/differential paths rely on,
    /// byte-for-byte unchanged. When `true`, each flag producer resolves its
    /// undefined bits to the deterministic value real silicon leaves (see
    /// [`Emulator::apply`]), so a later `jcc`/`cmov`/`setcc` reads a concrete bit
    /// and proceeds. Observed/quarantined tier only — never sound static output.
    pub resolve_undefined_flags: bool,
    cur: (u64, Mnemonic),
    det: (Option<u64>, Option<u64>),
    /// Retired-instruction count (always maintained, independent of `trace`).
    steps: u64,
    /// Direct-mapped decode cache indexed by RIP. A hit is only honoured when
    /// the current guest bytes still match the stored ones, so self-modifying/
    /// unpacking code re-decodes instead of reusing a stale entry. A colliding
    /// RIP simply evicts; a miss always re-decodes, so results are unchanged.
    dcache: Box<[DEntry]>,
    /// Direct-mapped cache of straight-line blocks of pre-resolved ops, keyed
    /// by start RIP (see [`Block`]); used by `run`/`run_until` when no trace is
    /// being recorded.
    bcache: Vec<Option<Box<Block>>>,
    /// Which hook events `run`/`run_until` still need to deliver (see [`Interest`]).
    gate: Gate,
}

/// A straight-line run of pre-resolved instructions starting at `start`,
/// ending at the first control transfer / unmodeled instruction (included), a
/// page-end fetch boundary, or `BLOCK_MAX` ops.
///
/// Soundness: the block records the exact instruction bytes it was built from
/// and the [`Memory::code_gen`] it was last validated at. While the generation
/// is unchanged no executable-cached page was written, so the bytes cannot
/// have changed; otherwise they are re-compared before reuse, and a store that
/// bumps the generation mid-block ends the block so the next instruction is
/// re-fetched fresh.
struct Block {
    start: u64,
    epoch: u64,
    /// The instruction bytes the ops were decoded from, segment by segment.
    bytes: Vec<u8>,
    segs: Vec<Seg>,
    /// Ops in execution order; a direct `jmp`/`call` is followed into its
    /// target, a `jcc` falls through (taken branches simply leave the block).
    ops: Vec<Pre>,
    /// Generic-path instructions for `K::Slow` ops (indexed by `Pre::t`).
    slow: Vec<Instruction>,
    /// One bit per `bloom_bit(op rip)`: a `run_until` target absent from it is
    /// not any op's address.
    bloom: u64,
}

/// A contiguous run of guest code inside a [`Block`].
struct Seg {
    start: u64,
    off: u32,
    len: u32,
}

const BLOCK_MAX: usize = 64;

#[inline(always)]
fn bloom_bit(rip: u64) -> u32 {
    ((rip ^ (rip >> 6)) & 63) as u32
}
const BCACHE_SIZE: usize = 1 << 13;

#[inline]
fn bslot(rip: u64) -> usize {
    ((rip ^ (rip >> 13)) as usize) & (BCACHE_SIZE - 1)
}

#[derive(Clone, Copy)]
struct DEntry {
    rip: u64,
    /// Instruction byte length; 0 marks an empty slot.
    len: u8,
    bytes: [u8; 15],
    ins: Instruction,
    /// Pre-resolved execution form of `ins`; valid exactly while `bytes` match.
    pre: Pre,
}

/// Decode-cache slots (power of two).
const DCACHE_SIZE: usize = 1 << 13;

/// `code[..len] == stored[..len]`, with `len <= code.len()`. A full 15-byte
/// fetch (the common case) compares via two 8-byte loads instead of a memcmp.
#[inline]
fn same_bytes(code: &[u8], stored: &[u8; 15], len: usize) -> bool {
    if code.len() < 15 {
        return code[..len] == stored[..len];
    }
    let ld = |b: &[u8], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
    let lo = |n: usize| if n >= 8 { u64::MAX } else { (1u64 << (n * 8)) - 1 };
    if len <= 8 {
        (ld(code, 0) ^ ld(stored, 0)) & lo(len) == 0
    } else {
        // Bytes 0..8 via the first load, 8..len via a load starting at byte 7.
        ld(code, 0) == ld(stored, 0) && (ld(code, 7) ^ ld(stored, 7)) & lo(len - 7) == 0
    }
}

#[inline]
fn dslot(rip: u64) -> usize {
    ((rip ^ (rip >> 15)) as usize) & (DCACHE_SIZE - 1)
}

impl Block {
    /// Whether `target` is the address of any op after the first.
    #[inline]
    fn has(&self, target: Option<u64>) -> bool {
        target.is_some_and(|t| {
            self.bloom >> bloom_bit(t) & 1 != 0 && self.ops[1..].iter().any(|p| p.rip == t)
        })
    }
}

/// Flag bits an op defines when it can neither fault, stop, call a hook nor
/// read the flags; `None` for anything else.
fn flag_def(k: K) -> Option<u64> {
    Some(match k {
        K::AddRR | K::AddRI | K::SubRR | K::SubRI | K::AndRR | K::AndRI | K::OrRR | K::OrRI
        | K::XorRR | K::XorRI | K::CmpRR | K::CmpRI | K::TestRR | K::TestRI | K::NegR => STATUS,
        K::IncR | K::DecR => STATUS & !CF,
        K::Clc | K::Stc => CF,
        K::Nop | K::MovRR | K::MovRI | K::ZxRR | K::SxRR | K::Lea | K::ShR | K::NotR | K::Cdq
        | K::Cqo | K::Cdqe | K::Cwde | K::Cld | K::Std | K::Bswap => 0,
        _ => return None,
    })
}

/// Flag bits a producer that honours `Pre::nf` may change.
fn flag_prod(k: K) -> Option<u64> {
    Some(match k {
        K::AddRR | K::AddRI | K::SubRR | K::SubRI | K::AndRR | K::AndRI | K::OrRR | K::OrRI
        | K::XorRR | K::XorRI | K::CmpRR | K::CmpRI | K::TestRR | K::TestRI | K::AdcSbbRR
        | K::AdcSbbRI | K::ShR => STATUS,
        K::Stc | K::Clc | K::Cmc => CF,
        _ => return None,
    })
}

/// Mark producers whose flags a later straight-line op fully overwrites before
/// anything (a fault, a hook, a flag reader, the block end) can observe them.
fn mark_dead_flags(ops: &mut [Pre]) {
    for i in 0..ops.len() {
        let Some(prod) = flag_prod(ops[i].kind) else { continue };
        let mut over = 0;
        for j in &ops[i + 1..] {
            let Some(def) = flag_def(j.kind) else { break };
            over |= def;
            if over & prod == prod {
                ops[i].nf = true;
                break;
            }
        }
    }
}

impl Emulator {
    #[must_use]
    pub fn new(cpu: Cpu, mem: Memory) -> Self {
        Self {
            audit: InsnAudit::default(),
            cpu,
            mem,
            trace: Vec::new(),
            record_trace: true,
            hook: None,
            observe: ObserveConfig::default(),
            resolve_undefined_flags: false,
            cur: (0, Mnemonic::INVALID),
            det: (None, None),
            steps: 0,
            dcache: vec![
                DEntry { rip: 0, len: 0, bytes: [0; 15], ins: Instruction::default(), pre: Pre::INVALID };
                DCACHE_SIZE
            ]
            .into_boxed_slice(),
            bcache: (0..BCACHE_SIZE).map(|_| None).collect(),
            gate: Gate::ALL,
        }
    }

    /// Instructions retired so far (equals `trace.len()` while recording).
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Execute one instruction.
    ///
    /// # Errors
    /// A [`Stop`]; the state is unchanged when it is returned.
    pub fn step(&mut self) -> Result<(), Stop> {
        let rip = self.cpu.rip;
        self.cur = (rip, Mnemonic::INVALID);
        // `det` is only ever set and read while `detail` is on.
        if self.observe.detail {
            self.det = (None, None);
        }
        let slot = if let Some(code) = self.mem.fetch15(rip) {
            Self::decode(&mut self.dcache, rip, code)
        } else {
            let mut buf = [0u8; 15];
            let n = self.mem.fetch(rip, &mut buf).map_err(|f| self.fault(f.addr, f.access))?;
            Self::decode(&mut self.dcache, rip, &buf[..n])
        };
        let Some(slot) = slot else {
            return Err(Stop::DecodeError { rip });
        };
        let p = self.dcache[slot].pre;
        self.cur.1 = p.mn;
        let next = rip.wrapping_add(u64::from(p.len));
        let new_rip = if p.kind == K::Slow {
            let ins = self.dcache[slot].ins;
            self.exec(&ins, next)?
        } else {
            self.exec_fast(&p, next)?
        };
        self.cpu.rip = new_rip;
        self.steps += 1;
        if self.record_trace {
            let (ea, value, vsp) = if self.observe.detail {
                (self.det.0, self.det.1, self.observe.vsp.and_then(|r| self.cpu.reg(r)))
            } else {
                (None, None, None)
            };
            self.trace.push(TraceEntry { rip, mnemonic: p.mn, ea, value, vsp });
        }
        Ok(())
    }

    /// Test oracle: the original decode-every-step interpreter (no decode cache,
    /// no pre-resolved ops, no blocks).
    #[cfg(test)]
    fn step_ref(&mut self) -> Result<(), Stop> {
        let rip = self.cpu.rip;
        self.cur = (rip, Mnemonic::INVALID);
        self.det = (None, None);
        let mut buf = [0u8; 15];
        let n = self.mem.fetch(rip, &mut buf).map_err(|f| self.fault(f.addr, f.access))?;
        let ins = Decoder::with_ip(64, &buf[..n], rip, DecoderOptions::NONE).decode();
        if ins.is_invalid() {
            return Err(Stop::DecodeError { rip });
        }
        self.cur.1 = ins.mnemonic();
        let next = rip.wrapping_add(ins.len() as u64);
        let new_rip = self.exec(&ins, next)?;
        self.cpu.rip = new_rip;
        self.steps += 1;
        if self.record_trace {
            let (ea, value, vsp) = if self.observe.detail {
                (self.det.0, self.det.1, self.observe.vsp.and_then(|r| self.cpu.reg(r)))
            } else {
                (None, None, None)
            };
            self.trace.push(TraceEntry { rip, mnemonic: ins.mnemonic(), ea, value, vsp });
        }
        Ok(())
    }

    /// Step until a stop; returns the reason ([`Stop::StepLimit`] after `max_steps`).
    pub fn run(&mut self, max_steps: u64) -> Stop {
        self.run_to(None, max_steps)
    }

    /// Like [`Emulator::run`] but stops with [`Stop::Reached`] when rip == `addr`
    /// (checked before executing there).
    pub fn run_until(&mut self, addr: u64, max_steps: u64) -> Stop {
        self.run_to(Some(addr), max_steps)
    }

    fn run_to(&mut self, target: Option<u64>, max_steps: u64) -> Stop {
        self.gate.wq = [EMPTY_PAGE; GATE_SLOTS];
        self.gate.fq = [EMPTY_PAGE; GATE_SLOTS];
        self.gate.active = true;
        self.refresh_gate();
        let stop = self.run_loop(target, max_steps);
        self.gate = Gate::ALL;
        stop
    }

    #[allow(clippy::collapsible_if)]
    fn run_loop(&mut self, target: Option<u64>, max_steps: u64) -> Stop {
        let mut left = max_steps;
        // Per-step trace/detail capture needs the single-step path.
        let blocks = !self.record_trace && !self.observe.detail;
        while left > 0 {
            let rip = self.cpu.rip;
            if target == Some(rip) {
                return Stop::Reached;
            }
            if blocks {
                if let Some((slot, b)) = self.block_for(rip) {
                    // Run whole only if it fits the budget (it may still exit
                    // early at a taken branch, a redirect, the target or a
                    // code write).
                    if b.ops.len() as u64 <= left && !b.has(target) {
                        let (done, stop) = self.exec_block(&b);
                        self.bcache[slot] = Some(b);
                        left -= done;
                        if let Some(s) = stop {
                            return s;
                        }
                        continue;
                    }
                    self.bcache[slot] = Some(b);
                }
            }
            if let Err(s) = self.step() {
                return s;
            }
            left -= 1;
        }
        Stop::StepLimit
    }

    /// Cached (revalidated) or freshly built block starting at `rip`, taken
    /// out of its slot (the caller puts it back).
    fn block_for(&mut self, rip: u64) -> Option<(usize, Box<Block>)> {
        // Two-way set: the block lives in `s0` or its neighbour.
        let s0 = bslot(rip);
        for slot in [s0, s0 ^ 1] {
            if self.bcache[slot].as_ref().is_some_and(|b| b.start == rip) {
                let mut b = self.bcache[slot].take()?;
                let g = self.mem.code_gen();
                if b.epoch == g {
                    return Some((slot, b));
                }
                let intact = b.segs.iter().all(|s| {
                    self.mem.exec_slice(s.start, s.len as usize)
                        == Some(&b.bytes[s.off as usize..(s.off + s.len) as usize])
                });
                if intact {
                    b.epoch = g;
                    return Some((slot, b));
                }
                return self.build_block(rip).map(|b| (slot, b));
            }
        }
        let slot = if self.bcache[s0].is_none() || self.bcache[s0 ^ 1].is_some() { s0 } else { s0 ^ 1 };
        self.build_block(rip).map(|b| (slot, b))
    }

    #[cold]
    #[inline(never)]
    fn build_block(&mut self, rip: u64) -> Option<Box<Block>> {
        let mut b = Block {
            start: rip,
            epoch: 0,
            bytes: Vec::new(),
            segs: Vec::new(),
            ops: Vec::new(),
            slow: Vec::new(),
            bloom: 0,
        };
        let mut seg_ip = rip;
        'segments: while let Some(page) = self.mem.code_bytes(seg_ip) {
            let (seg_off, mut off, mut ip) = (b.bytes.len(), 0usize, seg_ip);
            let mut follow = None;
            while b.ops.len() < BLOCK_MAX && off + 15 <= page.len() {
                let ins = Decoder::with_ip(64, &page[off..off + 15], ip, DecoderOptions::NONE).decode();
                if ins.is_invalid() {
                    break;
                }
                let len = ins.len();
                let mut pre = predecode::resolve(&ins);
                pre.rip = ip;
                pre.flow = matches!(pre.kind, K::Slow | K::Jmp | K::JmpD | K::Call | K::CallD | K::Ret | K::Jcc);
                // `stop`: nothing after this op is statically known.
                let mut stop = false;
                match pre.kind {
                    K::Slow => {
                        pre.t = b.slow.len() as u64;
                        b.slow.push(ins);
                        stop = ins.flow_control() != FlowControl::Next;
                    }
                    // Direct transfers: keep building at the target.
                    K::JmpD | K::CallD => follow = Some(pre.a.v),
                    K::Jmp | K::Call | K::Ret => stop = true,
                    _ => {}
                }
                b.bytes.extend_from_slice(&page[off..off + len]);
                b.ops.push(pre);
                off += len;
                ip = ip.wrapping_add(len as u64);
                if stop || follow.is_some() {
                    break;
                }
            }
            if b.bytes.len() > seg_off {
                b.segs.push(Seg { start: seg_ip, off: seg_off as u32, len: (b.bytes.len() - seg_off) as u32 });
            }
            match follow {
                Some(t) if b.ops.len() < BLOCK_MAX => seg_ip = t,
                _ => break 'segments,
            }
        }
        if b.ops.is_empty() {
            return None;
        }
        for i in 0..b.ops.len().saturating_sub(2) {
            let (p, j, t) = (&b.ops[i], &b.ops[i + 1], &b.ops[i + 2]);
            let fuse = !p.flow
                && j.kind == K::JmpD
                && j.rip == p.rip.wrapping_add(u64::from(p.len))
                && t.rip == j.a.v;
            b.ops[i].fj = fuse;
        }
        mark_dead_flags(&mut b.ops);
        for p in &mut b.ops {
            p.chk = !(flag_def(p.kind).is_some() || matches!(p.kind, K::AdcSbbRR | K::AdcSbbRI | K::Cmc));
        }
        b.epoch = self.mem.code_gen();
        b.bloom = b.ops.iter().fold(0, |m, p| m | 1 << bloom_bit(p.rip));
        Some(Box::new(b))
    }

    /// Execute `b` from its first op; returns `(ops retired, stop)`. State on a
    /// stop is exactly as after the last retired op (as `step` leaves it). The
    /// block is left as soon as control does not reach the next op in the block
    /// (taken branch, hook redirect), reaches `target`, or the guest code was
    /// written.
    fn exec_block(&mut self, b: &Block) -> (u64, Option<Stop>) {
        let base = self.steps;
        let epoch = b.epoch;
        let ops = &b.ops[..];
        let n = ops.len();
        let mut i = 0usize;
        let end = loop {
            let p = &ops[i];
            let rip = p.rip;
            self.cur.0 = rip;
            let next = rip.wrapping_add(u64::from(p.len));
            let r = if p.kind == K::Slow {
                self.cpu.rip = rip;
                self.cur.1 = p.mn;
                self.steps = base + i as u64;
                let ins = b.slow[p.t as usize];
                self.exec(&ins, next).map_err(Box::new)
            } else {
                self.exec_fast(p, next)
            };
            let nr = match r {
                Ok(nr) => nr,
                Err(s) => {
                    self.cpu.rip = rip;
                    self.steps = base + i as u64;
                    return (i as u64, Some(*s));
                }
            };
            i += 1;
            if p.flow {
                if i == n || nr != ops[i].rip || self.mem.code_gen() != epoch {
                    break nr;
                }
            } else {
                if i == n || (p.chk && self.mem.code_gen() != epoch) {
                    break next;
                }
                if p.fj {
                    // Direct `jmp` right after this op, run without a dispatch.
                    let j = &ops[i];
                    let to = j.a.v;
                    let jnext = next.wrapping_add(u64::from(j.len));
                    if self.gate.wants_flow(next, to) {
                        self.cur.0 = next;
                        match self.flow_hook_slow(FlowKind::Jmp, to, jnext) {
                            Err(s) => {
                                self.cpu.rip = next;
                                self.steps = base + i as u64;
                                return (i as u64, Some(*s));
                            }
                            Ok(out) => {
                                i += 1;
                                let nr = if let HookOutcome::Redirect = out { self.cpu.rip } else { to };
                                if i == n || nr != ops[i].rip || self.mem.code_gen() != epoch {
                                    break nr;
                                }
                            }
                        }
                    } else {
                        i += 1;
                        if i == n {
                            break to;
                        }
                    }
                }
            }
        };
        self.cpu.rip = end;
        self.steps = base + i as u64;
        (i as u64, None)
    }

    /// Decode the instruction at `rip` from `code` (the executable prefix, up
    /// to 15 bytes), reusing a cached decode when the guest bytes are
    /// unchanged. On a byte mismatch (self-modifying/unpacking code) or a miss,
    /// decode afresh and store the result. Invalid decodes are not cached; the
    /// caller surfaces them as [`Stop::DecodeError`].
    #[inline]
    fn decode(dcache: &mut [DEntry], rip: u64, code: &[u8]) -> Option<usize> {
        let n = code.len();
        let slot = dslot(rip);
        let e = &mut dcache[slot];
        let len = usize::from(e.len);
        if e.rip == rip && len != 0 && len <= n && same_bytes(code, &e.bytes, len) {
            return Some(slot);
        }
        Self::fill(e, rip, code)
    }

    /// Cache miss / stale entry: decode afresh and store the decode plus its
    /// pre-resolved form. `None` for an invalid decode (never cached).
    #[cold]
    #[inline(never)]
    fn fill(e: &mut DEntry, rip: u64, code: &[u8]) -> Option<usize> {
        let ins = Decoder::with_ip(64, code, rip, DecoderOptions::NONE).decode();
        let len = ins.len();
        if ins.is_invalid() || len > code.len() {
            return None;
        }
        e.rip = rip;
        e.len = len as u8;
        e.bytes[..len].copy_from_slice(&code[..len]);
        e.ins = ins;
        e.pre = predecode::resolve(&ins);
        Some(dslot(rip))
    }

    fn unsup(&self) -> Stop {
        Stop::Unsupported { rip: self.cur.0, mnemonic: self.cur.1 }
    }

    #[cold]
    fn fault(&self, addr: u64, access: Access) -> Trap {
        Box::new(Stop::Fault { rip: self.cur.0, addr, access })
    }

    /// Re-read the hook's interest (it may have changed in a callback).
    #[cold]
    #[inline(never)]
    fn refresh_gate(&mut self) {
        let ix = self.hook.as_ref().map_or(Interest::NONE, |h| h.interest());
        self.gate.set(&ix);
    }

    /// Deliver a memory event the gate let through.
    #[cold]
    #[inline(never)]
    fn mem_hook(&mut self, ev: MemEvent) -> Result<(), Trap> {
        // Disjoint field borrows: the hook mutates cpu/mem while it fires.
        self.cpu.rip = self.cur.0;
        let Self { hook, cpu, mem, .. } = self;
        let act = hook.as_mut().map(|h| h.on_mem(&ev, cpu, mem));
        if self.gate.active {
            self.gate.fq = [EMPTY_PAGE; GATE_SLOTS];
            if ev.access == Access::Write && act != Some(HookAction::Stop) {
                self.gate.note_write(ev.addr, ev.size);
            }
            self.refresh_gate();
        }
        match act {
            Some(HookAction::Stop) => Err(Box::new(Stop::Hook)),
            _ => Ok(()),
        }
    }

    /// Deliver a flow event the gate let through.
    #[cold]
    #[inline(never)]
    fn flow_hook_slow(&mut self, kind: FlowKind, to: u64, next: u64) -> Result<HookOutcome, Trap> {
        let ev = FlowEvent { kind, from: self.cur.0, to, next };
        self.cpu.rip = self.cur.0;
        let Self { hook, cpu, mem, .. } = self;
        let act = hook.as_mut().map(|h| h.on_flow(&ev, cpu, mem));
        if self.gate.active {
            let routed = !self.gate.flow_all && !Gate::hit(&self.gate.flow, to);
            if routed && matches!(act, None | Some(HookAction::Continue)) {
                let p = to >> PAGE_SHIFT;
                self.gate.fq[p as usize % GATE_SLOTS] = p;
            } else {
                self.gate.fq = [EMPTY_PAGE; GATE_SLOTS];
                self.refresh_gate();
            }
        }
        match act {
            Some(HookAction::Stop) => Err(Box::new(Stop::Hook)),
            Some(HookAction::Redirect) => Ok(HookOutcome::Redirect),
            _ => Ok(HookOutcome::Continue),
        }
    }

    #[inline(always)]
    fn flow_hook(&mut self, kind: FlowKind, to: u64, next: u64) -> Result<HookOutcome, Trap> {
        if self.gate.wants_flow(self.cur.0, to) {
            return self.flow_hook_slow(kind, to, next);
        }
        Ok(HookOutcome::Continue)
    }

    #[inline(always)]
    fn load(&mut self, addr: u64, size: u32) -> Result<u64, Trap> {
        if self.gate.wants_read(addr) {
            self.mem_hook(MemEvent { addr, size, access: Access::Read, value: None })?;
        }
        let v = self.mem.read_uint(addr, size as usize).map_err(|f| self.fault(f.addr, f.access))?;
        if self.observe.detail {
            self.det = (Some(addr), Some(v));
        }
        Ok(v)
    }

    #[inline(always)]
    fn store(&mut self, addr: u64, size: u32, val: u64) -> Result<(), Trap> {
        let value = val & mask(size);
        if self.gate.wants_write(addr, size) {
            self.mem_hook(MemEvent { addr, size, access: Access::Write, value: Some(value) })?;
        }
        self.mem.write_uint(addr, size as usize, val).map_err(|f| self.fault(f.addr, f.access))?;
        if self.observe.detail {
            self.det = (Some(addr), Some(value));
        }
        Ok(())
    }

    #[inline(always)]
    fn apply(&mut self, f: Flags) {
        if self.resolve_undefined_flags {
            // Detonation concretization: an architecturally-undefined bit takes
            // the deterministic hardware value its producer carries in `f.val`
            // and becomes defined, so a later flag consumer reads a concrete
            // bit. `f.val` already holds the modeled value for every undef bit
            // (shift/rotate/shld OF via the 1-bit formula; mul/imul SF/ZF/PF
            // from the low result, AF=0; bt*/bsf/bsr/div status bits cleared).
            let all = f.def | f.undef;
            self.cpu.rflags = (self.cpu.rflags & !all) | (f.val & all);
            self.cpu.undef_flags &= !all;
        } else {
            self.cpu.rflags = (self.cpu.rflags & !(f.def | f.undef)) | (f.val & f.def);
            self.cpu.undef_flags = (self.cpu.undef_flags & !f.def) | f.undef;
        }
    }

    fn ea(&self, ins: &Instruction) -> Result<u64, Stop> {
        // Only gs: is modeled (base added below); fs: and other segments stop.
        let seg = ins.memory_segment();
        if seg == Register::FS {
            return Err(self.unsup());
        }
        let base = ins.memory_base();
        if base == Register::RIP {
            return Ok(ins.memory_displacement64());
        }
        let mut a = ins.memory_displacement64();
        // 0x67 address-size override: the effective address wraps to 32 bits.
        let mut a32 = false;
        if base != Register::None {
            if !(base.is_gpr64() || base.is_gpr32()) {
                return Err(self.unsup());
            }
            a32 |= base.is_gpr32();
            a = a.wrapping_add(self.cpu.reg(base).ok_or_else(|| self.unsup())?);
        }
        let idx = ins.memory_index();
        if idx != Register::None {
            if !(idx.is_gpr64() || idx.is_gpr32()) {
                return Err(self.unsup());
            }
            a32 |= idx.is_gpr32();
            let v = self.cpu.reg(idx).ok_or_else(|| self.unsup())?;
            a = a.wrapping_add(v.wrapping_mul(u64::from(ins.memory_index_scale())));
        }
        if a32 {
            a &= 0xffff_ffff;
        }
        if seg == Register::GS {
            a = a.wrapping_add(self.cpu.gs_base);
        }
        Ok(a)
    }

    fn op_bytes(&self, ins: &Instruction, i: u32) -> Result<u32, Stop> {
        let n = match ins.op_kind(i) {
            OpKind::Register => ins.op_register(i).size(),
            OpKind::Memory => ins.memory_size().size(),
            _ => 0,
        };
        match n {
            1 | 2 | 4 | 8 => Ok(n as u32),
            _ => Err(self.unsup()),
        }
    }

    fn read_op(&mut self, ins: &Instruction, i: u32, bytes: u32) -> Result<u64, Stop> {
        match ins.op_kind(i) {
            OpKind::Register => self.cpu.reg(ins.op_register(i)).ok_or_else(|| self.unsup()),
            OpKind::Memory => {
                let a = self.ea(ins)?;
                Ok(self.load(a, bytes)?)
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => Ok(ins.immediate(i) & mask(bytes)),
            _ => Err(self.unsup()),
        }
    }

    fn write_op(&mut self, ins: &Instruction, i: u32, bytes: u32, val: u64) -> Result<(), Stop> {
        match ins.op_kind(i) {
            OpKind::Register => {
                if self.cpu.set_reg(ins.op_register(i), val) {
                    Ok(())
                } else {
                    Err(self.unsup())
                }
            }
            OpKind::Memory => {
                let a = self.ea(ins)?;
                Ok(self.store(a, bytes, val)?)
            }
            _ => Err(self.unsup()),
        }
    }

    fn cond(&self, cc: ConditionCode) -> Result<bool, Stop> {
        use ConditionCode as C;
        let need = match cc {
            C::o | C::no => OF,
            C::b | C::ae => CF,
            C::e | C::ne => ZF,
            C::be | C::a => CF | ZF,
            C::s | C::ns => SF,
            C::p | C::np => PF,
            C::l | C::ge => SF | OF,
            C::le | C::g => ZF | SF | OF,
            C::None => return Err(self.unsup()),
        };
        if self.cpu.undef_flags & need != 0 {
            return Err(Stop::UndefinedFlags { rip: self.cur.0, mask: self.cpu.undef_flags & need });
        }
        let f = |m| self.cpu.flag(m);
        Ok(match cc {
            C::o => f(OF),
            C::no => !f(OF),
            C::b => f(CF),
            C::ae => !f(CF),
            C::e => f(ZF),
            C::ne => !f(ZF),
            C::be => f(CF) || f(ZF),
            C::a => !f(CF) && !f(ZF),
            C::s => f(SF),
            C::ns => !f(SF),
            C::p => f(PF),
            C::np => !f(PF),
            C::l => f(SF) != f(OF),
            C::ge => f(SF) == f(OF),
            C::le => f(ZF) || f(SF) != f(OF),
            C::g => !f(ZF) && f(SF) == f(OF),
            C::None => return Err(self.unsup()),
        })
    }

    fn branch_target(&mut self, ins: &Instruction) -> Result<u64, Stop> {
        match ins.op0_kind() {
            OpKind::NearBranch64 => Ok(ins.near_branch_target()),
            OpKind::Register if ins.op0_register().is_gpr64() => {
                self.cpu.reg(ins.op0_register()).ok_or_else(|| self.unsup())
            }
            OpKind::Memory if ins.memory_size().size() == 8 => {
                let a = self.ea(ins)?;
                Ok(self.load(a, 8)?)
            }
            _ => Err(self.unsup()),
        }
    }

    #[inline(always)]
    fn rreg(&self, o: &Opnd) -> u64 {
        (self.cpu.regs[usize::from(o.r & 15)] >> o.sh) & o.v
    }

    /// Register write with x64 partial-write rules (matches `Cpu::set_reg`).
    #[inline(always)]
    fn wreg(&mut self, o: &Opnd, v: u64) {
        let vm = o.v << o.sh;
        let keep = if o.bytes >= 4 { 0 } else { !vm };
        let r = &mut self.cpu.regs[usize::from(o.r & 15)];
        *r = (*r & keep) | ((v << o.sh) & vm);
    }

    /// Effective address of a pre-resolved memory operand (branchless; matches `ea`).
    #[inline(always)]
    fn ea_of(&self, o: &Opnd) -> u64 {
        let neg = |b: u8| 0u64.wrapping_sub(u64::from(b));
        let regs = &self.cpu.regs;
        o.v.wrapping_add(regs[usize::from(o.r & 15)] & neg(o.fl & 1))
            .wrapping_add(regs[usize::from(o.x & 15)].wrapping_mul(u64::from(o.sc)))
            .wrapping_add(self.cpu.gs_base & neg((o.fl >> 1) & 1))
    }

    #[inline(always)]
    fn rd(&mut self, o: &Opnd) -> Result<u64, Trap> {
        match o.tag {
            T_REG => Ok(self.rreg(o)),
            T_MEM => {
                let a = self.ea_of(o);
                self.load(a, u32::from(o.bytes))
            }
            _ => Ok(o.v),
        }
    }

    #[inline(always)]
    fn wr(&mut self, o: &Opnd, v: u64) -> Result<(), Trap> {
        if o.tag == T_REG {
            self.wreg(o, v);
            Ok(())
        } else {
            let a = self.ea_of(o);
            self.store(a, u32::from(o.bytes), v)
        }
    }

    /// Low-`N`-byte register read / write with x64 partial-write rules. Only for
    /// operands without a high-byte shift (those resolve to the generic path).
    #[inline(always)]
    fn rn<const N: u32>(&self, o: &Opnd) -> u64 {
        self.cpu.regs[usize::from(o.r & 15)] & mask(N)
    }

    #[inline(always)]
    fn wn<const N: u32>(&mut self, o: &Opnd, v: u64) {
        let r = &mut self.cpu.regs[usize::from(o.r & 15)];
        *r = if N >= 4 { v & mask(N) } else { (*r & !mask(N)) | (v & mask(N)) };
    }

    /// `wn` with the width picked at run time (`n` is 1, 2, 4 or 8).
    #[inline(always)]
    fn wdst(&mut self, o: &Opnd, n: u8, v: u64) {
        match n {
            8 => self.wn::<8>(o, v),
            4 => self.wn::<4>(o, v),
            2 => self.wn::<2>(o, v),
            _ => self.wn::<1>(o, v),
        }
    }

    /// Register `shl`/`shr`/`sar` (`KIND` 0/1/2) at width `N`.
    #[inline(always)]
    fn shift_r<const KIND: u8, const N: u32>(&mut self, p: &Pre) -> Result<(), Trap> {
        let a = self.rn::<N>(&p.a);
        let c = self.rd(&p.b)?;
        let kind = match KIND {
            0 => Shift::Shl,
            1 => Shift::Shr,
            _ => Shift::Sar,
        };
        if let Some((r, fl)) = alu::shift(kind, a, c, N) {
            self.wn::<N>(&p.a, r);
            if !p.nf {
                self.apply(fl);
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn alu_rr<const OP: u8, const W: bool>(&mut self, p: &Pre) {
        match p.n {
            8 => self.alu_rr_n::<OP, W, 8>(p),
            4 => self.alu_rr_n::<OP, W, 4>(p),
            2 => self.alu_rr_n::<OP, W, 2>(p),
            _ => self.alu_rr_n::<OP, W, 1>(p),
        }
    }

    #[inline(always)]
    fn alu_rr_n<const OP: u8, const W: bool, const N: u32>(&mut self, p: &Pre) {
        let a = self.rn::<N>(&p.a);
        let b = self.rn::<N>(&p.b);
        let (r, fl) = alu::arith(alu_op(OP), a, b, N);
        if W {
            self.wn::<N>(&p.a, r);
        }
        if !p.nf {
            self.apply(fl);
        }
    }

    #[inline(always)]
    fn alu_ri<const OP: u8, const W: bool>(&mut self, p: &Pre) {
        match p.n {
            8 => self.alu_ri_n::<OP, W, 8>(p),
            4 => self.alu_ri_n::<OP, W, 4>(p),
            2 => self.alu_ri_n::<OP, W, 2>(p),
            _ => self.alu_ri_n::<OP, W, 1>(p),
        }
    }

    #[inline(always)]
    fn alu_ri_n<const OP: u8, const W: bool, const N: u32>(&mut self, p: &Pre) {
        let a = self.rn::<N>(&p.a);
        let (r, fl) = alu::arith(alu_op(OP), a, p.b.v, N);
        if W {
            self.wn::<N>(&p.a, r);
        }
        if !p.nf {
            self.apply(fl);
        }
    }

    /// `adc`/`sbb` with a register destination and register/immediate source.
    #[inline(always)]
    fn adc_sbb_r<const RR: bool, const N: u32>(&mut self, p: &Pre) {
        let a = self.rn::<N>(&p.a);
        let b = if RR { self.rn::<N>(&p.b) } else { p.b.v };
        let carry = self.cpu.flag(CF);
        let (r, fl) = if p.sub == 0 { alu::adc(a, b, carry, N) } else { alu::sbb(a, b, carry, N) };
        self.wn::<N>(&p.a, r);
        if !p.nf {
            self.apply(fl);
        }
    }

    #[inline(always)]
    fn alu_rm<const OP: u8, const W: bool>(&mut self, p: &Pre) -> Result<(), Trap> {
        let a = self.rreg(&p.a);
        let ea = self.ea_of(&p.b);
        let b = self.load(ea, u32::from(p.n))?;
        let (r, fl) = alu::arith(alu_op(OP), a, b, u32::from(p.n));
        if W {
            self.wreg(&p.a, r);
        }
        self.apply(fl);
        Ok(())
    }

    /// Memory-destination ALU op; `src` is a register or an immediate.
    #[inline(always)]
    fn alu_m<const OP: u8, const W: bool>(&mut self, p: &Pre) -> Result<(), Trap> {
        let ea = self.ea_of(&p.a);
        let a = self.load(ea, u32::from(p.n))?;
        let b = if p.b.tag == T_REG { self.rreg(&p.b) } else { p.b.v };
        let (r, fl) = alu::arith(alu_op(OP), a, b, u32::from(p.n));
        if W {
            let ea = self.ea_of(&p.a);
            self.store(ea, u32::from(p.n), r)?;
        }
        self.apply(fl);
        Ok(())
    }

    /// Unary r/m op (`inc`/`dec`/`neg`/`not` = 0/1/2/3); flags unchanged for `not`.
    #[inline(always)]
    fn unary(&mut self, p: &Pre, kind: u8) -> Result<(), Trap> {
        let n = u32::from(p.n);
        let a = self.rd(&p.a)?;
        let (r, fl) = match kind {
            0 => alu::incdec(a, false, n),
            1 => alu::incdec(a, true, n),
            2 => alu::neg(a, n),
            _ => (!a & mask(n), Flags { def: 0, val: 0, undef: 0 }),
        };
        self.wr(&p.a, r)?;
        if kind != 3 {
            self.apply(fl);
        }
        Ok(())
    }

    #[inline(always)]
    fn exec_fast(&mut self, p: &Pre, next: u64) -> Result<u64, Trap> {
        let n = u32::from(p.n);
        match p.kind {
            K::Nop => {}
            K::Slow => return Err(self.unsup().into()),
            K::MovRR => match p.n {
                8 => self.cpu.regs[usize::from(p.a.r & 15)] = self.cpu.regs[usize::from(p.b.r & 15)],
                4 => {
                    let v = self.rn::<4>(&p.b);
                    self.wn::<4>(&p.a, v);
                }
                2 => {
                    let v = self.rn::<2>(&p.b);
                    self.wn::<2>(&p.a, v);
                }
                _ => {
                    let v = self.rn::<1>(&p.b);
                    self.wn::<1>(&p.a, v);
                }
            },
            K::ZxRR => {
                let v = self.rreg(&p.b);
                self.wdst(&p.a, p.n, v);
            }
            K::MovRI => match p.n {
                8 => self.cpu.regs[usize::from(p.a.r & 15)] = p.b.v,
                4 => self.wn::<4>(&p.a, p.b.v),
                2 => self.wn::<2>(&p.a, p.b.v),
                _ => self.wn::<1>(&p.a, p.b.v),
            },
            K::MovRM => {
                let ea = self.ea_of(&p.b);
                let v = self.load(ea, n)?;
                self.wdst(&p.a, p.n, v);
            }
            K::MovMR => {
                let v = self.rreg(&p.b);
                let ea = self.ea_of(&p.a);
                self.store(ea, n, v)?;
            }
            K::MovMI => {
                let ea = self.ea_of(&p.a);
                self.store(ea, n, p.b.v)?;
            }
            K::ZxRM => {
                let ea = self.ea_of(&p.b);
                let v = self.load(ea, u32::from(p.b.bytes))?;
                self.wdst(&p.a, p.n, v);
            }
            K::SxRR => {
                let v = self.rreg(&p.b);
                self.wdst(&p.a, p.n, (((v << p.sub) as i64) >> p.sub) as u64);
            }
            K::SxRM => {
                let ea = self.ea_of(&p.b);
                let v = self.load(ea, u32::from(p.b.bytes))?;
                self.wdst(&p.a, p.n, (((v << p.sub) as i64) >> p.sub) as u64);
            }
            K::Lea => {
                let a = self.ea_of(&p.b);
                self.wdst(&p.a, p.n, a);
            }
            K::AddRR => self.alu_rr::<0, true>(p),
            K::AddRI => self.alu_ri::<0, true>(p),
            K::AddRM => self.alu_rm::<0, true>(p)?,
            K::AddMR | K::AddMI => self.alu_m::<0, true>(p)?,
            K::SubRR => self.alu_rr::<1, true>(p),
            K::SubRI => self.alu_ri::<1, true>(p),
            K::SubRM => self.alu_rm::<1, true>(p)?,
            K::SubMR | K::SubMI => self.alu_m::<1, true>(p)?,
            K::AndRR => self.alu_rr::<2, true>(p),
            K::AndRI => self.alu_ri::<2, true>(p),
            K::AndRM => self.alu_rm::<2, true>(p)?,
            K::AndMR | K::AndMI => self.alu_m::<2, true>(p)?,
            K::OrRR => self.alu_rr::<3, true>(p),
            K::OrRI => self.alu_ri::<3, true>(p),
            K::OrRM => self.alu_rm::<3, true>(p)?,
            K::OrMR | K::OrMI => self.alu_m::<3, true>(p)?,
            K::XorRR => self.alu_rr::<4, true>(p),
            K::XorRI => self.alu_ri::<4, true>(p),
            K::XorRM => self.alu_rm::<4, true>(p)?,
            K::XorMR | K::XorMI => self.alu_m::<4, true>(p)?,
            K::CmpRR => self.alu_rr::<1, false>(p),
            K::CmpRI => self.alu_ri::<1, false>(p),
            K::CmpRM => self.alu_rm::<1, false>(p)?,
            K::CmpMR | K::CmpMI => self.alu_m::<1, false>(p)?,
            K::TestRR => self.alu_rr::<2, false>(p),
            K::TestRI => self.alu_ri::<2, false>(p),
            K::TestRM => self.alu_rm::<2, false>(p)?,
            K::TestMR | K::TestMI => self.alu_m::<2, false>(p)?,
            K::IncR | K::IncM => self.unary(p, 0)?,
            K::DecR | K::DecM => self.unary(p, 1)?,
            K::NegR | K::NegM => self.unary(p, 2)?,
            K::NotR | K::NotM => self.unary(p, 3)?,
            K::ShR => match (p.sub, p.n) {
                (0, 8) => self.shift_r::<0, 8>(p)?,
                (0, 4) => self.shift_r::<0, 4>(p)?,
                (0, 2) => self.shift_r::<0, 2>(p)?,
                (0, _) => self.shift_r::<0, 1>(p)?,
                (1, 8) => self.shift_r::<1, 8>(p)?,
                (1, 4) => self.shift_r::<1, 4>(p)?,
                (1, 2) => self.shift_r::<1, 2>(p)?,
                (1, _) => self.shift_r::<1, 1>(p)?,
                (_, 8) => self.shift_r::<2, 8>(p)?,
                (_, 4) => self.shift_r::<2, 4>(p)?,
                (_, 2) => self.shift_r::<2, 2>(p)?,
                (_, _) => self.shift_r::<2, 1>(p)?,
            },
            K::ShM => {
                let a = self.rd(&p.a)?;
                let c = self.rd(&p.b)?;
                let kind = match p.sub {
                    0 => Shift::Shl,
                    1 => Shift::Shr,
                    _ => Shift::Sar,
                };
                if let Some((r, fl)) = alu::shift(kind, a, c, n) {
                    self.wr(&p.a, r)?;
                    self.apply(fl);
                }
            }
            K::RotR | K::RotM => {
                let a = self.rd(&p.a)?;
                let c = self.rd(&p.b)?;
                let kind = match p.sub {
                    0 => Rotate::Rol,
                    1 => Rotate::Ror,
                    2 => Rotate::Rcl,
                    _ => Rotate::Rcr,
                };
                if p.sub >= 2 && self.cpu.undef_flags & CF != 0 {
                    return Err(Box::new(Stop::UndefinedFlags { rip: self.cur.0, mask: CF }));
                }
                let cf_in = self.cpu.flag(CF);
                if let Some((r, fl)) = alu::rotate(kind, a, c, cf_in, n) {
                    self.wr(&p.a, r)?;
                    self.apply(fl);
                }
            }
            K::PushR => {
                let v = self.rreg(&p.a);
                self.push64(v)?;
            }
            K::PushI => self.push64(p.a.v)?,
            K::PopR => {
                let sp = self.cpu.regs[RSP];
                let v = self.load(sp, 8)?;
                self.cpu.regs[RSP] = sp.wrapping_add(8);
                self.wreg(&p.a, v);
            }
            K::Jmp => {
                let t = self.rd(&p.a)?;
                return Ok(match self.flow_hook(FlowKind::Jmp, t, next)? {
                    HookOutcome::Redirect => self.cpu.rip,
                    HookOutcome::Continue => t,
                });
            }
            K::JmpD => {
                let t = p.a.v;
                return Ok(match self.flow_hook(FlowKind::Jmp, t, next)? {
                    HookOutcome::Redirect => self.cpu.rip,
                    HookOutcome::Continue => t,
                });
            }
            K::CallD => {
                let t = p.a.v;
                if let HookOutcome::Redirect = self.flow_hook(FlowKind::Call, t, next)? {
                    return Ok(self.cpu.rip);
                }
                self.push64(next)?;
                return Ok(t);
            }
            K::Call => {
                let t = self.rd(&p.a)?;
                if let HookOutcome::Redirect = self.flow_hook(FlowKind::Call, t, next)? {
                    return Ok(self.cpu.rip);
                }
                self.push64(next)?;
                return Ok(t);
            }
            K::Ret => {
                let sp = self.cpu.regs[RSP];
                let t = self.load(sp, 8)?;
                if let HookOutcome::Redirect = self.flow_hook(FlowKind::Ret, t, next)? {
                    return Ok(self.cpu.rip);
                }
                self.cpu.regs[RSP] = sp.wrapping_add(8).wrapping_add(p.t);
                return Ok(t);
            }
            K::Jcc => {
                let und = self.cpu.undef_flags & u64::from(p.m);
                if und != 0 {
                    return Err(Box::new(Stop::UndefinedFlags { rip: self.cur.0, mask: und }));
                }
                let taken = (cc_val(self.cpu.rflags, p.sub) ^ u64::from(p.n)) != 0;
                let t = if taken { p.t } else { next };
                return Ok(match self.flow_hook(FlowKind::Jcc { taken }, t, next)? {
                    HookOutcome::Redirect => self.cpu.rip,
                    HookOutcome::Continue => t,
                });
            }
            K::Setcc => {
                let und = self.cpu.undef_flags & u64::from(p.m);
                if und != 0 {
                    return Err(Box::new(Stop::UndefinedFlags { rip: self.cur.0, mask: und }));
                }
                let taken = cc_val(self.cpu.rflags, p.sub) ^ u64::from(p.n);
                self.wr(&p.a, taken)?;
            }
            K::Cmov => {
                let und = self.cpu.undef_flags & u64::from(p.m);
                if und != 0 {
                    return Err(Box::new(Stop::UndefinedFlags { rip: self.cur.0, mask: und }));
                }
                let taken = (cc_val(self.cpu.rflags, p.sub & 15) ^ u64::from(p.sub >> 4)) != 0;
                let v = self.rd(&p.b)?;
                if taken {
                    self.wreg(&p.a, v);
                }
            }
            K::Clc => {
                if !p.nf {
                    self.apply(Flags { def: CF, val: 0, undef: 0 });
                }
            }
            K::Stc => {
                if !p.nf {
                    self.apply(Flags { def: CF, val: CF, undef: 0 });
                }
            }
            K::Cmc => {
                if !p.nf && self.cpu.undef_flags & CF == 0 {
                    let v = if self.cpu.flag(CF) { 0 } else { CF };
                    self.apply(Flags { def: CF, val: v, undef: 0 });
                }
            }
            K::Cld => self.cpu.rflags &= !DF,
            K::Std => self.cpu.rflags |= DF,
            K::AdcSbbRR | K::AdcSbbRI => {
                if self.cpu.undef_flags & CF != 0 {
                    return Err(Box::new(Stop::UndefinedFlags { rip: self.cur.0, mask: CF }));
                }
                match (p.kind == K::AdcSbbRR, p.n) {
                    (true, 8) => self.adc_sbb_r::<true, 8>(p),
                    (true, 4) => self.adc_sbb_r::<true, 4>(p),
                    (true, 2) => self.adc_sbb_r::<true, 2>(p),
                    (true, _) => self.adc_sbb_r::<true, 1>(p),
                    (false, 8) => self.adc_sbb_r::<false, 8>(p),
                    (false, 4) => self.adc_sbb_r::<false, 4>(p),
                    (false, 2) => self.adc_sbb_r::<false, 2>(p),
                    (false, _) => self.adc_sbb_r::<false, 1>(p),
                }
            }
            K::AdcSbb => {
                if self.cpu.undef_flags & CF != 0 {
                    return Err(Box::new(Stop::UndefinedFlags { rip: self.cur.0, mask: CF }));
                }
                let a = self.rd(&p.a)?;
                let b = self.rd(&p.b)?;
                let carry = self.cpu.flag(CF);
                let (r, fl) =
                    if p.sub == 0 { alu::adc(a, b, carry, n) } else { alu::sbb(a, b, carry, n) };
                self.wr(&p.a, r)?;
                self.apply(fl);
            }
            K::Xadd => {
                let dest = self.rd(&p.a)?;
                let src = self.rd(&p.b)?;
                let (sum, fl) = alu::arith(Alu::Add, dest, src, n);
                self.wr(&p.a, sum)?;
                self.wr(&p.b, dest)?;
                self.apply(fl);
            }
            K::Bswap => {
                let v = self.rreg(&p.a);
                let r = if n == 4 { u64::from((v as u32).swap_bytes()) } else { v.swap_bytes() };
                self.wreg(&p.a, r);
            }
            K::Cwde => self.cpu.regs[RAX] = alu_sext(self.cpu.regs[RAX], 2) & 0xffff_ffff,
            K::Cdqe => self.cpu.regs[RAX] = alu_sext(self.cpu.regs[RAX], 4),
            K::Cdq => {
                self.cpu.regs[RDX] = if self.cpu.regs[RAX] & 0x8000_0000 != 0 { 0xffff_ffff } else { 0 };
            }
            K::Cqo => self.cpu.regs[RDX] = if self.cpu.regs[RAX] >> 63 != 0 { u64::MAX } else { 0 },
            K::Pushfq => self.push64(self.cpu.rflags)?,
        }
        Ok(next)
    }

    fn exec(&mut self, ins: &Instruction, next: u64) -> Result<u64, Stop> {
        // `lock` is transparent in a single-threaded model. `rep`/`repne` are
        // only meaningful on the string ops that consume them below; anywhere
        // else they stop rather than being silently ignored — except on `ret`,
        // where `F3` is the architectural no-op the `rep ret` / branch-alignment
        // idiom emits (semantically identical to a plain near return).
        if (ins.has_rep_prefix() || ins.has_repne_prefix())
            && string_kind(ins.code()).is_none()
            && ins.mnemonic() != Mnemonic::Ret
        {
            return Err(self.unsup());
        }
        let mn = ins.mnemonic();
        match mn {
            Mnemonic::Nop => {}
            Mnemonic::Mov => {
                let n = self.op_bytes(ins, 0)?;
                let v = self.read_op(ins, 1, n)?;
                self.write_op(ins, 0, n, v)?;
            }
            Mnemonic::Movzx | Mnemonic::Movsx | Mnemonic::Movsxd => {
                let (dn, sn) = (self.op_bytes(ins, 0)?, self.op_bytes(ins, 1)?);
                if sn >= dn {
                    return Err(self.unsup());
                }
                let v = self.read_op(ins, 1, sn)?;
                let v = if mn == Mnemonic::Movzx {
                    v
                } else {
                    let sh = 64 - sn * 8;
                    (((v << sh) as i64) >> sh) as u64
                };
                self.write_op(ins, 0, dn, v)?;
            }
            Mnemonic::Lea => {
                let n = self.op_bytes(ins, 0)?;
                if ins.op_kind(1) != OpKind::Memory || ins.op_kind(0) != OpKind::Register {
                    return Err(self.unsup());
                }
                let a = self.ea(ins)?;
                self.write_op(ins, 0, n, a)?;
            }
            Mnemonic::Add
            | Mnemonic::Sub
            | Mnemonic::And
            | Mnemonic::Or
            | Mnemonic::Xor
            | Mnemonic::Cmp
            | Mnemonic::Test => {
                let n = self.op_bytes(ins, 0)?;
                let a = self.read_op(ins, 0, n)?;
                let b = self.read_op(ins, 1, n)?;
                let op = match mn {
                    Mnemonic::Add => Alu::Add,
                    Mnemonic::Sub | Mnemonic::Cmp => Alu::Sub,
                    Mnemonic::And | Mnemonic::Test => Alu::And,
                    Mnemonic::Or => Alu::Or,
                    _ => Alu::Xor,
                };
                let (r, fl) = alu::arith(op, a, b, n);
                if !matches!(mn, Mnemonic::Cmp | Mnemonic::Test) {
                    self.write_op(ins, 0, n, r)?;
                }
                self.apply(fl);
            }
            Mnemonic::Inc | Mnemonic::Dec | Mnemonic::Neg | Mnemonic::Not => {
                let n = self.op_bytes(ins, 0)?;
                let a = self.read_op(ins, 0, n)?;
                let (r, fl) = match mn {
                    Mnemonic::Inc => alu::incdec(a, false, n),
                    Mnemonic::Dec => alu::incdec(a, true, n),
                    Mnemonic::Neg => alu::neg(a, n),
                    _ => (!a & mask(n), Flags { def: 0, val: 0, undef: 0 }),
                };
                self.write_op(ins, 0, n, r)?;
                self.apply(fl);
            }
            Mnemonic::Shl | Mnemonic::Sal | Mnemonic::Shr | Mnemonic::Sar => {
                let n = self.op_bytes(ins, 0)?;
                let a = self.read_op(ins, 0, n)?;
                let c = match ins.op_count() {
                    1 => 1,
                    _ => self.read_op(ins, 1, 1)?,
                };
                let kind = match mn {
                    Mnemonic::Shl | Mnemonic::Sal => Shift::Shl,
                    Mnemonic::Shr => Shift::Shr,
                    _ => Shift::Sar,
                };
                if let Some((r, fl)) = alu::shift(kind, a, c, n) {
                    self.write_op(ins, 0, n, r)?;
                    self.apply(fl);
                }
            }
            Mnemonic::Push => {
                let ok = match ins.op0_kind() {
                    OpKind::Register => ins.op0_register().is_gpr64(),
                    OpKind::Memory => ins.memory_size().size() == 8,
                    OpKind::Immediate8to64 | OpKind::Immediate32to64 => true,
                    _ => false,
                };
                if !ok {
                    return Err(self.unsup());
                }
                let v = self.read_op(ins, 0, 8)?;
                self.push64(v)?;
            }
            Mnemonic::Pop => {
                let reg64 = ins.op0_kind() == OpKind::Register && ins.op0_register().is_gpr64();
                if !reg64 && ins.code() != Code::Pop_rm64 {
                    return Err(self.unsup());
                }
                let sp = self.cpu.regs[RSP];
                let v = self.load(sp, 8)?;
                self.cpu.regs[RSP] = sp.wrapping_add(8);
                self.write_op(ins, 0, 8, v)?;
            }
            Mnemonic::Jmp => {
                let t = self.branch_target(ins)?;
                return Ok(match self.flow_hook(FlowKind::Jmp, t, next)? {
                    HookOutcome::Redirect => self.cpu.rip,
                    HookOutcome::Continue => t,
                });
            }
            Mnemonic::Call => {
                let t = self.branch_target(ins)?;
                if let HookOutcome::Redirect = self.flow_hook(FlowKind::Call, t, next)? {
                    return Ok(self.cpu.rip);
                }
                self.push64(next)?;
                return Ok(t);
            }
            Mnemonic::Ret => {
                if !matches!(ins.code(), Code::Retnq | Code::Retnq_imm16) {
                    return Err(self.unsup());
                }
                let extra = if ins.op_count() == 1 { ins.immediate(0) } else { 0 };
                let sp = self.cpu.regs[RSP];
                let t = self.load(sp, 8)?;
                if let HookOutcome::Redirect = self.flow_hook(FlowKind::Ret, t, next)? {
                    return Ok(self.cpu.rip);
                }
                self.cpu.regs[RSP] = sp.wrapping_add(8).wrapping_add(extra);
                return Ok(t);
            }
            Mnemonic::Adc | Mnemonic::Sbb => {
                let n = self.op_bytes(ins, 0)?;
                if self.cpu.undef_flags & CF != 0 {
                    return Err(Stop::UndefinedFlags { rip: self.cur.0, mask: CF });
                }
                let a = self.read_op(ins, 0, n)?;
                let b = self.read_op(ins, 1, n)?;
                let carry = self.cpu.flag(CF);
                let (r, fl) = if mn == Mnemonic::Adc {
                    alu::adc(a, b, carry, n)
                } else {
                    alu::sbb(a, b, carry, n)
                };
                self.write_op(ins, 0, n, r)?;
                self.apply(fl);
            }
            Mnemonic::Mul => {
                let n = self.op_bytes(ins, 0)?;
                let a = self.cpu.regs[RAX] & mask(n);
                let b = self.read_op(ins, 0, n)?;
                let (lo, hi, fl) = alu::mul(a, b, n);
                self.put_mul_result(n, lo, hi);
                self.apply(fl);
            }
            Mnemonic::Imul => {
                let n = self.op_bytes(ins, 0)?;
                match ins.op_count() {
                    1 => {
                        let a = self.cpu.regs[RAX] & mask(n);
                        let b = self.read_op(ins, 0, n)?;
                        let (lo, hi, fl) = alu::imul1(a, b, n);
                        self.put_mul_result(n, lo, hi);
                        self.apply(fl);
                    }
                    2 | 3 => {
                        let (si, ii) = if ins.op_count() == 2 { (0, 1) } else { (1, 2) };
                        let a = self.read_op(ins, si, n)?;
                        let b = self.read_op(ins, ii, n)?;
                        let (r, fl) = alu::imul_n(a, b, n);
                        self.write_op(ins, 0, n, r)?;
                        self.apply(fl);
                    }
                    _ => return Err(self.unsup()),
                }
            }
            Mnemonic::Div | Mnemonic::Idiv => {
                let n = self.op_bytes(ins, 0)?;
                let divisor = self.read_op(ins, 0, n)?;
                let (hi, lo) = if n == 1 {
                    ((self.cpu.regs[RAX] >> 8) & 0xff, self.cpu.regs[RAX] & 0xff)
                } else {
                    (self.cpu.regs[RDX] & mask(n), self.cpu.regs[RAX] & mask(n))
                };
                match alu::divmod(hi, lo, divisor, n, mn == Mnemonic::Idiv) {
                    None => return Err(Stop::DivideError { rip: self.cur.0 }),
                    Some((q, r)) => {
                        if n == 1 {
                            self.cpu.set_reg(Register::AX, (q & 0xff) | ((r & 0xff) << 8));
                        } else {
                            self.cpu.set_reg(acc_reg(n), q);
                            self.cpu.set_reg(dx_reg(n), r);
                        }
                        self.apply(Flags { def: 0, val: 0, undef: STATUS });
                    }
                }
            }
            Mnemonic::Rol | Mnemonic::Ror | Mnemonic::Rcl | Mnemonic::Rcr => {
                let n = self.op_bytes(ins, 0)?;
                let a = self.read_op(ins, 0, n)?;
                let c = if ins.op_count() == 1 { 1 } else { self.read_op(ins, 1, 1)? };
                let kind = match mn {
                    Mnemonic::Rol => Rotate::Rol,
                    Mnemonic::Ror => Rotate::Ror,
                    Mnemonic::Rcl => Rotate::Rcl,
                    _ => Rotate::Rcr,
                };
                if matches!(mn, Mnemonic::Rcl | Mnemonic::Rcr) && self.cpu.undef_flags & CF != 0 {
                    return Err(Stop::UndefinedFlags { rip: self.cur.0, mask: CF });
                }
                let cf_in = self.cpu.flag(CF);
                if let Some((r, fl)) = alu::rotate(kind, a, c, cf_in, n) {
                    self.write_op(ins, 0, n, r)?;
                    self.apply(fl);
                }
            }
            Mnemonic::Shld | Mnemonic::Shrd => {
                let n = self.op_bytes(ins, 0)?;
                let bits = n * 8;
                let dest = self.read_op(ins, 0, n)?;
                let src = self.read_op(ins, 1, n)?;
                let cnt = (self.read_op(ins, 2, 1)? & if n == 8 { 63 } else { 31 }) as u32;
                if cnt >= bits {
                    // Count beyond the operand width (only reachable for 16-bit,
                    // whose count still masks mod 32) leaves both the result and
                    // the status flags architecturally undefined; VMProtect emits
                    // these as junk. Match the bsf/bsr convention: destination
                    // unchanged, status group flagged undefined.
                    self.apply(Flags { def: 0, val: 0, undef: STATUS });
                } else if cnt != 0 {
                    let (r, fl) = alu::dshift(mn == Mnemonic::Shld, dest, src, cnt, n);
                    self.write_op(ins, 0, n, r)?;
                    self.apply(fl);
                }
            }
            Mnemonic::Bt | Mnemonic::Bts | Mnemonic::Btr | Mnemonic::Btc => {
                self.exec_bit(ins, mn)?;
            }
            Mnemonic::Bsf | Mnemonic::Bsr => {
                let n = self.op_bytes(ins, 0)?;
                let s = self.read_op(ins, 1, n)? & mask(n);
                if s == 0 {
                    // Destination is architecturally undefined here; left
                    // unchanged by convention (no per-register undef tracking).
                    self.apply(Flags { def: ZF, val: ZF, undef: STATUS & !ZF });
                } else {
                    let idx = if mn == Mnemonic::Bsf { s.trailing_zeros() } else { s.ilog2() };
                    self.write_op(ins, 0, n, u64::from(idx))?;
                    self.apply(Flags { def: ZF, val: 0, undef: STATUS & !ZF });
                }
            }
            Mnemonic::Xchg => {
                let n = self.op_bytes(ins, 0)?;
                let a = self.read_op(ins, 0, n)?;
                let b = self.read_op(ins, 1, n)?;
                // Commit the memory operand (if any) first, so a store fault
                // leaves the register operand untouched.
                if ins.op_kind(1) == OpKind::Memory {
                    self.write_op(ins, 1, n, a)?;
                    self.write_op(ins, 0, n, b)?;
                } else {
                    self.write_op(ins, 0, n, b)?;
                    self.write_op(ins, 1, n, a)?;
                }
            }
            Mnemonic::Cmpxchg => {
                let n = self.op_bytes(ins, 0)?;
                let dest = self.read_op(ins, 0, n)?;
                let acc = self.cpu.regs[RAX] & mask(n);
                let (_, fl) = alu::arith(Alu::Sub, acc, dest, n);
                if acc == dest & mask(n) {
                    let src = self.read_op(ins, 1, n)?;
                    self.write_op(ins, 0, n, src)?;
                    self.apply(fl);
                } else {
                    self.apply(fl);
                    self.cpu.set_reg(acc_reg(n), dest);
                }
            }
            Mnemonic::Xadd => {
                let n = self.op_bytes(ins, 0)?;
                let dest = self.read_op(ins, 0, n)?;
                let src = self.read_op(ins, 1, n)?;
                let (sum, fl) = alu::arith(Alu::Add, dest, src, n);
                self.write_op(ins, 0, n, sum)?;
                self.write_op(ins, 1, n, dest)?;
                self.apply(fl);
            }
            Mnemonic::Leave => {
                let bp = self.cpu.regs[RBP];
                let v = self.load(bp, 8)?;
                self.cpu.regs[RSP] = bp.wrapping_add(8);
                self.cpu.regs[RBP] = v;
            }
            // Deterministic offline environment layer: no host clock or real
            // CPU is consulted, so detonation is repeatable. See `tsc`/`cpuid`.
            Mnemonic::Rdtsc => {
                self.audit.rdtsc += 1;
                if self.audit.tsc_sites.len() < 64 {
                    self.audit.tsc_sites.push((self.cur.0, self.steps));
                }
                let t = self.tsc();
                self.cpu.set_reg(Register::EAX, t & 0xffff_ffff);
                self.cpu.set_reg(Register::EDX, t >> 32);
            }
            Mnemonic::Rdtscp => {
                self.audit.rdtscp += 1;
                if self.audit.tsc_sites.len() < 64 {
                    self.audit.tsc_sites.push((self.cur.0, self.steps));
                }
                let t = self.tsc();
                self.cpu.set_reg(Register::EAX, t & 0xffff_ffff);
                self.cpu.set_reg(Register::EDX, t >> 32);
                self.cpu.set_reg(Register::ECX, TSC_AUX);
            }
            Mnemonic::Cpuid => {
                let leaf = self.cpu.regs[RAX] & 0xffff_ffff;
                let sub = self.cpu.regs[RCX] & 0xffff_ffff;
                *self.audit.cpuid.entry((leaf, sub)).or_insert(0) += 1;
                let (a, b, c, d) = cpuid(leaf, sub);
                self.cpu.set_reg(Register::EAX, a);
                self.cpu.set_reg(Register::EBX, b);
                self.cpu.set_reg(Register::ECX, c);
                self.cpu.set_reg(Register::EDX, d);
            }
            Mnemonic::Bswap => {
                let n = self.op_bytes(ins, 0)?;
                match n {
                    4 => {
                        let v = self.read_op(ins, 0, n)?;
                        self.write_op(ins, 0, n, u64::from((v as u32).swap_bytes()))?;
                    }
                    8 => {
                        let v = self.read_op(ins, 0, n)?;
                        self.write_op(ins, 0, n, v.swap_bytes())?;
                    }
                    // bswap r16 is architecturally undefined (it does not touch
                    // flags); VMProtect emits it as junk. Destination left
                    // unchanged, matching the bsf/bsr undefined-result convention.
                    2 => {}
                    _ => return Err(self.unsup()),
                }
            }
            // Bulk flag save/restore: pushfq copies the concrete flag image (as
            // hardware does); it does not stop on an undefined bit. Only a
            // consumer that branches/computes on a flag stops (see Lahf note).
            Mnemonic::Pushfq => self.push64(self.cpu.rflags)?,
            Mnemonic::Popfq => {
                let sp = self.cpu.regs[RSP];
                let v = self.load(sp, 8)?;
                // Setting TF arms a single-step #DB after the next instruction,
                // which a no-debugger process would catch in its own handler.
                // No exception delivery is modeled: fail closed rather than let
                // TF linger (rflags never holds TF, so pushfq always reads TF=0).
                if v & TF != 0 {
                    return Err(self.unsup());
                }
                self.cpu.regs[RSP] = sp.wrapping_add(8);
                self.cpu.rflags = v | 2; // bit 1 reads as 1
                self.cpu.undef_flags = 0; // all flags now have concrete values
            }
            Mnemonic::Cbw => {
                self.cpu.set_reg(Register::AX, alu_sext(self.cpu.regs[RAX], 1));
            }
            Mnemonic::Cwde => {
                self.cpu.set_reg(Register::EAX, alu_sext(self.cpu.regs[RAX], 2));
            }
            Mnemonic::Cdqe => {
                self.cpu.regs[RAX] = alu_sext(self.cpu.regs[RAX], 4);
            }
            Mnemonic::Cwd => {
                let s = if self.cpu.regs[RAX] & 0x8000 != 0 { 0xffff } else { 0 };
                self.cpu.set_reg(Register::DX, s);
            }
            Mnemonic::Cdq => {
                let s = if self.cpu.regs[RAX] & 0x8000_0000 != 0 { 0xffff_ffff } else { 0 };
                self.cpu.set_reg(Register::EDX, s);
            }
            Mnemonic::Cqo => {
                self.cpu.regs[RDX] = if self.cpu.regs[RAX] >> 63 != 0 { u64::MAX } else { 0 };
            }
            // Carry/direction flag controls (exact, pure).
            Mnemonic::Clc => self.apply(Flags { def: CF, val: 0, undef: 0 }),
            Mnemonic::Stc => self.apply(Flags { def: CF, val: CF, undef: 0 }),
            Mnemonic::Cmc => {
                // Pure CF -> !CF. When CF is architecturally undefined (e.g. a
                // VMProtect junk `bsf`/over-width shift feeds it), complementing
                // it yields an equally-undefined CF, so propagate that undefined
                // state rather than inventing a value: a later real consumer
                // (jcc/adc/sbb/setcc/cmovcc) still stops on it, but a sequence
                // that overwrites CF first sails through as on hardware.
                if self.cpu.undef_flags & CF == 0 {
                    let v = if self.cpu.flag(CF) { 0 } else { CF };
                    self.apply(Flags { def: CF, val: v, undef: 0 });
                }
            }
            Mnemonic::Cld => self.cpu.rflags &= !DF,
            Mnemonic::Std => self.cpu.rflags |= DF,
            // AH <-> low flag byte (SF:ZF:0:AF:0:PF:1:CF). A bulk flag transfer
            // copies the concrete flag image (as hardware does) rather than
            // stopping on an undefined bit: only a *consumer* that branches or
            // computes on a flag (jcc/setcc/cmov/adc/sbb/rcl/rcr/cmc) must stop.
            Mnemonic::Lahf => {
                let m5 = CF | PF | AF | ZF | SF;
                self.cpu.set_reg(Register::AH, (self.cpu.rflags & m5) | 2);
            }
            Mnemonic::Sahf => {
                let m5 = CF | PF | AF | ZF | SF;
                let ah = self.cpu.reg(Register::AH).ok_or_else(|| self.unsup())?;
                self.cpu.rflags = (self.cpu.rflags & !m5) | (ah & m5) | 2;
                self.cpu.undef_flags &= !m5;
            }
            _ if string_kind(ins.code()).is_some() => self.exec_string(ins)?,
            _ if ins.condition_code() != ConditionCode::None
                && ins.flow_control() == FlowControl::Next =>
            {
                // setcc (1 operand) / cmovcc (2 operands); both read flags via
                // `cond`, which stops on an undefined flag.
                let taken = self.cond(ins.condition_code())?;
                if ins.op_count() == 1 {
                    self.write_op(ins, 0, 1, u64::from(taken))?;
                } else {
                    let n = self.op_bytes(ins, 0)?;
                    let v = self.read_op(ins, 1, n)?;
                    if taken {
                        self.write_op(ins, 0, n, v)?;
                    }
                }
            }
            _ if ins.flow_control() == FlowControl::ConditionalBranch
                && ins.op0_kind() == OpKind::NearBranch64
                && ins.condition_code() != ConditionCode::None =>
            {
                let taken = self.cond(ins.condition_code())?;
                let t = if taken { ins.near_branch_target() } else { next };
                return Ok(match self.flow_hook(FlowKind::Jcc { taken }, t, next)? {
                    HookOutcome::Redirect => self.cpu.rip,
                    HookOutcome::Continue => t,
                });
            }
            _ if sse::is_vector(ins) => self.exec_sse(ins, mn)?,
            _ => return Err(self.unsup()),
        }
        Ok(next)
    }

    /// Deterministic, monotonic timestamp counter: the retired-instruction
    /// index scaled by a fixed stride. No host clock is read, so a given run
    /// always yields the same sequence; because each read is its own retired
    /// step, two reads are strictly increasing (a positive delta for code that
    /// times itself). The value fits comfortably in EAX across the step budget,
    /// so EDX reads back 0.
    fn tsc(&self) -> u64 {
        self.steps * TSC_STRIDE
    }

    #[inline(always)]
    fn push64(&mut self, v: u64) -> Result<(), Trap> {
        let sp = self.cpu.regs[RSP].wrapping_sub(8);
        self.store(sp, 8, v)?;
        self.cpu.regs[RSP] = sp;
        Ok(())
    }

    /// Store a `mul`/`imul` double-wide result: low half in the accumulator,
    /// high half in rDX (or AH for byte operands).
    fn put_mul_result(&mut self, n: u32, lo: u64, hi: u64) {
        if n == 1 {
            self.cpu.set_reg(Register::AX, (lo & 0xff) | ((hi & 0xff) << 8));
        } else {
            self.cpu.set_reg(acc_reg(n), lo);
            self.cpu.set_reg(dx_reg(n), hi);
        }
    }

    /// `bt`/`bts`/`btr`/`btc`. CF takes the selected bit; OF/SF/AF/PF become
    /// undefined and ZF is preserved.
    fn exec_bit(&mut self, ins: &Instruction, mn: Mnemonic) -> Result<(), Stop> {
        let n = self.op_bytes(ins, 0)?;
        let bits = n * 8;
        let imm = matches!(
            ins.op_kind(1),
            OpKind::Immediate8 | OpKind::Immediate8to16 | OpKind::Immediate8to32
        );
        let off = if imm {
            ins.immediate(1)
        } else if ins.op_register(1).is_gpr() {
            self.cpu.reg(ins.op_register(1)).ok_or_else(|| self.unsup())?
        } else {
            return Err(self.unsup());
        };

        // Read the targeted bit: `addr == Some(a)` means the bit lives in a byte
        // at `a` in memory; `None` means it lives in register operand 0.
        let (cur, addr, bitpos) = if ins.op_kind(0) == OpKind::Register {
            let bp = (off & u64::from(bits - 1)) as u32;
            (self.read_op(ins, 0, n)?, None, bp)
        } else {
            // Memory bit base: a (possibly signed) bit offset addresses a byte.
            let ea = self.ea(ins)?;
            let so = if imm { off as i64 } else { alu_sext(off, n) as i64 };
            let a = ea.wrapping_add((so >> 3) as u64);
            (self.load(a, 1)?, Some(a), (so & 7) as u32)
        };
        let cf = (cur >> bitpos) & 1;
        let newval = match mn {
            Mnemonic::Bts => cur | (1 << bitpos),
            Mnemonic::Btr => cur & !(1 << bitpos),
            Mnemonic::Btc => cur ^ (1 << bitpos),
            _ => cur,
        };
        if mn != Mnemonic::Bt {
            match addr {
                Some(a) => self.store(a, 1, newval)?,
                None => self.write_op(ins, 0, n, newval)?,
            }
        }
        self.apply(Flags { def: CF, val: if cf != 0 { CF } else { 0 }, undef: OF | SF | AF | PF });
        Ok(())
    }

    /// `movs`/`stos`/`scas`/`cmps`/`lods`, honouring `rep`/`repe`/`repne` and
    /// DF. Each element commits before RSI/RDI/RCX advance, so a mid-loop fault
    /// or hook stop leaves the string pointers at a restartable point.
    fn exec_string(&mut self, ins: &Instruction) -> Result<(), Stop> {
        let (size, kind) = string_kind(ins.code()).ok_or_else(|| self.unsup())?;
        let rep = ins.has_rep_prefix();
        let repne = ins.has_repne_prefix();
        let delta =
            if self.cpu.rflags & DF != 0 { u64::from(size).wrapping_neg() } else { u64::from(size) };
        loop {
            if (rep || repne) && self.cpu.regs[RCX] == 0 {
                break;
            }
            match kind {
                StrOp::Movs => {
                    let v = self.load(self.cpu.regs[RSI], size)?;
                    self.store(self.cpu.regs[RDI], size, v)?;
                    self.cpu.regs[RSI] = self.cpu.regs[RSI].wrapping_add(delta);
                    self.cpu.regs[RDI] = self.cpu.regs[RDI].wrapping_add(delta);
                }
                StrOp::Stos => {
                    let v = self.cpu.regs[RAX] & mask(size);
                    self.store(self.cpu.regs[RDI], size, v)?;
                    self.cpu.regs[RDI] = self.cpu.regs[RDI].wrapping_add(delta);
                }
                StrOp::Lods => {
                    let v = self.load(self.cpu.regs[RSI], size)?;
                    self.cpu.set_reg(acc_reg(size), v);
                    self.cpu.regs[RSI] = self.cpu.regs[RSI].wrapping_add(delta);
                }
                StrOp::Scas => {
                    let a = self.cpu.regs[RAX] & mask(size);
                    let v = self.load(self.cpu.regs[RDI], size)?;
                    let (_, fl) = alu::arith(Alu::Sub, a, v, size);
                    self.apply(fl);
                    self.cpu.regs[RDI] = self.cpu.regs[RDI].wrapping_add(delta);
                }
                StrOp::Cmps => {
                    let s = self.load(self.cpu.regs[RSI], size)?;
                    let d = self.load(self.cpu.regs[RDI], size)?;
                    let (_, fl) = alu::arith(Alu::Sub, s, d, size);
                    self.apply(fl);
                    self.cpu.regs[RSI] = self.cpu.regs[RSI].wrapping_add(delta);
                    self.cpu.regs[RDI] = self.cpu.regs[RDI].wrapping_add(delta);
                }
            }
            if rep || repne {
                self.cpu.regs[RCX] = self.cpu.regs[RCX].wrapping_sub(1);
                if matches!(kind, StrOp::Scas | StrOp::Cmps) {
                    let zf = self.cpu.flag(ZF);
                    // repe stops when ZF clears; repne stops when ZF sets.
                    if (rep && !zf) || (repne && zf) {
                        break;
                    }
                }
            } else {
                break;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum StrOp {
    Movs,
    Stos,
    Scas,
    Cmps,
    Lods,
}

/// Element size and family of a string-op [`Code`], or `None` otherwise.
fn string_kind(code: Code) -> Option<(u32, StrOp)> {
    use Code as C;
    Some(match code {
        C::Movsb_m8_m8 => (1, StrOp::Movs),
        C::Movsw_m16_m16 => (2, StrOp::Movs),
        C::Movsd_m32_m32 => (4, StrOp::Movs),
        C::Movsq_m64_m64 => (8, StrOp::Movs),
        C::Stosb_m8_AL => (1, StrOp::Stos),
        C::Stosw_m16_AX => (2, StrOp::Stos),
        C::Stosd_m32_EAX => (4, StrOp::Stos),
        C::Stosq_m64_RAX => (8, StrOp::Stos),
        C::Lodsb_AL_m8 => (1, StrOp::Lods),
        C::Lodsw_AX_m16 => (2, StrOp::Lods),
        C::Lodsd_EAX_m32 => (4, StrOp::Lods),
        C::Lodsq_RAX_m64 => (8, StrOp::Lods),
        C::Scasb_AL_m8 => (1, StrOp::Scas),
        C::Scasw_AX_m16 => (2, StrOp::Scas),
        C::Scasd_EAX_m32 => (4, StrOp::Scas),
        C::Scasq_RAX_m64 => (8, StrOp::Scas),
        C::Cmpsb_m8_m8 => (1, StrOp::Cmps),
        C::Cmpsw_m16_m16 => (2, StrOp::Cmps),
        C::Cmpsd_m32_m32 => (4, StrOp::Cmps),
        C::Cmpsq_m64_m64 => (8, StrOp::Cmps),
        _ => return None,
    })
}

/// ALU operation selected by a const index (0..=4 = add/sub/and/or/xor).
const fn alu_op(i: u8) -> Alu {
    match i {
        0 => Alu::Add,
        1 => Alu::Sub,
        2 => Alu::And,
        3 => Alu::Or,
        _ => Alu::Xor,
    }
}

/// 1 when condition base `base` (o/b/e/be/s/p/l/le) holds in `rflags`, else 0.
/// Branchless: every base is computed and one bit selected.
#[inline(always)]
fn cc_val(rflags: u64, base: u8) -> u64 {
    let cf = rflags & 1;
    let pf = (rflags >> 2) & 1;
    let zf = (rflags >> 6) & 1;
    let sf = (rflags >> 7) & 1;
    let of = (rflags >> 11) & 1;
    let lt = sf ^ of;
    let bits = of | cf << 1 | zf << 2 | (cf | zf) << 3 | sf << 4 | pf << 5 | lt << 6 | (zf | lt) << 7;
    (bits >> base) & 1
}

/// Sign-extend the low `bytes` of `v` to 64 bits (shared with alu's helper).
fn alu_sext(v: u64, bytes: u32) -> u64 {
    if bytes >= 8 {
        v
    } else {
        let sh = 64 - bytes * 8;
        (((v << sh) as i64) >> sh) as u64
    }
}

/// `rdtsc` tick per retired instruction (see [`Emulator::tsc`]).
const TSC_STRIDE: u64 = 64;
/// Fixed `rdtscp` processor id written to ECX (single generic core).
const TSC_AUX: u64 = 0;

/// Fixed, deterministic `cpuid` signature: a generic `GenuineIntel` family-6
/// processor advertising only the x87/TSC/CMOV/MMX/SSE/SSE2 baseline (no
/// AVX/SSE4), so the protector's `cpuid` gate takes one definite path and no
/// feature we cannot model is claimed. Returns `(EAX, EBX, ECX, EDX)`.
///
/// - leaf 0: max standard leaf + vendor string `GenuineIntel`.
/// - leaf 1: version `0x000606A7`; EDX = FPU|TSC|MSR|CMOV|MMX|FXSR|SSE|SSE2.
/// - leaf `0x8000_0000`: max extended leaf.
/// - leaf `0x8000_0001`: EDX = SYSCALL | Long-Mode (64-bit capable).
/// - any other leaf/subleaf: all zero.
fn cpuid(leaf: u64, _sub: u64) -> (u64, u64, u64, u64) {
    match leaf {
        0x0 => (0x16, 0x756e_6547, 0x6c65_746e, 0x4965_6e69),
        0x1 => (0x0006_06a7, 0, 0, 0x0780_8031),
        0x8000_0000 => (0x8000_0001, 0, 0, 0),
        0x8000_0001 => (0, 0, 0, 0x2000_0800),
        _ => (0, 0, 0, 0),
    }
}

/// Accumulator register for an operand width (AL/AX/EAX/RAX).
fn acc_reg(bytes: u32) -> Register {
    match bytes {
        1 => Register::AL,
        2 => Register::AX,
        4 => Register::EAX,
        _ => Register::RAX,
    }
}

/// rDX register for an operand width (used for mul/div high half).
fn dx_reg(bytes: u32) -> Register {
    match bytes {
        2 => Register::DX,
        4 => Register::EDX,
        _ => Register::RDX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CF;
    use crate::memory::{PAGE_SIZE, Perm};

    const ADDR: u64 = 0x1000;

    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    /// Records every event it is given (default interest: all). `.1` counts deliveries.
    #[derive(Default)]
    struct Log(Rc<RefCell<Vec<String>>>, Rc<Cell<u64>>);
    impl Hook for Log {
        fn on_mem(&mut self, ev: &MemEvent, _c: &mut Cpu, _m: &mut Memory) -> HookAction {
            self.1.set(self.1.get() + 1);
            self.0.borrow_mut().push(format!("{ev:?}"));
            HookAction::Continue
        }
        fn on_flow(&mut self, ev: &FlowEvent, _c: &mut Cpu, _m: &mut Memory) -> HookAction {
            self.1.set(self.1.get() + 1);
            self.0.borrow_mut().push(format!("{ev:?}"));
            HookAction::Continue
        }
    }

    /// A state-dependent hook shaped like `OepHook` + `WinApiHook`: logs events in
    /// its spans, tracks written pages, and notices a cross-page transfer into
    /// one (optionally stopping). Declares exactly that interest, so eliding
    /// anything else must leave its log (and the run) unchanged.
    struct Smart {
        log: Rc<RefCell<Vec<String>>>,
        delivered: Rc<Cell<u64>>,
        spans: [Span; 3],
        written: std::collections::BTreeSet<u64>,
        stop: bool,
        seen: bool,
    }
    impl Hook for Smart {
        fn on_mem(&mut self, ev: &MemEvent, _c: &mut Cpu, _m: &mut Memory) -> HookAction {
            self.delivered.set(self.delivered.get() + 1);
            let inside = |s: Span| ev.addr >= s.0 && ev.addr < s.1;
            if ev.access == Access::Read {
                if inside(self.spans[0]) {
                    self.log.borrow_mut().push(format!("{ev:?}"));
                }
                return HookAction::Continue;
            }
            if inside(self.spans[1]) {
                self.log.borrow_mut().push(format!("{ev:?}"));
            }
            let end = ev.addr.saturating_add(u64::from(ev.size.max(1)));
            let mut p = ev.addr & !0xfff;
            while p < end {
                if self.written.insert(p) {
                    self.log.borrow_mut().push(format!("new page {p:#x}"));
                }
                match p.checked_add(0x1000) {
                    Some(n) => p = n,
                    None => break,
                }
            }
            HookAction::Continue
        }
        fn on_flow(&mut self, ev: &FlowEvent, _c: &mut Cpu, _m: &mut Memory) -> HookAction {
            self.delivered.set(self.delivered.get() + 1);
            if ev.to >= self.spans[2].0 && ev.to < self.spans[2].1 {
                self.log.borrow_mut().push(format!("{ev:?}"));
            }
            if !self.seen
                && (ev.from & !0xfff) != (ev.to & !0xfff)
                && self.written.contains(&(ev.to & !0xfff))
            {
                self.seen = true;
                self.log.borrow_mut().push(format!("hit {ev:?}"));
                if self.stop {
                    return HookAction::Stop;
                }
            }
            HookAction::Continue
        }
        fn interest(&self) -> Interest {
            Interest {
                read: [self.spans[0], (0, 0)],
                write: [self.spans[1], (0, 0)],
                flow: [self.spans[2], (0, 0)],
                write_pages: true,
                flow_pages: !self.seen,
                ..Interest::NONE
            }
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const CODE: u64 = 0x10000;
    const DATA: u64 = 0x20000;

    /// A randomized emulator: random code bytes biased toward common opcodes,
    /// registers that mostly point into mapped data/code, random flags.
    fn fuzz_emu(seed: u64, log: &Log) -> Emulator {
        const COMMON: &[u8] = &[
            0x48, 0x49, 0x4c, 0x41, 0x66, 0x89, 0x8b, 0x01, 0x03, 0x83, 0x81, 0x0f, 0xb6, 0xb7,
            0xbe, 0xbf, 0x63, 0x8d, 0xff, 0xc1, 0xd1, 0xd3, 0xf7, 0x85, 0x39, 0x3b, 0x29, 0x2b,
            0x21, 0x23, 0x09, 0x0b, 0x31, 0x33, 0x90, 0xeb, 0x74, 0x75, 0x72, 0x7f, 0x0f, 0x84,
            0x50, 0x51, 0x58, 0x59, 0xc3, 0xc7, 0xc6, 0x88, 0x8a, 0xf8, 0xf9, 0xf5, 0x9c, 0x98,
            0x99, 0xa9, 0xa8, 0xc0, 0xd0, 0x87, 0x0f, 0xc1, 0xc8, 0xbc, 0xbd, 0xa3, 0xab, 0xba,
            0x00, 0x04, 0x24, 0x44, 0x80, 0x84, 0x05, 0x15, 0x1d, 0xe8, 0xe9, 0x6a, 0x68, 0xb8,
        ];
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut mem = Memory::new();
        let code_perm = if r.below(2) == 0 { Perm::RWX } else { Perm::RX };
        mem.map(CODE, PAGE_SIZE, code_perm).unwrap();
        if r.below(2) == 0 {
            mem.map(CODE + PAGE_SIZE, PAGE_SIZE, if r.below(2) == 0 { Perm::RX } else { Perm::RW }).unwrap();
        }
        mem.map(DATA, PAGE_SIZE * 2, Perm::RW).unwrap();
        let bytes: Vec<u8> = (0..PAGE_SIZE * 2)
            .map(|_| if r.below(4) < 3 { COMMON[r.below(COMMON.len() as u64) as usize] } else { r.next() as u8 })
            .collect();
        mem.poke(CODE, &bytes[..PAGE_SIZE as usize]).ok();
        mem.poke(CODE + PAGE_SIZE, &bytes[PAGE_SIZE as usize..]).ok();
        let data: Vec<u8> = (0..PAGE_SIZE * 2).map(|_| r.next() as u8).collect();
        mem.poke(DATA, &data).unwrap();
        let near_end = r.below(8) == 0;
        let start = CODE + if near_end { PAGE_SIZE - 1 - r.below(20) } else { r.below(PAGE_SIZE - 8) };
        let mut cpu = Cpu::new(start);
        for reg in &mut cpu.regs {
            *reg = match r.below(8) {
                0 => r.next(),
                1 => r.below(300),
                2 => u64::MAX - r.below(300),
                3 => CODE + r.below(PAGE_SIZE * 2),
                _ => DATA + r.below(PAGE_SIZE * 2 - 16),
            };
        }
        cpu.rflags = 2 | (r.next() & (CF | PF | AF | ZF | SF | OF | DF));
        cpu.undef_flags = if r.below(4) == 0 { r.next() & (CF | PF | AF | ZF | SF | OF) } else { 0 };
        cpu.gs_base = if r.below(4) == 0 { DATA } else { 0 };
        let mut emu = Emulator::new(cpu, mem);
        emu.resolve_undefined_flags = r.below(2) == 0;
        emu.hook = Some(if seed % 3 == 0 {
            Box::new(Log(log.0.clone(), log.1.clone()))
        } else {
            let mut span = |base: u64| {
                let lo = base + r.below(PAGE_SIZE * 2);
                (lo, lo + r.below(600))
            };
            Box::new(Smart {
                log: log.0.clone(),
                delivered: log.1.clone(),
                spans: [span(DATA), span(DATA), span(CODE)],
                written: std::collections::BTreeSet::new(),
                stop: r.below(2) == 0,
                seen: false,
            })
        });
        emu
    }

    fn same_state(a: &Emulator, b: &Emulator) -> bool {
        a.cpu == b.cpu
            && a.steps == b.steps
            && a.mem.pages().eq(b.mem.pages())
    }

    /// Fast path (pre-resolved ops via `step`, and blocks via `run`) must match
    /// the original interpreter exactly: result, registers/flags, memory and the
    /// hook event stream, including self-modifying and page-crossing code.
    #[test]
    fn fast_paths_match_reference_interpreter() {
        let mut ran = 0u64;
        let (mut full, mut elided) = (0u64, 0u64);
        let seeds: u64 = std::env::var("FUZZ_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(2_500);
        for seed in 1..seeds {
            let (la, lb, lc) = (Log::default(), Log::default(), Log::default());
            let mut a = fuzz_emu(seed, &la);
            let mut b = fuzz_emu(seed, &lb);
            let mut c = fuzz_emu(seed, &lc);
            b.record_trace = seed % 2 == 0;
            a.record_trace = b.record_trace;
            c.record_trace = false;
            let k = 1 + (seed % 24);
            // `a` = reference, `b` = step(), `c` = run() (blocks).
            let mut ra = Ok(());
            let mut rb = Ok(());
            let mut rips = vec![a.cpu.rip];
            for i in 0..k {
                ra = a.step_ref();
                rips.push(a.cpu.rip);
                rb = b.step();
                assert_eq!(ra, rb, "seed {seed} step {i} result");
                assert!(same_state(&a, &b), "seed {seed} step {i} state");
                ran += 1;
                if ra.is_err() {
                    break;
                }
            }
            assert_eq!(*la.0.borrow(), *lb.0.borrow(), "seed {seed} events");
            assert_eq!(a.trace, b.trace, "seed {seed} trace");
            let sc = c.run(a.steps);
            if ra.is_err() {
                // c may stop later than `a` only if budget remains; give it room.
                let sc2 = if sc == Stop::StepLimit { c.run(1) } else { sc };
                assert_eq!(Err(sc2), ra, "seed {seed} run stop");
            } else {
                assert_eq!(sc, Stop::StepLimit, "seed {seed} run limit");
            }
            assert!(same_state(&a, &c), "seed {seed} block state");
            assert_eq!(*la.0.borrow(), *lc.0.borrow(), "seed {seed} block events");
            // Same run on the per-step (trace recording) path.
            let lg = Log::default();
            let mut g = fuzz_emu(seed, &lg);
            let sg = g.run(a.steps);
            let sg = if ra.is_err() && sg == Stop::StepLimit { g.run(1) } else { sg };
            assert_eq!(sg, if ra.is_err() { ra.clone().unwrap_err() } else { Stop::StepLimit }, "seed {seed} traced run stop");
            assert!(same_state(&a, &g), "seed {seed} traced run state");
            assert_eq!(*la.0.borrow(), *lg.0.borrow(), "seed {seed} traced run events");
            (full, elided) = (full + la.1.get(), elided + lc.1.get());
            let _ = rb;
            // run_until: stop at a rip the reference visited (possibly mid-block).
            let tgt = rips[(seed as usize) % rips.len()];
            let (le, ld) = (Log::default(), Log::default());
            let (mut e, mut d) = (fuzz_emu(seed, &le), fuzz_emu(seed, &ld));
            e.record_trace = false;
            d.record_trace = false;
            let mut want = Stop::StepLimit;
            for _ in 0..k {
                if e.cpu.rip == tgt {
                    want = Stop::Reached;
                    break;
                }
                if let Err(s) = e.step_ref() {
                    want = s;
                    break;
                }
            }
            assert_eq!(d.run_until(tgt, k), want, "seed {seed} run_until stop");
            assert!(same_state(&e, &d), "seed {seed} run_until state");
            assert_eq!(*le.0.borrow(), *ld.0.borrow(), "seed {seed} run_until events");
        }
        assert!(ran > seeds, "fuzz exercised too little: {ran}");
        assert!(elided < full, "interest elision never kicked in: {elided} vs {full}");
    }

    /// A transfer into a page that was quiet (memoized) must be delivered again
    /// once that page has been written.
    #[test]
    fn elided_transfer_is_delivered_after_the_page_is_written() {
        const X: u64 = 0x31000;
        let build = |log: &Log| {
            let mut mem = Memory::new();
            mem.map(CODE, PAGE_SIZE, Perm::RWX).unwrap();
            mem.map(X, PAGE_SIZE, Perm::RWX).unwrap();
            let mut main = vec![0x48, 0xBB];
            main.extend_from_slice(&X.to_le_bytes());
            main.extend_from_slice(&[0xFF, 0xE3]); // mov rbx, X; jmp rbx
            mem.poke(CODE, &main).unwrap();
            let mut jb = vec![0xE9];
            jb.extend_from_slice(&((CODE + 0x20).wrapping_sub(X + 5) as u32).to_le_bytes());
            mem.poke(X, &jb).unwrap(); // jmp CODE+0x20
            mem.poke(CODE + 0x20, &[0x48, 0x89, 0x9B, 0, 8, 0, 0, 0xFF, 0xE3]).unwrap(); // mov [rbx+0x800], rbx; jmp rbx
            let mut emu = Emulator::new(Cpu::new(CODE), mem);
            emu.record_trace = false;
            emu.hook = Some(Box::new(Smart {
                log: log.0.clone(),
                delivered: log.1.clone(),
                spans: [(0, 0); 3],
                written: std::collections::BTreeSet::new(),
                stop: false,
                seen: false,
            }));
            emu
        };
        let (lr, lf) = (Log::default(), Log::default());
        let (mut r, mut f) = (build(&lr), build(&lf));
        for _ in 0..8 {
            r.step_ref().unwrap();
        }
        assert_eq!(f.run(8), Stop::StepLimit);
        assert!(lr.0.borrow().iter().any(|e| e.starts_with("hit")), "scenario never transferred into the written page");
        assert_eq!(*lr.0.borrow(), *lf.0.borrow());
        assert!(same_state(&r, &f));
    }

    /// The decode cache must not reuse a stale entry when the guest bytes at an
    /// address change underneath it (self-modifying / write-then-execute code).
    #[test]
    fn decode_cache_invalidates_on_self_modifying_code() {
        let mut mem = Memory::new();
        mem.map(ADDR, PAGE_SIZE, Perm::RWX).unwrap();
        // mov eax, 1 — primes the cache at ADDR.
        mem.poke(ADDR, &[0xB8, 0x01, 0x00, 0x00, 0x00]).unwrap();

        let mut emu = Emulator::new(Cpu::new(ADDR), mem);
        emu.step().unwrap();
        assert_eq!(emu.cpu.reg(Register::RAX), Some(1));

        // Overwrite the same bytes with mov eax, 2 and re-execute at ADDR.
        emu.mem.poke(ADDR, &[0xB8, 0x02, 0x00, 0x00, 0x00]).unwrap();
        emu.cpu.rip = ADDR;
        emu.step().unwrap();
        assert_eq!(emu.cpu.reg(Register::RAX), Some(2), "cache served stale decode");
    }

    /// The 8-byte-load fast compare must agree with a plain slice compare for
    /// every length and every single differing byte position.
    #[test]
    fn same_bytes_matches_slice_compare() {
        let base: [u8; 15] = std::array::from_fn(|i| i as u8 + 1);
        for len in 1..=15usize {
            assert!(same_bytes(&base, &base, len));
            for pos in 0..15 {
                let mut other = base;
                other[pos] ^= 0x80;
                assert_eq!(same_bytes(&other, &base, len), pos >= len, "len {len} pos {pos}");
                // Shorter fetch (page-crossing slow path) uses the slice compare.
                assert_eq!(same_bytes(&other[..len], &base, len), pos >= len);
            }
        }
    }

    /// Without trace recording the retired count and TSC are unchanged.
    #[test]
    fn untraced_run_keeps_step_count_and_tsc() {
        let code = [0x0F, 0x31, 0x0F, 0x31, 0x90]; // rdtsc; rdtsc; nop
        let run = |record: bool| {
            let mut mem = Memory::new();
            mem.map(ADDR, PAGE_SIZE, Perm::RWX).unwrap();
            mem.poke(ADDR, &code).unwrap();
            let mut emu = Emulator::new(Cpu::new(ADDR), mem);
            emu.record_trace = record;
            for _ in 0..3 {
                emu.step().unwrap();
            }
            (emu.steps(), emu.cpu.regs[RAX], emu.trace.len())
        };
        let (traced, untraced) = (run(true), run(false));
        assert_eq!(traced.0, 3);
        assert_eq!(traced.0, untraced.0);
        assert_eq!(traced.1, untraced.1, "rdtsc value depends only on the step count");
        assert_eq!((traced.2, untraced.2), (3, 0));
    }
}
