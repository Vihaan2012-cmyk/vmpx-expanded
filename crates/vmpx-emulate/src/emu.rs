use iced_x86::{
    Code, ConditionCode, Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind,
    Register,
};

use crate::alu::{self, Alu, Flags, Rotate, Shift};
use crate::cpu::{AF, CF, Cpu, OF, PF, RSP, SF, ZF, mask};
use crate::memory::{Access, Memory};

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

/// Why a `step`/`run` did not retire an instruction. On any `Stop` the
/// architectural state is exactly as before the offending instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    Unsupported { rip: u64, mnemonic: Mnemonic },
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
}

/// Internal resolution of a hook callback (a `Stop` is surfaced as `Err`).
enum HookOutcome {
    Continue,
    Redirect,
}

pub struct Emulator {
    pub cpu: Cpu,
    pub mem: Memory,
    pub trace: Vec<TraceEntry>,
    pub hook: Option<Box<dyn Hook>>,
    pub observe: ObserveConfig,
    cur: (u64, Mnemonic),
    det: (Option<u64>, Option<u64>),
}

impl Emulator {
    #[must_use]
    pub fn new(cpu: Cpu, mem: Memory) -> Self {
        Self {
            cpu,
            mem,
            trace: Vec::new(),
            hook: None,
            observe: ObserveConfig::default(),
            cur: (0, Mnemonic::INVALID),
            det: (None, None),
        }
    }

    /// Execute one instruction.
    ///
    /// # Errors
    /// A [`Stop`]; the state is unchanged when it is returned.
    pub fn step(&mut self) -> Result<(), Stop> {
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
        let (ea, value, vsp) = if self.observe.detail {
            (self.det.0, self.det.1, self.observe.vsp.and_then(|r| self.cpu.reg(r)))
        } else {
            (None, None, None)
        };
        self.trace.push(TraceEntry { rip, mnemonic: ins.mnemonic(), ea, value, vsp });
        Ok(())
    }

    /// Step until a stop; returns the reason ([`Stop::StepLimit`] after `max_steps`).
    pub fn run(&mut self, max_steps: u64) -> Stop {
        for _ in 0..max_steps {
            if let Err(s) = self.step() {
                return s;
            }
        }
        Stop::StepLimit
    }

    /// Like [`Emulator::run`] but stops with [`Stop::Reached`] when rip == `addr`
    /// (checked before executing there).
    pub fn run_until(&mut self, addr: u64, max_steps: u64) -> Stop {
        for _ in 0..max_steps {
            if self.cpu.rip == addr {
                return Stop::Reached;
            }
            if let Err(s) = self.step() {
                return s;
            }
        }
        Stop::StepLimit
    }

    fn unsup(&self) -> Stop {
        Stop::Unsupported { rip: self.cur.0, mnemonic: self.cur.1 }
    }

    fn fault(&self, addr: u64, access: Access) -> Stop {
        Stop::Fault { rip: self.cur.0, addr, access }
    }

    fn mem_hook(&mut self, ev: MemEvent) -> Result<(), Stop> {
        // Disjoint field borrows: the hook mutates cpu/mem while it fires.
        let Self { hook, cpu, mem, .. } = self;
        match hook.as_mut().map(|h| h.on_mem(&ev, cpu, mem)) {
            Some(HookAction::Stop) => Err(Stop::Hook),
            _ => Ok(()),
        }
    }

    fn flow_hook(&mut self, kind: FlowKind, to: u64, next: u64) -> Result<HookOutcome, Stop> {
        let ev = FlowEvent { kind, from: self.cur.0, to, next };
        let Self { hook, cpu, mem, .. } = self;
        match hook.as_mut().map(|h| h.on_flow(&ev, cpu, mem)) {
            Some(HookAction::Stop) => Err(Stop::Hook),
            Some(HookAction::Redirect) => Ok(HookOutcome::Redirect),
            _ => Ok(HookOutcome::Continue),
        }
    }

    fn load(&mut self, addr: u64, size: u32) -> Result<u64, Stop> {
        self.mem_hook(MemEvent { addr, size, access: Access::Read, value: None })?;
        let v = self.mem.read_uint(addr, size as usize).map_err(|f| self.fault(f.addr, f.access))?;
        if self.observe.detail {
            self.det = (Some(addr), Some(v));
        }
        Ok(v)
    }

    fn store(&mut self, addr: u64, size: u32, val: u64) -> Result<(), Stop> {
        let value = val & mask(size);
        self.mem_hook(MemEvent { addr, size, access: Access::Write, value: Some(value) })?;
        self.mem.write_uint(addr, size as usize, val).map_err(|f| self.fault(f.addr, f.access))?;
        if self.observe.detail {
            self.det = (Some(addr), Some(value));
        }
        Ok(())
    }

    fn apply(&mut self, f: Flags) {
        self.cpu.rflags = (self.cpu.rflags & !(f.def | f.undef)) | (f.val & f.def);
        self.cpu.undef_flags = (self.cpu.undef_flags & !f.def) | f.undef;
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
        if base != Register::None {
            if !base.is_gpr64() {
                return Err(self.unsup());
            }
            a = a.wrapping_add(self.cpu.reg(base).ok_or_else(|| self.unsup())?);
        }
        let idx = ins.memory_index();
        if idx != Register::None {
            if !idx.is_gpr64() {
                return Err(self.unsup());
            }
            let v = self.cpu.reg(idx).ok_or_else(|| self.unsup())?;
            a = a.wrapping_add(v.wrapping_mul(u64::from(ins.memory_index_scale())));
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
                self.load(a, bytes)
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
                self.store(a, bytes, val)
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
                self.load(a, 8)
            }
            _ => Err(self.unsup()),
        }
    }

    fn exec(&mut self, ins: &Instruction, next: u64) -> Result<u64, Stop> {
        // `lock` is transparent in a single-threaded model. `rep`/`repne` are
        // only meaningful on the string ops that consume them below; anywhere
        // else they stop rather than being silently ignored.
        if (ins.has_rep_prefix() || ins.has_repne_prefix()) && string_kind(ins.code()).is_none() {
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
                if ins.op0_kind() != OpKind::Register || !ins.op0_register().is_gpr64() {
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
                    // Count beyond the operand width (only reachable for 16-bit)
                    // is architecturally undefined — fail closed.
                    return Err(self.unsup());
                }
                if cnt != 0 {
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
            Mnemonic::Bswap => {
                let n = self.op_bytes(ins, 0)?;
                if n != 4 && n != 8 {
                    return Err(self.unsup());
                }
                let v = self.read_op(ins, 0, n)?;
                let r = if n == 4 { u64::from((v as u32).swap_bytes()) } else { v.swap_bytes() };
                self.write_op(ins, 0, n, r)?;
            }
            // Bulk flag save/restore: pushfq copies the concrete flag image (as
            // hardware does); it does not stop on an undefined bit. Only a
            // consumer that branches/computes on a flag stops (see Lahf note).
            Mnemonic::Pushfq => self.push64(self.cpu.rflags)?,
            Mnemonic::Popfq => {
                let sp = self.cpu.regs[RSP];
                let v = self.load(sp, 8)?;
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
                if self.cpu.undef_flags & CF != 0 {
                    return Err(Stop::UndefinedFlags { rip: self.cur.0, mask: CF });
                }
                let v = if self.cpu.flag(CF) { 0 } else { CF };
                self.apply(Flags { def: CF, val: v, undef: 0 });
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
            _ => return Err(self.unsup()),
        }
        Ok(next)
    }

    fn push64(&mut self, v: u64) -> Result<(), Stop> {
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

/// Sign-extend the low `bytes` of `v` to 64 bits (shared with alu's helper).
fn alu_sext(v: u64, bytes: u32) -> u64 {
    if bytes >= 8 {
        v
    } else {
        let sh = 64 - bytes * 8;
        (((v << sh) as i64) >> sh) as u64
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
