use iced_x86::{
    Code, ConditionCode, Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind,
    Register,
};

use crate::alu::{self, Alu, Flags, Shift};
use crate::cpu::{CF, Cpu, OF, PF, RSP, SF, ZF, mask};
use crate::memory::{Access, Memory};

/// Why a `step`/`run` did not retire an instruction. On any `Stop` the
/// architectural state is exactly as before the offending instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    Unsupported { rip: u64, mnemonic: Mnemonic },
    DecodeError { rip: u64 },
    Fault { rip: u64, addr: u64, access: Access },
    UndefinedFlags { rip: u64, mask: u64 },
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
        if ins.has_lock_prefix() || ins.has_rep_prefix() || ins.has_repne_prefix() {
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
            Mnemonic::Shl | Mnemonic::Shr | Mnemonic::Sar => {
                let n = self.op_bytes(ins, 0)?;
                let a = self.read_op(ins, 0, n)?;
                let c = match ins.op_count() {
                    1 => 1,
                    _ => self.read_op(ins, 1, 1)?,
                };
                let kind = match mn {
                    Mnemonic::Shl => Shift::Shl,
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
}
