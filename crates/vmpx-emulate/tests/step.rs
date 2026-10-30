use iced_x86::Mnemonic as M;
use iced_x86::code_asm::*;
use vmpx_emulate::{
    CF, Cpu, Emulator, FlowEvent, FlowKind, Hook, HookAction, MemEvent, Memory, OF, Perm, SF, Stop,
    ZF,
};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x2000;
const STACK_TOP: u64 = 0x9000;

fn emu(bytes: &[u8]) -> Emulator {
    let mut mem = Memory::new();
    mem.map(CODE, 0x1000, Perm::RWX).unwrap();
    mem.map(DATA, 0x1000, Perm::RW).unwrap();
    mem.map(0x8000, 0x1000, Perm::RW).unwrap();
    mem.poke(CODE, bytes).unwrap();
    let mut cpu = Cpu::new(CODE);
    cpu.regs[4] = STACK_TOP;
    Emulator::new(cpu, mem)
}

fn asm(f: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    f(&mut a);
    a.assemble(CODE).unwrap()
}

#[test]
fn program_state_and_trace() {
    let code = asm(|a| {
        let mut skip = a.create_label();
        let mut f = a.create_label();
        let mut done = a.create_label();
        a.mov(rax, 5u64).unwrap();
        a.mov(rbx, 7u64).unwrap();
        a.add(rax, rbx).unwrap();
        a.push(rax).unwrap();
        a.pop(rcx).unwrap();
        a.mov(rdx, DATA).unwrap();
        a.mov(qword_ptr(rdx), rcx).unwrap();
        a.xor(esi, esi).unwrap();
        a.cmp(rcx, 12).unwrap();
        a.je(skip).unwrap();
        a.mov(rsi, 0xdeadu64).unwrap();
        a.set_label(&mut skip).unwrap();
        a.shl(rcx, 2).unwrap();
        a.call(f).unwrap();
        a.jmp(done).unwrap();
        a.set_label(&mut f).unwrap();
        a.lea(rdi, ptr(rcx + rcx * 2)).unwrap();
        a.ret().unwrap();
        a.set_label(&mut done).unwrap();
        a.mov(r8d, -1).unwrap();
        a.add(r8d, 1).unwrap();
        a.nop().unwrap();
    });
    let end = CODE + code.len() as u64 - 1;
    let mut e = emu(&code);
    assert_eq!(e.run_until(end, 100), Stop::Reached);

    let r = &e.cpu.regs;
    assert_eq!((r[0], r[3], r[1], r[2]), (12, 7, 48, DATA));
    assert_eq!((r[6], r[7], r[8]), (0, 144, 0));
    assert_eq!(r[4], STACK_TOP);
    assert_eq!(e.mem.read_uint(DATA, 8).unwrap(), 12);
    assert!(e.cpu.flag(CF) && e.cpu.flag(ZF));
    assert!(!e.cpu.flag(SF) && !e.cpu.flag(OF));

    let mns: Vec<M> = e.trace.iter().map(|t| t.mnemonic).collect();
    assert_eq!(
        mns,
        [
            M::Mov, M::Mov, M::Add, M::Push, M::Pop, M::Mov, M::Mov, M::Xor, M::Cmp, M::Je,
            M::Shl, M::Call, M::Lea, M::Ret, M::Jmp, M::Mov, M::Add,
        ]
    );
    assert_eq!(e.trace[0].rip, CODE);
}

#[test]
fn partial_register_writes() {
    let code = asm(|a| {
        a.mov(rax, -1i64).unwrap();
        a.mov(eax, 1).unwrap();
        a.mov(rbx, -1i64).unwrap();
        a.mov(bl, 0x12).unwrap();
        a.mov(bh, 0x34).unwrap();
        a.mov(sil, 0x56).unwrap();
        a.mov(r9w, 0x7777).unwrap();
        a.movsx(rcx, bl).unwrap();
        a.movzx(edx, bh).unwrap();
        a.nop().unwrap();
    });
    let mut e = emu(&code);
    assert_eq!(e.run_until(CODE + code.len() as u64 - 1, 50), Stop::Reached);
    let r = &e.cpu.regs;
    assert_eq!(r[0], 1);
    assert_eq!(r[3], 0xffff_ffff_ffff_3412);
    assert_eq!(r[6], 0x56);
    assert_eq!(r[9], 0x7777);
    assert_eq!(r[1], 0x12);
    assert_eq!(r[2], 0x34);
}

#[test]
fn unsupported_stops_cleanly() {
    let mut e = emu(&[0x0f, 0xa2]);
    e.cpu.regs[0] = 7;
    let before = e.cpu.clone();
    assert_eq!(e.step(), Err(Stop::Unsupported { rip: CODE, mnemonic: M::Cpuid }));
    assert_eq!(e.cpu, before);
    assert!(e.trace.is_empty());
}

#[test]
fn invalid_encoding_is_decode_error() {
    let mut e = emu(&[0x06]);
    assert_eq!(e.step(), Err(Stop::DecodeError { rip: CODE }));
}

#[test]
fn undefined_flag_read_stops() {
    let code = asm(|a| {
        let mut l = a.create_label();
        a.mov(eax, 1).unwrap();
        a.shl(eax, 3).unwrap();
        a.jo(l).unwrap();
        a.set_label(&mut l).unwrap();
        a.nop().unwrap();
    });
    let mut e = emu(&code);
    let s = e.run(10);
    assert!(matches!(s, Stop::UndefinedFlags { mask: OF, .. }), "{s:?}");
    assert_eq!(e.trace.len(), 2);
}

#[test]
fn faulting_write_leaves_state_untouched() {
    let code = asm(|a| {
        a.mov(rdx, 0x5000u64).unwrap();
        a.add(qword_ptr(rdx), rcx).unwrap();
    });
    let mut e = emu(&code);
    e.cpu.regs[1] = 9;
    e.step().unwrap();
    let before = e.cpu.clone();
    let s = e.step().unwrap_err();
    assert!(matches!(s, Stop::Fault { addr: 0x5000, .. }));
    assert_eq!(e.cpu, before);
}

#[test]
fn non_executable_fetch_faults() {
    let mut e = emu(&[0x90]);
    e.cpu.rip = DATA;
    assert!(matches!(e.step(), Err(Stop::Fault { addr: DATA, .. })));
}

#[derive(Default)]
struct Rec {
    flows: Vec<FlowEvent>,
    writes: Vec<MemEvent>,
}

struct Shared(std::rc::Rc<std::cell::RefCell<Rec>>);

impl Hook for Shared {
    fn on_mem(&mut self, ev: &MemEvent) -> HookAction {
        if ev.value.is_some() {
            self.0.borrow_mut().writes.push(*ev);
        }
        HookAction::Continue
    }
    fn on_flow(&mut self, ev: &FlowEvent) -> HookAction {
        self.0.borrow_mut().flows.push(*ev);
        if ev.kind == FlowKind::Call { HookAction::Stop } else { HookAction::Continue }
    }
}

#[test]
fn hook_observes_and_can_stop() {
    let code = asm(|a| {
        let mut f = a.create_label();
        a.push(1).unwrap();
        a.call(f).unwrap();
        a.set_label(&mut f).unwrap();
        a.nop().unwrap();
    });
    let rec = std::rc::Rc::new(std::cell::RefCell::new(Rec::default()));
    let mut e = emu(&code);
    e.hook = Some(Box::new(Shared(rec.clone())));
    assert_eq!(e.run(10), Stop::Hook);
    let r = rec.borrow();
    assert_eq!(r.writes.len(), 1);
    assert_eq!((r.writes[0].addr, r.writes[0].value), (STACK_TOP - 8, Some(1)));
    assert_eq!(r.flows.len(), 1);
    assert_eq!(r.flows[0].kind, FlowKind::Call);
    assert_eq!(e.cpu.regs[4], STACK_TOP - 8);
    assert_eq!(e.trace.len(), 1);
}
