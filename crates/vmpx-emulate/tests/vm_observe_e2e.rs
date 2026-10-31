//! Layer-5 end-to-end: drive hand-assembled guest code through a
//! [`VmObserveHook`] and assert (1) a dispatcher-style indirect-transfer
//! sequence records the visited handler targets in order, and (2) a loop that
//! iterates a concrete number of times records that exact back-edge depth into
//! `ObservedFacts.loop_depths`.

use iced_x86::code_asm::*;
use vmpx_emulate::{
    Cpu, Emulator, Memory, Observed, ObservedFacts, Perm, Stop, SurveyHook, VipLoc, VmObserveHook,
};

const CODE: u64 = 0x1000;

fn asm_at(base: u64, build: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    a.assemble(base).unwrap()
}

fn emu_over(blobs: &[(u64, Vec<u8>)]) -> Emulator {
    let mut mem = Memory::new();
    mem.map(CODE, 0x1000, Perm::RX).unwrap();
    for (addr, bytes) in blobs {
        mem.poke(*addr, bytes).unwrap();
    }
    Emulator::new(Cpu::new(CODE), mem)
}

/// A central dispatcher (`jmp rax`) that three handlers enter in turn; the
/// recorded handler trace is the visited targets in dispatch order.
#[test]
fn dispatcher_handler_trace_records_visited_targets_in_order() {
    const DISPATCH: u64 = 0x1200;
    const H0: u64 = 0x1300;
    const H1: u64 = 0x1340;
    const H2: u64 = 0x1380;
    const EXIT: u64 = 0x1400;

    // Entry: load first handler, enter the dispatcher.
    let entry = asm_at(CODE, |a| {
        a.mov(rax, H0).unwrap();
        a.jmp(DISPATCH).unwrap();
    });
    // Dispatcher: indirect jmp to the selected handler.
    let dispatch = asm_at(DISPATCH, |a| {
        a.jmp(rax).unwrap();
    });
    // Handlers select the next target and re-enter the dispatcher; the last
    // one leaves directly (so EXIT is not mistaken for a handler).
    let h0 = asm_at(H0, |a| {
        a.mov(rax, H1).unwrap();
        a.jmp(DISPATCH).unwrap();
    });
    let h1 = asm_at(H1, |a| {
        a.mov(rax, H2).unwrap();
        a.jmp(DISPATCH).unwrap();
    });
    let h2 = asm_at(H2, |a| {
        a.jmp(EXIT).unwrap();
    });

    let hook = VmObserveHook::new(Some(DISPATCH), None);
    let obs = hook.observations();
    let mut emu = emu_over(&[
        (CODE, entry),
        (DISPATCH, dispatch),
        (H0, h0),
        (H1, h1),
        (H2, h2),
    ]);
    emu.hook = Some(Box::new(hook));

    assert_eq!(emu.run_until(EXIT, 1000), Stop::Reached, "ran to exit");

    let o = obs.borrow();
    assert_eq!(o.handlers, vec![H0, H1, H2], "handler entries in dispatch order");
    assert_eq!(o.back_edges, 0, "no loop head configured");
}

/// A down-counter loop: the back-edge to the loop head fires exactly
/// `TRIPS - 1` times for this concrete input, and that count lands in
/// `ObservedFacts.loop_depths`, tagged `Observed`.
#[test]
fn back_edge_counter_records_concrete_recursion_depth() {
    const TRIPS: u64 = 8;

    // Loop head sits right after the prologue; measure the prologue to know it.
    let plen = asm_at(CODE, |a| {
        a.mov(rcx, TRIPS).unwrap();
    })
    .len() as u64;
    let loop_head = CODE + plen;

    let code = asm_at(CODE, |a| {
        let mut head = a.create_label();
        a.mov(rcx, TRIPS).unwrap();
        a.set_label(&mut head).unwrap();
        a.dec(rcx).unwrap();
        a.jnz(head).unwrap();
    });
    let end = CODE + code.len() as u64;

    let hook = VmObserveHook::new(None, Some(loop_head));
    let obs = hook.observations();
    let mut emu = emu_over(&[(CODE, code)]);
    emu.hook = Some(Box::new(hook));

    assert_eq!(emu.run_until(end, 1000), Stop::Reached, "loop ran to completion");

    let mut facts = ObservedFacts::new();
    {
        let o = obs.borrow();
        assert_eq!(o.back_edges, TRIPS - 1, "back-edge taken TRIPS-1 times");
        assert!(o.handlers.is_empty(), "no dispatcher configured");
        o.seed_loop_depth(&mut facts);
    }

    assert_eq!(facts.loop_depths.len(), 1, "one observed depth seeded");
    let d = facts.loop_depths[0];
    assert_eq!(d.loop_head, loop_head);
    assert_eq!(d.depth, TRIPS - 1, "observed concrete depth for this input");
    assert_eq!(d.provenance, Observed, "stamped Observed, never sound output");
}

/// A threaded VM whose VIP lives only in a stack slot (`[rsp+0x20]`): three
/// handlers chain through a `push rbp; ret` stub, each stepping the slot by -4.
/// The survey must find the stub and the memory VIP on its own, and the
/// observer must then record the real bytecode edges.
#[test]
fn survey_then_observe_a_memory_resident_vip_vm() {
    const STUB: u64 = 0x1100;
    const HANDLERS: [u64; 3] = [0x1200, 0x1240, 0x1280];
    const EXIT: u64 = 0x1400;
    const STACK: u64 = 0x20_0000;
    const BYTECODE_TOP: u64 = 0x1_8000;
    const DISPATCHES: u64 = 3_000;

    let entry = asm_at(CODE, |a| {
        a.mov(rbp, HANDLERS[0]).unwrap();
        a.jmp(STUB).unwrap();
    });
    let stub = asm_at(STUB, |a| {
        a.push(rbp).unwrap();
        a.ret().unwrap();
    });
    let mut blobs = vec![(CODE, entry), (STUB, stub)];
    for (k, &h) in HANDLERS.iter().enumerate() {
        let next = HANDLERS[(k + 1) % 3];
        blobs.push((
            h,
            asm_at(h, |a| {
                a.sub(qword_ptr(rsp + 0x20), 4).unwrap();
                a.dec(rcx).unwrap();
                a.jz(EXIT).unwrap();
                a.mov(rbp, next).unwrap();
                a.jmp(STUB).unwrap();
            }),
        ));
    }
    let fresh = |hook: Box<dyn vmpx_emulate::Hook>| {
        let mut emu = emu_over(&blobs);
        emu.mem.map(STACK, 0x1000, Perm::RW).unwrap();
        emu.mem.write_uint(STACK + 0x800 + 0x20, 8, BYTECODE_TOP).unwrap();
        emu.cpu.set_reg(iced_x86::Register::RSP, STACK + 0x800);
        emu.cpu.set_reg(iced_x86::Register::RCX, DISPATCHES);
        emu.hook = Some(hook);
        emu
    };

    // Survey run: discover the stub and the VIP location.
    let survey_hook = SurveyHook::new();
    let survey = survey_hook.survey();
    let mut emu = fresh(Box::new(survey_hook));
    assert_eq!(emu.run_until(EXIT, 100_000), Stop::Reached);
    let vms = survey.borrow().discover((0x1_0000, 0x2_0000));
    assert_eq!(vms.len(), 1, "{vms:?}");
    assert_eq!(vms[0].entries.iter().map(|e| e.va).collect::<Vec<_>>(), vec![STUB + 1], "the stub's ret");
    assert_eq!(
        vms[0].vip.map(|v| v.loc),
        Some(VipLoc::Mem { base: iced_x86::Register::RSP, disp: 0x20 })
    );

    // Observe run on a fresh emulator with the discovered specs.
    let hook = VmObserveHook::with_instances(0, vms[0].specs(0, 1, (0x1_0000, 0x2_0000)), None);
    let obs = hook.observations();
    let mut emu = fresh(Box::new(hook));
    assert_eq!(emu.run_until(EXIT, 100_000), Stop::Reached);

    let o = obs.borrow();
    assert_eq!(o.unattributed, 0);
    assert_eq!(o.edges.len() as u64, DISPATCHES - 1, "one edge per consecutive dispatch pair");
    assert!(o.edges.iter().all(|e| e.to_vip + 4 == e.from_vip), "VIP steps -4 every handler");
    let first = o.edges[0];
    assert_eq!((first.from_vip, first.to_vip, first.handler_va), (BYTECODE_TOP, BYTECODE_TOP - 4, HANDLERS[0]));
    assert_eq!(o.edges[1].handler_va, HANDLERS[1], "handlers follow the threaded order");
}
