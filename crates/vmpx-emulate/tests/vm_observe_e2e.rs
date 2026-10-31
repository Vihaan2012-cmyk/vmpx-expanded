//! Layer-5 end-to-end: drive hand-assembled guest code through a
//! [`VmObserveHook`] and assert (1) a dispatcher-style indirect-transfer
//! sequence records the visited handler targets in order, and (2) a loop that
//! iterates a concrete number of times records that exact back-edge depth into
//! `ObservedFacts.loop_depths`.

use iced_x86::code_asm::*;
use vmpx_emulate::{
    Cpu, Emulator, Memory, Observed, ObservedFacts, Perm, Stop, VmObserveHook,
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
