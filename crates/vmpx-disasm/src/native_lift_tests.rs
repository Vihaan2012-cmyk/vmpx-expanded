#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use iced_x86::code_asm::{
    CodeAssembler, ah, al, ax, bl, bx, byte_ptr, cl, cx, dl, dword_ptr, eax, ebx, ecx, edi, edx,
    esi, ptr, qword_ptr, r8b, rax, rbp, rbx, rcx, rdi, rdx, rsi, rsp,
};

use crate::{
    Algebra, BinOp, BlockSemantics, Flag, FlagState, Gpr, LiftError, LiftOptions, NativeAlgebra,
    NativeExpr, Resize, Width, lift_bytes, lift_bytes_with,
};

type Sem = BlockSemantics<NativeExpr>;

fn lift(build: impl FnOnce(&mut CodeAssembler)) -> Result<Sem, LiftError> {
    let mut asm = CodeAssembler::new(64).expect("assembler");
    build(&mut asm);
    let bytes = asm.assemble(0x1000).expect("assemble");
    lift_bytes(&NativeAlgebra, &bytes, 0x1000)
}

fn ok(build: impl FnOnce(&mut CodeAssembler)) -> Sem {
    lift(build).expect("lift")
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn input_value(seed: u64, name: &str) -> u64 {
    let tag = name.bytes().fold(0u64, |acc, b| {
        acc.wrapping_mul(131).wrapping_add(u64::from(b))
    });
    let value = mix(seed.wrapping_mul(0x1_0000_0001).wrapping_add(tag));
    if matches!(
        name,
        "cf_in" | "pf_in" | "af_in" | "zf_in" | "sf_in" | "of_in"
    ) {
        value & 1
    } else {
        value
    }
}

fn eval_expr(expr: &NativeExpr, seed: u64) -> u64 {
    expr.eval(&|name| input_value(seed, name), &|addr, _| {
        mix(addr ^ 0xABCD)
    })
    .expect("evaluable")
}

fn reg(sem: &Sem, seed: u64, gpr: Gpr) -> u64 {
    eval_expr(&sem.regs[&gpr], seed)
}

fn inp(seed: u64, gpr: Gpr) -> u64 {
    input_value(seed, &gpr.input_name())
}

const SEEDS: std::ops::Range<u64> = 0..32;

fn input(name: &str) -> NativeExpr {
    NativeAlgebra.input(Width::W64, name)
}

fn trunc(w: Width, e: NativeExpr) -> NativeExpr {
    NativeAlgebra.resize(Resize::Truncate, w, e).unwrap()
}

fn zext64(e: NativeExpr) -> NativeExpr {
    NativeAlgebra
        .resize(Resize::ZeroExtend, Width::W64, e)
        .unwrap()
}

fn bin(op: BinOp, l: NativeExpr, r: NativeExpr) -> NativeExpr {
    NativeAlgebra.binary(op, l, r).unwrap()
}

#[test]
fn lea_eax_sums_in_64_then_truncates_and_zero_extends() {
    let sem = ok(|a| a.lea(eax, ptr(rcx + rdx)).unwrap());
    let expected = zext64(trunc(
        Width::W32,
        bin(BinOp::Add, input("rcx_in"), input("rdx_in")),
    ));
    assert_eq!(sem.regs[&Gpr::Rax], expected);
    assert!(sem.written.contains(&Gpr::Rax));
    assert_eq!(sem.written.len(), 1);
    assert_eq!(sem.regs[&Gpr::Rcx], input("rcx_in"));
}

#[test]
fn lea_scale_disp_and_16_bit_destination() {
    let sem = ok(|a| {
        a.lea(rax, ptr(rcx + rdx * 4 + 8)).unwrap();
        a.lea(bx, ptr(rcx - 1)).unwrap();
    });
    for seed in SEEDS {
        let (c, d) = (inp(seed, Gpr::Rcx), inp(seed, Gpr::Rdx));
        assert_eq!(
            reg(&sem, seed, Gpr::Rax),
            c.wrapping_add(d.wrapping_mul(4)).wrapping_add(8)
        );
        let b = inp(seed, Gpr::Rbx);
        let low = u64::from((c.wrapping_sub(1)) as u16);
        assert_eq!(reg(&sem, seed, Gpr::Rbx), (b & !0xFFFF) | low);
    }
}

#[test]
fn mov_then_xor_32_bit_zero_extends() {
    let sem = ok(|a| {
        a.mov(eax, ecx).unwrap();
        a.xor(eax, edx).unwrap();
    });
    let expected = zext64(bin(
        BinOp::Xor,
        trunc(Width::W32, input("rcx_in")),
        trunc(Width::W32, input("rdx_in")),
    ));
    assert_eq!(sem.regs[&Gpr::Rax], expected);
    for seed in SEEDS {
        let want = (inp(seed, Gpr::Rcx) ^ inp(seed, Gpr::Rdx)) & 0xFFFF_FFFF;
        assert_eq!(reg(&sem, seed, Gpr::Rax), want);
    }
}

#[test]
fn substitution_identities_round_trip() {
    // neg(x) == not(x) + 1
    let neg = ok(|a| {
        a.mov(eax, ecx).unwrap();
        a.not(eax).unwrap();
        a.add(eax, 1).unwrap();
    });
    // x ^ y == (x | y) - (x & y)
    let xor = ok(|a| {
        a.mov(eax, ecx).unwrap();
        a.or(eax, edx).unwrap();
        a.mov(ebx, ecx).unwrap();
        a.and(ebx, edx).unwrap();
        a.sub(eax, ebx).unwrap();
    });
    // x + y == x - (~y) - 1
    let add = ok(|a| {
        a.mov(rax, rcx).unwrap();
        a.mov(rbx, rdx).unwrap();
        a.not(rbx).unwrap();
        a.sub(rax, rbx).unwrap();
        a.sub(rax, 1).unwrap();
    });
    for seed in SEEDS {
        let (c, d) = (inp(seed, Gpr::Rcx), inp(seed, Gpr::Rdx));
        assert_eq!(reg(&neg, seed, Gpr::Rax), (c as u32).wrapping_neg() as u64);
        assert_eq!(reg(&xor, seed, Gpr::Rax), (c ^ d) & 0xFFFF_FFFF);
        assert_eq!(reg(&add, seed, Gpr::Rax), c.wrapping_add(d));
    }
    let direct = ok(|a| {
        a.mov(eax, ecx).unwrap();
        a.not(eax).unwrap();
    });
    let expected = zext64(
        NativeAlgebra
            .unary(crate::UnOp::Not, trunc(Width::W32, input("rcx_in")))
            .unwrap(),
    );
    assert_eq!(direct.regs[&Gpr::Rax], expected);
}

#[test]
fn sub_register_writes_preserve_untouched_bits() {
    let sem = ok(|a| {
        a.mov(al, cl).unwrap();
        a.mov(ah, dl).unwrap();
        a.mov(bx, cx).unwrap();
        a.mov(dl, ah).unwrap();
    });
    for seed in SEEDS {
        let (a0, b0, c, d0) = (
            inp(seed, Gpr::Rax),
            inp(seed, Gpr::Rbx),
            inp(seed, Gpr::Rcx),
            inp(seed, Gpr::Rdx),
        );
        let mut rax_v = (a0 & !0xFF) | (c & 0xFF);
        rax_v = (rax_v & !0xFF00) | ((d0 & 0xFF) << 8);
        assert_eq!(reg(&sem, seed, Gpr::Rax), rax_v);
        assert_eq!(reg(&sem, seed, Gpr::Rbx), (b0 & !0xFFFF) | (c & 0xFFFF));
        assert_eq!(
            reg(&sem, seed, Gpr::Rdx),
            (d0 & !0xFF) | ((rax_v >> 8) & 0xFF)
        );
    }
}

#[test]
fn extensions() {
    let sem = ok(|a| {
        a.movsx(rax, cl).unwrap();
        a.movsxd(rbx, ecx).unwrap();
        a.movzx(edx, cx).unwrap();
        a.movsx(ecx, bl).unwrap();
    });
    for seed in SEEDS {
        let c = inp(seed, Gpr::Rcx);
        assert_eq!(reg(&sem, seed, Gpr::Rax), i64::from(c as i8) as u64);
        assert_eq!(reg(&sem, seed, Gpr::Rbx), i64::from(c as i32) as u64);
        assert_eq!(reg(&sem, seed, Gpr::Rdx), c & 0xFFFF);
        // bl was overwritten by movsxd, so its low byte is c's low byte
        assert_eq!(
            reg(&sem, seed, Gpr::Rcx),
            u64::from(i32::from(c as i8) as u32)
        );
    }
}

#[test]
fn shifts_and_rotates_use_hardware_count_semantics() {
    let sem = ok(|a| {
        a.shl(eax, 3).unwrap();
        a.sar(rbx, 70).unwrap();
        a.rol(cl, 9).unwrap();
        a.shr(dl, 9).unwrap();
        a.ror(ax, 1).unwrap();
    });
    for seed in SEEDS {
        let (a0, b0, c, d) = (
            inp(seed, Gpr::Rax),
            inp(seed, Gpr::Rbx),
            inp(seed, Gpr::Rcx),
            inp(seed, Gpr::Rdx),
        );
        let eax_v = u64::from((a0 as u32) << 3);
        let ax_v = ((eax_v as u16).rotate_right(1)) as u64;
        assert_eq!(reg(&sem, seed, Gpr::Rax), (eax_v & !0xFFFF) | ax_v);
        assert_eq!(reg(&sem, seed, Gpr::Rbx), ((b0 as i64) >> 6) as u64);
        assert_eq!(
            reg(&sem, seed, Gpr::Rcx),
            (c & !0xFF) | u64::from((c as u8).rotate_left(1))
        );
        assert_eq!(reg(&sem, seed, Gpr::Rdx), d & !0xFF);
    }
}

#[test]
fn shift_forms_that_are_not_safely_modeled_fail_closed() {
    assert!(matches!(
        lift(|a| a.shl(eax, 32).unwrap()),
        Err(LiftError::Unsupported { .. })
    ));
    assert!(lift(|a| a.shl(rax, 64).unwrap()).is_ok());
}

#[test]
fn imul_inc_dec_neg() {
    let sem = ok(|a| {
        a.imul_2(eax, ecx).unwrap();
        a.imul_3(rbx, rdx, 7).unwrap();
        a.inc(cl).unwrap();
        a.dec(edx).unwrap();
        a.neg(rcx).unwrap();
    });
    for seed in SEEDS {
        let (a0, c, d) = (
            inp(seed, Gpr::Rax),
            inp(seed, Gpr::Rcx),
            inp(seed, Gpr::Rdx),
        );
        assert_eq!(
            reg(&sem, seed, Gpr::Rax),
            u64::from((a0 as u32).wrapping_mul(c as u32))
        );
        assert_eq!(reg(&sem, seed, Gpr::Rbx), d.wrapping_mul(7));
        let c1 = (c & !0xFF) | u64::from((c as u8).wrapping_add(1));
        assert_eq!(reg(&sem, seed, Gpr::Rcx), c1.wrapping_neg());
        assert_eq!(
            reg(&sem, seed, Gpr::Rdx),
            u64::from((d as u32).wrapping_sub(1))
        );
    }
}

#[test]
fn xchg_swaps_and_zero_extends_32_bit() {
    let sem = ok(|a| a.xchg(eax, ecx).unwrap());
    for seed in SEEDS {
        let (a0, c) = (inp(seed, Gpr::Rax), inp(seed, Gpr::Rcx));
        assert_eq!(reg(&sem, seed, Gpr::Rax), c & 0xFFFF_FFFF);
        assert_eq!(reg(&sem, seed, Gpr::Rcx), a0 & 0xFFFF_FFFF);
    }
    let mem = ok(|a| a.xchg(qword_ptr(rax), rax).unwrap());
    assert_eq!(mem.writes.len(), 1);
    assert_eq!(mem.writes[0].address, input("rax_in"));
    assert_eq!(mem.writes[0].value, input("rax_in"));
    assert_eq!(
        mem.regs[&Gpr::Rax],
        NativeAlgebra.load(Width::W64, input("rax_in"))
    );
}

#[test]
fn nops_and_junk_are_identity() {
    let sem = ok(|a| {
        a.nop().unwrap();
        a.nop_1(eax).unwrap();
        a.add(rax, 0).unwrap();
        a.xor(rcx, 0).unwrap();
        a.mov(rdx, rdx).unwrap();
    });
    for g in [Gpr::Rax, Gpr::Rcx, Gpr::Rdx] {
        assert_eq!(sem.regs[&g], input(&g.input_name()));
    }
}

#[test]
fn memory_store_forwarding_disjointness_and_aliasing() {
    let sem = ok(|a| {
        a.mov(ptr(rsp + 8), ecx).unwrap();
        a.mov(eax, ptr(rsp + 8)).unwrap();
        a.mov(rbx, qword_ptr(rsp + 16)).unwrap();
        a.add(ptr(rsp + 8), edx).unwrap();
    });
    for seed in SEEDS {
        assert_eq!(reg(&sem, seed, Gpr::Rax), inp(seed, Gpr::Rcx) & 0xFFFF_FFFF);
    }
    assert_eq!(sem.writes.len(), 2);
    assert_eq!(sem.reads.len(), 1);
    assert_eq!(
        sem.regs[&Gpr::Rbx],
        NativeAlgebra.load(
            Width::W64,
            bin(
                BinOp::Add,
                input("rsp_in"),
                NativeAlgebra.constant(Width::W64, 16)
            )
        )
    );
    let sum = &sem.writes[1].value;
    for seed in SEEDS {
        let want = (inp(seed, Gpr::Rcx).wrapping_add(inp(seed, Gpr::Rdx))) & 0xFFFF_FFFF;
        assert_eq!(eval_expr(sum, seed), want);
    }

    assert!(matches!(
        lift(|a| {
            a.mov(qword_ptr(rsp), rcx).unwrap();
            a.mov(rax, qword_ptr(rdx)).unwrap();
        }),
        Err(LiftError::MemoryAliasing { .. })
    ));
    assert!(
        lift(|a| {
            a.mov(qword_ptr(rsp), rcx).unwrap();
            a.mov(eax, ptr(rsp + 8)).unwrap();
        })
        .is_ok()
    );
    let low = ok(|a| {
        a.mov(qword_ptr(rsp), rcx).unwrap();
        a.mov(eax, ptr(rsp)).unwrap();
    });
    for seed in SEEDS {
        assert_eq!(reg(&low, seed, Gpr::Rax), inp(seed, Gpr::Rcx) & 0xFFFF_FFFF);
    }
}

#[test]
fn flags_are_modeled_or_marked_clobbered() {
    let cmp = ok(|a| a.cmp(eax, ecx).unwrap());
    assert!(cmp.written.is_empty());
    let Some(FlagState::Value(zf)) = cmp.flags.get(&Flag::Zf) else {
        panic!("zf");
    };
    let Some(FlagState::Value(cf)) = cmp.flags.get(&Flag::Cf) else {
        panic!("cf");
    };
    for seed in SEEDS {
        let (a0, c) = (inp(seed, Gpr::Rax) as u32, inp(seed, Gpr::Rcx) as u32);
        assert_eq!(eval_expr(zf, seed), u64::from(a0 == c));
        assert_eq!(eval_expr(cf, seed), u64::from(a0 < c));
    }
    let xor = ok(|a| a.xor(eax, eax).unwrap());
    assert!(matches!(xor.flags[&Flag::Af], FlagState::Clobbered));
    let FlagState::Value(zf) = &xor.flags[&Flag::Zf] else {
        panic!()
    };
    assert_eq!(eval_expr(zf, 1), 1);
    let inc = ok(|a| a.inc(eax).unwrap());
    assert!(!inc.flags.contains_key(&Flag::Cf));
    let shl = ok(|a| a.shl(eax, 1).unwrap());
    assert!(matches!(shl.flags[&Flag::Cf], FlagState::Clobbered));
    let mov = ok(|a| a.mov(eax, ecx).unwrap());
    assert!(mov.flags.is_empty());
}

#[test]
fn unmodeled_instructions_fail_closed_naming_the_instruction() {
    type Case = (&'static str, Box<dyn FnOnce(&mut CodeAssembler)>);
    let cases: Vec<Case> = vec![
        ("cpuid", Box::new(|a| a.cpuid().unwrap())),
        ("pushfq", Box::new(|a| a.pushfq().unwrap())),
    ];
    for (name, build) in cases {
        match lift(build) {
            Err(LiftError::Unsupported { instruction, .. }) => {
                assert!(instruction.contains(name), "{instruction}");
            }
            other => panic!("{name}: {other:?}"),
        }
    }
    assert!(matches!(
        lift(|a| a.mov(rax, qword_ptr(0x28).fs()).unwrap()),
        Err(LiftError::Unsupported { .. })
    ));
    assert!(matches!(
        lift_bytes(&NativeAlgebra, &[0xEB, 0x00], 0),
        Err(LiftError::Unsupported { .. })
    ));
    assert!(matches!(
        lift_bytes(&NativeAlgebra, &[0x90, 0x06], 0),
        Err(LiftError::Decode { offset: 1 })
    ));
}

#[test]
fn rip_relative_is_an_absolute_address() {
    let sem = lift_bytes(&NativeAlgebra, &[0x8B, 0x05, 0x20, 0, 0, 0], 0x1000).unwrap();
    assert_eq!(sem.reads.len(), 1);
    assert_eq!(
        sem.reads[0].address,
        NativeAlgebra.constant(Width::W64, 0x1006 + 0x20)
    );
}

fn w_of(bits: u32) -> Width {
    Width::from_bytes((bits / 8) as usize).unwrap()
}

fn eval_with(expr: &NativeExpr, input: &dyn Fn(&str) -> u64) -> Option<u64> {
    expr.eval(input, &|addr, _| mix(addr ^ 0xABCD))
}

fn rsp_off(off: i64) -> NativeExpr {
    bin(
        BinOp::Add,
        input("rsp_in"),
        NativeAlgebra.constant(Width::W64, off as u64),
    )
}

#[test]
fn push_pop_round_trip_restores_values_and_rsp() {
    let sem = ok(|a| {
        a.push(rcx).unwrap();
        a.push(rdx).unwrap();
        a.pop(rax).unwrap();
        a.pop(rbx).unwrap();
    });
    assert_eq!(sem.regs[&Gpr::Rsp], input("rsp_in"));
    assert!(sem.written.contains(&Gpr::Rsp));
    assert!(sem.reads.is_empty());
    assert_eq!(sem.writes.len(), 2);
    assert_eq!(sem.writes[0].address, rsp_off(-8));
    assert_eq!(sem.writes[1].address, rsp_off(-16));
    assert_eq!(sem.writes[0].value, input("rcx_in"));
    assert_eq!(sem.writes[1].value, input("rdx_in"));
    for seed in SEEDS {
        assert_eq!(reg(&sem, seed, Gpr::Rax), inp(seed, Gpr::Rdx));
        assert_eq!(reg(&sem, seed, Gpr::Rbx), inp(seed, Gpr::Rcx));
    }
}

#[test]
fn push_forms_pop_to_memory_and_pop_rsp() {
    let sem = ok(|a| {
        a.push(qword_ptr(rbx)).unwrap();
        a.push(-2i32).unwrap();
        a.push(rsp).unwrap();
        a.pop(rsi).unwrap();
        a.pop(rdi).unwrap();
        a.pop(rax).unwrap();
    });
    assert_eq!(sem.reads.len(), 1);
    assert_eq!(sem.regs[&Gpr::Rsp], input("rsp_in"));
    assert_eq!(
        sem.regs[&Gpr::Rax],
        NativeAlgebra.load(Width::W64, input("rbx_in"))
    );
    for seed in SEEDS {
        assert_eq!(
            reg(&sem, seed, Gpr::Rsi),
            inp(seed, Gpr::Rsp).wrapping_sub(16)
        );
        assert_eq!(reg(&sem, seed, Gpr::Rdi), (-2i64) as u64);
    }

    // Intel: pop [rsp] computes its address after rsp has been incremented.
    let sem = ok(|a| {
        a.push(rcx).unwrap();
        a.pop(qword_ptr(rsp)).unwrap();
    });
    assert_eq!(sem.regs[&Gpr::Rsp], input("rsp_in"));
    assert_eq!(sem.writes.len(), 2);
    assert_eq!(sem.writes[1].address, input("rsp_in"));
    assert_eq!(sem.writes[1].value, input("rcx_in"));

    let sem = ok(|a| {
        a.push(rcx).unwrap();
        a.pop(rsp).unwrap();
    });
    assert_eq!(sem.regs[&Gpr::Rsp], input("rcx_in"));
    assert!(
        lift(|a| {
            a.push(rcx).unwrap();
            a.pop(rsp).unwrap();
            a.push(rax).unwrap();
        })
        .is_err()
    );
}

#[test]
fn push_pop_16_bit_operands() {
    let sem = ok(|a| {
        a.push(cx).unwrap();
        a.pop(bx).unwrap();
    });
    assert_eq!(sem.regs[&Gpr::Rsp], input("rsp_in"));
    assert_eq!(sem.writes[0].width, Width::W16);
    assert_eq!(sem.writes[0].address, rsp_off(-2));
    for seed in SEEDS {
        let (b0, c) = (inp(seed, Gpr::Rbx), inp(seed, Gpr::Rcx));
        assert_eq!(reg(&sem, seed, Gpr::Rbx), (b0 & !0xFFFF) | (c & 0xFFFF));
    }
}

#[test]
fn rsp_tracks_constant_adjustments_and_fails_closed_otherwise() {
    let sem = ok(|a| {
        a.sub(rsp, 0x28).unwrap();
        a.push(rcx).unwrap();
        a.mov(rax, qword_ptr(rsp)).unwrap();
        a.mov(rbx, qword_ptr(rsp + 8)).unwrap();
        a.add(rsp, 0x28).unwrap();
    });
    assert_eq!(sem.regs[&Gpr::Rax], input("rcx_in"));
    assert_eq!(
        sem.regs[&Gpr::Rbx],
        NativeAlgebra.load(Width::W64, rsp_off(-0x28))
    );
    assert_eq!(sem.regs[&Gpr::Rsp], rsp_off(-8));
    assert_eq!(sem.writes[0].address, rsp_off(-0x30));
    let untracked = lift(|a| {
        a.mov(rsp, rbp).unwrap();
        a.push(rax).unwrap();
    });
    assert!(matches!(untracked, Err(LiftError::Unsupported { .. })));
    let masked = lift(|a| {
        a.and(rsp, -16).unwrap();
        a.pop(rax).unwrap();
    });
    assert!(matches!(masked, Err(LiftError::Unsupported { .. })));
}

struct Ref {
    result: u64,
    cf: u64,
    of: u64,
    zf: u64,
    sf: u64,
    pf: u64,
    af: u64,
}

fn carry_ref(bits: u32, sub: bool, a: u64, b: u64, c: u64) -> Ref {
    let w = w_of(bits);
    let (a, b) = (w.truncate(a), w.truncate(b));
    let (result, cf) = if sub {
        (
            w.truncate(a.wrapping_sub(b).wrapping_sub(c)),
            u128::from(a) < u128::from(b) + u128::from(c),
        )
    } else {
        let full = u128::from(a) + u128::from(b) + u128::from(c);
        (w.truncate(full as u64), (full >> bits) & 1 == 1)
    };
    let top = |x: u64| (x >> (bits - 1)) & 1;
    let of = if sub {
        top((a ^ b) & (a ^ result))
    } else {
        top((a ^ result) & (b ^ result))
    };
    Ref {
        result,
        cf: u64::from(cf),
        of,
        zf: u64::from(result == 0),
        sf: top(result),
        pf: u64::from((result & 0xFF).count_ones().is_multiple_of(2)),
        af: ((a ^ b ^ result) >> 4) & 1,
    }
}

fn check_flags(sem: &Sem, seed: u64, want: &Ref) {
    for (flag, v) in [
        (Flag::Cf, want.cf),
        (Flag::Of, want.of),
        (Flag::Zf, want.zf),
        (Flag::Sf, want.sf),
        (Flag::Pf, want.pf),
        (Flag::Af, want.af),
    ] {
        let FlagState::Value(e) = &sem.flags[&flag] else {
            panic!("{flag:?}")
        };
        assert_eq!(eval_expr(e, seed), v, "{flag:?}");
    }
}

#[test]
fn adc_sbb_use_carry_in_and_set_flags() {
    let adc32 = ok(|a| a.adc(eax, ecx).unwrap());
    let sbb64 = ok(|a| a.sbb(rbx, rdx).unwrap());
    let adc8 = ok(|a| a.adc(cl, dl).unwrap());
    let sbb8 = ok(|a| a.sbb(al, bl).unwrap());
    for seed in SEEDS {
        let cin = input_value(seed, "cf_in");
        let (a0, b0, c, d) = (
            inp(seed, Gpr::Rax),
            inp(seed, Gpr::Rbx),
            inp(seed, Gpr::Rcx),
            inp(seed, Gpr::Rdx),
        );
        let w = carry_ref(32, false, a0, c, cin);
        assert_eq!(reg(&adc32, seed, Gpr::Rax), w.result);
        check_flags(&adc32, seed, &w);
        let w = carry_ref(64, true, b0, d, cin);
        assert_eq!(reg(&sbb64, seed, Gpr::Rbx), w.result);
        check_flags(&sbb64, seed, &w);
        let w = carry_ref(8, false, c, d, cin);
        assert_eq!(reg(&adc8, seed, Gpr::Rcx), (c & !0xFF) | w.result);
        check_flags(&adc8, seed, &w);
        let w = carry_ref(8, true, a0, b0, cin);
        assert_eq!(reg(&sbb8, seed, Gpr::Rax), (a0 & !0xFF) | w.result);
        check_flags(&sbb8, seed, &w);
    }
}

#[test]
fn adc_consumes_an_in_block_carry_and_memory_destination() {
    let sem = ok(|a| {
        a.cmp(rax, rcx).unwrap();
        a.adc(rdx, 0).unwrap();
    });
    let mem = ok(|a| a.adc(qword_ptr(rbx), rcx).unwrap());
    for seed in SEEDS {
        let (a0, c, d) = (
            inp(seed, Gpr::Rax),
            inp(seed, Gpr::Rcx),
            inp(seed, Gpr::Rdx),
        );
        assert_eq!(reg(&sem, seed, Gpr::Rdx), d.wrapping_add(u64::from(a0 < c)));
        let old = mix(inp(seed, Gpr::Rbx) ^ 0xABCD);
        let want = old.wrapping_add(c).wrapping_add(input_value(seed, "cf_in"));
        assert_eq!(eval_expr(&mem.writes[0].value, seed), want);
    }
    let shifted = lift(|a| {
        a.shl(eax, 1).unwrap();
        a.adc(ebx, ecx).unwrap();
    });
    assert!(matches!(shifted, Err(LiftError::Unsupported { .. })));
    let multiplied = lift(|a| {
        a.imul_2(eax, ecx).unwrap();
        a.sbb(ebx, ecx).unwrap();
    });
    assert!(matches!(multiplied, Err(LiftError::Unsupported { .. })));
}

#[allow(clippy::struct_excessive_bools)]
struct CmpFlags {
    cf: bool,
    zf: bool,
    sf: bool,
    of: bool,
    pf: bool,
}

type FlagPred = fn(&CmpFlags) -> bool;

fn cmp_flags(a: u32, c: u32) -> CmpFlags {
    let r = a.wrapping_sub(c);
    CmpFlags {
        cf: a < c,
        zf: r == 0,
        sf: r >> 31 == 1,
        of: (((a ^ c) & (a ^ r)) >> 31) & 1 == 1,
        pf: (r & 0xFF).count_ones().is_multiple_of(2),
    }
}

macro_rules! cc_cases {
    ($(($cmov:ident, $set:ident, $pred:expr)),* $(,)?) => {
        vec![$((
            stringify!($cmov),
            Box::new(|a: &mut CodeAssembler| {
                a.cmp(eax, ecx).unwrap();
                a.$cmov(edx, ebx).unwrap();
                a.$set(r8b).unwrap();
            }) as Box<dyn Fn(&mut CodeAssembler)>,
            $pred as FlagPred,
        )),*]
    };
}

#[test]
fn cmovcc_and_setcc_follow_every_condition() {
    let cases = cc_cases![
        (cmovo, seto, |f| f.of),
        (cmovno, setno, |f| !f.of),
        (cmovb, setb, |f| f.cf),
        (cmovae, setae, |f| !f.cf),
        (cmove, sete, |f| f.zf),
        (cmovne, setne, |f| !f.zf),
        (cmovbe, setbe, |f| f.cf || f.zf),
        (cmova, seta, |f| !(f.cf || f.zf)),
        (cmovs, sets, |f| f.sf),
        (cmovns, setns, |f| !f.sf),
        (cmovp, setp, |f| f.pf),
        (cmovnp, setnp, |f| !f.pf),
        (cmovl, setl, |f| f.sf != f.of),
        (cmovge, setge, |f| f.sf == f.of),
        (cmovle, setle, |f| f.zf || f.sf != f.of),
        (cmovg, setg, |f| !f.zf && f.sf == f.of),
    ];
    for (name, build, pred) in cases {
        let sem = ok(|a| build(a));
        for seed in SEEDS {
            // random 32-bit values are never equal, so also force rcx == rax
            for equal in [false, true] {
                let f = |n: &str| {
                    if equal && n == "rcx_in" {
                        input_value(seed, "rax_in")
                    } else {
                        input_value(seed, n)
                    }
                };
                let (a0, c) = (f("rax_in"), f("rcx_in"));
                let (b, d, r) = (f("rbx_in"), f("rdx_in"), f("r8_in"));
                let taken = pred(&cmp_flags(a0 as u32, c as u32));
                let want_d = if taken { b } else { d } & 0xFFFF_FFFF;
                let got_d = eval_with(&sem.regs[&Gpr::Rdx], &f).unwrap();
                let got_r = eval_with(&sem.regs[&Gpr::R8], &f).unwrap();
                assert_eq!(got_d, want_d, "{name} cmov seed {seed} eq {equal}");
                assert_eq!(
                    got_r,
                    (r & !0xFF) | u64::from(taken),
                    "{name} set seed {seed} eq {equal}"
                );
            }
        }
    }
}

#[test]
fn cmov_setcc_read_entry_flags_fold_constants_and_fail_closed() {
    let sem = ok(|a| {
        a.cmove(rax, rcx).unwrap();
        a.sete(byte_ptr(rdx)).unwrap();
    });
    assert_eq!(sem.writes[0].width, Width::W8);
    for seed in SEEDS {
        let zf = input_value(seed, "zf_in");
        let want = if zf == 1 {
            inp(seed, Gpr::Rcx)
        } else {
            inp(seed, Gpr::Rax)
        };
        assert_eq!(reg(&sem, seed, Gpr::Rax), want);
        assert_eq!(eval_expr(&sem.writes[0].value, seed), zf);
    }
    let sem = ok(|a| {
        a.xor(eax, eax).unwrap();
        a.cmovb(ebx, ecx).unwrap();
        a.cmovae(edx, ecx).unwrap();
    });
    assert_eq!(
        sem.regs[&Gpr::Rbx],
        zext64(trunc(Width::W32, input("rbx_in")))
    );
    assert_eq!(
        sem.regs[&Gpr::Rdx],
        zext64(trunc(Width::W32, input("rcx_in")))
    );
    let cmov = lift(|a| {
        a.shl(eax, 1).unwrap();
        a.cmovb(ebx, ecx).unwrap();
    });
    assert!(matches!(cmov, Err(LiftError::Unsupported { .. })));
    let set = lift(|a| {
        a.shl(eax, 1).unwrap();
        a.sete(dl).unwrap();
    });
    assert!(matches!(set, Err(LiftError::Unsupported { .. })));
}

fn wide_mul_ref(bits: u32, signed: bool, a: u64, b: u64) -> (u64, u64) {
    let w = w_of(bits);
    let p: u128 = if signed {
        (i128::from(w.sign_extend(a) as i64) * i128::from(w.sign_extend(b) as i64)) as u128
    } else {
        u128::from(w.truncate(a)) * u128::from(w.truncate(b))
    };
    (w.truncate(p as u64), w.truncate((p >> bits) as u64))
}

type Build = Box<dyn Fn(&mut CodeAssembler)>;

#[test]
fn one_operand_mul_and_imul_write_the_wide_product() {
    let cases: Vec<(u32, bool, Build)> = vec![
        (8, false, Box::new(|a| a.mul(cl).unwrap())),
        (16, false, Box::new(|a| a.mul(cx).unwrap())),
        (32, false, Box::new(|a| a.mul(ecx).unwrap())),
        (64, false, Box::new(|a| a.mul(rcx).unwrap())),
        (8, true, Box::new(|a| a.imul(cl).unwrap())),
        (16, true, Box::new(|a| a.imul(cx).unwrap())),
        (32, true, Box::new(|a| a.imul(ecx).unwrap())),
        (64, true, Box::new(|a| a.imul(rcx).unwrap())),
    ];
    let corners = [
        (u64::MAX, u64::MAX, 7),
        (1 << 63, u64::MAX, 7),
        (0x8000_0000, 0x8000_0000, 7),
        (0x80, 0x80, 7),
        (0x8000, 0xFFFF, 7),
    ];
    for (bits, signed, build) in &cases {
        let sem = ok(|a| build(a));
        assert!(matches!(sem.flags[&Flag::Cf], FlagState::Clobbered));
        let mut inputs: Vec<(u64, u64, u64)> = corners.to_vec();
        inputs.extend(SEEDS.map(|s| (mix(s), mix(s + 100), mix(s + 200))));
        for (a0, c, d) in inputs {
            let f = |n: &str| match n {
                "rax_in" => a0,
                "rcx_in" => c,
                "rdx_in" => d,
                _ => 0,
            };
            let (lo, hi) = wide_mul_ref(*bits, *signed, a0, c);
            let got = |g: Gpr| eval_with(&sem.regs[&g], &f).unwrap();
            let ctx = format!("{bits} {signed} {a0:x} {c:x}");
            match bits {
                8 => {
                    assert_eq!(got(Gpr::Rax), (a0 & !0xFFFF) | (hi << 8) | lo, "{ctx}");
                    assert_eq!(got(Gpr::Rdx), d, "{ctx}");
                }
                16 => {
                    assert_eq!(got(Gpr::Rax), (a0 & !0xFFFF) | lo, "{ctx}");
                    assert_eq!(got(Gpr::Rdx), (d & !0xFFFF) | hi, "{ctx}");
                }
                _ => {
                    assert_eq!(got(Gpr::Rax), lo, "{ctx}");
                    assert_eq!(got(Gpr::Rdx), hi, "{ctx}");
                }
            }
        }
    }
}

#[test]
fn div_and_idiv_consume_the_wide_dividend() {
    let cases: Vec<(u32, bool, Build)> = vec![
        (8, false, Box::new(|a| a.div(cl).unwrap())),
        (16, false, Box::new(|a| a.div(cx).unwrap())),
        (32, false, Box::new(|a| a.div(ecx).unwrap())),
        (64, false, Box::new(|a| a.div(rcx).unwrap())),
        (8, true, Box::new(|a| a.idiv(cl).unwrap())),
        (16, true, Box::new(|a| a.idiv(cx).unwrap())),
        (32, true, Box::new(|a| a.idiv(ecx).unwrap())),
        (64, true, Box::new(|a| a.idiv(rcx).unwrap())),
    ];
    for (bits, signed, build) in &cases {
        let w = w_of(*bits);
        let sem = ok(|a| build(a));
        let mut checked = 0;
        for s in 0..64u64 {
            let (r1, r2, r3) = (mix(s), mix(s + 1000), mix(s + 2000));
            let (j1, j2, j3) = (mix(s + 3000), mix(s + 4000), mix(s + 5000));
            let d = w.truncate(r1).max(1);
            // dividend = q*d + r (feasible quotient by construction); expected values come from Rust division
            let (dividend, q, r) = if *signed {
                let dsx = i128::from(w.sign_extend(d) as i64);
                let qsx = i128::from(w.sign_extend(r2) as i64);
                let prod = qsx * dsx;
                let mag = i128::from(r3) % dsx.abs();
                let n = prod + if prod < 0 { -mag } else { mag };
                let half = 1i128 << (bits - 1);
                if n / dsx < -half || n / dsx >= half {
                    continue;
                }
                (
                    n as u128,
                    w.truncate((n / dsx) as u64),
                    w.truncate((n % dsx) as u64),
                )
            } else {
                let n = u128::from(w.truncate(r2)) * u128::from(d) + u128::from(r3) % u128::from(d);
                (
                    n,
                    w.truncate((n / u128::from(d)) as u64),
                    w.truncate((n % u128::from(d)) as u64),
                )
            };
            let (hi, lo) = (
                w.truncate((dividend >> bits) as u64),
                w.truncate(dividend as u64),
            );
            let (rax_in, rdx_in) = match bits {
                8 => ((j1 & !0xFFFF) | (hi << 8) | lo, j2),
                16 => ((j1 & !0xFFFF) | lo, (j2 & !0xFFFF) | hi),
                32 => ((j1 << 32) | lo, (j2 << 32) | hi),
                _ => (lo, hi),
            };
            let rcx_in = (j3 & !w.mask()) | d;
            let f = |n: &str| match n {
                "rax_in" => rax_in,
                "rdx_in" => rdx_in,
                "rcx_in" => rcx_in,
                _ => 0,
            };
            let got = |g: Gpr| eval_with(&sem.regs[&g], &f).unwrap();
            let ctx = format!("{bits} {signed} seed {s}");
            match bits {
                8 => {
                    assert_eq!(got(Gpr::Rax), (rax_in & !0xFFFF) | (r << 8) | q, "{ctx}");
                    assert_eq!(got(Gpr::Rdx), rdx_in, "{ctx}");
                }
                16 => {
                    assert_eq!(got(Gpr::Rax), (rax_in & !0xFFFF) | q, "{ctx}");
                    assert_eq!(got(Gpr::Rdx), (rdx_in & !0xFFFF) | r, "{ctx}");
                }
                _ => {
                    assert_eq!(got(Gpr::Rax), q, "{ctx}");
                    assert_eq!(got(Gpr::Rdx), r, "{ctx}");
                }
            }
            checked += 1;
        }
        assert!(checked > 20, "{bits} {signed}: only {checked} cases");
    }
}

#[test]
fn div_is_undefined_where_hardware_faults() {
    let udiv = ok(|a| a.div(ecx).unwrap());
    let idiv = ok(|a| a.idiv(ecx).unwrap());
    let eval = |sem: &Sem, rax_in: u64, rdx_in: u64, rcx_in: u64| {
        let f = |n: &str| match n {
            "rax_in" => rax_in,
            "rdx_in" => rdx_in,
            "rcx_in" => rcx_in,
            _ => 0,
        };
        eval_with(&sem.regs[&Gpr::Rax], &f)
    };
    assert_eq!(eval(&udiv, 100, 0, 7), Some(14));
    assert_eq!(eval(&udiv, 100, 0, 0), None);
    assert_eq!(eval(&udiv, 0, 5, 3), None);
    assert_eq!(eval(&idiv, 0x8000_0000, 0, 1), None);
    assert_eq!(eval(&idiv, 0x8000_0000, 0xFFFF_FFFF, 0xFFFF_FFFF), None);
    assert_eq!(
        eval(&idiv, u64::from((-100i32) as u32), 0xFFFF_FFFF, 7),
        Some(u64::from((-14i32) as u32))
    );
}

type Case = (&'static str, Box<dyn Fn(&mut CodeAssembler)>);

fn lift_assuming(build: impl FnOnce(&mut CodeAssembler)) -> Result<Sem, LiftError> {
    let mut asm = CodeAssembler::new(64).expect("assembler");
    build(&mut asm);
    let bytes = asm.assemble(0x1000).expect("assemble");
    let options = LiftOptions {
        assume_stack_disjoint: true,
    };
    lift_bytes_with(&NativeAlgebra, &bytes, 0x1000, options)
}

fn shift_ref(m: &str, bits: u32, v: u64, c: u64) -> u64 {
    let w = w_of(bits);
    let cnt = c & if bits == 64 { 63 } else { 31 };
    let v = w.truncate(v);
    let r = match m {
        "shl" => v.checked_shl(cnt as u32).unwrap_or(0),
        "shr" => v.checked_shr(cnt as u32).unwrap_or(0),
        "sar" => (w.sign_extend(v) as i64 >> cnt.min(63)) as u64,
        "rol" => {
            let k = (cnt % u64::from(bits)) as u32;
            if k == 0 {
                v
            } else {
                (v << k) | (v >> (bits - k))
            }
        }
        _ => {
            let k = (cnt % u64::from(bits)) as u32;
            if k == 0 {
                v
            } else {
                (v >> k) | (v << (bits - k))
            }
        }
    };
    w.truncate(r)
}

#[test]
fn variable_count_shifts_and_rotates_match_hardware() {
    type Build = fn(&mut CodeAssembler);
    macro_rules! case {
        ($name:literal, $bits:literal, $m:ident, $dst:expr) => {
            (
                $name,
                $bits,
                (|a: &mut CodeAssembler| a.$m($dst, cl).unwrap()) as Build,
            )
        };
    }
    let cases: Vec<(&str, u32, Build)> = vec![
        case!("shl", 8, shl, bl),
        case!("shr", 8, shr, bl),
        case!("sar", 8, sar, bl),
        case!("rol", 8, rol, bl),
        case!("ror", 8, ror, bl),
        case!("shl", 16, shl, bx),
        case!("shr", 16, shr, bx),
        case!("sar", 16, sar, bx),
        case!("rol", 16, rol, bx),
        case!("ror", 16, ror, bx),
        case!("shl", 32, shl, ebx),
        case!("shr", 32, shr, ebx),
        case!("sar", 32, sar, ebx),
        case!("rol", 32, rol, ebx),
        case!("ror", 32, ror, ebx),
        case!("shl", 64, shl, rbx),
        case!("shr", 64, shr, rbx),
        case!("sar", 64, sar, rbx),
        case!("rol", 64, rol, rbx),
        case!("ror", 64, ror, rbx),
    ];
    for (m, bits, build) in cases {
        let sem = ok(build);
        assert!(sem.written.contains(&Gpr::Rbx));
        let rotate = m == "rol" || m == "ror";
        for flag in [Flag::Cf, Flag::Of] {
            assert!(
                matches!(sem.flags[&flag], FlagState::Clobbered),
                "{m}{bits}"
            );
        }
        assert_eq!(sem.flags.contains_key(&Flag::Zf), !rotate);
        for seed in 0..8 {
            for count in 0..=70u64 {
                let get = |name: &str| {
                    if name == "rcx_in" {
                        (mix(seed) & !0xFF) | count
                    } else {
                        input_value(seed, name)
                    }
                };
                let b0 = get("rbx_in");
                let want = shift_ref(m, bits, b0, count);
                let want = match bits {
                    64 | 32 => want,
                    _ => (b0 & !w_of(bits).mask()) | want,
                };
                assert_eq!(
                    eval_with(&sem.regs[&Gpr::Rbx], &get),
                    Some(want),
                    "{m}{bits} count {count}"
                );
            }
        }
    }
}

#[test]
fn loads_forward_from_a_wider_store_and_skip_disjoint_ranges() {
    let sem = ok(|a| {
        a.mov(qword_ptr(rcx), rdx).unwrap();
        a.mov(eax, dword_ptr(rcx + 4)).unwrap();
        a.mov(bl, byte_ptr(rcx + 1)).unwrap();
        a.mov(rsi, qword_ptr(rcx + 8)).unwrap();
        a.mov(edi, dword_ptr(rcx + 2)).unwrap();
    });
    for seed in SEEDS {
        let d = inp(seed, Gpr::Rdx);
        assert_eq!(reg(&sem, seed, Gpr::Rax), d >> 32);
        assert_eq!(reg(&sem, seed, Gpr::Rbx) & 0xFF, (d >> 8) & 0xFF);
        assert_eq!(reg(&sem, seed, Gpr::Rdi), (d >> 16) & 0xFFFF_FFFF);
    }
    assert_eq!(sem.reads.len(), 1);
    assert!(!sem.assumes_stack_disjoint);
    // straddles the end of the store: still unprovable
    assert!(matches!(
        lift(|a| {
            a.mov(dword_ptr(rcx), edx).unwrap();
            a.mov(rax, qword_ptr(rcx)).unwrap();
        }),
        Err(LiftError::MemoryAliasing { .. })
    ));
}

#[test]
fn stack_frame_versus_foreign_pointer_needs_the_labeled_assumption() {
    let frame = |a: &mut CodeAssembler| {
        a.sub(rsp, 0x28).unwrap();
        a.mov(qword_ptr(rsp + 0x20), rcx).unwrap();
        a.mov(rax, qword_ptr(rdx)).unwrap();
        a.mov(rbx, qword_ptr(rsp + 0x20)).unwrap();
        a.mov(esi, dword_ptr(0x4000)).unwrap();
    };
    assert!(matches!(lift(frame), Err(LiftError::MemoryAliasing { .. })));
    let sem = lift_assuming(frame).expect("lift");
    assert!(sem.assumes_stack_disjoint);
    for seed in SEEDS {
        assert_eq!(reg(&sem, seed, Gpr::Rbx), inp(seed, Gpr::Rcx));
    }
    assert_eq!(sem.reads.len(), 2);

    let refused: Vec<Case> = vec![
        (
            "two foreign pointers",
            Box::new(|a| {
                a.mov(qword_ptr(rcx), rbx).unwrap();
                a.mov(rax, qword_ptr(rdx)).unwrap();
            }),
        ),
        (
            "frame at or above entry rsp",
            Box::new(|a| {
                a.mov(qword_ptr(rsp + 8), rcx).unwrap();
                a.mov(rax, qword_ptr(rdx)).unwrap();
            }),
        ),
        (
            "rsp-derived pointer",
            Box::new(|a| {
                a.mov(qword_ptr(rsp - 0x30), rbx).unwrap();
                a.lea(rcx, ptr(rsp + rdx)).unwrap();
                a.mov(rax, qword_ptr(rcx)).unwrap();
            }),
        ),
        (
            "rsp-derived pointer laundered through memory",
            Box::new(|a| {
                a.lea(rax, ptr(rsp - 0x10)).unwrap();
                a.mov(qword_ptr(rdx), rax).unwrap();
                a.mov(rcx, qword_ptr(rdx)).unwrap();
                a.mov(qword_ptr(rsp - 0x10), rbx).unwrap();
                a.mov(rsi, qword_ptr(rcx + rdi)).unwrap();
            }),
        ),
        (
            "rsp-derived pointer through a sub-register write",
            Box::new(|a| {
                a.mov(rax, rsp).unwrap();
                a.mov(qword_ptr(rsp - 0x20), rbx).unwrap();
                a.mov(al, cl).unwrap();
                a.mov(rsi, qword_ptr(rax + rdi)).unwrap();
            }),
        ),
    ];
    for (name, build) in refused {
        assert!(
            matches!(lift_assuming(&build), Err(LiftError::MemoryAliasing { .. })),
            "{name}"
        );
    }
    // restoring a pushed register does not make it rsp-derived
    let sem = lift_assuming(|a| {
        a.push(rbx).unwrap();
        a.mov(rax, qword_ptr(rcx)).unwrap();
        a.pop(rbx).unwrap();
        a.mov(rax, qword_ptr(rbx)).unwrap();
    })
    .expect("lift");
    assert!(sem.assumes_stack_disjoint);
}

#[test]
fn sign_extension_to_the_upper_half() {
    let cdq = ok(|a| a.cdq().unwrap());
    let cqo = ok(|a| a.cqo().unwrap());
    let cwd = ok(|a| a.cwd().unwrap());
    let chain = ok(|a| {
        a.cbw().unwrap();
        a.cwde().unwrap();
    });
    let cdqe = ok(|a| a.cdqe().unwrap());
    for seed in SEEDS {
        let (a0, d0) = (inp(seed, Gpr::Rax), inp(seed, Gpr::Rdx));
        assert_eq!(
            reg(&cdq, seed, Gpr::Rdx),
            u64::from(((a0 as i32) >> 31) as u32)
        );
        assert_eq!(reg(&cdq, seed, Gpr::Rax), a0);
        assert_eq!(reg(&cqo, seed, Gpr::Rdx), ((a0 as i64) >> 63) as u64);
        let dx = u64::from(((a0 as i16) >> 15) as u16);
        assert_eq!(reg(&cwd, seed, Gpr::Rdx), (d0 & !0xFFFF) | dx);
        let ax1 = u64::from(i16::from(a0 as i8) as u16);
        let eax1 = u64::from(i32::from(ax1 as u16 as i16) as u32);
        assert_eq!(reg(&chain, seed, Gpr::Rax), eax1);
        assert_eq!(reg(&cdqe, seed, Gpr::Rax), i64::from(a0 as i32) as u64);
    }
    assert!(cdq.flags.is_empty());
    assert_eq!(cdq.written.len(), 1);
}

fn bit_ref(m: &str, bits: u32, v: u64, off: u64) -> (u64, u64) {
    let k = off % u64::from(bits);
    let bit = (v >> k) & 1;
    let new = match m {
        "bts" => v | (1 << k),
        "btr" => v & !(1 << k),
        "btc" => v ^ (1 << k),
        _ => v,
    };
    (bit, w_of(bits).truncate(new))
}

#[test]
fn bit_test_family_sets_cf_and_updates_the_operand() {
    type Build = fn(&mut CodeAssembler);
    macro_rules! reg_case {
        ($m:literal, $f:ident, $bits:literal, $dst:expr, $src:expr) => {
            (
                $m,
                $bits,
                (|a: &mut CodeAssembler| a.$f($dst, $src).unwrap()) as Build,
            )
        };
    }
    let cases: Vec<(&str, u32, Build)> = vec![
        reg_case!("bt", bt, 64, rbx, rcx),
        reg_case!("bts", bts, 64, rbx, rcx),
        reg_case!("btr", btr, 32, ebx, ecx),
        reg_case!("btc", btc, 32, ebx, ecx),
        reg_case!("bts", bts, 16, bx, cx),
        reg_case!("btc", btc, 16, bx, cx),
    ];
    for (m, bits, build) in cases {
        let sem = ok(build);
        let FlagState::Value(cf) = &sem.flags[&Flag::Cf] else {
            panic!("cf {m}{bits}")
        };
        assert!(matches!(sem.flags[&Flag::Of], FlagState::Clobbered));
        assert!(!sem.flags.contains_key(&Flag::Zf));
        for seed in SEEDS {
            let (b0, c) = (inp(seed, Gpr::Rbx), inp(seed, Gpr::Rcx));
            let w = w_of(bits);
            let (bit, new) = bit_ref(m, bits, w.truncate(b0), w.truncate(c));
            assert_eq!(eval_expr(cf, seed), bit, "{m}{bits}");
            let want = match (m, bits) {
                ("bt", _) => b0,
                (_, 16) => (b0 & !0xFFFF) | new,
                _ => new,
            };
            assert_eq!(reg(&sem, seed, Gpr::Rbx), want, "{m}{bits}");
        }
    }
    let sem = ok(|a| {
        a.bts(rbx, 70).unwrap();
        a.btr(dword_ptr(rcx), 33).unwrap();
        a.bt(rdx, 3).unwrap();
    });
    for seed in SEEDS {
        assert_eq!(reg(&sem, seed, Gpr::Rbx), inp(seed, Gpr::Rbx) | (1 << 6));
    }
    let FlagState::Value(cf) = &sem.flags[&Flag::Cf] else {
        panic!("cf")
    };
    for seed in SEEDS {
        assert_eq!(eval_expr(cf, seed), (inp(seed, Gpr::Rdx) >> 3) & 1);
    }
    assert_eq!(sem.writes.len(), 1);
    assert_eq!(sem.writes[0].width, Width::W32);
    for seed in SEEDS {
        let old = mix(inp(seed, Gpr::Rcx) ^ 0xABCD) & 0xFFFF_FFFF;
        assert_eq!(eval_expr(&sem.writes[0].value, seed), old & !(1 << 1));
    }
    assert!(matches!(
        lift(|a| a.bts(dword_ptr(rdx), ecx).unwrap()),
        Err(LiftError::Unsupported { .. })
    ));
}

#[test]
fn bsf_bsr_are_undefined_for_a_zero_source() {
    let sem = ok(|a| {
        a.bsf(ebx, ecx).unwrap();
        a.bsr(rdx, rsi).unwrap();
    });
    let FlagState::Value(zf) = &sem.flags[&Flag::Zf] else {
        panic!("zf")
    };
    assert!(matches!(sem.flags[&Flag::Cf], FlagState::Clobbered));
    for seed in SEEDS {
        let (c, s) = (inp(seed, Gpr::Rcx) as u32, inp(seed, Gpr::Rsi));
        assert_eq!(reg(&sem, seed, Gpr::Rbx), u64::from(c.trailing_zeros()));
        assert_eq!(reg(&sem, seed, Gpr::Rdx), u64::from(s.ilog2()));
        assert_eq!(eval_expr(zf, seed), u64::from(s == 0));
    }
    let zero = |name: &str| {
        if name == "rcx_in" {
            0
        } else {
            input_value(1, name)
        }
    };
    assert_eq!(eval_with(&sem.regs[&Gpr::Rbx], &zero), None);
    let zero_rsi = |name: &str| u64::from(name != "rsi_in");
    assert_eq!(eval_with(zf, &zero_rsi), Some(1));
}
