#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless
)]

use iced_x86::code_asm::{
    CodeAssembler, ah, al, ax, bl, bx, cl, cx, dl, eax, ebx, ecx, edx, ptr, qword_ptr, rax, rbx, rcx,
    rdx, rsp,
};

use crate::{
    Algebra, BinOp, BlockSemantics, Flag, FlagState, Gpr, LiftError, NativeAlgebra, NativeExpr,
    Resize, Width, lift_bytes,
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
    let tag = name
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(u64::from(b)));
    mix(seed.wrapping_mul(0x1_0000_0001).wrapping_add(tag))
}

fn eval_expr(expr: &NativeExpr, seed: u64) -> u64 {
    expr.eval(&|name| input_value(seed, name), &|addr, _| mix(addr ^ 0xABCD))
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
    NativeAlgebra.resize(Resize::ZeroExtend, Width::W64, e).unwrap()
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
    let expected = zext64(NativeAlgebra
        .unary(crate::UnOp::Not, trunc(Width::W32, input("rcx_in")))
        .unwrap());
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
        assert_eq!(reg(&sem, seed, Gpr::Rcx), u64::from(i32::from(c as i8) as u32));
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
        lift(|a| a.shl(eax, cl).unwrap()),
        Err(LiftError::Unsupported { .. })
    ));
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
        assert_eq!(reg(&sem, seed, Gpr::Rdx), u64::from((d as u32).wrapping_sub(1)));
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
    assert_eq!(mem.regs[&Gpr::Rax], NativeAlgebra.load(Width::W64, input("rax_in")));
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
    assert!(matches!(
        lift(|a| {
            a.mov(qword_ptr(rsp), rcx).unwrap();
            a.mov(eax, ptr(rsp)).unwrap();
        }),
        Err(LiftError::MemoryAliasing { .. })
    ));
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
        ("div", Box::new(|a| a.div(ecx).unwrap())),
        ("push", Box::new(|a| a.push(rax).unwrap())),
        ("adc", Box::new(|a| a.adc(eax, ecx).unwrap())),
        ("bsf", Box::new(|a| a.bsf(eax, ecx).unwrap())),
        ("mul", Box::new(|a| a.mul(rcx).unwrap())),
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
    let imul1 = lift(|a| a.imul(ecx).unwrap());
    assert!(matches!(imul1, Err(LiftError::Unsupported { .. })));
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
