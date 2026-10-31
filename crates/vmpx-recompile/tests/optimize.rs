use std::collections::{BTreeMap, BTreeSet};
use vmpx_recompile::opt::expr::{Op, Rule, bin, konst, not, shl, standard_rules, var};
use vmpx_recompile::opt::{
    AlgebraicRules, Alias, Certificate, ConstantFold, CopyPropagation, DeadDefinitionElimination,
    DeadStoreElimination, EmissionTier, Instr, OptimizeOptions, Optimizer, Pass, PassStatus,
    ProofMethod, Rejection, Verdict, alias, certify_rules, compile_optimized, verify_rule,
};
use vmpx_recompile::{
    Address, BinaryOp, BinarySource, CodeAddress, Condition, Function, Label, Operation,
    PhysicalRegister, RegisterAssignment, ResizeOp, ShiftCount, ShiftOp, SpillSlot, StackFrame,
    UnaryOp, VirtualRegister, Width, compile,
};

const REGS: [PhysicalRegister; 4] = [
    PhysicalRegister::Rax,
    PhysicalRegister::Rcx,
    PhysicalRegister::Rdx,
    PhysicalRegister::R8,
];

fn r(id: u32, width: Width) -> VirtualRegister {
    VirtualRegister::new(id, width)
}

fn q(id: u32) -> VirtualRegister {
    VirtualRegister::qword(id)
}

fn function(slots: u16, operations: Vec<Operation>) -> Function {
    Function {
        operations,
        assignments: REGS
            .iter()
            .enumerate()
            .map(|(i, p)| RegisterAssignment::new(q(i as u32), *p))
            .collect(),
        frame: StackFrame {
            stack_allocation: u32::from(slots) * 8,
            preinitialized_slots: (0..slots).map(SpillSlot).collect(),
            ..StackFrame::default()
        },
    }
}

fn optimize(f: &Function) -> (Function, vmpx_recompile::opt::OptimizationReport) {
    Optimizer::standard().run(f, &OptimizeOptions::default())
}

fn run_pass(pass: &dyn Pass, f: &Function, options: &OptimizeOptions) -> Vec<Operation> {
    let mut program: Vec<Instr> = f
        .operations
        .iter()
        .enumerate()
        .map(|(i, op)| Instr {
            op: op.clone(),
            volatile: options.volatile.contains(&i),
        })
        .collect();
    pass.run(&mut program, options);
    program.into_iter().map(|i| i.op).collect()
}

fn add_imm(d: VirtualRegister, v: i32) -> Operation {
    Operation::Binary {
        operation: BinaryOp::Add,
        destination: d,
        source: BinarySource::Immediate(v),
    }
}

fn mov_imm(d: VirtualRegister, value: u64) -> Operation {
    Operation::MoveImmediate {
        destination: d,
        value,
    }
}

fn flag_kill() -> Operation {
    Operation::Compare {
        left: q(3),
        right: BinarySource::Immediate(0),
    }
}

// ------------------------------------------------------------ rule proofs

#[test]
fn every_shipped_rule_has_a_proof() {
    let certificates = certify_rules(&standard_rules()).expect("all standard rules proven");
    assert_eq!(certificates.len(), standard_rules().len());
    let method_of = |name: &str| {
        certificates.iter().find_map(|c| match c {
            Certificate::Rule { name: n, method } if *n == name => Some(*method),
            _ => None,
        })
    };
    assert_eq!(method_of("xor-self"), Some(ProofMethod::BitSlice));
    assert_eq!(method_of("de-morgan"), Some(ProofMethod::BitSlice));
    assert_eq!(method_of("sub-self"), Some(ProofMethod::Ring));
    assert_eq!(method_of("not-not"), Some(ProofMethod::BitSlice));
    assert_eq!(method_of("shl1-add"), Some(ProofMethod::Ring));
    assert_eq!(method_of("add-neg"), Some(ProofMethod::Ring));
}

#[test]
fn wrong_rules_are_refuted_with_a_counterexample() {
    let x = || var(0);
    let y = || var(1);
    for rule in [
        Rule::new("or-is-add", bin(Op::Or, x(), y()), bin(Op::Add, x(), y())),
        Rule::new("and-self-zero", bin(Op::And, x(), x()), konst(0)),
        Rule::new("shl1-is-shl2", shl(x(), 1), shl(x(), 2)),
        Rule::new(
            "sub-commutes",
            bin(Op::Sub, x(), y()),
            bin(Op::Sub, y(), x()),
        ),
        Rule::new("not-is-neg", not(x()), bin(Op::Sub, konst(0), x())),
    ] {
        let Verdict::Refuted(cex) = verify_rule(&rule) else {
            panic!("{} must be refuted", rule.name);
        };
        assert_ne!(cex.pattern_value, cex.replacement_value, "{}", rule.name);
        // The stored counterexample really separates the two sides.
        assert_ne!(
            rule.pattern.eval(&cex.inputs, cex.bits),
            rule.replacement.eval(&cex.inputs, cex.bits)
        );
    }
}

#[test]
fn true_but_unprovable_rules_are_unknown_not_shipped() {
    // (x & y) + (x | y) == x + y holds, but mixes carries with bit logic.
    let rule = Rule::new(
        "and-or-add",
        bin(
            Op::Add,
            bin(Op::And, var(0), var(1)),
            bin(Op::Or, var(0), var(1)),
        ),
        bin(Op::Add, var(0), var(1)),
    );
    assert_eq!(verify_rule(&rule), Verdict::Unknown);
    assert!(matches!(
        certify_rules(&[rule]),
        Err(Rejection::Unproven { rule: "and-or-add" })
    ));
}

#[test]
fn pipeline_refuses_a_pass_with_a_wrong_rule_and_leaves_code_alone() {
    // "x + 0 -> 0" is wrong; a pass carrying it must not run.
    let wrong = Rule::new("add-zero-is-zero", bin(Op::Add, var(0), konst(0)), konst(0));
    let optimizer = Optimizer::new(vec![Box::new(AlgebraicRules::with_rules(vec![wrong]))]);
    let f = function(0, vec![add_imm(q(0), 0), flag_kill(), Operation::Return]);
    let (out, report) = optimizer.run(&f, &OptimizeOptions::default());
    assert_eq!(out, f);
    let pass = report.pass("algebraic-rules").unwrap();
    assert_eq!(pass.rewrites, 0);
    let PassStatus::Rejected(Rejection::Refuted {
        rule,
        counterexample,
    }) = &pass.status
    else {
        panic!("expected a stored counterexample, got {:?}", pass.status);
    };
    assert_eq!(*rule, "add-zero-is-zero");
    assert_ne!(
        counterexample.pattern_value,
        counterexample.replacement_value
    );
}

// ------------------------------------------------------------ constant folding

fn ref_binary(op: BinaryOp, a: u64, b: u64, width: Width) -> u64 {
    macro_rules! typed {
        ($t:ty) => {{
            let (a, b) = (a as $t, b as $t);
            u64::from(match op {
                BinaryOp::Add => a.wrapping_add(b),
                BinaryOp::Subtract => a.wrapping_sub(b),
                BinaryOp::Multiply => a.wrapping_mul(b),
                BinaryOp::Xor => a ^ b,
                BinaryOp::And => a & b,
                BinaryOp::Or => a | b,
            })
        }};
    }
    match width {
        Width::W8 => typed!(u8),
        Width::W16 => typed!(u16),
        Width::W32 => typed!(u32),
        Width::W64 => {
            let (a, b) = (a, b);
            match op {
                BinaryOp::Add => a.wrapping_add(b),
                BinaryOp::Subtract => a.wrapping_sub(b),
                BinaryOp::Multiply => a.wrapping_mul(b),
                BinaryOp::Xor => a ^ b,
                BinaryOp::And => a & b,
                BinaryOp::Or => a | b,
            }
        }
    }
}

/// Reference built on std typed integer operations (x86 masks the count to 5
/// bits, or 6 for 64-bit operands, before applying it).
fn ref_shift(op: ShiftOp, value: u64, count: u8, width: Width) -> u64 {
    macro_rules! typed {
        ($u:ty, $i:ty) => {{
            let v = value as $u;
            let k = u32::from(count) & if width == Width::W64 { 63 } else { 31 };
            let bits = <$u>::BITS;
            u64::from(match op {
                ShiftOp::Left => v.checked_shl(k).unwrap_or(0),
                ShiftOp::RightLogical => v.checked_shr(k).unwrap_or(0),
                ShiftOp::RightArithmetic => ((v as $i) >> k.min(bits - 1)) as $u,
                ShiftOp::RotateLeft => v.rotate_left(k % bits),
                ShiftOp::RotateRight => v.rotate_right(k % bits),
            })
        }};
    }
    match width {
        Width::W8 => typed!(u8, i8),
        Width::W16 => typed!(u16, i16),
        Width::W32 => typed!(u32, i32),
        Width::W64 => typed!(u64, i64),
    }
}

const WIDTHS: [Width; 4] = [Width::W8, Width::W16, Width::W32, Width::W64];

fn folded_value(operations: Vec<Operation>, destination: VirtualRegister) -> Option<u64> {
    let f = function(0, operations);
    let (out, _) = optimize(&f);
    out.operations.iter().rev().find_map(|op| match op {
        Operation::MoveImmediate {
            destination: d,
            value,
        } if d.id == destination.id => Some(*value),
        _ => None,
    })
}

#[test]
fn folding_matches_std_typed_arithmetic_at_every_width() {
    let samples = [0u64, 1, 0x7f, 0x80, 0x8000, 0xffff_ffff, u64::MAX];
    let ops = [
        BinaryOp::Add,
        BinaryOp::Subtract,
        BinaryOp::Multiply,
        BinaryOp::Xor,
        BinaryOp::And,
        BinaryOp::Or,
    ];
    for width in WIDTHS {
        let d = r(0, width);
        let s = r(1, width);
        let mask = if width == Width::W64 {
            u64::MAX
        } else {
            (1u64 << width.bits()) - 1
        };
        // Narrow writes only merge into a known register, so prime both full-width.
        let prime = || [mov_imm(q(0), 0), mov_imm(q(1), 0)];
        for &a in &samples {
            for &b in &samples {
                for op in ops {
                    if op == BinaryOp::Multiply && width == Width::W8 {
                        continue;
                    }
                    let mut ops = prime().to_vec();
                    ops.extend([
                        mov_imm(d, a),
                        mov_imm(s, b),
                        Operation::Binary {
                            operation: op,
                            destination: d,
                            source: BinarySource::Register(s),
                        },
                        flag_kill(),
                        Operation::Return,
                    ]);
                    let got = folded_value(ops, d).expect("folded to an immediate");
                    assert_eq!(
                        got,
                        ref_binary(op, a & mask, b & mask, width),
                        "{op:?} {width:?} {a:#x} {b:#x}"
                    );
                }
            }
        }
        for &a in &samples {
            for shift in [
                ShiftOp::Left,
                ShiftOp::RightLogical,
                ShiftOp::RightArithmetic,
                ShiftOp::RotateLeft,
                ShiftOp::RotateRight,
            ] {
                for count in [1u8, 7, 8, 9, 17, 31, 32, 33, 63, 64, 255] {
                    let masked = u32::from(count) & if width == Width::W64 { 63 } else { 31 };
                    if masked == 0 {
                        continue;
                    }
                    let mut ops = prime().to_vec();
                    ops.extend([
                        mov_imm(d, a),
                        Operation::Shift {
                            operation: shift,
                            destination: d,
                            count: ShiftCount::Immediate(count),
                        },
                        flag_kill(),
                        Operation::Return,
                    ]);
                    let got = folded_value(ops, d).expect("folded to an immediate");
                    assert_eq!(
                        got,
                        ref_shift(shift, a & mask, count, width),
                        "{shift:?} {width:?} {a:#x} {count}"
                    );
                }
            }
        }
    }
}

#[test]
fn folding_is_withheld_while_flags_are_live() {
    // The add's flags feed the branch, so it must stay a real add.
    let f = function(
        0,
        vec![
            mov_imm(q(0), 5),
            add_imm(q(0), 3),
            Operation::BranchIf {
                condition: Condition::Equal,
                target: Label(0),
            },
            Operation::Label(Label(0)),
            Operation::Return,
        ],
    );
    let (out, _) = optimize(&f);
    assert!(
        out.operations
            .iter()
            .any(|op| matches!(op, Operation::Binary { .. }))
    );

    // Same add, flags redefined before any reader: folds.
    let g = function(
        0,
        vec![
            mov_imm(q(0), 5),
            add_imm(q(0), 3),
            flag_kill(),
            Operation::Return,
        ],
    );
    assert_eq!(folded_value(g.operations, q(0)), Some(8));
}

#[test]
fn register_source_becomes_immediate_when_dest_is_unknown() {
    let f = function(
        0,
        vec![
            mov_imm(q(1), 7),
            Operation::Binary {
                operation: BinaryOp::Add,
                destination: q(0),
                source: BinarySource::Register(q(1)),
            },
            Operation::Return,
        ],
    );
    let ops = run_pass(&ConstantFold, &f, &OptimizeOptions::default());
    assert_eq!(ops[1], add_imm(q(0), 7));
}

#[test]
fn algebraic_rules_rewrite_identities_only_when_flags_dead() {
    let xor_self = Operation::Binary {
        operation: BinaryOp::Xor,
        destination: q(0),
        source: BinarySource::Register(q(0)),
    };
    // xor leaves AF undefined -> reader sees it: kept.
    let live = function(
        0,
        vec![
            xor_self.clone(),
            Operation::BranchIf {
                condition: Condition::Equal,
                target: Label(0),
            },
            Operation::Label(Label(0)),
            Operation::Return,
        ],
    );
    assert_eq!(
        run_pass(
            &AlgebraicRules::standard(),
            &live,
            &OptimizeOptions::default()
        ),
        live.operations
    );
    // Identity at W32 still zero-extends, so it stays as a self move.
    let wide_identity = function(
        0,
        vec![
            Operation::Binary {
                operation: BinaryOp::Or,
                destination: r(0, Width::W32),
                source: BinarySource::Register(r(0, Width::W32)),
            },
            flag_kill(),
            Operation::Return,
        ],
    );
    let ops = run_pass(
        &AlgebraicRules::standard(),
        &wide_identity,
        &OptimizeOptions::default(),
    );
    assert_eq!(
        ops[0],
        Operation::Move {
            destination: r(0, Width::W32),
            source: r(0, Width::W32)
        }
    );
    let dead = function(0, vec![xor_self, flag_kill(), Operation::Return]);
    let ops = run_pass(
        &AlgebraicRules::standard(),
        &dead,
        &OptimizeOptions::default(),
    );
    assert_eq!(ops[0], mov_imm(q(0), 0));

    // x + 0 vanishes; x & -1 (as an 8-bit 0xff-equivalent immediate) vanishes.
    let noop = function(
        0,
        vec![
            add_imm(q(0), 0),
            flag_kill(),
            Operation::Binary {
                operation: BinaryOp::And,
                destination: r(1, Width::W8),
                source: BinarySource::Immediate(-1),
            },
            flag_kill(),
            Operation::Return,
        ],
    );
    let ops = run_pass(
        &AlgebraicRules::standard(),
        &noop,
        &OptimizeOptions::default(),
    );
    assert_eq!(ops, vec![flag_kill(), flag_kill(), Operation::Return]);
}

// ------------------------------------------------------------ copy propagation

#[test]
fn copy_propagation_forwards_sources_and_exposes_dead_moves() {
    let f = function(
        0,
        vec![
            Operation::Move {
                destination: q(1),
                source: q(0),
            },
            Operation::Binary {
                operation: BinaryOp::Add,
                destination: q(2),
                source: BinarySource::Register(q(1)),
            },
            mov_imm(q(1), 9),
            Operation::Return,
        ],
    );
    let ops = run_pass(&CopyPropagation, &f, &OptimizeOptions::default());
    assert_eq!(
        ops[1],
        Operation::Binary {
            operation: BinaryOp::Add,
            destination: q(2),
            source: BinarySource::Register(q(0)),
        }
    );
    let (out, report) = optimize(&f);
    assert!(
        !out.operations
            .iter()
            .any(|op| matches!(op, Operation::Move { .. }))
    );
    assert!(report.pass("copy-propagation").unwrap().rewrites > 0);
}

#[test]
fn copy_propagation_respects_width_and_redefinition() {
    // 32-bit move zero-extends: a 64-bit read of the copy is NOT the source.
    let wide = function(
        0,
        vec![
            Operation::Move {
                destination: r(1, Width::W32),
                source: r(0, Width::W32),
            },
            Operation::Binary {
                operation: BinaryOp::Add,
                destination: q(2),
                source: BinarySource::Register(q(1)),
            },
            Operation::Return,
        ],
    );
    assert_eq!(
        run_pass(&CopyPropagation, &wide, &OptimizeOptions::default()),
        wide.operations
    );
    // Source redefined between the copy and the use.
    let stale = function(
        0,
        vec![
            Operation::Move {
                destination: q(1),
                source: q(0),
            },
            mov_imm(q(0), 1),
            Operation::Binary {
                operation: BinaryOp::Add,
                destination: q(2),
                source: BinarySource::Register(q(1)),
            },
            Operation::Return,
        ],
    );
    assert_eq!(
        run_pass(&CopyPropagation, &stale, &OptimizeOptions::default()),
        stale.operations
    );
}

#[test]
fn dead_definition_kept_unless_provably_overwritten() {
    // Value of r1 escapes through Return: must stay.
    let live_out = function(0, vec![mov_imm(q(1), 3), Operation::Return]);
    assert_eq!(
        run_pass(
            &DeadDefinitionElimination,
            &live_out,
            &OptimizeOptions::default()
        ),
        live_out.operations
    );
    // A 8-bit overwrite does not kill the 64-bit definition.
    let partial = function(
        0,
        vec![
            mov_imm(q(1), 3),
            mov_imm(r(1, Width::W8), 4),
            Operation::Return,
        ],
    );
    assert_eq!(
        run_pass(
            &DeadDefinitionElimination,
            &partial,
            &OptimizeOptions::default()
        ),
        partial.operations
    );
}

// ------------------------------------------------------------ volatile

#[test]
fn volatile_operations_are_never_removed_or_rewritten() {
    let f = function(
        0,
        vec![mov_imm(q(1), 3), mov_imm(q(1), 4), Operation::Return],
    );
    let (plain, _) = optimize(&f);
    assert_eq!(plain.operations.len(), 2);

    let options = OptimizeOptions {
        volatile: BTreeSet::from([0]),
        ..OptimizeOptions::default()
    };
    let (out, _) = Optimizer::standard().run(&f, &options);
    assert_eq!(out.operations, f.operations);

    // A volatile store is never dropped, nor stepped over by another store's DSE.
    let base = q(0);
    let store = |d| Operation::Store {
        address: Address::new(base, d),
        source: q(1),
    };
    let g = function(0, vec![store(0), store(0), Operation::Return]);
    let options = OptimizeOptions {
        volatile: BTreeSet::from([0]),
        ..OptimizeOptions::default()
    };
    assert_eq!(run_pass(&DeadStoreElimination, &g, &options), g.operations);
    let options = OptimizeOptions {
        volatile: BTreeSet::from([1]),
        ..OptimizeOptions::default()
    };
    assert_eq!(run_pass(&DeadStoreElimination, &g, &options), g.operations);
}

// ------------------------------------------------------------ dead stores / aliasing

fn store(base: u32, disp: i32, width: Width) -> Operation {
    Operation::Store {
        address: Address::new(q(base), disp),
        source: r(1, width),
    }
}

fn load(base: u32, disp: i32, dest: u32) -> Operation {
    Operation::Load {
        destination: q(dest),
        address: Address::new(q(base), disp),
    }
}

#[test]
fn overwritten_store_is_removed() {
    let f = function(
        0,
        vec![
            store(0, 8, Width::W64),
            add_imm(q(2), 1),
            store(0, 8, Width::W64),
            Operation::Return,
        ],
    );
    let ops = run_pass(&DeadStoreElimination, &f, &OptimizeOptions::default());
    assert_eq!(
        ops,
        vec![add_imm(q(2), 1), store(0, 8, Width::W64), Operation::Return]
    );
}

#[test]
fn uncertain_aliasing_keeps_the_store() {
    let keep = |operations: Vec<Operation>, options: &OptimizeOptions| {
        let f = function(0, operations);
        assert_eq!(run_pass(&DeadStoreElimination, &f, options), f.operations);
    };
    let strict = OptimizeOptions::default();
    let relaxed = OptimizeOptions {
        assume_fault_free_disjoint_access: true,
        ..OptimizeOptions::default()
    };
    // A load through a different base may read the stored bytes.
    keep(
        vec![
            store(0, 8, Width::W64),
            load(2, 8, 3),
            store(0, 8, Width::W64),
            Operation::Return,
        ],
        &strict,
    );
    keep(
        vec![
            store(0, 8, Width::W64),
            load(2, 8, 3),
            store(0, 8, Width::W64),
            Operation::Return,
        ],
        &relaxed,
    );
    // A load of the same base but partially overlapping bytes.
    keep(
        vec![
            store(0, 8, Width::W64),
            load(0, 12, 3),
            store(0, 8, Width::W64),
            Operation::Return,
        ],
        &relaxed,
    );
    // A store through a different base could be the same address.
    keep(
        vec![
            store(0, 8, Width::W64),
            store(2, 8, Width::W64),
            store(0, 8, Width::W64),
            Operation::Return,
        ],
        &relaxed,
    );
    // A narrower overwrite does not cover the first store.
    keep(
        vec![
            store(0, 8, Width::W64),
            store(0, 8, Width::W32),
            Operation::Return,
        ],
        &strict,
    );
    // The base register changes between the two stores: different address.
    keep(
        vec![
            store(0, 8, Width::W64),
            add_imm(q(0), 8),
            store(0, 8, Width::W64),
            Operation::Return,
        ],
        &strict,
    );
    // A call may read the memory.
    keep(
        vec![
            store(0, 8, Width::W64),
            Operation::Call {
                target: vmpx_recompile::CallTarget::Direct(vmpx_recompile::NativeAddress(0x1000)),
                arguments: vec![],
                result: None,
            },
            store(0, 8, Width::W64),
            Operation::Return,
        ],
        &strict,
    );
}

#[test]
fn provably_disjoint_access_is_stepped_over_only_when_opted_in() {
    let f = function(
        0,
        vec![
            store(0, 8, Width::W64),
            load(0, 24, 3),
            store(0, 8, Width::W64),
            Operation::Return,
        ],
    );
    assert_eq!(
        run_pass(&DeadStoreElimination, &f, &OptimizeOptions::default()),
        f.operations,
        "the in-between load may fault and expose the store"
    );
    let relaxed = OptimizeOptions {
        assume_fault_free_disjoint_access: true,
        ..OptimizeOptions::default()
    };
    assert_eq!(run_pass(&DeadStoreElimination, &f, &relaxed).len(), 3);
}

#[test]
fn alias_classification() {
    let a = Address::new(q(0), 0);
    assert_eq!(alias(a, Width::W64, a, Width::W64), Alias::Must);
    assert_eq!(
        alias(a, Width::W64, Address::new(q(0), 8), Width::W64),
        Alias::Disjoint
    );
    assert_eq!(
        alias(a, Width::W64, Address::new(q(0), 4), Width::W64),
        Alias::May
    );
    assert_eq!(alias(a, Width::W64, a, Width::W32), Alias::May);
    assert_eq!(
        alias(a, Width::W64, Address::new(q(1), 0), Width::W64),
        Alias::May
    );
}

#[test]
fn spill_stores_follow_slot_reads() {
    let spill = |slot, width| Operation::SpillStore {
        slot: SpillSlot(slot),
        source: r(1, width),
    };
    let reload = |slot| Operation::SpillLoad {
        destination: q(2),
        slot: SpillSlot(slot),
    };
    let f = function(
        2,
        vec![
            spill(0, Width::W64),
            spill(1, Width::W64),
            spill(0, Width::W64),
            Operation::Return,
        ],
    );
    let ops = run_pass(&DeadStoreElimination, &f, &OptimizeOptions::default());
    assert_eq!(
        ops,
        vec![
            spill(1, Width::W64),
            spill(0, Width::W64),
            Operation::Return
        ]
    );
    // Read in between: kept.
    let g = function(
        1,
        vec![
            spill(0, Width::W64),
            reload(0),
            spill(0, Width::W64),
            Operation::Return,
        ],
    );
    assert_eq!(
        run_pass(&DeadStoreElimination, &g, &OptimizeOptions::default()),
        g.operations
    );
    // An address-based access may alias the frame: kept.
    let h = function(
        1,
        vec![
            spill(0, Width::W64),
            load(0, 0, 3),
            spill(0, Width::W64),
            Operation::Return,
        ],
    );
    assert_eq!(
        run_pass(&DeadStoreElimination, &h, &OptimizeOptions::default()),
        h.operations
    );
}

// ------------------------------------------------------------ differential fuzz

#[derive(Default, Clone, PartialEq, Debug)]
struct State {
    regs: [u64; 4],
    memory: BTreeMap<u64, u8>,
    spill: BTreeMap<u64, u8>,
}

fn mask(w: Width) -> u64 {
    if w == Width::W64 {
        u64::MAX
    } else {
        (1u64 << w.bits()) - 1
    }
}

fn sext(v: u64, w: Width) -> u64 {
    let s = 64 - w.bits();
    if s == 0 {
        v
    } else {
        (((v << s) as i64) >> s) as u64
    }
}

impl State {
    fn read(&self, reg: VirtualRegister) -> u64 {
        self.regs[reg.id as usize] & mask(reg.width)
    }

    fn write(&mut self, reg: VirtualRegister, v: u64) {
        let slot = &mut self.regs[reg.id as usize];
        *slot = match reg.width {
            Width::W64 => v,
            Width::W32 => v & 0xffff_ffff,
            w => (*slot & !mask(w)) | (v & mask(w)),
        };
    }

    fn mem_byte(&self, addr: u64) -> u8 {
        self.memory
            .get(&addr)
            .copied()
            .unwrap_or_else(|| (addr.wrapping_mul(0x9e37_79b9) >> 5) as u8)
    }

    fn load(&self, addr: u64, w: Width) -> u64 {
        (0..w.bits() / 8).fold(0u64, |acc, i| {
            acc | u64::from(self.mem_byte(addr.wrapping_add(u64::from(i)))) << (8 * i)
        })
    }

    fn store(&mut self, addr: u64, w: Width, v: u64) {
        for i in 0..w.bits() / 8 {
            self.memory
                .insert(addr.wrapping_add(u64::from(i)), (v >> (8 * i)) as u8);
        }
    }

    fn spill_load(&self, slot: u16, w: Width) -> u64 {
        (0..w.bits() / 8).fold(0u64, |acc, i| {
            let key = u64::from(slot) * 8 + u64::from(i);
            acc | u64::from(self.spill.get(&key).copied().unwrap_or(0)) << (8 * i)
        })
    }

    fn spill_store(&mut self, slot: u16, w: Width, v: u64) {
        for i in 0..w.bits() / 8 {
            self.spill
                .insert(u64::from(slot) * 8 + u64::from(i), (v >> (8 * i)) as u8);
        }
    }
}

fn execute(f: &Function, mut st: State) -> State {
    for op in &f.operations {
        match op {
            Operation::MoveImmediate { destination, value } => st.write(*destination, *value),
            Operation::Move {
                destination,
                source,
            } => {
                let v = st.read(*source);
                st.write(*destination, v);
            }
            Operation::Binary {
                operation,
                destination,
                source,
            } => {
                let a = st.read(*destination);
                let b = match source {
                    BinarySource::Register(s) => st.read(*s),
                    BinarySource::Immediate(i) => (i64::from(*i) as u64) & mask(destination.width),
                };
                st.write(
                    *destination,
                    ref_binary(*operation, a, b, destination.width),
                );
            }
            Operation::Unary {
                operation,
                destination,
            } => {
                let a = st.read(*destination);
                st.write(
                    *destination,
                    match operation {
                        UnaryOp::Not => !a,
                        UnaryOp::Negate => a.wrapping_neg(),
                    },
                );
            }
            Operation::Shift {
                operation,
                destination,
                count: ShiftCount::Immediate(c),
            } => {
                let a = st.read(*destination);
                st.write(
                    *destination,
                    ref_shift(*operation, a, *c, destination.width),
                );
            }
            Operation::Resize {
                operation,
                destination,
                source,
            } => {
                let v = st.read(*source);
                st.write(
                    *destination,
                    match operation {
                        ResizeOp::Truncate | ResizeOp::ZeroExtend => v,
                        ResizeOp::SignExtend => sext(v, source.width),
                    },
                );
            }
            Operation::Load {
                destination,
                address,
            } => {
                let addr = st
                    .read(address.base)
                    .wrapping_add(i64::from(address.displacement) as u64);
                let v = st.load(addr, destination.width);
                st.write(*destination, v);
            }
            Operation::Store { address, source } => {
                let addr = st
                    .read(address.base)
                    .wrapping_add(i64::from(address.displacement) as u64);
                let v = st.read(*source);
                st.store(addr, source.width, v);
            }
            Operation::SpillLoad { destination, slot } => {
                let v = st.spill_load(slot.0, destination.width);
                st.write(*destination, v);
            }
            Operation::SpillStore { slot, source } => {
                let v = st.read(*source);
                st.spill_store(slot.0, source.width, v);
            }
            Operation::Compare { .. } | Operation::Return => {}
            other => panic!("generator never emits {other:?}"),
        }
    }
    st
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % bound
    }

    fn width(&mut self) -> Width {
        WIDTHS[self.next(4) as usize]
    }

    fn value(&mut self) -> u64 {
        match self.next(4) {
            0 => self.next(4),
            1 => u64::MAX - self.next(3),
            2 => self.next(256),
            _ => self.next(u64::MAX),
        }
    }
}

fn dump(f: &Function) -> String {
    f.operations
        .iter()
        .map(|o| format!("{o:?}"))
        .collect::<Vec<_>>()
        .join(
            "
",
        )
}

fn random_program(rng: &mut Lcg) -> Function {
    let len = 10 + rng.next(24) as usize;
    let mut ops = Vec::new();
    for _ in 0..len {
        let width = rng.width();
        let d = r(rng.next(4) as u32, width);
        let s = r(rng.next(4) as u32, width);
        let disp = [0, 4, 8, 16][rng.next(4) as usize];
        let base = q(rng.next(4) as u32);
        ops.push(match rng.next(16) {
            0 | 1 => mov_imm(d, rng.value()),
            2 | 3 => Operation::Move {
                destination: d,
                source: s,
            },
            4..=6 => Operation::Binary {
                operation: [
                    BinaryOp::Add,
                    BinaryOp::Subtract,
                    BinaryOp::Multiply,
                    BinaryOp::Xor,
                    BinaryOp::And,
                    BinaryOp::Or,
                ][rng.next(6) as usize],
                destination: d,
                source: if rng.next(2) == 0 {
                    BinarySource::Register(s)
                } else {
                    BinarySource::Immediate([0, 1, -1, 5, 100, -100][rng.next(6) as usize])
                },
            },
            7 => Operation::Unary {
                operation: if rng.next(2) == 0 {
                    UnaryOp::Not
                } else {
                    UnaryOp::Negate
                },
                destination: d,
            },
            8 => Operation::Shift {
                operation: [
                    ShiftOp::Left,
                    ShiftOp::RightLogical,
                    ShiftOp::RightArithmetic,
                    ShiftOp::RotateLeft,
                    ShiftOp::RotateRight,
                ][rng.next(5) as usize],
                destination: d,
                count: ShiftCount::Immediate([0, 1, 3, 8, 9, 31, 33, 64][rng.next(8) as usize]),
            },
            9 => {
                let (small, big) = if rng.next(2) == 0 {
                    (Width::W8, Width::W32)
                } else {
                    (Width::W16, Width::W64)
                };
                let operation = [
                    ResizeOp::Truncate,
                    ResizeOp::ZeroExtend,
                    ResizeOp::SignExtend,
                ][rng.next(3) as usize];
                let (destination, source) = if operation == ResizeOp::Truncate {
                    (r(rng.next(4) as u32, small), r(rng.next(4) as u32, big))
                } else {
                    (r(rng.next(4) as u32, big), r(rng.next(4) as u32, small))
                };
                Operation::Resize {
                    operation,
                    destination,
                    source,
                }
            }
            10 | 11 => Operation::Store {
                address: Address::new(base, disp),
                source: s,
            },
            12 => Operation::Load {
                destination: d,
                address: Address::new(base, disp),
            },
            13 => Operation::SpillStore {
                slot: SpillSlot(rng.next(3) as u16),
                source: s,
            },
            14 => Operation::SpillLoad {
                destination: d,
                slot: SpillSlot(rng.next(3) as u16),
            },
            _ => Operation::Compare {
                left: s,
                right: BinarySource::Immediate(0),
            },
        });
    }
    ops.push(Operation::Return);
    function(3, ops)
}

#[test]
fn optimized_programs_match_originals_on_random_inputs() {
    let mut rng = Lcg(0x5eed_1234);
    let (mut tested, mut rewrites, mut shrunk) = (0, 0, 0);
    for _ in 0..600 {
        let f = random_program(&mut rng);
        let volatile: BTreeSet<usize> = (0..f.operations.len())
            .filter(|_| rng.next(10) == 0)
            .collect();
        let options = OptimizeOptions {
            volatile: volatile.clone(),
            ..OptimizeOptions::default()
        };
        let (out, report) = Optimizer::standard().run(&f, &options);
        if report.input_error.is_some() {
            continue;
        }
        for pass in &report.passes {
            assert!(
                !matches!(pass.status, PassStatus::Reverted(_)),
                "pass {} produced invalid output: {:?}",
                pass.name,
                pass.status
            );
        }
        tested += 1;
        rewrites += report.total_rewrites();
        shrunk += usize::from(out.operations.len() < f.operations.len());
        // Volatile operations survive, in order.
        let kept: Vec<&Operation> = volatile.iter().map(|&i| &f.operations[i]).collect();
        let mut cursor = out.operations.iter();
        for op in kept {
            assert!(cursor.any(|o| o == op), "volatile op lost: {op:?}");
        }
        for _ in 0..3 {
            let mut st = State::default();
            for reg in &mut st.regs {
                *reg = rng.value();
            }
            assert_eq!(
                execute(&f, st.clone()),
                execute(&out, st),
                "behaviour changed\n{}\n--\n{}",
                dump(&f),
                dump(&out)
            );
        }
    }
    assert!(tested > 400, "only {tested} programs validated");
    assert!(
        rewrites > 100 && shrunk > 50,
        "optimizer too inert: {rewrites} rewrites, {shrunk} shrunk"
    );
}

// ------------------------------------------------------------ emission

#[test]
fn optimized_function_emits_smaller_runnable_code() {
    let f = function(
        0,
        vec![
            mov_imm(q(0), 0x1000),
            mov_imm(q(1), 0x234),
            Operation::Binary {
                operation: BinaryOp::Add,
                destination: q(0),
                source: BinarySource::Register(q(1)),
            },
            mov_imm(q(1), 0),
            Operation::Return,
        ],
    );
    let address = CodeAddress(0x1_4000_1000);
    let result = compile_optimized(&f, address, &OptimizeOptions::default()).unwrap();
    assert_eq!(result.tier, EmissionTier::Optimized);
    assert!(result.report.total_rewrites() > 0);
    let baseline = compile(&f, address).unwrap();
    assert!(result.compiled.bytes.len() < baseline.bytes.len());
    assert_eq!(result.compiled, compile(&result.emitted, address).unwrap());
}

struct CorruptingPass;

impl Pass for CorruptingPass {
    fn name(&self) -> &'static str {
        "test-corrupt"
    }
    fn certify(&self) -> Result<Vec<Certificate>, Rejection> {
        Ok(vec![Certificate::Structural { invariant: "test" }])
    }
    fn run(&self, program: &mut Vec<Instr>, _: &OptimizeOptions) -> usize {
        // Valid for the validator, unencodable for the assembler.
        program.insert(
            0,
            Instr {
                op: Operation::Binary {
                    operation: BinaryOp::Add,
                    destination: r(0, Width::W16),
                    source: BinarySource::Immediate(1 << 20),
                },
                volatile: false,
            },
        );
        1
    }
}

#[test]
fn emission_falls_back_when_the_optimized_form_does_not_encode() {
    let f = function(0, vec![mov_imm(q(0), 1), Operation::Return]);
    let optimizer = Optimizer::new(vec![Box::new(CorruptingPass)]);
    let result = vmpx_recompile::opt::compile_optimized_with(
        &optimizer,
        &f,
        CodeAddress(0x1000),
        &OptimizeOptions::default(),
    )
    .unwrap();
    assert!(matches!(result.tier, EmissionTier::FellBack { .. }));
    assert_eq!(result.emitted, f);
    assert_eq!(result.compiled, compile(&f, CodeAddress(0x1000)).unwrap());
}
