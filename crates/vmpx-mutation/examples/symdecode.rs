//! Symbolic decode diagnostic: trace a VM entry's bootstrap->dispatch path
//! (following opaque-predicate conditionals), compose the per-instruction
//! semantics symbolically, and print the simplified Expr for each register that
//! derives from the bytecode load -- i.e. the handler decode, with the MBA
//! mutation folded away by Expr::simplify.
use std::collections::BTreeMap;

use iced_x86::FlowControl;
use vmpx_disasm::{DirectJumpTraceLimits, Gpr, trace_direct_jump_fragments};
use vmpx_mutation::{lift_native_block, BitWidth, Expr};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::ExprKind;

fn pp(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Const { value } => format!("{value:#x}"),
        ExprKind::Register { name } => name.clone(),
        ExprKind::Load { address } => format!("load[{}]", pp(address)),
        ExprKind::Unary { op, value } => format!("{op:?}({})", pp(value)),
        ExprKind::Binary { op, left, right } => format!("({} {op:?} {})", pp(left), pp(right)),
        ExprKind::Extract { value, lsb } => format!("ext{}@{lsb}({})", e.width.bits(), pp(value)),
        ExprKind::Concat { high, low } => format!("cat({},{})", pp(high), pp(low)),
        ExprKind::Resize { op, value } => format!("{op:?}{}({})", e.width.bits(), pp(value)),
        ExprKind::Intrinsic { name, arguments } => {
            format!("{name}({})", arguments.iter().map(pp).collect::<Vec<_>>().join(","))
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let start = Rva(u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap());
    let handler_reg = args.get(3).cloned().unwrap_or_else(|| "rdi".to_string());

    let mut limits = DirectJumpTraceLimits::default();
    limits.follow_conditional_fallthrough = true;
    limits.max_conditional_continuations = 128;
    let trace = trace_direct_jump_fragments(&pe, start, limits).unwrap();

    let mut regs: BTreeMap<Gpr, Expr> =
        Gpr::ALL.iter().map(|&g| (g, Expr::register(BitWidth::W64, &g.input_name()))).collect();

    let (mut lifted, mut ctrl, mut failed) = (0usize, 0usize, 0usize);
    for ins in trace.instructions() {
        if ins.instruction.flow_control() != FlowControl::Next {
            ctrl += 1;
            continue;
        }
        match lift_native_block(&ins.bytes, u64::from(ins.rva.0)) {
            Ok(sem) => {
                let repl: BTreeMap<String, Expr> =
                    Gpr::ALL.iter().map(|&g| (g.input_name(), regs[&g].clone())).collect();
                let mut next = regs.clone();
                for (g, e) in &sem.regs {
                    if let Ok(c) = e.substitute(&repl) {
                        next.insert(*g, c.simplify());
                    }
                }
                regs = next;
                lifted += 1;
                // stop once the handler register first becomes token-derived:
                // that is exactly one handler's decode, isolated from the rest.
                let h = Gpr::ALL.iter().find(|g| gpr_name(**g) == handler_reg).copied();
                if let Some(h) = h {
                    if pp(&regs[&h]).contains("load[") {
                        break;
                    }
                }
            }
            Err(_) => failed += 1,
        }
    }
    eprintln!("path: lifted={lifted} ctrl-skipped={ctrl} lift-failed={failed}");
    println!("=== registers derived from the bytecode load (the decode) ===");
    for &g in &Gpr::ALL {
        let e = regs[&g].simplify();
        let s = pp(&e);
        if s.contains("load[") {
            let shown = if s.len() > 300 { format!("{}...", &s[..300]) } else { s };
            println!("{g:?} = {shown}");
        }
    }

    // --- codec extraction: peel handler = anchor + sext(W, chain(token, key)) ---
    println!("\n=== recovered codec (symbolic, string peel) ===");
    if let Some(h) = Gpr::ALL.iter().find(|g| gpr_name(**g) == handler_reg).copied() {
        let e = regs[&h].simplify();
        let mut transforms = Vec::new();
        let mut key = None;
        peel(&e, &mut transforms, &mut key);
        transforms.reverse();
        println!("handler = {handler_reg}_in + decode(token, key)");
        println!("  key-mix : {}", key.clone().unwrap_or_else(|| "none".into()));
        println!("  decode chain (bottom=token .. top=displacement), {} steps:", transforms.len());
        for t in &transforms {
            println!("    {t}");
        }
    }

    // --- TYPED codec via the library (what the pipeline consumes) ---
    println!("\n=== recovered KeyedTransformCodec (typed, library) ===");
    let key_reg = args.get(4).cloned().unwrap_or_else(|| "rbp".to_string());
    let h = Gpr::ALL.iter().copied().find(|g| gpr_name(*g) == handler_reg);
    let k = Gpr::ALL.iter().copied().find(|g| gpr_name(*g) == key_reg);
    match (h, k) {
        (Some(h), Some(k)) => {
            match vmpx_mutation::extract_symbolic_codec(&pe, start, h, k) {
                Some(codec) => {
                    println!("width                 : {:?}", codec.width);
                    println!("input_key_operation   : {:?}", codec.input_key_operation);
                    println!("input_key_on_left     : {}", codec.input_key_on_left);
                    println!("update_key_operation  : {:?}", codec.update_key_operation);
                    println!("transforms ({}):", codec.transforms.transforms.len());
                    for t in &codec.transforms.transforms {
                        println!("    {:?} operand={:#x} @{:?}", t.op, t.operand, t.width);
                    }
                    println!("\n(debug) {codec:?}");
                }
                None => println!("(no codec recovered — handler not token-derived or unmodellable decode)"),
            }
        }
        _ => println!("(bad handler/key register name)"),
    }

    // --- auto key-selection (item 3): discover the rolling-key register ---
    println!("\n=== auto key-selection ===");
    if let Some(h) = h {
        let keys: Vec<Gpr> = Gpr::ALL.to_vec();
        match vmpx_mutation::extract_symbolic_codec_auto(&pe, start, h, &keys) {
            Some((codec, key)) => println!(
                "chosen key register = {} ; key-mix = {:?} ; transforms = {}",
                gpr_name(key),
                codec.input_key_operation,
                codec.transforms.transforms.len()
            ),
            None => println!("(auto: no codec recovered)"),
        }
    }
}

fn gpr_name(g: Gpr) -> String {
    g.input_name().trim_end_matches("_in").to_string()
}

/// Peel a decode displacement Expr into an ordered transform list + the key-mix,
/// stripping width casts. Records the token load when reached.
fn peel(e: &Expr, out: &mut Vec<String>, key: &mut Option<String>) {
    match &e.kind {
        ExprKind::Resize { value, .. } => peel(value, out, key),
        ExprKind::Extract { value, .. } => peel(value, out, key),
        ExprKind::Concat { low, .. } => peel(low, out, key),
        ExprKind::Unary { op, value } => {
            out.push(format!("{op:?}"));
            peel(value, out, key);
        }
        ExprKind::Load { .. } => out.push(format!("load[token]: {}", pp(e))),
        ExprKind::Register { name } => out.push(format!("input {name}")),
        ExprKind::Binary { op, left, right } => {
            let is_const = |x: &Expr| matches!(x.kind, ExprKind::Const { .. });
            let as_const = |x: &Expr| {
                if let ExprKind::Const { value } = &x.kind {
                    Some(*value)
                } else {
                    None
                }
            };
            let reg_name = |x: &Expr| {
                if let ExprKind::Register { name } = &x.kind {
                    Some(name.clone())
                } else {
                    None
                }
            };
            if let Some(c) = as_const(right) {
                out.push(format!("{op:?} {c:#x}"));
                peel(left, out, key);
            } else if is_const(left) {
                out.push(format!("{op:?} {:#x} (lhs)", as_const(left).unwrap()));
                peel(right, out, key);
            } else if let Some(k) = reg_name(right).or_else(|| reg_name(left)) {
                *key = Some(format!("token {op:?} {k}"));
                // recurse into the non-key side (the token path)
                if reg_name(right).is_some() {
                    peel(left, out, key);
                } else {
                    peel(right, out, key);
                }
            } else {
                // two computed operands (e.g. token mixed with a second load/index)
                out.push(format!("{op:?} <{}>", pp(right)));
                peel(left, out, key);
            }
        }
        ExprKind::Const { value } => out.push(format!("const {value:#x}")),
        ExprKind::Intrinsic { name, .. } => out.push(format!("intrinsic {name}")),
    }
}
