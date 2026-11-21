//! Stage 2 -- handler-body symbolic lift.
//!
//! `symdecode` recovered the *dispatch* decode (registers only). A VM handler
//! BODY also reads and writes the VM stack, so recovering what it computes needs
//! memory composition on top of the register composition: compose each straight
//! line instruction's `BlockSemantics` symbolically, forwarding earlier stores
//! into later loads, and let `Expr::simplify` fold the MBA. The net register
//! deltas + the surviving stack writes ARE the VM operation (e.g. a store of
//! `load[vsp+8] + load[vsp+0x10]` with `vsp += 8` is a VM ADD).
//!
//! Usage: handlerbody <img> <body_rva> [max_insns]
use std::collections::BTreeMap;

use iced_x86::FlowControl;
use vmpx_disasm::{
    DirectJumpTraceLimits, Flag, FlagState, Gpr, trace_direct_jump_fragments,
};
use vmpx_mutation::{lift_native_block, BitWidth, Expr};
use vmpx_pe::{ImageBase, PeImage, Rva, Va};
use vmpx_vm::ExprKind;

/// Read-only image backing: a load from a constant image address resolves to the
/// actual mapped bytes (RVA-layout image, so file offset == RVA). Correct for the
/// read-only bootstrap data VMProtect's own decode reads (bytecode, handler table).
struct Img<'a> {
    data: &'a [u8],
    base: ImageBase,
}

impl Img<'_> {
    fn read(&self, va: u64, bits: u32) -> Option<u64> {
        let rva = self.base.checked_rva_of(Va(va))?;
        let off = rva.0 as usize;
        let wb = (bits / 8) as usize;
        if off + wb > self.data.len() {
            return None;
        }
        let mut v = 0u64;
        for i in 0..wb {
            v |= u64::from(self.data[off + i]) << (8 * i);
        }
        Some(v)
    }
}

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

/// Rewrite `Load` leaves using the ordered store log: the most recent store with
/// the same (simplified) address and width supplies the loaded value. A store's
/// value was already forwarded when it was recorded, so no re-forwarding needed.
fn forward(e: &Expr, mem: &[(Expr, Expr, u32)], img: &Img) -> Expr {
    let rebuilt = match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Load { address } => {
            let a = forward(address, mem, img).simplify();
            let w = e.width.bits();
            for (saddr, sval, sw) in mem.iter().rev() {
                if *sw == w && *saddr == a {
                    return sval.clone();
                }
            }
            // read-only image-backed constant load
            if let ExprKind::Const { value } = &a.kind {
                if let Some(v) = img.read(*value, w) {
                    return Expr::constant(e.width, v);
                }
            }
            Expr::load(e.width, a)
        }
        ExprKind::Unary { op, value } => Expr::unary(*op, forward(value, mem, img)),
        ExprKind::Binary { op, left, right } => {
            Expr::binary(*op, forward(left, mem, img), forward(right, mem, img))
                .expect("width preserved")
        }
        ExprKind::Extract { value, lsb } => {
            Expr::extract(e.width, *lsb, forward(value, mem, img)).expect("width preserved")
        }
        ExprKind::Concat { high, low } => {
            Expr::concat(forward(high, mem, img), forward(low, mem, img)).expect("width preserved")
        }
        ExprKind::Resize { op, value } => {
            Expr::resize(*op, e.width, forward(value, mem, img)).expect("width preserved")
        }
        ExprKind::Intrinsic { name, arguments } => Expr::intrinsic(
            e.width,
            name.clone(),
            arguments.iter().map(|a| forward(a, mem, img)).collect(),
        ),
    };
    rebuilt.simplify()
}

fn subst(e: &Expr, repl: &BTreeMap<String, Expr>) -> Expr {
    e.substitute(repl).unwrap_or_else(|_| e.clone())
}

/// Does this expr tree contain a load? (a real data read vs a pure constant/decoy).
fn has_load(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Load { .. } => true,
        ExprKind::Const { .. } | ExprKind::Register { .. } => false,
        ExprKind::Unary { value, .. } | ExprKind::Extract { value, .. } | ExprKind::Resize { value, .. } => {
            has_load(value)
        }
        ExprKind::Binary { left, right, .. } | ExprKind::Concat { high: left, low: right } => {
            has_load(left) || has_load(right)
        }
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().any(has_load),
    }
}

/// Strip width-cast wrappers (Resize/Extract/Concat-low) to reach the core op.
fn core(e: &Expr) -> &Expr {
    match &e.kind {
        ExprKind::Resize { value, .. } | ExprKind::Extract { value, .. } => core(value),
        ExprKind::Concat { low, .. } => core(low),
        _ => e,
    }
}

/// Reduce one write's value expression to a canonical VM opcode mnemonic.
fn classify_value(e: &Expr) -> String {
    let c = core(e);
    match &c.kind {
        ExprKind::Binary { op, left, right } => {
            let (l, r) = (core(left), core(right));
            let is_not = |x: &Expr| matches!(core(x).kind, ExprKind::Unary { op: o, .. } if format!("{o:?}") == "Not");
            // NOR = Not(a) And Not(b); NAND = Not(a) Or Not(b)
            if format!("{op:?}") == "And" && is_not(l) && is_not(r) {
                return "NOR".into();
            }
            if format!("{op:?}") == "Or" && is_not(l) && is_not(r) {
                return "NAND".into();
            }
            let opn = format!("{op:?}");
            // shift by constant
            if matches!(&r.kind, ExprKind::Const { .. })
                && matches!(opn.as_str(), "Shl" | "Shr" | "Sar" | "Rol" | "Ror")
            {
                if let ExprKind::Const { value } = &r.kind {
                    return format!("{} {value:#x}", opn.to_uppercase());
                }
            }
            // binary data op between two reads
            if has_load(l) && has_load(r) {
                return opn.to_uppercase();
            }
            if has_load(l) || has_load(r) {
                return format!("{}i", opn.to_uppercase()); // op with immediate
            }
            "const".into()
        }
        ExprKind::Unary { op, value } => {
            if has_load(value) {
                format!("{}", format!("{op:?}").to_uppercase())
            } else {
                "const".into()
            }
        }
        ExprKind::Load { .. } => "COPY".into(),
        ExprKind::Const { .. } => "IMM".into(),
        _ => {
            if has_load(c) {
                "expr".into()
            } else {
                "const".into()
            }
        }
    }
}

fn base_tag(addr: &str) -> &'static str {
    if addr.contains("rbx_in") {
        "vreg"
    } else if addr.contains("rbp_in") {
        "vreg2"
    } else if addr.contains("rsp_in") {
        "vstk"
    } else {
        "mem"
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let img = Img { data: &bytes, base: pe.optional.image_base };
    let start = Rva(u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap());
    let max_insns: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4000);

    let mut limits = DirectJumpTraceLimits::default();
    limits.follow_conditional_fallthrough = true;
    limits.max_conditional_continuations = 256;
    let trace = trace_direct_jump_fragments(&pe, start, limits).unwrap();

    // Composition state over the *_in inputs.
    let mut regs: BTreeMap<Gpr, Expr> =
        Gpr::ALL.iter().map(|&g| (g, Expr::register(BitWidth::W64, &g.input_name()))).collect();
    let mut flags: BTreeMap<Flag, Expr> = BTreeMap::new();
    let mut mem: Vec<(Expr, Expr, u32)> = Vec::new();

    let (mut lifted, mut ctrl, mut failed, mut seen) = (0usize, 0usize, 0usize, 0usize);
    for ins in trace.instructions() {
        if seen >= max_insns {
            break;
        }
        seen += 1;
        if std::env::var_os("DUMP").is_some() {
            eprintln!("  {:#x}: {}  [{:?}]", ins.rva.0, ins.instruction, ins.instruction.flow_control());
        }
        if ins.instruction.flow_control() != FlowControl::Next {
            ctrl += 1;
            continue;
        }
        let sem = match lift_native_block(&ins.bytes, u64::from(ins.rva.0)) {
            Ok(s) => s,
            Err(_) => {
                failed += 1;
                continue;
            }
        };
        // Replacement map: register inputs + any flag this body has written.
        let mut repl: BTreeMap<String, Expr> =
            Gpr::ALL.iter().map(|&g| (g.input_name(), regs[&g].clone())).collect();
        for (f, e) in &flags {
            repl.insert(format!("{}_in", f.name()), e.clone());
        }
        // New register values (read old regs via repl, commit after).
        let mut next = regs.clone();
        for (g, e) in &sem.regs {
            next.insert(*g, forward(&subst(e, &repl), &mem, &img));
        }
        // Flags.
        let mut nflags = flags.clone();
        for (f, st) in &sem.flags {
            match st {
                FlagState::Value(e) => {
                    nflags.insert(*f, forward(&subst(e, &repl), &mem, &img));
                }
                FlagState::Clobbered => {
                    nflags.remove(f);
                }
            }
        }
        // Stores, in program order, each forwarded through the log so far.
        for w in &sem.writes {
            let a = forward(&subst(&w.address, &repl), &mem, &img);
            let v = forward(&subst(&w.value, &repl), &mem, &img);
            mem.push((a, v, w.width.bits()));
        }
        regs = next;
        flags = nflags;
        lifted += 1;
    }

    // Net memory effect (last write wins per slot), computed up-front for classify.
    let mut final_writes_c: BTreeMap<(String, u32), Expr> = BTreeMap::new();
    for (a, v, w) in &mem {
        final_writes_c.insert((pp(a), *w), v.clone());
    }

    // --- CLASSIFY mode: one canonical opcode line (for building the opcode table) ---
    if std::env::var_os("CLASSIFY").is_some() {
        let vip = regs[&Gpr::ALL.iter().find(|g| g.input_name() == "rsi_in").copied().unwrap()].simplify();
        let rsp = regs[&Gpr::ALL.iter().find(|g| g.input_name() == "rsp_in").copied().unwrap()].simplify();
        let vip_s = pp(&vip);
        let rsp_s = pp(&rsp);
        // operation writes = writes whose value reads memory (not pure decoy constants)
        let mut ops: Vec<String> = Vec::new();
        for ((addr, w), v) in &final_writes_c {
            if has_load(v) {
                ops.push(format!("{}=w{} {}", base_tag(addr), w, classify_value(v)));
            }
        }
        ops.sort();
        ops.dedup();
        let op = if ops.is_empty() { "(no-data-write)".to_string() } else { ops.join(",") };
        // compact deltas
        let vip_d = vip_s.strip_prefix("(rsi_in Add ").map(|s| s.trim_end_matches(')').to_string());
        let sp_d = rsp_s.strip_prefix("(rsp_in Add ").map(|s| s.trim_end_matches(')').to_string());
        println!(
            "OP {op} | vip+={} sp+={}",
            vip_d.unwrap_or_else(|| if vip_s == "rsi_in" { "0".into() } else { "?".into() }),
            sp_d.unwrap_or_else(|| if rsp_s == "rsp_in" { "0".into() } else { "?".into() }),
        );
        return;
    }

    eprintln!("body: insns={seen} lifted={lifted} ctrl-skipped={ctrl} lift-failed={failed} stores={}", mem.len());

    // --- net register deltas (the handler's register effect) ---
    println!("=== register deltas (final != entry) ===");
    for &g in &Gpr::ALL {
        let entry = Expr::register(BitWidth::W64, g.input_name()).simplify();
        let now = regs[&g].simplify();
        if now != entry {
            let s = pp(&now);
            let shown = if s.len() > 240 { format!("{}...", &s[..240]) } else { s };
            println!("  {g:?} = {shown}");
        }
    }

    // --- net memory effect: last write wins per (address,width) ---
    println!("\n=== net stack writes (last-write-wins per slot) ===");
    let mut final_writes: BTreeMap<(String, u32), Expr> = BTreeMap::new();
    for (a, v, w) in &mem {
        final_writes.insert((pp(a), *w), v.clone());
    }
    for ((addr, w), v) in &final_writes {
        let s = pp(v);
        let shown = if s.len() > 240 { format!("{}...", &s[..240]) } else { s };
        println!("  [{addr}] <-(w{w}) {shown}");
    }
}
