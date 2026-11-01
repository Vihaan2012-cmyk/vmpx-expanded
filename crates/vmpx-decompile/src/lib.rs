//! Decompilation SPIKE: raises already-recovered IR (native CFG blocks, or the
//! recovered VM CFG) to C-like pseudocode. Fail-closed: a shape the structurer
//! cannot prove falls back to labelled gotos, never to guessed structure.
#![allow(clippy::pedantic, clippy::all, dead_code)]

pub mod ctext;
pub mod exec;
pub mod ir;
pub mod native;
pub mod opt;
pub mod render;
pub mod structure;
pub mod vm;
pub mod xu;

use std::collections::{BTreeMap, BTreeSet};

use vmpx_vm::BitWidth;

use crate::exec::{Env, Rng, run_reference, run_structured};
use crate::ir::{Func, NodeId, Term};
use crate::opt::{LBlock, Stmt};
use crate::render::{Ctx, Renderer, ty};

#[derive(Debug)]
pub struct Decompiled {
    pub text: String,
    /// True when the structurer placed every block; false = flat goto form.
    pub structured: bool,
    pub fallback_reason: Option<String>,
    /// `Ok(n)`: n random vectors agreed with the reference run. `Err`: why the
    /// check could not run (or the first disagreement).
    pub check: Result<usize, String>,
    /// Rendered C text re-evaluated with C semantics vs the IR: (checked, skipped).
    pub text_check: Result<(usize, usize), String>,
    pub loops: usize,
    pub ifs: usize,
    pub gotos: usize,
}

fn x86_name(reg: &str, w: BitWidth) -> Option<String> {
    let (b64, b32, b16, b8) = match reg {
        "rax" => ("rax", "eax", "ax", "al"),
        "rcx" => ("rcx", "ecx", "cx", "cl"),
        "rdx" => ("rdx", "edx", "dx", "dl"),
        "rbx" => ("rbx", "ebx", "bx", "bl"),
        "rbp" => ("rbp", "ebp", "bp", "bpl"),
        "rsi" => ("rsi", "esi", "si", "sil"),
        "rdi" => ("rdi", "edi", "di", "dil"),
        "rsp" => ("rsp", "esp", "sp", "spl"),
        r if r.starts_with('r') && r[1..].chars().all(|c| c.is_ascii_digit()) && r.len() > 1 => {
            return Some(match w {
                BitWidth::W64 => r.to_owned(),
                BitWidth::W32 => format!("{r}d"),
                BitWidth::W16 => format!("{r}w"),
                BitWidth::W8 => format!("{r}b"),
            });
        }
        _ => return None,
    };
    Some(
        match w {
            BitWidth::W64 => b64,
            BitWidth::W32 => b32,
            BitWidth::W16 => b16,
            BitWidth::W8 => b8,
        }
        .to_owned(),
    )
}

fn display_name(v: &str, w: BitWidth) -> String {
    if let Some(base) = v.strip_suffix("_prev") {
        return format!("{}_prev", display_name(base, w));
    }
    x86_name(v, w).unwrap_or_else(|| v.to_owned())
}

const WIN64_ARGS: [&str; 4] = ["rcx", "rdx", "r8", "r9"];

fn flat(blocks: &BTreeMap<NodeId, LBlock>, entry: NodeId) -> Vec<Stmt> {
    let mut out = Vec::new();
    let mut order: Vec<NodeId> = blocks.keys().copied().collect();
    order.retain(|k| *k != entry);
    order.insert(0, entry);
    for (i, id) in order.iter().enumerate() {
        let b = &blocks[id];
        out.push(Stmt::Label(*id));
        out.extend(b.stmts.clone());
        let next = order.get(i + 1).copied();
        match &b.term {
            Term::Return(v) => out.push(Stmt::Return(v.clone())),
            Term::Goto(t) => {
                if Some(*t) != next {
                    out.push(Stmt::Goto(*t));
                }
            }
            Term::Branch { cond, then, els } => {
                out.push(Stmt::IfGoto { cond: cond.clone(), target: *then });
                if Some(*els) != next {
                    out.push(Stmt::Goto(*els));
                }
            }
        }
    }
    out
}

fn count(stmts: &[Stmt], loops: &mut usize, ifs: &mut usize, gotos: &mut usize) {
    for s in stmts {
        match s {
            Stmt::If { then, els, .. } => {
                *ifs += 1;
                count(then, loops, ifs, gotos);
                count(els, loops, ifs, gotos);
            }
            Stmt::Loop(b) | Stmt::While { body: b, .. } | Stmt::DoWhile { body: b, .. } => {
                *loops += 1;
                count(b, loops, ifs, gotos);
            }
            Stmt::Goto(_) | Stmt::IfGoto { .. } => *gotos += 1,
            _ => {}
        }
    }
}

fn collect_decls(stmts: &[Stmt], out: &mut BTreeMap<String, BitWidth>) {
    for s in stmts {
        match s {
            Stmt::Assign(v, e) => {
                out.insert(v.clone(), e.width);
            }
            Stmt::If { then, els, .. } => {
                collect_decls(then, out);
                collect_decls(els, out);
            }
            Stmt::Loop(b) | Stmt::While { body: b, .. } | Stmt::DoWhile { body: b, .. } => {
                collect_decls(b, out);
            }
            _ => {}
        }
    }
}

const REG_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
];

fn input_env(rng: &mut Rng, k: usize) -> Env {
    REG_NAMES
        .iter()
        .map(|r| {
            let x = match k {
                0 => 0,
                1 => u64::MAX,
                _ => rng.next(),
            };
            ((*r).to_owned(), x)
        })
        .collect()
}

/// Runs `f` and each `targets` expression (over register names / loads) on the
/// same random register inputs; the return value must equal every target.
pub fn linkage_vs_exprs(f: &Func, targets: &[vmpx_vm::Expr], vectors: usize) -> Result<usize, String> {
    let mut rng = Rng(0x0BAD_5EED_1234_5678);
    let mut f = f.clone();
    opt::cleanup_all(&mut f);
    for k in 0..vectors {
        let env = input_env(&mut rng, k);
        let want = run_reference(&f, &env)?.ret;
        for (i, t) in targets.iter().enumerate() {
            let got = exec::eval(t, &env).map_err(|e| format!("target {i}: {e}"))?;
            if got != want {
                return Err(format!("vector {k}: decompiled function returns {want:#x}, target {i} evaluates to {got:#x}"));
            }
        }
    }
    Ok(vectors)
}

/// Two IR functions on identical random inputs.
pub fn linkage_vs_func(a: &Func, b: &Func, vectors: usize) -> Result<usize, String> {
    let mut rng = Rng(0x5EED_0BAD_8765_4321);
    let (mut a, mut b) = (a.clone(), b.clone());
    opt::cleanup_all(&mut a);
    opt::cleanup_all(&mut b);
    for k in 0..vectors {
        let env = input_env(&mut rng, k);
        let (x, y) = (run_reference(&a, &env)?, run_reference(&b, &env)?);
        if x.ret != y.ret {
            return Err(format!("vector {k}: {:#x} vs {:#x}", x.ret, y.ret));
        }
    }
    Ok(vectors)
}

/// Human-readable dump of the IR (debugging aid).
pub fn dump_func(f: &Func, depth: u32) -> String {
    let p = |e: &vmpx_vm::Expr| vmpx_verify::differential::pretty(e, depth);
    let mut s = String::new();
    for b in f.blocks.values() {
        s.push_str(&format!("block {:#x}:\n", b.id));
        for (v, e) in &b.assigns {
            s.push_str(&format!("  {v} = {}\n", p(e)));
        }
        match &b.term {
            Term::Goto(t) => s.push_str(&format!("  goto {t:#x}\n")),
            Term::Branch { cond, then, els } => {
                let c: Vec<String> = cond.exprs().into_iter().map(|e| p(e)).collect();
                s.push_str(&format!("  branch {cond:?} [{}] then {then:#x} else {els:#x}\n", c.join(" ; ")));
            }
            Term::Return(v) => s.push_str(&format!("  return {}\n", p(&v[0].1))),
        }
    }
    s
}

/// Runs the whole pipeline on `func` (consumed as the reference).
pub fn decompile(func: &Func) -> Result<Decompiled, String> {
    let reference = {
        let mut r = func.clone();
        opt::cleanup_all(&mut r);
        r
    };
    if std::env::var_os("VMPX_DECOMP_DUMP").is_some() {
        eprintln!("{}", dump_func(&reference, 14));
    }
    let mut f = reference.clone();
    let live_in = opt::dce(&mut f);
    let widths = opt::narrow(&mut f, &live_in);
    // a return value that is a zero-extension everywhere is returned at its narrow type
    let all_zext = f.blocks.values().all(|b| match &b.term {
        Term::Return(v) => matches!(
            v[0].1.kind,
            vmpx_vm::ExprKind::Resize { op: vmpx_vm::ResizeOp::ZeroExtend, .. }
        ),
        _ => true,
    });
    if all_zext {
        for b in f.blocks.values_mut() {
            if let Term::Return(v) = &mut b.term {
                if let vmpx_vm::ExprKind::Resize { value, .. } = &v[0].1.kind {
                    let inner = (**value).clone();
                    v[0].1 = inner;
                }
            }
        }
    }
    let mut seqs: BTreeMap<NodeId, LBlock> = BTreeMap::new();
    for (id, b) in &f.blocks {
        seqs.insert(*id, opt::sequentialize(b)?);
    }

    // parameters in Win64 order, then any other live-in register
    let mut params: Vec<String> = WIN64_ARGS
        .iter()
        .filter(|a| live_in.contains(**a))
        .map(|s| (*s).to_owned())
        .collect();
    for v in &live_in {
        if !params.contains(v) {
            params.push(v.clone());
        }
    }

    let (body, structured, fallback_reason) =
        match structure::Structurer::new(&seqs, f.entry).run() {
            Ok(b) => (b, true, None),
            Err(why) => (flat(&seqs, f.entry), false, Some(why)),
        };

    // variable widths for declarations
    let mut decl: BTreeMap<String, BitWidth> = BTreeMap::new();
    collect_decls(&body, &mut decl);
    let mut assigned: BTreeSet<String> = decl.keys().cloned().collect();
    // flat form: assignments live in label sections; collect from them
    let _ = &mut assigned;

    // names
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    let var_w = |v: &str| -> BitWidth { decl.get(v).copied().or_else(|| widths.get(v).copied()).unwrap_or(BitWidth::W64) };
    let mut all_vars: BTreeSet<String> = decl.keys().cloned().collect();
    all_vars.extend(live_in.iter().cloned());
    let mut prologue: Vec<Stmt> = Vec::new();
    let mut sig: Vec<String> = Vec::new();
    for (i, p) in params.iter().enumerate() {
        let w = var_w(p);
        let arg = format!("a{i}");
        sig.push(format!("{} {arg}", ty(w)));
        if decl.contains_key(p) {
            let dn = display_name(p, w);
            names.insert(p.clone(), dn);
            prologue.push(Stmt::Assign(p.clone(), vmpx_vm::Expr::register(w, format!("\u{1}{arg}"))));
        } else {
            names.insert(p.clone(), arg);
        }
    }
    for v in &all_vars {
        names.entry(v.clone()).or_insert_with(|| display_name(v, var_w(v)));
    }
    names.insert("\u{1}a0".into(), "a0".into());
    for i in 0..params.len() {
        names.insert(format!("\u{1}a{i}"), format!("a{i}"));
    }

    // return type
    fn ret_width(stmts: &[Stmt], w: &mut Option<BitWidth>) {
        for s in stmts {
            match s {
                Stmt::Return(v) => {
                    let ew = v[0].1.width;
                    *w = Some(w.map_or(ew, |o| if o.bits() >= ew.bits() { o } else { ew }));
                }
                Stmt::If { then, els, .. } => {
                    ret_width(then, w);
                    ret_width(els, w);
                }
                Stmt::Loop(b) | Stmt::While { body: b, .. } | Stmt::DoWhile { body: b, .. } => ret_width(b, w),
                _ => {}
            }
        }
    }
    let mut rw = None;
    ret_width(&body, &mut rw);
    let ret_ty = rw.map_or("void", ty);

    let mut ctx = Ctx { r: Renderer { names: &names }, out: String::new(), log: Vec::new() };
    let (mut loops, mut ifs, mut gotos) = (0, 0, 0);
    count(&body, &mut loops, &mut ifs, &mut gotos);
    let mut text = String::new();
    for n in f.notes.iter().chain(func.notes.iter()) {
        text.push_str(&format!("// {n}\n"));
    }
    text.push_str(&format!("{ret_ty} {}({})\n{{\n", func.name, sig.join(", ")));
    let mut by_ty: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for (v, w) in &decl {
        by_ty.entry(ty(*w)).or_default().push(names[v].clone());
    }
    for (t, vs) in by_ty {
        text.push_str(&format!("    {t} {};\n", vs.join(", ")));
    }
    ctx.stmts(&prologue, 1, rw);
    ctx.stmts(&body, 1, rw);
    text.push_str(&ctx.out);
    text.push_str("}\n");

    let text_check = check_text(&ctx.log, &names);

    // cross-check against the reference run
    let check = (|| -> Result<usize, String> {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut full: Vec<Stmt> = prologue.clone();
        full.extend(body.clone());
        let mut n = 0;
        for k in 0..48 {
            let mut inputs = Env::new();
            for v in &live_in {
                let x = match k {
                    0 => 0,
                    1 => u64::MAX,
                    _ => rng.next(),
                };
                inputs.insert(v.clone(), x);
            }
            let mut sinputs = inputs.clone();
            for (i, p) in params.iter().enumerate() {
                sinputs.insert(format!("\u{1}a{i}"), inputs[p]);
            }
            let want = run_reference(&reference, &inputs)?;
            // parameters are typed: mask to the declared width
            let mut pw: BTreeMap<String, BitWidth> = BTreeMap::new();
            for (i, p) in params.iter().enumerate() {
                pw.insert(format!("\u{1}a{i}"), var_w(p));
                if !decl.contains_key(p) {
                    pw.insert(p.clone(), var_w(p));
                }
            }
            let got = run_structured(&full, &sinputs, &pw)?;
            let got_ret = got.ret;
            let mask = rw.map_or(u64::MAX, |w| w.mask());
            if (want.ret & mask) != got_ret || (rw.is_some_and(|w| w.bits() < 64) && want.ret >> rw.unwrap().bits() != 0) || want.stores != got.stores {
                return Err(format!(
                    "MISMATCH on vector {k}: reference ret {:#x}, decompiled ret {:#x}",
                    want.ret, got_ret
                ));
            }
            n += 1;
        }
        Ok(n)
    })();

    Ok(Decompiled { text, structured, fallback_reason, check, text_check, loops, ifs, gotos })
}

fn registers_of(e: &vmpx_vm::Expr, out: &mut BTreeMap<String, BitWidth>) {
    xu::rewrite(e, &mut |n| {
        if let vmpx_vm::ExprKind::Register { name } = &n.kind {
            out.insert(name.clone(), n.width);
            return Some(n.clone());
        }
        None
    });
}

/// Re-evaluates every rendered expression/condition as C text (real C typing
/// rules, UB detection) against the IR expression it came from.
fn check_text(log: &[(render::Logged, String)], names: &BTreeMap<String, String>) -> Result<(usize, usize), String> {
    let mut rng = Rng(0xC0DE_C0DE_1357_9BDF);
    let (mut checked, mut skipped) = (0, 0);
    for (item, text) in log {
        let mut regs = BTreeMap::new();
        match item {
            render::Logged::E(e) => registers_of(e, &mut regs),
            render::Logged::C(c) => c.exprs().into_iter().for_each(|e| registers_of(e, &mut regs)),
        }
        if let render::Logged::C(ir::Cond::Opaque { .. }) = item {
            skipped += 1;
            continue;
        }
        let mut unsupported = false;
        for k in 0..24 {
            let mut env_ir = Env::new();
            let mut env_c = std::collections::HashMap::new();
            for (n, w) in &regs {
                let x = w.truncate(match k {
                    0 => 0,
                    1 => u64::MAX,
                    _ => rng.next(),
                });
                env_ir.insert(n.clone(), x);
                let d = names.get(n).cloned().unwrap_or_else(|| n.clone());
                env_c.insert(d, (x, w.bits()));
            }
            let (want, width_mask) = match item {
                render::Logged::E(e) => match exec::eval(e, &env_ir) {
                    Ok(v) => (v, e.width.mask()),
                    Err(_) => {
                        unsupported = true;
                        break;
                    }
                },
                render::Logged::C(c) => match exec::eval_cond(c, &env_ir) {
                    Ok(b) => (u64::from(b), u64::MAX),
                    Err(_) => {
                        unsupported = true;
                        break;
                    }
                },
            };
            match ctext::eval(text, &env_c) {
                Ok(got) => {
                    let got = if matches!(item, render::Logged::C(_)) { u64::from(got != 0) } else { got & width_mask };
                    if got != want {
                        return Err(format!("rendered text `{text}` evaluates to {got:#x}, IR says {want:#x} (sample {k})"));
                    }
                }
                Err(e) if e.contains("unsupported call") || e.contains("unknown identifier") => {
                    unsupported = true;
                    break;
                }
                Err(e) => return Err(format!("rendered text `{text}`: {e}")),
            }
        }
        if unsupported {
            skipped += 1;
        } else {
            checked += 1;
        }
    }
    Ok((checked, skipped))
}
