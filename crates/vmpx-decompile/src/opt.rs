//! Function-level passes over [`Func`]: cleanup, dead-assign elimination,
//! variable width narrowing, and per-block sequentialization.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp};

use crate::ir::{Block, Cond, Func, NodeId, Store, Term};
use crate::xu::{as_trunc, cleanup, is_leaf, rewrite, trunc_to};

/// Every expression slot of a block, mutably.
fn for_each_expr_mut(b: &mut Block, f: &mut dyn FnMut(&mut Expr)) {
    for (_, e) in &mut b.assigns {
        f(e);
    }
    for s in &mut b.stores {
        f(&mut s.address);
        f(&mut s.value);
    }
    match &mut b.term {
        Term::Goto(_) => {}
        Term::Branch { cond, .. } => match cond {
            Cond::Cmp { a, b, .. } => {
                f(a);
                f(b);
            }
            Cond::Opaque { args, .. } => args.iter_mut().for_each(&mut *f),
        },
        Term::Return(v) => v.iter_mut().for_each(|(_, e)| f(e)),
    }
}

pub fn cleanup_all(func: &mut Func) {
    for b in func.blocks.values_mut() {
        for_each_expr_mut(b, &mut |e| *e = cleanup(e));
    }
}

fn regs_of(e: &Expr, out: &mut BTreeSet<String>) {
    out.extend(e.dependencies());
}

fn term_reads(t: &Term) -> BTreeSet<String> {
    let mut s = BTreeSet::new();
    match t {
        Term::Goto(_) => {}
        Term::Branch { cond, .. } => cond.exprs().into_iter().for_each(|e| regs_of(e, &mut s)),
        Term::Return(v) => v.iter().for_each(|(_, e)| regs_of(e, &mut s)),
    }
    s
}

/// Live-in variable sets per block, with assignments dead at their block's exit
/// dropped. Returns the live-in set of the entry.
pub fn dce(func: &mut Func) -> BTreeSet<String> {
    let ids: Vec<NodeId> = func.blocks.keys().copied().collect();
    let mut live_in: BTreeMap<NodeId, BTreeSet<String>> =
        ids.iter().map(|i| (*i, BTreeSet::new())).collect();
    loop {
        let mut changed = false;
        for id in ids.iter().rev() {
            let b = &func.blocks[id];
            let mut live_out: BTreeSet<String> = BTreeSet::new();
            for s in func.successors(*id) {
                live_out.extend(live_in[&s].iter().cloned());
            }
            let mut uses = term_reads(&b.term);
            for s in &b.stores {
                regs_of(&s.address, &mut uses);
                regs_of(&s.value, &mut uses);
            }
            let mut defs = BTreeSet::new();
            for (v, d) in &b.assigns {
                defs.insert(v.clone());
                if live_out.contains(v) {
                    regs_of(d, &mut uses);
                }
            }
            let mut li = uses;
            li.extend(live_out.difference(&defs).cloned());
            if li != live_in[id] {
                live_in.insert(*id, li);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for id in &ids {
        let mut live_out: BTreeSet<String> = BTreeSet::new();
        for s in func.successors(*id) {
            live_out.extend(live_in[&s].iter().cloned());
        }
        let b = func.blocks.get_mut(id).expect("block");
        b.assigns.retain(|(v, d)| {
            live_out.contains(v) && !matches!(&d.kind, ExprKind::Register { name } if name == v)
        });
    }
    live_in[&func.entry].clone()
}

#[derive(Default)]
struct ReadInfo {
    max_trunc: u32,
    full: bool,
}

fn scan(
    e: &Expr,
    ctx: Option<u32>,
    reads: &mut HashMap<String, ReadInfo>,
    seen: &mut HashSet<(Expr, Option<u32>)>,
) {
    if !seen.insert((e.clone(), ctx)) {
        return;
    }
    match &e.kind {
        ExprKind::Const { .. } => {}
        ExprKind::Register { name } => {
            let r = reads.entry(name.clone()).or_default();
            match ctx {
                Some(w) => r.max_trunc = r.max_trunc.max(w),
                None => r.full = true,
            }
        }
        _ => {
            if let Some(inner) = as_trunc(e) {
                if matches!(inner.kind, ExprKind::Register { .. }) {
                    scan(inner, Some(e.width.bits()), reads, seen);
                    return;
                }
            }
            let mut kids: Vec<&Expr> = Vec::new();
            match &e.kind {
                ExprKind::Unary { value, .. }
                | ExprKind::Extract { value, .. }
                | ExprKind::Resize { value, .. } => kids.push(value),
                ExprKind::Binary { left, right, .. } => {
                    kids.push(left);
                    kids.push(right);
                }
                ExprKind::Concat { high, low } => {
                    kids.push(high);
                    kids.push(low);
                }
                ExprKind::Load { address } => kids.push(address),
                ExprKind::Intrinsic { arguments, .. } => kids.extend(arguments.iter()),
                ExprKind::Const { .. } | ExprKind::Register { .. } => {}
            }
            for k in kids {
                scan(k, None, reads, seen);
            }
        }
    }
}

fn bw(bits: u32) -> BitWidth {
    BitWidth::from_bits(bits).expect("width")
}

fn round_up(bits: u32) -> u32 {
    match bits {
        0..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

/// Returns an expression equal to the low `min(k, e.width)` bits of `e`,
/// simplified under that demand (bits above are never looked at).
pub fn narrow_expr(e: &Expr, k: u32) -> Expr {
    let mut memo: HashMap<(Expr, u32), Expr> = HashMap::new();
    ne(e, k, &mut memo)
}

fn zext_to(t: BitWidth, x: Expr) -> Expr {
    if x.width == t {
        x
    } else {
        Expr::resize(ResizeOp::ZeroExtend, t, x.clone()).unwrap_or(x)
    }
}

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::binary(op, l.clone(), r).unwrap_or(l)
}

fn ne(e: &Expr, k: u32, memo: &mut HashMap<(Expr, u32), Expr>) -> Expr {
    if let Some(r) = memo.get(&(e.clone(), k)) {
        return r.clone();
    }
    let out = ne_inner(e, k, memo);
    memo.insert((e.clone(), k), out.clone());
    out
}

fn generic(e: &Expr, t: BitWidth, memo: &mut HashMap<(Expr, u32), Expr>) -> Expr {
    let e = &peel_shift_count(e);
    let kids = crate::xu::map_children(e, &mut |c| ne(c, c.width.bits(), memo));
    trunc_to(t, &kids)
}

fn peel_shift_count(e: &Expr) -> Expr {
    if let ExprKind::Binary { op, left, right } = &e.kind
        && matches!(op, BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Sar | BinaryOp::Rol | BinaryOp::Ror)
        && let Some(p) = crate::xu::peel_count(right, e.width, matches!(op, BinaryOp::Rol | BinaryOp::Ror))
        && p != **right
        && let Ok(n) = Expr::binary(*op, (**left).clone(), p)
    {
        return n;
    }
    e.clone()
}

fn ne_inner(e: &Expr, k: u32, memo: &mut HashMap<(Expr, u32), Expr>) -> Expr {
    let t = bw(k.min(e.width.bits()));
    match &e.kind {
        ExprKind::Const { value } => Expr::constant(t, *value),
        ExprKind::Register { .. } => trunc_to(t, e),
        ExprKind::Unary { op, value } => Expr::unary(*op, ne(value, t.bits(), memo)),
        ExprKind::Binary { op, left, right } => match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Xor | BinaryOp::Or => {
                bin(*op, ne(left, t.bits(), memo), ne(right, t.bits(), memo))
            }
            BinaryOp::And => {
                let (x, c) = match (&left.kind, &right.kind) {
                    (_, ExprKind::Const { value }) => (left, Some(*value)),
                    (ExprKind::Const { value }, _) => (right, Some(*value)),
                    _ => (left, None),
                };
                if let Some(c) = c {
                    let ct = t.truncate(c);
                    if ct == 0 {
                        return Expr::constant(t, 0);
                    }
                    if ct == t.mask() {
                        return ne(x, t.bits(), memo);
                    }
                    let dmd = round_up(64 - ct.leading_zeros()).min(t.bits());
                    let inner = ne(x, dmd, memo);
                    let masked = bin(BinaryOp::And, inner.clone(), Expr::constant(inner.width, ct));
                    return zext_to(t, masked);
                }
                bin(BinaryOp::And, ne(left, t.bits(), memo), ne(right, t.bits(), memo))
            }
            BinaryOp::Shl => {
                if let ExprKind::Const { value } = right.kind {
                    let mask = if e.width == BitWidth::W64 { 63 } else { 31 };
                    let cm = value & mask;
                    if cm >= u64::from(t.bits()) {
                        return Expr::constant(t, 0);
                    }
                    let dmd = round_up(t.bits() - u32::try_from(cm).unwrap_or(0));
                    let inner = zext_to(t, ne(left, dmd, memo));
                    return bin(BinaryOp::Shl, inner, Expr::constant(t, cm));
                }
                generic(e, t, memo)
            }
            BinaryOp::Shr => {
                if let ExprKind::Const { value } = right.kind {
                    let mask = if e.width == BitWidth::W64 { 63 } else { 31 };
                    let cm = value & mask;
                    let need = u32::try_from(cm).unwrap_or(0) + t.bits();
                    if need <= e.width.bits() {
                        let m = bw(round_up(need));
                        if m.bits() < e.width.bits() {
                            let inner = ne(left, m.bits(), memo);
                            let shifted = bin(BinaryOp::Shr, inner, Expr::constant(m, cm));
                            return trunc_to(t, &shifted);
                        }
                    }
                }
                generic(e, t, memo)
            }
            _ => generic(e, t, memo),
        },
        ExprKind::Resize { op: ResizeOp::Truncate, value } => ne(value, t.bits(), memo),
        ExprKind::Extract { value, lsb: 0 } => ne(value, t.bits(), memo),
        ExprKind::Resize { op: ResizeOp::ZeroExtend, value } => {
            if t.bits() <= value.width.bits() {
                ne(value, t.bits(), memo)
            } else {
                zext_to(t, ne(value, value.width.bits(), memo))
            }
        }
        _ => generic(e, t, memo),
    }
}

/// Shrinks 64-bit register variables to their natural width via demanded-bits
/// narrowing. A variable of width `w` is sound when every read looks only at its
/// low `w` bits after each use is narrowed (A), or when every definition is a
/// zero-extension from `w` bits and it has no entry value (B).
pub fn narrow(func: &mut Func, live_in: &BTreeSet<String>) -> BTreeMap<String, BitWidth> {
    let mut vars: BTreeSet<String> = live_in.clone();
    for b in func.blocks.values() {
        vars.extend(b.assigns.iter().map(|(v, _)| v.clone()));
    }
    let mut widths: BTreeMap<String, BitWidth> = vars.iter().map(|v| (v.clone(), bw(8))).collect();
    let mut full_vars: BTreeSet<String> = BTreeSet::new();
    let narrowed_func = |widths: &BTreeMap<String, BitWidth>, func: &Func| -> Func {
        let mut f = func.clone();
        let nar = |e: &Expr| cleanup(&narrow_expr(e, e.width.bits()));
        for b in f.blocks.values_mut() {
            for (v, d) in &mut b.assigns {
                let w = widths.get(v).map_or(64, |w| w.bits());
                *d = cleanup(&narrow_expr(d, w));
            }
            for_each_expr_mut_nonassign(b, &nar);
        }
        f
    };
    for _ in 0..16 {
        let f = narrowed_func(&widths, func);
        let mut reads: HashMap<String, ReadInfo> = HashMap::new();
        let mut seen = HashSet::new();
        for b in f.blocks.values() {
            for (_, d) in &b.assigns {
                scan(d, None, &mut reads, &mut seen);
            }
            for s in &b.stores {
                scan(&s.address, None, &mut reads, &mut seen);
                scan(&s.value, None, &mut reads, &mut seen);
            }
            match &b.term {
                Term::Goto(_) => {}
                Term::Branch { cond, .. } => cond
                    .exprs()
                    .into_iter()
                    .for_each(|e| scan(e, None, &mut reads, &mut seen)),
                Term::Return(v) => v.iter().for_each(|(_, e)| scan(e, None, &mut reads, &mut seen)),
            }
        }
        let mut changed = false;
        full_vars.clear();
        for v in &vars {
            let cur = widths[v];
            let new = match reads.get(v) {
                Some(r) if !r.full => bw(r.max_trunc.max(8)),
                Some(_) => {
                    full_vars.insert(v.clone());
                    bw(64)
                }
                None => bw(8),
            };
            if new.bits() > cur.bits() {
                widths.insert(v.clone(), new);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // rule B for variables still read in full
    let nf = narrowed_func(&widths, func);
    for v in &full_vars {
        if live_in.contains(v) {
            continue;
        }
        let mut wmax = 0;
        let mut ok = true;
        for b in func.blocks.values() {
            for (u, d) in &b.assigns {
                if u != v {
                    continue;
                }
                match &cleanup(d).kind {
                    ExprKind::Resize { op: ResizeOp::ZeroExtend, value } => {
                        wmax = wmax.max(value.width.bits());
                    }
                    ExprKind::Const { value } => {
                        wmax = wmax.max(round_up(64 - value.leading_zeros()));
                    }
                    _ => ok = false,
                }
            }
        }
        if ok && wmax > 0 && wmax < 64 {
            widths.insert(v.clone(), bw(wmax));
        }
    }
    let _ = nf;
    // final rewrite: narrowed expressions, then retarget register reads
    let mut f = narrowed_func(&widths, func);
    let wmap = widths.clone();
    let fix = move |e: &Expr| -> Expr {
        let out = rewrite(e, &mut |n| {
            if let Some(inner) = as_trunc(n)
                && let ExprKind::Register { name } = &inner.kind
                && let Some(w) = wmap.get(name)
                && w.bits() < inner.width.bits()
            {
                let r = Expr::register(*w, name.clone());
                return Some(trunc_to(n.width, &r));
            }
            if let ExprKind::Register { name } = &n.kind
                && let Some(w) = wmap.get(name)
                && w.bits() < n.width.bits()
            {
                let r = Expr::register(*w, name.clone());
                return Some(
                    Expr::resize(ResizeOp::ZeroExtend, n.width, r).unwrap_or_else(|_| n.clone()),
                );
            }
            None
        });
        cleanup(&crate::xu::mba_fold(&cleanup(&out)))
    };
    for b in f.blocks.values_mut() {
        for (v, d) in &mut b.assigns {
            let nd = fix(d);
            let w = widths[v];
            *d = if nd.width.bits() > w.bits() { cleanup(&trunc_to(w, &nd)) } else { nd };
        }
        for_each_expr_mut_nonassign(b, &fix);
    }
    *func = f;
    widths
}

fn for_each_expr_mut_nonassign(b: &mut Block, f: &dyn Fn(&Expr) -> Expr) {
    for s in &mut b.stores {
        s.address = f(&s.address);
        s.value = f(&s.value);
    }
    match &mut b.term {
        Term::Goto(_) => {}
        Term::Branch { cond, .. } => match cond {
            Cond::Cmp { a, b, .. } => {
                *a = f(a);
                *b = f(b);
            }
            Cond::Opaque { args, .. } => args.iter_mut().for_each(|e| *e = f(e)),
        },
        Term::Return(v) => v.iter_mut().for_each(|(_, e)| *e = f(e)),
    }
}

// ---------------------------------------------------------------- sequentialize

#[derive(Clone, Debug)]
pub enum Stmt {
    Assign(String, Expr),
    Store(Store),
    If { cond: Cond, then: Vec<Stmt>, els: Vec<Stmt> },
    Loop(Vec<Stmt>),
    While { cond: Cond, body: Vec<Stmt> },
    DoWhile { body: Vec<Stmt>, cond: Cond },
    Break,
    Continue,
    Return(Vec<(String, Expr)>),
    Label(NodeId),
    Goto(NodeId),
    IfGoto { cond: Cond, target: NodeId },
    Comment(String),
}

#[derive(Clone, Debug)]
pub struct LBlock {
    pub id: NodeId,
    pub stmts: Vec<Stmt>,
    pub term: Term,
}

const NEW: &str = "@new";

fn new_ref(v: &str, w: BitWidth) -> Expr {
    Expr::register(w, format!("{v}{NEW}"))
}

fn subst_names(e: &Expr, map: &HashMap<String, Expr>) -> Expr {
    if map.is_empty() {
        return e.clone();
    }
    rewrite(e, &mut |n| match &n.kind {
        ExprKind::Register { name } => map.get(name).cloned(),
        _ => None,
    })
}

fn strip_new(e: &Expr) -> Expr {
    rewrite(e, &mut |n| match &n.kind {
        ExprKind::Register { name } => name
            .strip_suffix(NEW)
            .map(|b| Expr::register(n.width, b.to_owned())),
        _ => None,
    })
}

fn flatten_ac(e: &Expr, op: BinaryOp, out: &mut Vec<Expr>) {
    if let ExprKind::Binary { op: o, left, right } = &e.kind
        && *o == op
    {
        flatten_ac(left, op, out);
        flatten_ac(right, op, out);
    } else {
        out.push(e.clone());
    }
}

/// Replaces subtrees equal to another assignment's value by a reference to that
/// variable's new value; for associative-commutative chains also when the other
/// value's operands are a sub-multiset of the chain's.
fn cse(
    e: &Expr,
    own: Option<&str>,
    table: &HashMap<Expr, String>,
    defs: &[(String, Expr)],
    var_w: &HashMap<String, BitWidth>,
) -> Expr {
    rewrite(e, &mut |n| {
        if is_leaf(n) {
            return None;
        }
        if let Some(v) = table.get(n) {
            if Some(v.as_str()) == own {
                return None;
            }
            return Some(new_ref(v, var_w[v]));
        }
        let ExprKind::Binary { op, .. } = &n.kind else { return None };
        if !matches!(op, BinaryOp::Add | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or | BinaryOp::Mul) {
            return None;
        }
        let mut ln = Vec::new();
        flatten_ac(n, *op, &mut ln);
        let mut best: Option<(usize, &str, Vec<Expr>)> = None;
        for (v, d) in defs {
            if Some(v.as_str()) == own || d.width != n.width {
                continue;
            }
            if !matches!(&d.kind, ExprKind::Binary { op: o, .. } if o == op) {
                continue;
            }
            let mut li = Vec::new();
            flatten_ac(d, *op, &mut li);
            if li.len() < 2 || li.len() >= ln.len() {
                continue;
            }
            let mut rest = ln.clone();
            let mut ok = true;
            for x in &li {
                match rest.iter().position(|r| r == x) {
                    Some(p) => {
                        rest.remove(p);
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && best.as_ref().is_none_or(|(l, _, _)| li.len() > *l) {
                best = Some((li.len(), v.as_str(), rest));
            }
        }
        let (_, v, rest) = best?;
        let mut acc = new_ref(v, var_w[v]);
        for r in rest {
            let r = cse(&r, own, table, defs, var_w);
            acc = Expr::binary(*op, acc.clone(), r).unwrap_or(acc);
        }
        Some(acc)
    })
}

/// Turns a block's parallel assignments (all reads see block-entry values) into
/// an ordered statement list. Subtrees equal to another assignment's value are
/// replaced by that variable; entry values overwritten before a late read are
/// saved in `<v>_prev` temporaries.
pub fn sequentialize(b: &Block) -> Result<LBlock, String> {
    let assigns: Vec<(String, Expr)> = b.assigns.clone();
    // table: expr -> first variable that holds it (non-leaf only)
    let mut table: HashMap<Expr, String> = HashMap::new();
    for (v, d) in &assigns {
        if !is_leaf(d) {
            table.entry(d.clone()).or_insert_with(|| v.clone());
        }
    }
    let var_w: HashMap<String, BitWidth> = assigns.iter().map(|(v, d)| (v.clone(), d.width)).collect();
    let replace = |e: &Expr, own_root_of: Option<&str>| -> Expr {
        cse(e, own_root_of, &table, &assigns, &var_w)
    };
    let mut rhs: Vec<(String, Expr)> = assigns
        .iter()
        .map(|(v, d)| (v.clone(), replace(d, Some(v))))
        .collect();
    // equal definitions: later copies the earlier (handled by replace: root maps to first var)
    let term = match &b.term {
        Term::Goto(t) => Term::Goto(*t),
        Term::Branch { cond, then, els } => Term::Branch {
            cond: cond.map_exprs(&mut |e| replace(e, None)),
            then: *then,
            els: *els,
        },
        Term::Return(v) => Term::Return(v.iter().map(|(n, e)| (n.clone(), replace(e, None))).collect()),
    };
    let mut tails: Vec<Expr> = Vec::new();
    match &term {
        Term::Branch { cond, .. } => tails.extend(cond.exprs().into_iter().cloned()),
        Term::Return(v) => tails.extend(v.iter().map(|(_, e)| e.clone())),
        Term::Goto(_) => {}
    }
    let deps_new = |e: &Expr| -> BTreeSet<String> {
        e.dependencies()
            .into_iter()
            .filter_map(|d| d.strip_suffix(NEW).map(str::to_owned))
            .collect()
    };
    let assigned: BTreeSet<String> = rhs.iter().map(|(v, _)| v.clone()).collect();
    let mut stmts: Vec<Stmt> = b.stores.iter().cloned().map(Stmt::Store).collect();
    let mut remaining: Vec<usize> = (0..rhs.len()).collect();
    let mut emitted: BTreeSet<String> = BTreeSet::new();
    let mut tail_map: HashMap<String, Expr> = HashMap::new();
    let mut assign_maps: HashMap<usize, HashMap<String, Expr>> = HashMap::new();
    while !remaining.is_empty() {
        // candidates whose hard deps (new-refs) are all emitted
        let ready: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&j| deps_new(&rhs[j].1).iter().all(|u| emitted.contains(u) || *u == rhs[j].0))
            .collect();
        if ready.is_empty() {
            return Err(format!("block {:#x}: cyclic new-value dependencies", b.id));
        }
        let pending_readers = |j: usize| -> usize {
            let v = &rhs[j].0;
            remaining
                .iter()
                .filter(|&&k| k != j && rhs[k].1.dependencies().contains(v))
                .count()
        };
        // j reaches itself through "reads the entry value of" edges among the remaining
        let on_cycle = |j: usize| -> bool {
            let mut seen: BTreeSet<usize> = BTreeSet::new();
            let mut st: Vec<usize> = vec![j];
            while let Some(x) = st.pop() {
                // x must precede every remaining k whose variable x reads
                for &k in &remaining {
                    if k != x || k == j {
                        let kv = &rhs[k].0;
                        if k != x && rhs[x].1.dependencies().contains(kv) {
                            if k == j {
                                return true;
                            }
                            if seen.insert(k) {
                                st.push(k);
                            }
                        }
                    }
                }
            }
            false
        };
        let pick = if let Some(&j) = ready.iter().find(|&&j| pending_readers(j) == 0) {
            j
        } else if let Some(&j) = ready
            .iter()
            .filter(|&&j| on_cycle(j))
            .min_by_key(|&&j| pending_readers(j))
        {
            j
        } else {
            *ready.iter().min_by_key(|&&j| pending_readers(j)).expect("ready")
        };
        let v = rhs[pick].0.clone();
        let readers: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&k| k != pick && rhs[k].1.dependencies().contains(&v))
            .collect();
        let tail_reads = tails.iter().any(|t| t.dependencies().contains(&v));
        if !readers.is_empty() || tail_reads {
            let w = var_w[&v];
            let temp = format!("{v}_prev");
            let tr = Expr::register(w, temp.clone());
            // save the entry value only if some later reader still needs it
            let mut m = HashMap::new();
            m.insert(v.clone(), tr);
            stmts.push(Stmt::Assign(temp, Expr::register(w, v.clone())));
            for k in readers {
                rhs[k].1 = subst_names(&rhs[k].1, &m);
            }
            tail_map.insert(v.clone(), m[&v].clone());
            let _ = &assign_maps;
        }
        stmts.push(Stmt::Assign(v.clone(), strip_new(&rhs[pick].1)));
        emitted.insert(v);
        remaining.retain(|&k| k != pick);
        let _ = &assigned;
    }
    let fix_tail = |e: &Expr| strip_new(&subst_names(e, &tail_map));
    let term = match term {
        Term::Goto(t) => Term::Goto(t),
        Term::Branch { cond, then, els } => Term::Branch {
            cond: cond.map_exprs(&mut |e| fix_tail(e)),
            then,
            els,
        },
        Term::Return(v) => Term::Return(v.into_iter().map(|(n, e)| (n, fix_tail(&e))).collect()),
    };
    Ok(LBlock { id: b.id, stmts, term })
}
