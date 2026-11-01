//! Frontend: rolled VM recovery ([`RolledFunction`]) -> [`Func`].
//!
//! Steps: collapse protector environment gates, merge byte-lane loop phis into
//! wide variables, name entry registers, resolve `memory_load_epoch_n` reads,
//! fold MBA / double shifts, decode branch selectors into comparisons (by
//! evaluation on the project's own flag semantics), lower the tree to blocks.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, X64NativeRegister};
use vmpx_vmlift::{RExit, RNode, RPhi, RolledFunction};

use crate::exec::Rng;
use crate::ir::{Block, Cmp, Cond, Func, NodeId, Term};
use crate::xu::{cleanup, mba_fold, rewrite};

pub struct VmLowered {
    pub func: Func,
    pub notes: Vec<String>,
}

fn reg_name(r: X64NativeRegister) -> &'static str {
    use X64NativeRegister as R;
    match r {
        R::Rax => "rax",
        R::Rbx => "rbx",
        R::Rcx => "rcx",
        R::Rdx => "rdx",
        R::Rbp => "rbp",
        R::Rsi => "rsi",
        R::Rdi => "rdi",
        R::R8 => "r8",
        R::R9 => "r9",
        R::R10 => "r10",
        R::R11 => "r11",
        R::R12 => "r12",
        R::R13 => "r13",
        R::R14 => "r14",
        R::R15 => "r15",
    }
}

fn is_callee_saved(r: X64NativeRegister) -> bool {
    use X64NativeRegister as R;
    matches!(r, R::Rbx | R::Rbp | R::Rsi | R::Rdi | R::R12 | R::R13 | R::R14 | R::R15)
}

/// Entry symbols -> register variables; memory-epoch reads -> loads; double
/// shifts -> rotates; then width/MBA cleanups.
pub fn fix(e: &Expr) -> Expr {
    let renamed = rewrite(e, &mut |n| match &n.kind {
        ExprKind::Register { name } => {
            let base = name.strip_prefix("native_")?.strip_suffix("_initial")?;
            Some(Expr::register(n.width, base.to_owned()))
        }
        ExprKind::Intrinsic { name, arguments } if name.starts_with("memory_load_epoch_") => {
            name["memory_load_epoch_".len()..].parse::<u64>().ok()?;
            let [addr] = arguments.as_slice() else { return None };
            Some(Expr::load(n.width, fix(addr)))
        }
        _ => None,
    });
    let canon = vmpx_verify::differential::canonicalize_recovered(&renamed).unwrap_or(renamed);
    mba_fold(&cleanup(&canon))
}

fn hash_expr(e: &Expr, h: &mut impl Hasher) {
    e.hash(h);
}

// ---------------------------------------------------------------- gate collapse

fn is_live(n: &RNode) -> bool {
    match n {
        RNode::Exit(x) => !x.region,
        RNode::Fork { arms, .. } => arms.iter().any(|(_, a)| is_live(a)),
        RNode::Loop { body, .. } => is_live(body),
        RNode::BackEdge { .. } => true,
        RNode::Dead(_) => false,
    }
}

/// Structure hash with phi names replaced by their position in the enclosing loop.
fn signature(n: &RNode, rename: &BTreeMap<String, String>, h: &mut impl Hasher) {
    let sub = |e: &Expr| -> Expr {
        if rename.is_empty() {
            return e.clone();
        }
        rewrite(e, &mut |x| match &x.kind {
            ExprKind::Register { name } => rename
                .get(name)
                .map(|r| Expr::register(x.width, r.clone())),
            _ => None,
        })
    };
    match n {
        RNode::Exit(x) => {
            0_u8.hash(h);
            x.region.hash(h);
            for (r, e) in &x.registers {
                reg_name(*r).hash(h);
                hash_expr(&sub(e), h);
            }
        }
        RNode::Fork { selector, env, arms, .. } => {
            1_u8.hash(h);
            env.hash(h);
            hash_expr(&sub(selector), h);
            for (v, a) in arms {
                v.hash(h);
                signature(a, rename, h);
            }
        }
        RNode::Loop { phis, body, .. } => {
            2_u8.hash(h);
            let mut m = rename.clone();
            for (i, p) in phis.iter().enumerate() {
                hash_expr(&sub(&p.init), h);
                m.insert(p.name.clone(), format!("__p{}_{i}", rename.len()));
            }
            signature(body, &m, h);
        }
        RNode::BackEdge { next } => {
            3_u8.hash(h);
            for e in next {
                hash_expr(&sub(e), h);
            }
        }
        RNode::Dead(_) => 4_u8.hash(h),
    }
}

fn sig(n: &RNode) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    signature(n, &BTreeMap::new(), &mut h);
    h.finish()
}

/// Removes environment-gate forks: failure arms (region exit / dead) are dropped;
/// when several live arms remain they must be structurally identical.
fn collapse(n: &RNode, notes: &mut Vec<String>) -> Result<RNode, String> {
    match n {
        RNode::Fork { selector, env: true, block, arms } => {
            let _ = selector;
            let mut live: Vec<RNode> = Vec::new();
            for (_, a) in arms {
                if !is_live(a) {
                    continue;
                }
                let c = collapse(a, notes)?;
                if is_live(&c) {
                    live.push(c);
                }
            }
            match live.len() {
                0 => Ok(RNode::Dead(format!("every arm of the gate at {block} fails"))),
                1 => {
                    notes.push(format!("environment gate at {block}: {} failing arm(s) dropped", arms.len() - 1));
                    Ok(live.remove(0))
                }
                n => {
                    let first = sig(&live[0]);
                    if live.iter().all(|a| sig(a) == first) {
                        notes.push(format!("environment gate at {block}: {n} identical live arms collapsed"));
                        Ok(live.remove(0))
                    } else {
                        Err(format!("environment gate at {block} has {n} live arms that differ after collapse"))
                    }
                }
            }
        }
        RNode::Fork { selector, env, block, arms } => Ok(RNode::Fork {
            selector: selector.clone(),
            env: *env,
            block: *block,
            arms: arms
                .iter()
                .map(|(v, a)| Ok((*v, collapse(a, notes)?)))
                .collect::<Result<_, String>>()?,
        }),
        RNode::Loop { header, phis, body } => Ok(RNode::Loop {
            header: *header,
            phis: phis.clone(),
            body: Box::new(collapse(body, notes)?),
        }),
        other => Ok(other.clone()),
    }
}

// ----------------------------------------------------------------- lane merge

fn flatten_cat(e: &Expr, out: &mut Vec<Expr>) {
    if let ExprKind::Concat { high, low } = &e.kind {
        flatten_cat(high, out);
        flatten_cat(low, out);
    } else {
        out.push(e.clone());
    }
}

/// Concatenates a power-of-two list of equal-width pieces, first = most significant.
fn concat_balanced(parts: &[Expr]) -> Option<Expr> {
    if parts.len() == 1 {
        return Some(parts[0].clone());
    }
    let half = parts.len() / 2;
    let hi = concat_balanced(&parts[..half])?;
    let lo = concat_balanced(&parts[half..])?;
    Expr::concat(hi, lo).ok()
}

struct Group {
    wide: String,
    width: BitWidth,
    /// Set when the next value is simply another group's wide variable (a copy).
    next_from: Option<usize>,
    /// Lane phis, index = lane (0 = least significant byte).
    lanes: Vec<String>,
}

fn ext_lane(e: &Expr) -> Option<(&Expr, u32)> {
    match &e.kind {
        ExprKind::Extract { value, lsb } if e.width == BitWidth::W8 => Some((value, *lsb)),
        _ => None,
    }
}

/// Finds 8-lane byte-phi groups of one loop and the rewrite that replaces them
/// with one 64-bit phi each.
fn merge_lanes(phis: &[RPhi], backs: &[&Vec<Expr>], uid: &mut usize) -> (Vec<Group>, Vec<Option<Expr>>) {
    // key: hash of the inner expression of the first back edge's next, per phi
    let mut by_key: HashMap<u64, Vec<(usize, u32)>> = HashMap::new();
    let mut inner: HashMap<u64, Expr> = HashMap::new();
    for (i, p) in phis.iter().enumerate() {
        if p.width != BitWidth::W8 {
            continue;
        }
        let Some(first) = backs.first() else { continue };
        let Some((h, lane)) = ext_lane(&first[i]) else { continue };
        if lane % 8 != 0 {
            continue;
        }
        let mut hh = std::collections::hash_map::DefaultHasher::new();
        h.hash(&mut hh);
        let k = hh.finish();
        by_key.entry(k).or_default().push((i, lane / 8));
        inner.insert(k, h.clone());
    }
    let mut groups = Vec::new();
    let mut wide_inits: Vec<Option<Expr>> = vec![None; phis.len()];
    let mut keys: Vec<&u64> = by_key.keys().collect();
    keys.sort();
    for k in keys {
        let lanes = &by_key[k];
        if !matches!(lanes.len(), 2 | 4 | 8) {
            continue;
        }
        let mut order: Vec<Option<usize>> = vec![None; lanes.len()];
        if lanes.iter().any(|(_, l)| *l as usize >= lanes.len()) {
            continue;
        }
        for (i, l) in lanes {
            order[*l as usize] = Some(*i);
        }
        if order.iter().any(Option::is_none) {
            continue;
        }
        let order: Vec<usize> = order.into_iter().map(|x| x.expect("lane")).collect();
        // every back edge must agree on the inner expression and lane
        let consistent = backs.iter().all(|b| {
            let Some((h0, _)) = ext_lane(&b[order[0]]) else { return false };
            order.iter().enumerate().all(|(lane, &i)| {
                ext_lane(&b[i]).is_some_and(|(h, l)| l == (lane as u32) * 8 && h == h0)
            })
        });
        if !consistent {
            continue;
        }
        // init: concatenate lane inits (balanced; Expr widths are 8/16/32/64 only)
        let lane_inits: Vec<Expr> = order.iter().rev().map(|&i| phis[i].init.clone()).collect();
        let Some(init) = concat_balanced(&lane_inits) else { continue };
        let init = Some(init);
        *uid += 1;
        let wide = format!("v{}", *uid - 1);
        let first = order[0];
        wide_inits[first] = init;
        let width = BitWidth::from_bits(8 * order.len() as u32).expect("lane width");
        groups.push(Group { wide, width, next_from: None, lanes: order.iter().map(|&i| phis[i].name.clone()).collect() });
    }
    // copy groups: byte phis whose next is just lane l of an existing group
    let mut lane_of: HashMap<String, (usize, usize)> = HashMap::new();
    for (g, grp) in groups.iter().enumerate() {
        for (l, n) in grp.lanes.iter().enumerate() {
            lane_of.insert(n.clone(), (g, l));
        }
    }
    let grouped: std::collections::HashSet<&str> = lane_of.keys().map(String::as_str).collect();
    let mut buckets: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    if let Some(first) = backs.first() {
        for (i, p) in phis.iter().enumerate() {
            if p.width != BitWidth::W8 || grouped.contains(p.name.as_str()) {
                continue;
            }
            if let ExprKind::Register { name } = &first[i].kind
                && let Some(&(g, l)) = lane_of.get(name.as_str())
            {
                buckets.entry(g).or_default().push((i, l));
            }
        }
    }
    for (g, items) in buckets {
        let k = groups[g].lanes.len();
        if items.len() != k {
            continue;
        }
        let mut order: Vec<Option<usize>> = vec![None; k];
        for (i, l) in &items {
            order[*l] = Some(*i);
        }
        if order.iter().any(Option::is_none) {
            continue;
        }
        let order: Vec<usize> = order.into_iter().map(|x| x.expect("lane")).collect();
        let ok = backs.iter().all(|b| {
            order.iter().enumerate().all(|(l, &i)| {
                matches!(&b[i].kind, ExprKind::Register { name } if *name == groups[g].lanes[l])
            })
        });
        if !ok {
            continue;
        }
        let lane_inits: Vec<Expr> = order.iter().rev().map(|&i| phis[i].init.clone()).collect();
        let Some(init) = concat_balanced(&lane_inits) else { continue };
        *uid += 1;
        let wide = format!("v{}", *uid - 1);
        wide_inits[order[0]] = Some(init);
        let width = groups[g].width;
        groups.push(Group {
            wide,
            width,
            next_from: Some(g),
            lanes: order.iter().map(|&i| phis[i].name.clone()).collect(),
        });
    }
    (groups, wide_inits)
}

fn apply_groups(e: &Expr, groups: &[Group]) -> Expr {
    if groups.is_empty() {
        return e.clone();
    }
    let lane_of: HashMap<&str, (usize, usize)> = groups
        .iter()
        .enumerate()
        .flat_map(|(g, grp)| grp.lanes.iter().enumerate().map(move |(l, n)| (n.as_str(), (g, l))))
        .collect();
    rewrite(e, &mut |n| {
        if let ExprKind::Concat { .. } = &n.kind {
            let mut leaves = Vec::new();
            flatten_cat(n, &mut leaves);
            // leaves are high lane first; must be lane k-1 .. 0 of one group
            let mut found: Option<(usize, usize)> = None;
            let mut ok = true;
            for (pos, leaf) in leaves.iter().enumerate() {
                let ExprKind::Register { name } = &leaf.kind else {
                    ok = false;
                    break;
                };
                let Some(&(g, l)) = lane_of.get(name.as_str()) else {
                    ok = false;
                    break;
                };
                let expect_lane = leaves.len() - 1 - pos;
                if l != expect_lane || found.is_some_and(|(fg, _)| fg != g) {
                    ok = false;
                    break;
                }
                found = Some((g, leaves.len()));
            }
            if ok && let Some((g, count)) = found {
                let w = Expr::register(groups[g].width, groups[g].wide.clone());
                let bits = (count * 8) as u32;
                return Some(if bits == groups[g].width.bits() {
                    w
                } else {
                    crate::xu::trunc_to(BitWidth::from_bits(bits)?, &w)
                });
            }
        }
        if let ExprKind::Register { name } = &n.kind
            && let Some(&(g, l)) = lane_of.get(name.as_str())
        {
            let w = Expr::register(groups[g].width, groups[g].wide.clone());
            return Expr::extract(BitWidth::W8, (l * 8) as u32, w).ok();
        }
        None
    })
}

// ------------------------------------------------------------- selector decode

fn flags_nodes(sel: &Expr) -> Vec<Expr> {
    let mut out: Vec<Expr> = Vec::new();
    rewrite(sel, &mut |n| {
        if let ExprKind::Intrinsic { name, .. } = &n.kind
            && name.starts_with("vm_flags_")
        {
            if !out.contains(n) {
                out.push(n.clone());
            }
            return Some(n.clone());
        }
        None
    });
    out
}

fn eval_const(e: &Expr, env: &BTreeMap<String, Expr>) -> Option<u64> {
    let s = e.substitute(env).ok()?.simplify();
    match s.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// Decodes `selector == value` (as a condition on the state) into a comparison,
/// by evaluating the selector with its flag words replaced by the project's own
/// flag semantics for crafted operand pairs.
fn decode_selector(selector: &Expr, value: u64, notes: &mut Vec<String>) -> Cond {
    let opaque = |why: &str| Cond::Opaque {
        desc: format!("branch_selector_is_{value:#x}[{why}]"),
        args: vec![],
    };
    let nodes = flags_nodes(selector);
    if nodes.is_empty() {
        return opaque("no flag inputs");
    }
    // all flag words must be the same comparison of the same operands
    let first = &nodes[0];
    let ExprKind::Intrinsic { name, arguments } = &first.kind else { return opaque("shape") };
    let (x, y) = match arguments.as_slice() {
        [x, y] => (x.clone(), y.clone()),
        _ => return opaque("flag intrinsic arity"),
    };
    let tied = nodes.iter().all(|n| {
        let ExprKind::Intrinsic { name: n2, arguments: a2 } = &n.kind else { return false };
        if n2 != name || a2.len() != 2 {
            return false;
        }
        a2[0].simplify() == x.simplify() && a2[1].simplify() == y.simplify()
    });
    if !tied {
        return opaque("several different flag words");
    }
    let w = x.width;
    let is_sub = name.starts_with("vm_flags_sub_");
    let is_and = name.starts_with("vm_flags_and_");
    if !is_sub && !is_and {
        return opaque(name);
    }
    // crafted operand pairs
    let mut rng = Rng(0x00C0_FFEE_1234_5678);
    let mut pairs: Vec<(u64, u64)> = vec![(0, 0), (1, 0), (0, 1), (5, 5), (w.mask(), w.mask()), (w.mask(), 0), (0, w.mask())];
    let top = 1_u64 << (w.bits() - 1);
    pairs.extend([(top, 0), (0, top), (top, 1), (top - 1, w.mask()), (top, top), (top - 1, top)]);
    for _ in 0..40 {
        let a = w.truncate(rng.next());
        let b = if rng.next() % 4 == 0 { a } else { w.truncate(rng.next()) };
        pairs.push((a, b));
        pairs.push((a, w.truncate(a.wrapping_add(1))));
        pairs.push((w.truncate(a.wrapping_add(1)), a));
    }
    let eval_sel = |a: u64, b: u64| -> Option<u64> {
        let word = Expr::intrinsic(first.width, name.clone(), vec![Expr::constant(w, a), Expr::constant(w, b)]);
        let word = word.simplify();
        if !matches!(word.kind, ExprKind::Const { .. }) {
            return None;
        }
        let mut s = selector.clone();
        for n in &nodes {
            let n2 = n.clone();
            s = rewrite(&s, &mut |q| if q == &n2 { Some(word.clone()) } else { None });
        }
        match s.simplify().kind {
            ExprKind::Const { value } => Some(value),
            _ => None,
        }
    };
    let mut samples: Vec<(u64, u64, bool)> = Vec::new();
    for (a, b) in pairs {
        let Some(sv) = eval_sel(a, b) else { return opaque("flag word does not fold") };
        samples.push((a, b, sv == value));
    }
    let ops = [
        Cmp::Eq, Cmp::Ne, Cmp::Ult, Cmp::Uge, Cmp::Ule, Cmp::Ugt, Cmp::Slt, Cmp::Sge, Cmp::Sle, Cmp::Sgt,
    ];
    let sext = |v: u64| -> i64 {
        if w.bits() == 64 { v.cast_signed() } else if v & top != 0 { (v | (u64::MAX << w.bits())).cast_signed() } else { v.cast_signed() }
    };
    let truth = |op: Cmp, a: u64, b: u64| -> bool {
        match op {
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
            Cmp::Ult => a < b,
            Cmp::Ule => a <= b,
            Cmp::Ugt => a > b,
            Cmp::Uge => a >= b,
            Cmp::Slt => sext(a) < sext(b),
            Cmp::Sle => sext(a) <= sext(b),
            Cmp::Sgt => sext(a) > sext(b),
            Cmp::Sge => sext(a) >= sext(b),
        }
    };
    for op in ops {
        let ok = if is_sub {
            samples.iter().all(|(a, b, t)| truth(op, *a, *b) == *t)
        } else {
            samples.iter().all(|(a, b, t)| truth(op, a & b, 0) == *t)
        };
        if ok {
            let (a, b) = if is_sub {
                (x.clone(), y.clone())
            } else {
                (
                    Expr::binary(BinaryOp::And, x.clone(), y.clone()).unwrap_or(x.clone()),
                    Expr::constant(w, 0),
                )
            };
            notes.push(format!(
                "branch predicate decoded from flag word {name}: {} samples agree",
                samples.len()
            ));
            return Cond::Cmp { op, a, b };
        }
    }
    opaque("no comparison matches the flag-word samples")
}

// ------------------------------------------------------------------- lowering

struct Lower<'a> {
    blocks: BTreeMap<NodeId, Block>,
    next: NodeId,
    notes: &'a mut Vec<String>,
    /// innermost-first: (header block id, phi variable names)
    loops: Vec<(NodeId, Vec<String>)>,
    uid: usize,
}

impl Lower<'_> {
    fn alloc(&mut self) -> NodeId {
        self.next += 1;
        self.next
    }

    fn lower(&mut self, n: &RNode, at: Option<NodeId>) -> Result<NodeId, String> {
        let id = at.unwrap_or_else(|| self.alloc());
        match n {
            RNode::Exit(x) => {
                let rax = x.registers.get(&X64NativeRegister::Rax).ok_or("exit has no rax")?;
                for (r, e) in &x.registers {
                    if is_callee_saved(*r) {
                        let f = fix(e);
                        let same = matches!(&f.kind, ExprKind::Register { name } if name == reg_name(*r));
                        if !same {
                            self.notes.push(format!("exit: callee-saved {} is not restored verbatim", reg_name(*r)));
                        }
                    }
                }
                self.blocks.insert(
                    id,
                    Block { id, assigns: vec![], stores: vec![], term: Term::Return(vec![("rax".to_owned(), fix(rax))]) },
                );
            }
            RNode::BackEdge { next } => {
                let (header, names) = self.loops.last().cloned().ok_or("back edge outside a loop")?;
                if names.len() != next.len() {
                    return Err("back edge arity does not match the loop".to_owned());
                }
                let assigns = names.iter().cloned().zip(next.iter().map(fix)).collect();
                self.blocks.insert(id, Block { id, assigns, stores: vec![], term: Term::Goto(header) });
            }
            RNode::Fork { selector, arms, .. } => {
                if arms.len() != 2 {
                    return Err(format!("{}-way branch (switch) is not lowered", arms.len()));
                }
                let sel = fix(selector);
                let cond = decode_selector(&sel, arms[0].0, self.notes);
                let then = self.lower(&arms[0].1, None)?;
                let els = self.lower(&arms[1].1, None)?;
                self.blocks.insert(id, Block { id, assigns: vec![], stores: vec![], term: Term::Branch { cond, then, els } });
            }
            RNode::Loop { phis, body, .. } => {
                // merge byte lanes into wide phis
                let mut backs: Vec<&Vec<Expr>> = Vec::new();
                collect_backs(body, &mut backs);
                let (groups, wide_inits) = merge_lanes(phis, &backs, &mut self.uid);
                let mut names: Vec<String> = Vec::new();
                let mut inits: Vec<Expr> = Vec::new();
                let in_group: HashMap<&str, usize> = groups
                    .iter()
                    .enumerate()
                    .flat_map(|(g, grp)| grp.lanes.iter().map(move |n| (n.as_str(), g)))
                    .collect();
                let mut keep: Vec<usize> = Vec::new();
                for (i, p) in phis.iter().enumerate() {
                    if let Some(&g) = in_group.get(p.name.as_str()) {
                        if groups[g].lanes[0] == p.name {
                            names.push(groups[g].wide.clone());
                            inits.push(wide_inits[i].clone().ok_or("wide init missing")?);
                            keep.push(i);
                        }
                    } else {
                        let nm = format!("v{}", self.uid);
                        self.uid += 1;
                        names.push(nm);
                        inits.push(p.init.clone());
                        keep.push(i);
                    }
                }
                // per-phi rename for non-grouped phis
                let single: HashMap<String, String> = keep
                    .iter()
                    .zip(names.iter())
                    .filter(|(i, _)| !in_group.contains_key(phis[**i].name.as_str()))
                    .map(|(i, n)| (phis[*i].name.clone(), n.clone()))
                    .collect();
                let rewrite_all = |e: &Expr| -> Expr {
                    let g = apply_groups(e, &groups);
                    rewrite(&g, &mut |n| match &n.kind {
                        ExprKind::Register { name } => single.get(name).map(|r| Expr::register(n.width, r.clone())),
                        _ => None,
                    })
                };
                let body2 = rewrite_tree(body, &rewrite_all, &keep, &groups, phis);
                // pre-header block: phi := init
                let pre = id;
                let header = self.alloc();
                let assigns = names.iter().cloned().zip(inits.iter().map(|e| fix(&rewrite_all(e)))).collect();
                self.blocks.insert(pre, Block { id: pre, assigns, stores: vec![], term: Term::Goto(header) });
                self.loops.push((header, names));
                let r = self.lower(&body2, Some(header));
                self.loops.pop();
                r?;
            }
            RNode::Dead(why) => return Err(format!("a dead arm remained after gate collapse: {why}")),
        }
        Ok(id)
    }
}

fn collect_backs<'a>(n: &'a RNode, out: &mut Vec<&'a Vec<Expr>>) {
    match n {
        RNode::BackEdge { next } => out.push(next),
        RNode::Fork { arms, .. } => arms.iter().for_each(|(_, a)| collect_backs(a, out)),
        _ => {}
    }
}

/// Rewrites every expression of the loop body with `f`, and re-expresses back
/// edges over the kept (wide / single) phis.
fn rewrite_tree(
    n: &RNode,
    f: &dyn Fn(&Expr) -> Expr,
    keep: &[usize],
    groups: &[Group],
    phis: &[RPhi],
) -> RNode {
    match n {
        RNode::Exit(x) => RNode::Exit(RExit {
            registers: x.registers.iter().map(|(r, e)| (*r, f(e))).collect(),
            region: x.region,
        }),
        RNode::Fork { selector, env, block, arms } => RNode::Fork {
            selector: f(selector),
            env: *env,
            block: *block,
            arms: arms.iter().map(|(v, a)| (*v, rewrite_tree(a, f, keep, groups, phis))).collect(),
        },
        RNode::Loop { header, phis: ip, body } => RNode::Loop {
            header: *header,
            phis: ip.iter().map(|p| RPhi { name: p.name.clone(), width: p.width, init: f(&p.init) }).collect(),
            // nested loop: only expressions are rewritten (its own back edges keep their arity)
            body: Box::new(rewrite_exprs_only(body, f)),
        },
        RNode::BackEdge { next } => {
            // kept phi i takes the next value of its representative lane's inner
            // expression when it heads a group
            let in_group: HashMap<&str, usize> = groups
                .iter()
                .enumerate()
                .flat_map(|(g, grp)| grp.lanes.iter().map(move |n| (n.as_str(), g)))
                .collect();
            let out = keep
                .iter()
                .map(|&i| {
                    let e = &next[i];
                    if in_group.contains_key(phis[i].name.as_str()) {
                        // lane-0 next is Ext8@0(H): the wide next is H
                        let grp = &groups[in_group[phis[i].name.as_str()]];
                        if let Some(src) = grp.next_from {
                            return_register(&groups[src])
                        } else {
                        let gw = grp.width;
                        match &e.kind {
                            ExprKind::Extract { value, .. } => crate::xu::trunc_to(gw, &f(value)),
                            _ => f(e),
                        }
                        }
                    } else {
                        f(e)
                    }
                })
                .collect();
            RNode::BackEdge { next: out }
        }
        RNode::Dead(w) => RNode::Dead(w.clone()),
    }
}

fn return_register(g: &Group) -> Expr {
    Expr::register(g.width, g.wide.clone())
}

fn rewrite_exprs_only(n: &RNode, f: &dyn Fn(&Expr) -> Expr) -> RNode {
    match n {
        RNode::Exit(x) => RNode::Exit(RExit {
            registers: x.registers.iter().map(|(r, e)| (*r, f(e))).collect(),
            region: x.region,
        }),
        RNode::Fork { selector, env, block, arms } => RNode::Fork {
            selector: f(selector),
            env: *env,
            block: *block,
            arms: arms.iter().map(|(v, a)| (*v, rewrite_exprs_only(a, f))).collect(),
        },
        RNode::Loop { header, phis, body } => RNode::Loop {
            header: *header,
            phis: phis.iter().map(|p| RPhi { name: p.name.clone(), width: p.width, init: f(&p.init) }).collect(),
            body: Box::new(rewrite_exprs_only(body, f)),
        },
        RNode::BackEdge { next } => RNode::BackEdge { next: next.iter().map(|e| f(e)).collect() },
        RNode::Dead(w) => RNode::Dead(w.clone()),
    }
}

/// Lowers a rolled function (after gate collapse) to the frontend-neutral IR.
///
/// # Errors
/// Gate arms that differ, multi-way branches, or any shape listed above.
pub fn lower(rolled: &RolledFunction, name: &str) -> Result<VmLowered, String> {
    let mut notes = Vec::new();
    let tree = collapse(&rolled.root, &mut notes)?;
    let mut lw = Lower { blocks: BTreeMap::new(), next: 0, notes: &mut notes, loops: Vec::new(), uid: 0 };
    let entry = lw.lower(&tree, None)?;
    let blocks = lw.blocks;
    notes.dedup();
    Ok(VmLowered {
        func: Func { name: name.to_owned(), entry, blocks, params: vec![], notes: vec![] },
        notes,
    })
}
