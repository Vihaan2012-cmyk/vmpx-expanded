//! Control-flow structuring: dominator-based natural loops + post-dominator
//! merge points. Anything it cannot prove is reported as `Err`, and the caller
//! falls back to the flat labelled-goto form for the whole function.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::ir::{Cond, NodeId, Term};
use crate::opt::{LBlock, Stmt};

struct LoopCtx {
    header: NodeId,
    exit: Option<NodeId>,
    body: BTreeSet<NodeId>,
}

pub struct Structurer<'a> {
    blocks: &'a BTreeMap<NodeId, LBlock>,
    entry: NodeId,
    dom: BTreeMap<NodeId, BTreeSet<NodeId>>,
    headers: BTreeMap<NodeId, BTreeSet<NodeId>>,
    done: HashSet<NodeId>,
    entered: HashSet<NodeId>,
    loops: Vec<LoopCtx>,
}

fn succs(b: &LBlock) -> Vec<NodeId> {
    match &b.term {
        Term::Goto(t) => vec![*t],
        Term::Branch { then, els, .. } => {
            if then == els {
                vec![*then]
            } else {
                vec![*then, *els]
            }
        }
        Term::Return(_) => vec![],
    }
}

impl<'a> Structurer<'a> {
    pub fn new(blocks: &'a BTreeMap<NodeId, LBlock>, entry: NodeId) -> Self {
        let ids: Vec<NodeId> = blocks.keys().copied().collect();
        let all: BTreeSet<NodeId> = ids.iter().copied().collect();
        let mut preds: BTreeMap<NodeId, Vec<NodeId>> = ids.iter().map(|i| (*i, vec![])).collect();
        for (id, b) in blocks {
            for s in succs(b) {
                preds.entry(s).or_default().push(*id);
            }
        }
        let mut dom: BTreeMap<NodeId, BTreeSet<NodeId>> =
            ids.iter().map(|i| (*i, all.clone())).collect();
        dom.insert(entry, BTreeSet::from([entry]));
        loop {
            let mut changed = false;
            for id in &ids {
                if *id == entry {
                    continue;
                }
                let mut it = preds[id].iter().map(|p| dom[p].clone());
                let mut new = it.next().unwrap_or_default();
                for s in it {
                    new = new.intersection(&s).copied().collect();
                }
                new.insert(*id);
                if new != dom[id] {
                    dom.insert(*id, new);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // natural loops: back edge u -> h where h dominates u
        let mut headers: BTreeMap<NodeId, BTreeSet<NodeId>> = BTreeMap::new();
        for (u, b) in blocks {
            for h in succs(b) {
                if dom[u].contains(&h) {
                    let body = headers.entry(h).or_insert_with(|| BTreeSet::from([h]));
                    let mut stack = vec![*u];
                    while let Some(x) = stack.pop() {
                        if body.insert(x) {
                            stack.extend(preds[&x].iter().copied());
                        }
                    }
                }
            }
        }
        Self {
            blocks,
            entry,
            dom,
            headers,
            done: HashSet::new(),
            entered: HashSet::new(),
            loops: Vec::new(),
        }
    }

    pub fn run(mut self) -> Result<Vec<Stmt>, String> {
        let out = self.chain(self.entry, None)?;
        let reachable = self.reachable();
        if let Some(missed) = reachable.iter().find(|n| !self.done.contains(n)) {
            return Err(format!("block {missed:#x} was not placed by the structurer"));
        }
        let _ = &self.dom;
        Ok(simplify(out))
    }

    fn reachable(&self) -> BTreeSet<NodeId> {
        let mut seen = BTreeSet::new();
        let mut st = vec![self.entry];
        while let Some(n) = st.pop() {
            if seen.insert(n) {
                st.extend(succs(&self.blocks[&n]));
            }
        }
        seen
    }

    /// First node where `then` and `els` re-join, computed on the region graph in
    /// which loop continue/break targets and enclosing merges are sinks.
    fn merge_point(&self, n: NodeId, merge: Option<NodeId>) -> Option<NodeId> {
        let mut sinks: HashSet<NodeId> = HashSet::new();
        if let Some(m) = merge {
            sinks.insert(m);
        }
        for l in &self.loops {
            sinks.insert(l.header);
            if let Some(e) = l.exit {
                sinks.insert(e);
            }
        }
        // Edges INTO a sink go to a pseudo-node (SINK bit set) that leaves the
        // region, so a branch block that is itself a loop header is handled too.
        const EXIT: NodeId = u64::MAX;
        const SINK: NodeId = 1 << 63;
        let redirect = |s: NodeId| -> NodeId { if sinks.contains(&s) { s | SINK } else { s } };
        let mut region: BTreeSet<NodeId> = BTreeSet::new();
        let mut st = vec![n];
        while let Some(x) = st.pop() {
            if !region.insert(x) {
                continue;
            }
            if x & SINK != 0 {
                continue;
            }
            st.extend(succs(&self.blocks[&x]).into_iter().map(redirect));
        }
        let nodes: Vec<NodeId> = region.iter().copied().collect();
        let succ_of = |x: NodeId| -> Vec<NodeId> {
            if x & SINK != 0 {
                return vec![EXIT];
            }
            let s: Vec<NodeId> = succs(&self.blocks[&x]).into_iter().map(redirect).collect();
            if s.is_empty() { vec![EXIT] } else { s }
        };
        let mut all: BTreeSet<NodeId> = nodes.iter().copied().collect();
        all.insert(EXIT);
        let mut pdom: BTreeMap<NodeId, BTreeSet<NodeId>> =
            all.iter().map(|x| (*x, all.clone())).collect();
        pdom.insert(EXIT, BTreeSet::from([EXIT]));
        loop {
            let mut changed = false;
            for x in &nodes {
                let mut it = succ_of(*x).into_iter().map(|s| pdom[&s].clone());
                let mut new = it.next().unwrap_or_default();
                for s in it {
                    new = new.intersection(&s).copied().collect();
                }
                new.insert(*x);
                if new != pdom[x] {
                    pdom.insert(*x, new);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // immediate post-dominator: the strict pdom with the largest pdom set
        pdom[&n]
            .iter()
            .filter(|p| **p != n)
            .max_by_key(|p| pdom[p].len())
            .copied()
            .filter(|p| *p != EXIT)
            .map(|p| p & !SINK)
    }

    fn lower_stmts(&self, id: NodeId) -> Vec<Stmt> {
        self.blocks[&id].stmts.clone()
    }

    fn chain(&mut self, start: NodeId, merge: Option<NodeId>) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        let mut n = start;
        loop {
            if Some(n) == merge {
                return Ok(out);
            }
            if let Some(top) = self.loops.last() {
                if n == top.header {
                    out.push(Stmt::Continue);
                    return Ok(out);
                }
                if Some(n) == top.exit {
                    out.push(Stmt::Break);
                    return Ok(out);
                }
                if !top.body.contains(&n) {
                    return Err(format!("jump from loop to {n:#x}, which is not its single exit"));
                }
            }
            if self.done.contains(&n) {
                return Err(format!("block {n:#x} reached twice (needs a goto)"));
            }
            if self.headers.contains_key(&n) && !self.entered.contains(&n) {
                self.entered.insert(n);
                let body_set = self.headers[&n].clone();
                let mut exits: BTreeSet<NodeId> = BTreeSet::new();
                for b in &body_set {
                    for s in succs(&self.blocks[b]) {
                        if !body_set.contains(&s) {
                            exits.insert(s);
                        }
                    }
                }
                if exits.len() > 1 {
                    return Err(format!("loop at {n:#x} has {} exits", exits.len()));
                }
                let exit = exits.iter().next().copied();
                self.loops.push(LoopCtx {
                    header: n,
                    exit,
                    body: body_set,
                });
                let body = self.loop_body(n)?;
                self.loops.pop();
                out.push(Stmt::Loop(body));
                match exit {
                    Some(e) => {
                        n = e;
                        continue;
                    }
                    None => return Ok(out),
                }
            }
            let (stmts, next) = self.block_and_term(n, merge)?;
            out.extend(stmts);
            match next {
                Next::Done => return Ok(out),
                Next::Go(m) => n = m,
            }
        }
    }

    /// Emits the loop header block (first iteration entry) and what follows it
    /// inside the loop. The caller has temporarily disabled the header check.
    fn loop_body(&mut self, header: NodeId) -> Result<Vec<Stmt>, String> {
        let (mut stmts, next) = self.block_and_term(header, None)?;
        if let Next::Go(m) = next {
            let rest = self.chain(m, None)?;
            stmts.extend(rest);
        }
        Ok(stmts)
    }

    fn block_and_term(&mut self, n: NodeId, merge: Option<NodeId>) -> Result<(Vec<Stmt>, Next), String> {
        self.done.insert(n);
        let mut out = self.lower_stmts(n);
        let term = self.blocks[&n].term.clone();
        match term {
            Term::Return(v) => {
                out.push(Stmt::Return(v));
                Ok((out, Next::Done))
            }
            Term::Goto(t) => Ok((out, Next::Go(t))),
            Term::Branch { cond, then, els } => {
                if then == els {
                    return Ok((out, Next::Go(then)));
                }
                let m = self.merge_point(n, merge);
                let t_arm = self.chain(then, m)?;
                let e_arm = self.chain(els, m)?;
                out.push(Stmt::If {
                    cond,
                    then: t_arm,
                    els: e_arm,
                });
                Ok((
                    out,
                    match m {
                        Some(m) => Next::Go(m),
                        None => Next::Done,
                    },
                ))
            }
        }
    }
}

enum Next {
    Done,
    Go(NodeId),
}

// ------------------------------------------------------------------ peepholes

fn strip_tail_continue(body: &mut Vec<Stmt>) {
    if matches!(body.last(), Some(Stmt::Continue)) {
        body.pop();
    }
    if let Some(Stmt::If { then, els, .. }) = body.last_mut() {
        strip_tail_continue(then);
        strip_tail_continue(els);
    }
}

fn simplify(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.into_iter().map(simplify_one).collect()
}

fn negate(c: &Cond) -> Cond {
    c.negate()
}

fn simplify_one(s: Stmt) -> Stmt {
    match s {
        Stmt::If { cond, then, els } => {
            let then = simplify(then);
            let els = simplify(els);
            if then.is_empty() && !els.is_empty() {
                Stmt::If {
                    cond: negate(&cond),
                    then: els,
                    els: vec![],
                }
            } else {
                Stmt::If { cond, then, els }
            }
        }
        Stmt::Loop(body) => {
            let mut body = simplify(body);
            strip_tail_continue(&mut body);
            // do { ...; if (c) {} else break; } while (c)
            if let Some(Stmt::If { cond, then, els }) = body.last() {
                if then.is_empty() && matches!(els.as_slice(), [Stmt::Break]) {
                    let cond = cond.clone();
                    body.pop();
                    return Stmt::DoWhile { body, cond };
                }
                if els.is_empty() && matches!(then.as_slice(), [Stmt::Break]) {
                    let cond = negate(cond);
                    body.pop();
                    return Stmt::DoWhile { body, cond };
                }
            }
            if let Some(Stmt::If { cond, then, els }) = body.first() {
                if els.is_empty() && matches!(then.as_slice(), [Stmt::Break]) {
                    let cond = negate(cond);
                    let rest = body[1..].to_vec();
                    return Stmt::While { cond, body: rest };
                }
            }
            Stmt::Loop(body)
        }
        other => other,
    }
}
