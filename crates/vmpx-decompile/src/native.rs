//! Frontend: a recovered native CFG (`vmpx-mutation` `NativeCfg`) -> [`Func`].
//!
//! Each GPR is a variable named after the register; flags are not carried across
//! blocks (a branch whose flags are not defined in its own block becomes an
//! honest opaque condition).

use std::collections::BTreeMap;

use vmpx_disasm::{Flag, FlagState, Gpr, Width};
use vmpx_mutation::cfg_recovery::{CallTarget, NativeCfg, Terminator};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind};

use crate::ir::{Block, Cmp, Cond, Func, NodeId, Store, Term};
use crate::xu::cleanup;

fn reg(g: Gpr) -> Expr {
    Expr::register(BitWidth::W64, g.name())
}

fn bw(w: Width) -> BitWidth {
    match w {
        Width::W8 => BitWidth::W8,
        Width::W16 => BitWidth::W16,
        Width::W32 => BitWidth::W32,
        Width::W64 => BitWidth::W64,
    }
}

fn intr<'a>(e: &'a Expr, name: &str) -> Option<&'a [Expr]> {
    match &e.kind {
        ExprKind::Intrinsic { name: n, arguments } if n == name => Some(arguments),
        _ => None,
    }
}

fn is_const(e: &Expr, v: u64) -> bool {
    matches!(e.kind, ExprKind::Const { value } if value == v)
}

fn sub_args(e: &Expr) -> Option<(&Expr, &Expr)> {
    match &e.kind {
        ExprKind::Binary {
            op: BinaryOp::Sub,
            left,
            right,
        } => Some((left, right)),
        _ => None,
    }
}

/// Builds the branch condition of `cc` from the flag expressions defined by the
/// block (all over block-entry values).
fn build_cond(cc: &str, flags: &BTreeMap<Flag, Expr>) -> Cond {
    let opaque = |why: &str| Cond::Opaque {
        desc: format!("cc_{cc}[{why}]"),
        args: flags.values().cloned().collect(),
    };
    let zf_r = flags
        .get(&Flag::Zf)
        .and_then(|e| intr(e, "zf"))
        .and_then(|a| a.first());
    let sf_r = flags
        .get(&Flag::Sf)
        .and_then(|e| intr(e, "sf"))
        .and_then(|a| a.first());
    let cf = flags.get(&Flag::Cf);
    let of = flags.get(&Flag::Of);
    let cf_sub = cf.and_then(|e| intr(e, "cf_sub"));
    let of_sub = of.and_then(|e| intr(e, "of_sub"));
    let zero = |e: &Expr| Expr::constant(e.width, 0);
    let cmp = |op: Cmp, a: &Expr, b: &Expr| Cond::Cmp {
        op,
        a: a.clone(),
        b: b.clone(),
    };
    // the (a, b) of a `cmp a, b` / `sub a, b` whose result is `r`
    let sub_pair = |r: &Expr| sub_args(r).map(|(a, b)| (a.clone(), b.clone()));
    let logic = cf.is_some_and(|e| is_const(e, 0)) && of.is_some_and(|e| is_const(e, 0));
    match cc {
        "e" | "ne" => {
            let Some(r) = zf_r else {
                return opaque("zf undefined");
            };
            let op = if cc == "e" { Cmp::Eq } else { Cmp::Ne };
            match sub_pair(r) {
                Some((a, b)) => cmp(op, &a, &b),
                None => cmp(op, r, &zero(r)),
            }
        }
        "b" | "ae" | "be" | "a" => {
            let (Some(args), Some(r)) = (cf_sub, zf_r) else {
                if logic {
                    return opaque("cf=0 after logic op");
                }
                return opaque("cf/zf not from one sub/cmp");
            };
            let (Some((a, b)), [ca, cb]) = (sub_pair(r), args) else {
                return opaque("zf not a sub result");
            };
            if &a != ca || &b != cb {
                return opaque("cf/zf operands differ");
            }
            let op = match cc {
                "b" => Cmp::Ult,
                "ae" => Cmp::Uge,
                "be" => Cmp::Ule,
                _ => Cmp::Ugt,
            };
            cmp(op, &a, &b)
        }
        "l" | "ge" | "le" | "g" => {
            let op_s = match cc {
                "l" => Cmp::Slt,
                "ge" => Cmp::Sge,
                "le" => Cmp::Sle,
                _ => Cmp::Sgt,
            };
            if logic {
                let Some(r) = sf_r else {
                    return opaque("sf undefined");
                };
                return cmp(op_s, r, &zero(r));
            }
            let (Some(args), Some(r), Some(sr)) = (of_sub, zf_r, sf_r) else {
                return opaque("of/sf/zf not from one sub/cmp");
            };
            let (Some((a, b)), [oa, ob]) = (sub_pair(r), args) else {
                return opaque("zf not a sub result");
            };
            if &a != oa || &b != ob || sr != r {
                return opaque("of/sf operands differ");
            }
            cmp(op_s, &a, &b)
        }
        "s" | "ns" => match sf_r {
            Some(r) => cmp(if cc == "s" { Cmp::Slt } else { Cmp::Sge }, r, &zero(r)),
            None => opaque("sf undefined"),
        },
        _ => opaque("unmodeled condition"),
    }
}

/// Lowers the CFG. `name` is the function's display name.
///
/// # Errors
/// An unlifted/unsupported block, a block that reads flags defined elsewhere in
/// a value (not just in a branch), or a block with stores that also loads.
pub fn lower(cfg: &NativeCfg, name: &str) -> Result<Func, String> {
    let mut blocks: BTreeMap<NodeId, Block> = BTreeMap::new();
    let env: BTreeMap<String, Expr> = Gpr::ALL.iter().map(|g| (g.input_name(), reg(*g))).collect();
    let mut notes: Vec<String> = Vec::new();
    for (addr, nb) in &cfg.blocks {
        // Lower the block's data effect, falling back to an honest-opaque block
        // on ANY refusal (unlifted semantics, a carried-flag read, a store/load
        // mix): its recovered control flow is always kept, so one unliftable
        // block no longer sinks the whole function.
        let built: Result<Block, String> = (|| {
            let sem = nb
                .semantics
                .as_ref()
                .ok_or_else(|| format!("semantics refused ({:?})", nb.terminator))?;
            let subst = |e: &Expr| -> Result<Expr, String> {
                let s = e.substitute(&env).map_err(|x| x.to_string())?;
                let s = cleanup(&s);
                if let Some(stray) = s.dependencies().into_iter().find(|d| d.ends_with("_in")) {
                    return Err(format!("reads carried flag/input `{stray}`"));
                }
                Ok(s)
            };
            let mut assigns = Vec::new();
            for g in &sem.written {
                assigns.push((g.name().to_owned(), subst(&sem.regs[g])?));
            }
            let mut stores = Vec::new();
            for w in &sem.writes {
                let _ = bw(w.width);
                stores.push(Store { address: subst(&w.address)?, value: subst(&w.value)? });
            }
            let mut flags: BTreeMap<Flag, Expr> = BTreeMap::new();
            for (f, st) in &sem.flags {
                if let FlagState::Value(e) = st {
                    flags.insert(*f, subst(e)?);
                }
            }
            let term = match &nb.terminator {
                Terminator::FallThrough(t) | Terminator::Jump(t) => Term::Goto(*t),
                Terminator::CondJump { cc, taken, not_taken } => Term::Branch {
                    cond: build_cond(cc, &flags),
                    then: *taken,
                    els: *not_taken,
                },
                Terminator::Return => {
                    let rax = if sem.written.contains(&Gpr::Rax) {
                        subst(&sem.regs[&Gpr::Rax])?
                    } else {
                        reg(Gpr::Rax)
                    };
                    Term::Return(vec![("rax".to_owned(), rax)])
                }
                Terminator::Call { next, .. } => Term::Goto(*next),
                Terminator::TailCall { target } => {
                    let callee = match target {
                        CallTarget::Direct(a) => format!("sub_{a:x}"),
                        CallTarget::Import(a) => format!("import_at_{a:x}"),
                        CallTarget::Indirect(_) => "tailcall_indirect".to_owned(),
                    };
                    let args = [Gpr::Rcx, Gpr::Rdx, Gpr::R8, Gpr::R9]
                        .into_iter()
                        .map(|g| {
                            let v = if sem.written.contains(&g) { sem.regs[&g].clone() } else { reg(g) };
                            subst(&v)
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    Term::Return(vec![("rax".to_owned(), Expr::intrinsic(BitWidth::W64, callee, args))])
                }
                Terminator::Unsupported(why) => return Err(why.clone()),
            };
            if !stores.is_empty() && assigns.iter().any(|(_, e)| has_load(e)) {
                return Err("mixes stores and loads".to_owned());
            }
            Ok(Block { id: *addr, assigns, stores, term })
        })();
        match built {
            Ok(b) => {
                blocks.insert(*addr, b);
            }
            Err(why) => {
                notes.push(format!(
                    "block {addr:#x} not lifted ({why}); data effect opaque, control flow kept"
                ));
                blocks.insert(*addr, opaque_block(*addr, &nb.terminator));
            }
        }
    }
    // The Return block of a native function reads rax only; remaining
    // assignments in it are dead and removed by DCE.
    Ok(Func {
        name: name.to_owned(),
        entry: cfg.entry,
        blocks,
        params: Vec::new(),
        notes,
    })
}

/// An honest-opaque stand-in for a block the lifter refused: no data effect,
/// but the recovered control-flow edges are preserved (an opaque condition for
/// a conditional branch, since no flags were lifted).
fn opaque_block(addr: NodeId, terminator: &Terminator) -> Block {
    let empty: BTreeMap<Flag, Expr> = BTreeMap::new();
    let term = match terminator {
        Terminator::FallThrough(t) | Terminator::Jump(t) => Term::Goto(*t),
        Terminator::CondJump { cc, taken, not_taken } => Term::Branch {
            cond: build_cond(cc, &empty),
            then: *taken,
            els: *not_taken,
        },
        Terminator::Call { next, .. } => Term::Goto(*next),
        Terminator::Return | Terminator::TailCall { .. } | Terminator::Unsupported(_) => {
            Term::Return(vec![("rax".to_owned(), reg(Gpr::Rax))])
        }
    };
    Block { id: addr, assigns: Vec::new(), stores: Vec::new(), term }
}

fn has_load(e: &Expr) -> bool {
    let mut found = false;
    crate::xu::rewrite(e, &mut |n| {
        if matches!(n.kind, ExprKind::Load { .. }) {
            found = true;
            return Some(n.clone());
        }
        None
    });
    found
}
