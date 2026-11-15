//! Removes branches that decide nothing and duplicated branch tails.
//!
//! `VMProtect` fans one logical path out across many dispatch selectors, so the
//! structurer legitimately produces nests whose arms are byte-identical. On
//! `vm_c43d80e` the body was 130 lines in which one 4-line block appeared 12
//! times and the two top-level arms differed by a single store; the real logic
//! was there, just not findable.
//!
//! Two rules, applied innermost-first to a fixpoint:
//!
//! - identical arms: `if (c) { A } else { A }` is `A`. The condition decides
//!   nothing, and nothing in this IR makes evaluating it observable -- stores
//!   are statements, not expressions.
//! - common tail: `if (c) { P; T } else { Q; T }` is `if (c) { P } else { Q } T`.
//!   Safe even when `P` ends in `return`/`break`/`continue`: such an arm did not
//!   reach `T` before and does not reach the hoisted `T` either.

use crate::ir::{Cond, Store};
use crate::opt::Stmt;

/// Fixpoint rounds. Each round rewrites innermost-first, so a nest collapses one
/// level per round; the cap only stops a pathological shape from spinning.
const ROUNDS: usize = 16;

pub fn tidy_stmts(stmts: &mut Vec<Stmt>) {
    for _ in 0..ROUNDS {
        if !pass(stmts) {
            break;
        }
    }
}

fn pass(stmts: &mut Vec<Stmt>) -> bool {
    let mut changed = false;
    for stmt in stmts.iter_mut() {
        changed |= match stmt {
            Stmt::If { then, els, .. } => {
                let a = pass(then);
                pass(els) || a
            }
            Stmt::Loop(body) | Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => pass(body),
            _ => false,
        };
    }
    let mut out: Vec<Stmt> = Vec::with_capacity(stmts.len());
    for stmt in std::mem::take(stmts) {
        if debugging()
            && let Stmt::If { then, els, .. } = &stmt
        {
            eprintln!(
                "[tidy] if then={} els={} movable={} same={} tail={} first_diff={:?}",
                then.len(),
                els.len(),
                movable(then),
                same_list(then, els),
                common_tail(then, els),
                then.iter()
                    .zip(els)
                    .position(|(a, b)| !same(a, b))
                    .map(|i| (kind(&then[i]), kind(&els[i]))),
            );
        }
        match stmt {
            Stmt::If { then, els, .. } if movable(&then) && same_list(&then, &els) => {
                out.extend(then);
                changed = true;
            }
            Stmt::If {
                cond,
                mut then,
                mut els,
            } => {
                let tail = common_tail(&then, &els);
                if tail == 0 {
                    out.push(Stmt::If { cond, then, els });
                    continue;
                }
                let hoisted = then.split_off(then.len() - tail);
                els.truncate(els.len() - tail);
                out.push(Stmt::If { cond, then, els });
                out.extend(hoisted);
                changed = true;
            }
            other => out.push(other),
        }
    }
    *stmts = out;
    changed
}

/// Length of the longest identical suffix of the two arms that may be moved.
///
/// Stops at the shorter arm: hoisting everything out of an arm leaves it empty,
/// which the identical-arms rule then clears on the next round.
fn common_tail(a: &[Stmt], b: &[Stmt]) -> usize {
    let mut n = 0;
    while n < a.len()
        && n < b.len()
        && same(&a[a.len() - 1 - n], &b[b.len() - 1 - n])
        && movable(std::slice::from_ref(&a[a.len() - 1 - n]))
    {
        n += 1;
    }
    n
}

/// Whether these statements can be moved or dropped. A label may be the target
/// of a `goto` from outside the arm, and relocating one would retarget it, so
/// any statement list carrying label-based control flow is left alone.
fn movable(stmts: &[Stmt]) -> bool {
    stmts.iter().all(|stmt| match stmt {
        Stmt::Label(_) | Stmt::Goto(_) | Stmt::IfGoto { .. } => false,
        Stmt::If { then, els, .. } => movable(then) && movable(els),
        Stmt::Loop(body) | Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => movable(body),
        _ => true,
    })
}

/// `VMPX_TIDY_DEBUG` reports, per branch, why it was or was not reduced. It
/// earned its keep: the arms of this function's nests look identical in the
/// rendered C and are not, so without it the obvious conclusion is that the pass
/// is broken rather than that the text is hiding the difference.
fn debugging() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VMPX_TIDY_DEBUG").is_some())
}

/// Statement kind name, for `VMPX_TIDY_DEBUG`.
fn kind(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::Assign(..) => "assign",
        Stmt::Store(_) => "store",
        Stmt::If { .. } => "if",
        Stmt::Loop(_) => "loop",
        Stmt::While { .. } => "while",
        Stmt::DoWhile { .. } => "dowhile",
        Stmt::Break => "break",
        Stmt::Continue => "continue",
        Stmt::Return(_) => "return",
        Stmt::Label(_) => "label",
        Stmt::Goto(_) => "goto",
        Stmt::IfGoto { .. } => "ifgoto",
        Stmt::Comment(_) => "comment",
    }
}

fn same_list(a: &[Stmt], b: &[Stmt]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(x, y))
}

fn same(a: &Stmt, b: &Stmt) -> bool {
    match (a, b) {
        (Stmt::Assign(x, xe), Stmt::Assign(y, ye)) => x == y && xe == ye,
        (Stmt::Store(x), Stmt::Store(y)) => same_store(x, y),
        (
            Stmt::If {
                cond: ca,
                then: ta,
                els: ea,
            },
            Stmt::If {
                cond: cb,
                then: tb,
                els: eb,
            },
        ) => same_cond(ca, cb) && same_list(ta, tb) && same_list(ea, eb),
        (Stmt::Loop(x), Stmt::Loop(y)) => same_list(x, y),
        (Stmt::While { cond: ca, body: xa }, Stmt::While { cond: cb, body: xb })
        | (Stmt::DoWhile { body: xa, cond: ca }, Stmt::DoWhile { body: xb, cond: cb }) => {
            same_cond(ca, cb) && same_list(xa, xb)
        }
        (Stmt::Break, Stmt::Break) | (Stmt::Continue, Stmt::Continue) => true,
        (Stmt::Return(x), Stmt::Return(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|((xn, xe), (yn, ye))| xn == yn && xe == ye)
        }
        (Stmt::Label(x), Stmt::Label(y)) | (Stmt::Goto(x), Stmt::Goto(y)) => x == y,
        (
            Stmt::IfGoto {
                cond: ca,
                target: x,
            },
            Stmt::IfGoto {
                cond: cb,
                target: y,
            },
        ) => x == y && same_cond(ca, cb),
        (Stmt::Comment(x), Stmt::Comment(y)) => x == y,
        _ => false,
    }
}

fn same_store(a: &Store, b: &Store) -> bool {
    a.address == b.address && a.value == b.value
}

fn same_cond(a: &Cond, b: &Cond) -> bool {
    match (a, b) {
        (
            Cond::Cmp {
                op: oa,
                a: xa,
                b: ya,
            },
            Cond::Cmp {
                op: ob,
                a: xb,
                b: yb,
            },
        ) => oa == ob && xa == xb && ya == yb,
        (Cond::Opaque { desc: da, args: aa }, Cond::Opaque { desc: db, args: ab }) => {
            da == db && aa == ab
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::{BitWidth, Expr};

    fn ret(v: u64) -> Stmt {
        Stmt::Return(vec![("rax".to_owned(), Expr::constant(BitWidth::W64, v))])
    }

    fn eq_cond(v: u64) -> Cond {
        Cond::Cmp {
            op: crate::ir::Cmp::Eq,
            a: Expr::register(BitWidth::W64, "t"),
            b: Expr::constant(BitWidth::W64, v),
        }
    }

    fn store(address: u64, value: u64) -> Stmt {
        Stmt::Store(Store {
            address: Expr::constant(BitWidth::W64, address),
            value: Expr::constant(BitWidth::W64, value),
        })
    }

    #[test]
    fn identical_arms_collapse_to_one_copy() {
        let mut stmts = vec![Stmt::If {
            cond: eq_cond(1),
            then: vec![ret(7)],
            els: vec![ret(7)],
        }];
        tidy_stmts(&mut stmts);
        assert!(matches!(stmts.as_slice(), [Stmt::Return(_)]), "{stmts:?}");
    }

    #[test]
    fn differing_arms_keep_their_branch() {
        let mut stmts = vec![Stmt::If {
            cond: eq_cond(1),
            then: vec![ret(7)],
            els: vec![ret(8)],
        }];
        tidy_stmts(&mut stmts);
        assert!(matches!(stmts.as_slice(), [Stmt::If { .. }]), "{stmts:?}");
    }

    #[test]
    fn a_shared_tail_is_hoisted_and_the_difference_stays_branched() {
        // if (c) { store(0,1); ret } else { store(0,2); ret }
        let mut stmts = vec![Stmt::If {
            cond: eq_cond(1),
            then: vec![store(0, 1), ret(7)],
            els: vec![store(0, 2), ret(7)],
        }];
        tidy_stmts(&mut stmts);
        match stmts.as_slice() {
            [Stmt::If { then, els, .. }, Stmt::Return(_)] => {
                assert_eq!(then.len(), 1, "{stmts:?}");
                assert_eq!(els.len(), 1, "{stmts:?}");
            }
            other => panic!("expected a branch over the stores then one return: {other:?}"),
        }
    }

    #[test]
    fn a_nest_whose_arms_all_agree_collapses_entirely() {
        // Mirrors vm_c43d80e: four conditions over the same body.
        let leaf = || vec![ret(7)];
        let mut inner = Stmt::If {
            cond: eq_cond(1),
            then: leaf(),
            els: leaf(),
        };
        for level in 2..=4 {
            inner = Stmt::If {
                cond: eq_cond(level),
                then: vec![inner],
                els: leaf(),
            };
        }
        let mut stmts = vec![inner];
        tidy_stmts(&mut stmts);
        assert!(
            matches!(stmts.as_slice(), [Stmt::Return(_)]),
            "every arm returns the same value, so no branch survives: {stmts:?}"
        );
    }

    #[test]
    fn a_label_stays_in_its_arm() {
        // The arms are identical, but one statement is a label: collapsing them
        // would delete a `goto` target. Hoisting the return past it is still
        // fine -- an arm left holding only a label falls through to it.
        let mut stmts = vec![Stmt::If {
            cond: eq_cond(1),
            then: vec![Stmt::Label(9), ret(7)],
            els: vec![Stmt::Label(9), ret(7)],
        }];
        tidy_stmts(&mut stmts);
        let [Stmt::If { then, els, .. }, ..] = stmts.as_slice() else {
            panic!("the branch must survive: {stmts:?}");
        };
        assert!(
            matches!(then.as_slice(), [Stmt::Label(9)])
                && matches!(els.as_slice(), [Stmt::Label(9)]),
            "the label must not leave its arm: {stmts:?}"
        );
    }
}
