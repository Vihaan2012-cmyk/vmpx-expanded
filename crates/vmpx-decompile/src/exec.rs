//! Concrete interpreters used to cross-check the decompiled program against the
//! IR it was built from: the reference interpreter runs the original [`Func`]
//! (parallel-assignment blocks); the structured interpreter runs the emitted
//! statement tree. Equal return values and store traces on many input vectors
//! is evidence (not proof) that no pass changed the meaning.

use std::collections::{BTreeMap, HashMap};

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp};

use crate::ir::{Cmp, Cond, Func, NodeId, Term};
use crate::opt::Stmt;

pub type Env = HashMap<String, u64>;

fn sext(v: u64, from: BitWidth) -> u64 {
    let v = from.truncate(v);
    if v >> (from.bits() - 1) & 1 == 1 { v | !from.mask() } else { v }
}

pub fn mem_read(addr: u64, w: BitWidth) -> u64 {
    let mut z = addr.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    w.truncate(z ^ (z >> 31))
}

fn binop(op: BinaryOp, w: BitWidth, l: u64, r: u64) -> u64 {
    let mask = if w == BitWidth::W64 { 0x3F } else { 0x1F };
    let sc = u32::try_from(r & mask).unwrap_or(0);
    let bits = w.bits();
    let rc = sc % bits;
    let l = w.truncate(l);
    let out = match op {
        BinaryOp::Add => l.wrapping_add(r),
        BinaryOp::Sub => l.wrapping_sub(r),
        BinaryOp::Mul => l.wrapping_mul(r),
        BinaryOp::Xor => l ^ r,
        BinaryOp::And => l & r,
        BinaryOp::Or => l | r,
        BinaryOp::Shl => l.wrapping_shl(sc),
        BinaryOp::Shr => l.wrapping_shr(sc),
        BinaryOp::Sar => sext(l, w).cast_signed().wrapping_shr(sc).cast_unsigned(),
        BinaryOp::Rol => {
            if rc == 0 { l } else { (l << rc) | (l >> (bits - rc)) }
        }
        BinaryOp::Ror => {
            if rc == 0 { l } else { (l >> rc) | (l << (bits - rc)) }
        }
    };
    w.truncate(out)
}

pub fn eval(e: &Expr, env: &Env) -> Result<u64, String> {
    fn go(e: &Expr, env: &Env, memo: &mut HashMap<Expr, u64>) -> Result<u64, String> {
        if let Some(v) = memo.get(e) {
            return Ok(*v);
        }
        let v = match &e.kind {
            ExprKind::Const { value } => *value,
            ExprKind::Register { name } => e
                .width
                .truncate(*env.get(name).ok_or_else(|| format!("`{name}` is uninitialized"))?),
            ExprKind::Unary { op, value } => {
                let v = go(value, env, memo)?;
                e.width.truncate(match op {
                    UnaryOp::Not => !v,
                    UnaryOp::Neg => v.wrapping_neg(),
                })
            }
            ExprKind::Binary { op, left, right } => {
                binop(*op, e.width, go(left, env, memo)?, go(right, env, memo)?)
            }
            ExprKind::Extract { value, lsb } => e.width.truncate(go(value, env, memo)? >> lsb),
            ExprKind::Concat { high, low } => {
                e.width.truncate((go(high, env, memo)? << low.width.bits()) | go(low, env, memo)?)
            }
            ExprKind::Resize { op, value } => {
                let v = go(value, env, memo)?;
                e.width.truncate(match op {
                    ResizeOp::Truncate | ResizeOp::ZeroExtend => v,
                    ResizeOp::SignExtend => sext(v, value.width),
                })
            }
            ExprKind::Load { address } => mem_read(go(address, env, memo)?, e.width),
            ExprKind::Intrinsic { name, .. } => return Err(format!("intrinsic `{name}` not evaluable")),
        };
        memo.insert(e.clone(), v);
        Ok(v)
    }
    go(e, env, &mut HashMap::new())
}

pub fn eval_cond(c: &Cond, env: &Env) -> Result<bool, String> {
    match c {
        Cond::Opaque { desc, .. } => Err(format!("opaque condition `{desc}`")),
        Cond::Cmp { op, a, b } => {
            let (x, y) = (eval(a, env)?, eval(b, env)?);
            let w = a.width;
            let (sx, sy) = (sext(x, w).cast_signed(), sext(y, w).cast_signed());
            Ok(match op {
                Cmp::Eq => x == y,
                Cmp::Ne => x != y,
                Cmp::Ult => x < y,
                Cmp::Ule => x <= y,
                Cmp::Ugt => x > y,
                Cmp::Uge => x >= y,
                Cmp::Slt => sx < sy,
                Cmp::Sle => sx <= sy,
                Cmp::Sgt => sx > sy,
                Cmp::Sge => sx >= sy,
            })
        }
    }
}

pub type Trace = Vec<(u64, u32, u64)>;

#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub ret: u64,
    pub stores: Trace,
}

const BUDGET: usize = 2_000_000;

pub fn run_reference(f: &Func, inputs: &Env) -> Result<Outcome, String> {
    let mut env = inputs.clone();
    let mut id = f.entry;
    let mut stores = Vec::new();
    for _ in 0..BUDGET {
        let b = f.blocks.get(&id).ok_or("missing block")?;
        for s in &b.stores {
            stores.push((eval(&s.address, &env)?, s.value.width.bits(), eval(&s.value, &env)?));
        }
        let mut next: Vec<(String, u64)> = Vec::new();
        for (v, d) in &b.assigns {
            next.push((v.clone(), eval(d, &env)?));
        }
        match &b.term {
            Term::Return(vals) => {
                return Ok(Outcome { ret: eval(&vals[0].1, &env)?, stores });
            }
            Term::Goto(t) => id = *t,
            Term::Branch { cond, then, els } => id = if eval_cond(cond, &env)? { *then } else { *els },
        }
        for (v, x) in next {
            env.insert(v, x);
        }
    }
    Err("reference run exceeded its step budget".into())
}

enum Flow {
    Normal,
    Break,
    Continue,
    Return(u64),
    Goto(NodeId),
}

struct Interp {
    env: Env,
    stores: Trace,
    steps: usize,
}

impl Interp {
    fn seq(&mut self, list: &[Stmt]) -> Result<Flow, String> {
        for s in list {
            match self.one(s)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    fn tick(&mut self) -> Result<(), String> {
        self.steps += 1;
        if self.steps > BUDGET { Err("structured run exceeded its step budget".into()) } else { Ok(()) }
    }

    fn one(&mut self, s: &Stmt) -> Result<Flow, String> {
        self.tick()?;
        match s {
            Stmt::Assign(v, e) => {
                let x = eval(e, &self.env)?;
                self.env.insert(v.clone(), x);
            }
            Stmt::Store(st) => {
                let a = eval(&st.address, &self.env)?;
                let x = eval(&st.value, &self.env)?;
                self.stores.push((a, st.value.width.bits(), x));
            }
            Stmt::If { cond, then, els } => {
                return if eval_cond(cond, &self.env)? { self.seq(then) } else { self.seq(els) };
            }
            Stmt::Loop(body) => loop {
                self.tick()?;
                match self.seq(body)? {
                    Flow::Break => return Ok(Flow::Normal),
                    Flow::Normal | Flow::Continue => {}
                    other => return Ok(other),
                }
            },
            Stmt::While { cond, body } => {
                while eval_cond(cond, &self.env)? {
                    self.tick()?;
                    match self.seq(body)? {
                        Flow::Break => break,
                        Flow::Normal | Flow::Continue => {}
                        other => return Ok(other),
                    }
                }
            }
            Stmt::DoWhile { body, cond } => loop {
                self.tick()?;
                match self.seq(body)? {
                    Flow::Break => return Ok(Flow::Normal),
                    Flow::Normal | Flow::Continue => {}
                    other => return Ok(other),
                }
                if !eval_cond(cond, &self.env)? {
                    break;
                }
            },
            Stmt::Break => return Ok(Flow::Break),
            Stmt::Continue => return Ok(Flow::Continue),
            Stmt::Return(vals) => return Ok(Flow::Return(eval(&vals[0].1, &self.env)?)),
            Stmt::Label(_) | Stmt::Comment(_) => {}
            Stmt::Goto(n) => return Ok(Flow::Goto(*n)),
            Stmt::IfGoto { cond, target } => {
                if eval_cond(cond, &self.env)? {
                    return Ok(Flow::Goto(*target));
                }
            }
        }
        Ok(Flow::Normal)
    }
}

/// Runs a statement list that may contain top-level labels/gotos.
pub fn run_structured(
    body: &[Stmt],
    inputs: &Env,
    widths: &BTreeMap<String, BitWidth>,
) -> Result<Outcome, String> {
    let mut env = Env::new();
    for (k, v) in inputs {
        env.insert(k.clone(), widths.get(k).map_or(*v, |w| w.truncate(*v)));
    }
    let mut it = Interp { env, stores: Vec::new(), steps: 0 };
    let mut pc = 0;
    loop {
        match it.seq(&body[pc..])? {
            Flow::Return(v) => return Ok(Outcome { ret: v, stores: it.stores }),
            Flow::Goto(l) => {
                pc = body
                    .iter()
                    .position(|s| matches!(s, Stmt::Label(x) if *x == l))
                    .ok_or("goto to a missing label")?;
            }
            Flow::Normal => return Err("fell off the end of the function".into()),
            Flow::Break | Flow::Continue => return Err("break/continue outside a loop".into()),
        }
    }
}

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
