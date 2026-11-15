//! Frontend-neutral function IR: basic blocks of parallel assignments over named
//! variables, ending in goto / two-way branch / return. Both frontends (native
//! CFG, recovered VM CFG) lower to this; structuring and rendering only see it.
//!
//! Semantics of a block: every `Expr` read refers to the variable values at block
//! ENTRY (parallel assignment); stores are performed in order against entry
//! values; the terminator condition / return values read the values AFTER the
//! assignments (`Cond` and `Return` exprs are over the post-assignment state).

use std::collections::BTreeMap;

use vmpx_vm::Expr;

pub type NodeId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cmp {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
}

impl Cmp {
    #[must_use]
    pub fn negate(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Ult => Self::Uge,
            Self::Ule => Self::Ugt,
            Self::Ugt => Self::Ule,
            Self::Uge => Self::Ult,
            Self::Slt => Self::Sge,
            Self::Sle => Self::Sgt,
            Self::Sgt => Self::Sle,
            Self::Sge => Self::Slt,
        }
    }

    #[must_use]
    pub fn signed(self) -> bool {
        matches!(self, Self::Slt | Self::Sle | Self::Sgt | Self::Sge)
    }

    #[must_use]
    pub fn c_op(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Ult | Self::Slt => "<",
            Self::Ule | Self::Sle => "<=",
            Self::Ugt | Self::Sgt => ">",
            Self::Uge | Self::Sge => ">=",
        }
    }
}

#[derive(Clone, Debug)]
pub enum Cond {
    /// `a OP b`, both of the same width; signed ops compare as two's complement.
    Cmp { op: Cmp, a: Expr, b: Expr },
    /// A condition the frontend could not express as a comparison; rendered
    /// verbatim as `desc(args...)` and honest about it.
    Opaque { desc: String, args: Vec<Expr> },
}

impl Cond {
    #[must_use]
    pub fn negate(&self) -> Self {
        match self {
            Self::Cmp { op, a, b } => Self::Cmp {
                op: op.negate(),
                a: a.clone(),
                b: b.clone(),
            },
            Self::Opaque { desc, args } => Self::Opaque {
                desc: format!("!{desc}"),
                args: args.clone(),
            },
        }
    }

    pub fn exprs(&self) -> Vec<&Expr> {
        match self {
            Self::Cmp { a, b, .. } => vec![a, b],
            Self::Opaque { args, .. } => args.iter().collect(),
        }
    }

    pub fn map_exprs(&self, f: &mut impl FnMut(&Expr) -> Expr) -> Self {
        match self {
            Self::Cmp { op, a, b } => Self::Cmp {
                op: *op,
                a: f(a),
                b: f(b),
            },
            Self::Opaque { desc, args } => Self::Opaque {
                desc: desc.clone(),
                args: args.iter().map(f).collect(),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Store {
    pub address: Expr,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub enum Term {
    Goto(NodeId),
    Branch {
        cond: Cond,
        then: NodeId,
        els: NodeId,
    },
    /// Named return components (first is the C return value; the rest are
    /// reported as comments, e.g. callee-saved registers).
    Return(Vec<(String, Expr)>),
}

#[derive(Clone, Debug)]
pub struct Block {
    pub id: NodeId,
    pub assigns: Vec<(String, Expr)>,
    pub stores: Vec<Store>,
    pub term: Term,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub entry: NodeId,
    pub blocks: BTreeMap<NodeId, Block>,
    /// Variables live on entry, in argument order, with their display names.
    pub params: Vec<String>,
    /// Notes the frontend wants printed above the function (assumptions).
    pub notes: Vec<String>,
}

impl Func {
    #[must_use]
    pub fn successors(&self, id: NodeId) -> Vec<NodeId> {
        match &self.blocks[&id].term {
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
}
