//! C-like rendering of expressions, conditions and structured statements.
//! Every operator is rendered with the exact wrap-around semantics of the IR:
//! narrow (8/16-bit) results are explicitly truncated, shift counts are masked,
//! rotates go through `rolN`/`rorN` (count taken modulo N).

use std::collections::BTreeMap;

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp};

use crate::ir::{Cmp, Cond, NodeId};
use crate::opt::Stmt;

pub fn ty(w: BitWidth) -> &'static str {
    match w {
        BitWidth::W8 => "u8",
        BitWidth::W16 => "u16",
        BitWidth::W32 => "u32",
        BitWidth::W64 => "u64",
    }
}

fn sty(w: BitWidth) -> &'static str {
    match w {
        BitWidth::W8 => "i8",
        BitWidth::W16 => "i16",
        BitWidth::W32 => "i32",
        BitWidth::W64 => "i64",
    }
}

const P_ATOM: u8 = 15;
const P_UN: u8 = 14;

pub struct Renderer<'a> {
    pub names: &'a BTreeMap<String, String>,
    /// Subtrees that have been given a local name (see [`crate::cse`]). Any
    /// expression in here renders AS that name instead of being printed again,
    /// which is what keeps a shared value from appearing ten times in one line.
    pub temps: &'a std::collections::HashMap<Expr, String>,
}

fn hex(width: BitWidth, v: u64) -> String {
    if v <= 9 {
        v.to_string()
    } else if width == BitWidth::W64 && v > u64::from(u32::MAX) {
        format!("0x{v:x}ULL")
    } else {
        format!("0x{v:x}")
    }
}

fn shift_mask(w: BitWidth) -> u64 {
    if w == BitWidth::W64 { 63 } else { 31 }
}

impl Renderer<'_> {
    fn name(&self, n: &str) -> String {
        self.names.get(n).cloned().unwrap_or_else(|| n.to_owned())
    }

    fn child(&self, c: &Expr, parent: Option<BinaryOp>, right: bool, ring: bool) -> String {
        let (s, p) = self.rx(c, ring);
        if p >= P_UN {
            return s;
        }
        let ExprKind::Binary { op, .. } = &c.kind else {
            return format!("({s})");
        };
        let need = match parent {
            None => true,
            Some(pop) => {
                if *op == pop {
                    right
                        && !matches!(
                            pop,
                            BinaryOp::Add | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or | BinaryOp::Mul
                        )
                } else {
                    !(matches!(pop, BinaryOp::Add | BinaryOp::Sub) && *op == BinaryOp::Mul)
                }
            }
        };
        if need { format!("({s})") } else { s }
    }

    fn wrap_child(&self, child: &Expr, parent: Option<BinaryOp>, right: bool) -> String {
        self.child(child, parent, right, false)
    }

    /// Renders `e`; `ring_ok` means the consumer only looks at the low bits (an
    /// assignment into a narrow variable, or a parent low-bits-only operator),
    /// so a narrow ring operation need not truncate itself.
    /// Renders `e`, substituting any named subtree for its name.
    pub fn rx(&self, e: &Expr, ring_ok: bool) -> (String, u8) {
        if let Some(name) = self.temps.get(e) {
            return (name.clone(), P_ATOM);
        }
        self.rx_body(e, ring_ok)
    }

    /// Renders a temporary's DEFINITION: the root is printed in full (or the
    /// definition would be `t3 = t3`), while nested named subtrees still
    /// substitute.
    pub fn define(&self, e: &Expr) -> String {
        self.rx_body(e, true).0
    }

    fn rx_body(&self, e: &Expr, ring_ok: bool) -> (String, u8) {
        let narrow = e.width.bits() < 32;
        match &e.kind {
            ExprKind::Const { value } => (hex(e.width, *value), P_ATOM),
            ExprKind::Register { name } => (self.name(name), P_ATOM),
            ExprKind::Unary { op, value } => {
                let o = match op {
                    UnaryOp::Not => "~",
                    UnaryOp::Neg => "-",
                };
                let inner = self.wrap_child_ring(value, narrow);
                let s = format!("{o}{inner}");
                if narrow && !ring_ok {
                    (format!("({}){}", ty(e.width), paren(&s)), P_UN)
                } else {
                    (s, P_UN)
                }
            }
            ExprKind::Binary { op, left, right } => self.binary(e, *op, left, right, ring_ok),
            ExprKind::Load { address } => {
                let a = self.rx(address, false).0;
                (format!("*({} *)({a})", ty(e.width)), P_UN)
            }
            ExprKind::Intrinsic { name, arguments } => {
                let args: Vec<String> = arguments.iter().map(|a| self.rx(a, false).0).collect();
                (format!("{name}({})", args.join(", ")), P_ATOM)
            }
            ExprKind::Extract { value, lsb } => {
                let v = self.wrap_child(value, None, false);
                if *lsb == 0 {
                    (format!("({}){v}", ty(e.width)), P_UN)
                } else {
                    (format!("({})({v} >> {lsb})", ty(e.width)), P_UN)
                }
            }
            ExprKind::Concat { high, low } => {
                let n = ty(e.width);
                let h = self.wrap_child(high, None, false);
                let l = self.wrap_child(low, None, false);
                (
                    format!("((({n}){h} << {}) | ({n}){l})", low.width.bits()),
                    P_ATOM,
                )
            }
            ExprKind::Resize { op, value } => {
                let v = self.wrap_child(value, None, false);
                match op {
                    ResizeOp::Truncate => (format!("({}){v}", ty(e.width)), P_UN),
                    ResizeOp::ZeroExtend => (format!("({}){v}", ty(e.width)), P_UN),
                    ResizeOp::SignExtend => (
                        format!("({})({})({}){v}", ty(e.width), sty(e.width), sty(value.width)),
                        P_UN,
                    ),
                }
            }
        }
    }

    fn wrap_child_ring(&self, child: &Expr, parent_narrow: bool) -> String {
        let ring_child = parent_narrow && is_ring(child);
        let (s, p) = self.rx(child, ring_child);
        if p >= P_UN { s } else { format!("({s})") }
    }

    fn operand(&self, child: &Expr, parent: BinaryOp, right: bool, parent_narrow: bool) -> String {
        let ring_child = parent_narrow && is_ring(child) && is_low_bits_op(parent);
        self.child(child, Some(parent), right, ring_child)
    }

    fn count(&self, n: &Expr, w: BitWidth) -> String {
        let mask = shift_mask(w);
        if let ExprKind::Const { value } = n.kind {
            return (value & mask).to_string();
        }
        if let ExprKind::Binary { op: BinaryOp::And, right, .. } = &n.kind
            && let ExprKind::Const { value } = right.kind
            && value == mask
        {
            return format!("({})", self.rx(n, false).0);
        }
        let s = self.wrap_child(n, Some(BinaryOp::And), false);
        format!("({s} & {mask})")
    }

    fn binary(&self, e: &Expr, op: BinaryOp, l: &Expr, r: &Expr, ring_ok: bool) -> (String, u8) {
        let w = e.width;
        let narrow = w.bits() < 32;
        let wrap = |s: String, p: u8| -> (String, u8) {
            if narrow && !ring_ok {
                (format!("({}){}", ty(w), paren(&s)), P_UN)
            } else {
                (s, p)
            }
        };
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or => {
                // a + C with the top bit of C set reads better as a - |C|
                if op == BinaryOp::Add
                    && let ExprKind::Const { value } = r.kind
                    && value >> (w.bits() - 1) == 1
                    && w.bits() >= 32
                {
                    let neg = w.truncate(value.wrapping_neg());
                    let a = self.operand(l, BinaryOp::Sub, false, narrow);
                    return wrap(format!("{a} - {}", hex(w, neg)), 11);
                }
                let (sym, p) = match op {
                    BinaryOp::Add => ("+", 11),
                    BinaryOp::Sub => ("-", 11),
                    BinaryOp::Xor => ("^", 7),
                    BinaryOp::And => ("&", 8),
                    _ => ("|", 6),
                };
                let a = self.operand(l, op, false, narrow);
                let b = self.operand(r, op, true, narrow);
                wrap(format!("{a} {sym} {b}"), p)
            }
            BinaryOp::Mul => {
                if narrow {
                    let a = self.wrap_child(l, None, false);
                    let b = self.wrap_child(r, None, false);
                    let s = format!("(u32){a} * (u32){b}");
                    wrap(s, 12)
                } else {
                    let a = self.operand(l, op, false, false);
                    let b = self.operand(r, op, true, false);
                    (format!("{a} * {b}"), 12)
                }
            }
            BinaryOp::Shl | BinaryOp::Shr => {
                let sym = if op == BinaryOp::Shl { "<<" } else { ">>" };
                if let ExprKind::Const { value } = r.kind
                    && value & shift_mask(w) == 0
                {
                    return self.rx(l, ring_ok);
                }
                let a = self.wrap_child(l, None, false);
                let c = self.count(r, w);
                let a = if narrow && op == BinaryOp::Shl { format!("(u32){a}") } else { a };
                let s = format!("{a} {sym} {c}");
                if narrow {
                    (format!("({}){}", ty(w), paren(&s)), P_UN)
                } else {
                    (s, 10)
                }
            }
            BinaryOp::Sar => {
                let a = self.wrap_child(l, None, false);
                let c = self.count(r, w);
                let s = format!("({}){a} >> {c}", sty(w));
                if narrow {
                    (format!("({}){}", ty(w), paren(&s)), P_UN)
                } else {
                    (format!("({}){}", ty(w), paren(&s)), P_UN)
                }
            }
            BinaryOp::Rol | BinaryOp::Ror => {
                let bits = u64::from(w.bits());
                if let ExprKind::Const { value } = r.kind {
                    let mut c = (value & shift_mask(w)) % bits;
                    if c == 0 {
                        return self.rx(l, ring_ok);
                    }
                    let mut o = op;
                    if c > bits / 2 {
                        c = bits - c;
                        o = if op == BinaryOp::Rol { BinaryOp::Ror } else { BinaryOp::Rol };
                    }
                    let f = if o == BinaryOp::Rol { "rol" } else { "ror" };
                    let a = self.rx(l, false).0;
                    return (format!("{f}{}({a}, {c})", w.bits()), P_ATOM);
                }
                let f = if op == BinaryOp::Rol { "rol" } else { "ror" };
                let a = self.rx(l, false).0;
                let n = self.rx(r, false).0;
                (format!("{f}{}({a}, {n})", w.bits()), P_ATOM)
            }
        }
    }

    pub fn expr(&self, e: &Expr) -> String {
        self.rx(e, true).0
    }

    pub fn cond(&self, c: &Cond) -> String {
        match c {
            Cond::Cmp { op, a, b } => {
                let signed = op.signed();
                let side = |x: &Expr| -> String {
                    let s = self.wrap_child(x, None, false);
                    if signed && !matches!(x.kind, ExprKind::Const { .. }) {
                        format!("({}){s}", sty(x.width))
                    } else if signed {
                        // constants print as signed values
                        let v = x.width.truncate(match x.kind {
                            ExprKind::Const { value } => value,
                            _ => 0,
                        });
                        let sv = signed_value(x.width, v);
                        sv.to_string()
                    } else {
                        s
                    }
                };
                format!("{} {} {}", side(a), op.c_op(), side(b))
            }
            Cond::Opaque { desc, args } => {
                let a: Vec<String> = args.iter().map(|x| self.rx(x, false).0).collect();
                format!("{desc}({})", a.join(", "))
            }
        }
    }
}

fn signed_value(w: BitWidth, v: u64) -> i64 {
    let bits = w.bits();
    if bits == 64 {
        v.cast_signed()
    } else if v >> (bits - 1) & 1 == 1 {
        (v | (u64::MAX << bits)).cast_signed()
    } else {
        v.cast_signed()
    }
}

fn is_infix(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or | BinaryOp::Mul | BinaryOp::Shl | BinaryOp::Shr
    )
}

fn is_ring(e: &Expr) -> bool {
    matches!(&e.kind, ExprKind::Binary { op, .. } if matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or))
        || matches!(&e.kind, ExprKind::Unary { .. })
}

fn is_low_bits_op(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or)
}

fn is_cast_wrapped(_e: &Expr) -> bool {
    false
}

fn paren(s: &str) -> String {
    format!("({s})")
}

trait Wrapped {
    fn wrapped(self) -> (String, u8);
}
impl Wrapped for (String, u8) {
    fn wrapped(self) -> (String, u8) {
        (format!("({})", self.0), P_ATOM)
    }
}

// ---------------------------------------------------------------- statements

pub struct Ctx<'a> {
    pub r: Renderer<'a>,
    pub out: String,
    pub log: Vec<(Logged, String)>,
}

#[derive(Clone, Debug)]
pub enum Logged {
    E(Expr),
    C(Cond),
}

impl Ctx<'_> {
    fn lexpr(&mut self, e: &Expr) -> String {
        let t = self.r.expr(e);
        self.log.push((Logged::E(e.clone()), t.clone()));
        t
    }

    fn lcond(&mut self, c: &Cond) -> String {
        let t = self.r.cond(c);
        self.log.push((Logged::C(c.clone()), t.clone()));
        t
    }

    fn line(&mut self, depth: usize, s: &str) {
        for _ in 0..depth {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    pub fn stmts(&mut self, list: &[Stmt], depth: usize, ret_w: Option<BitWidth>) {
        for s in list {
            self.stmt(s, depth, ret_w);
        }
    }

    fn block(&mut self, head: &str, body: &[Stmt], depth: usize, ret_w: Option<BitWidth>) {
        self.line(depth, &format!("{head} {{"));
        self.stmts(body, depth + 1, ret_w);
        self.line(depth, "}");
    }

    fn stmt(&mut self, s: &Stmt, depth: usize, ret_w: Option<BitWidth>) {
        match s {
            Stmt::Assign(v, e) => {
                let name = self.r.name(v);
                let t = self.lexpr(e);
                self.line(depth, &format!("{name} = {t};"));
            }
            Stmt::Store(st) => {
                let a = self.r.rx(&st.address, false).0;
                let v = self.lexpr(&st.value);
                self.line(depth, &format!("*({} *)({a}) = {v};", ty(st.value.width)));
            }
            Stmt::If { cond, then, els } => {
                let c = self.lcond(cond);
                self.line(depth, &format!("if ({c}) {{"));
                self.stmts(then, depth + 1, ret_w);
                if els.is_empty() {
                    self.line(depth, "}");
                } else if let [Stmt::If { .. }] = els.as_slice() {
                    self.line(depth, "} else");
                    self.stmts(els, depth, ret_w);
                } else {
                    self.line(depth, "} else {");
                    self.stmts(els, depth + 1, ret_w);
                    self.line(depth, "}");
                }
            }
            Stmt::Loop(body) => self.block("for (;;)", body, depth, ret_w),
            Stmt::While { cond, body } => {
                let c = self.lcond(cond);
                self.block(&format!("while ({c})"), body, depth, ret_w);
            }
            Stmt::DoWhile { body, cond } => {
                self.line(depth, "do {");
                self.stmts(body, depth + 1, ret_w);
                let c = self.lcond(cond);
                self.line(depth, &format!("}} while ({c});"));
            }
            Stmt::Break => self.line(depth, "break;"),
            Stmt::Continue => self.line(depth, "continue;"),
            Stmt::Return(vals) => {
                let (_, first) = &vals[0];
                let e = self.lexpr(first);
                let _ = ret_w;
                self.line(depth, &format!("return {e};"));
            }
            Stmt::Label(n) => self.line(0, &format!("L_{n:x}:")),
            Stmt::Goto(n) => self.line(depth, &format!("goto L_{n:x};")),
            Stmt::IfGoto { cond, target } => {
                let c = self.lcond(cond);
                self.line(depth, &format!("if ({c}) goto L_{target:x};"));
            }
            Stmt::Comment(c) => self.line(depth, &format!("/* {c} */")),
        }
    }
}

pub fn node_label(n: NodeId) -> String {
    format!("L_{n:x}")
}

pub fn cmp_name(c: Cmp) -> &'static str {
    c.c_op()
}
