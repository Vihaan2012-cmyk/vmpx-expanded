//! Readable Rust rendering of a recovered function.
//!
//! A second printer over the same `opt::Stmt` / `vmpx_vm::Expr` tree the C
//! renderer uses, emitting faithful Rust: `let` bindings (no width types in the
//! body), explicit `.wrapping_*` for every arithmetic op the VM relies on to
//! wrap, `read_*`/`write_*` helpers for memory, `!` for bitwise-not, and named
//! constants. Opt-in via `VMPX_DECOMPILE_DIALECT=rust`; the default C output is
//! untouched.

use std::collections::{BTreeMap, HashMap};

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp};

use crate::ir::Cond;
use crate::opt::Stmt;
use crate::symbols::Symbols;

const ATOM: u8 = 100;
/// A conditional (`?:` / `if {} else {}`) binds looser than any binary operator,
/// so it parenthesises whenever it is embedded as an operand.
const COND: u8 = 5;

static DIALECT: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

/// Select the output dialect (`"rust"`, ...), or `None` for the default C. Set
/// by the CLI `--dialect` flag; falls back to `VMPX_DECOMPILE_DIALECT`.
pub fn set_dialect(d: Option<String>) {
    *DIALECT
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = d;
}

/// The selected dialect: the CLI flag if set, else the env var.
#[must_use]
pub fn dialect() -> Option<String> {
    if let Some(d) = DIALECT
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
    {
        return Some(d);
    }
    std::env::var("VMPX_DECOMPILE_DIALECT").ok()
}

/// The readable output dialects. All render the *same* recovered tree -- idioms,
/// names and proven folds are dialect-independent; only syntax differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    /// Faithful Rust: explicit `.wrapping_*`, `!` bitwise-not, no body types.
    Rust,
    /// Compilable C with `<stdint.h>` types; unsigned overflow is defined, so the
    /// body needs no wrapping verbs.
    C99,
    /// C23: like C99 but `auto`-typed temporaries and `constexpr` constants.
    C23,
    /// C++: like C99 but `auto`-typed temporaries and `constexpr` constants.
    Cpp,
    /// Maximally readable, not compilable: types and widening casts elided.
    Pseudo,
}

impl Dialect {
    /// Parse a `--dialect` value; `None` for the default C emitter (not readable).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "rust" => Some(Self::Rust),
            "c99" => Some(Self::C99),
            "c23" => Some(Self::C23),
            "cpp" | "c++" => Some(Self::Cpp),
            "pseudo" => Some(Self::Pseudo),
            _ => None,
        }
    }
    fn is_c_family(self) -> bool {
        matches!(self, Self::C99 | Self::C23 | Self::Cpp)
    }
    /// C23/C++ infer temp types with `auto`.
    fn auto_temps(self) -> bool {
        matches!(self, Self::C23 | Self::Cpp)
    }
}

fn uty(w: BitWidth) -> &'static str {
    match w {
        BitWidth::W8 => "u8",
        BitWidth::W16 => "u16",
        BitWidth::W32 => "u32",
        BitWidth::W64 => "u64",
    }
}

fn ity(w: BitWidth) -> &'static str {
    match w {
        BitWidth::W8 => "i8",
        BitWidth::W16 => "i16",
        BitWidth::W32 => "i32",
        BitWidth::W64 => "i64",
    }
}

fn cty(w: BitWidth) -> &'static str {
    match w {
        BitWidth::W8 => "uint8_t",
        BitWidth::W16 => "uint16_t",
        BitWidth::W32 => "uint32_t",
        BitWidth::W64 => "uint64_t",
    }
}

fn cty_signed(w: BitWidth) -> &'static str {
    match w {
        BitWidth::W8 => "int8_t",
        BitWidth::W16 => "int16_t",
        BitWidth::W32 => "int32_t",
        BitWidth::W64 => "int64_t",
    }
}

fn psty(w: BitWidth) -> &'static str {
    match w {
        BitWidth::W8 => "s8",
        BitWidth::W16 => "s16",
        BitWidth::W32 => "s32",
        BitWidth::W64 => "s64",
    }
}

/// Parenthesise `s` (rendered at precedence `have`) when embedded where `need`
/// binding is required.
fn paren(s: String, have: u8, need: u8) -> String {
    if have < need { format!("({s})") } else { s }
}

/// A constant's readable name, or `None` for a plain literal. Role-based, so the
/// same value in a different role is free to print as a literal.
fn const_name(w: BitWidth, v: u64, image_base: u64) -> Option<&'static str> {
    if v != 0 && v == image_base {
        return Some("IMAGE_BASE");
    }
    if w == BitWidth::W8 && v == 0xCC {
        return Some("INT3");
    }
    None
}

pub struct Renderer<'a> {
    pub dialect: Dialect,
    pub names: &'a BTreeMap<String, String>,
    pub image_base: u64,
    pub temps: &'a HashMap<Expr, String>,
    /// Role-named constants (a recurring XOR mask -> `DECODE_KEY`, etc.).
    pub consts: &'a BTreeMap<u64, String>,
    /// A sound `Ctx` struct layout was synthesized for this function, so a
    /// compilable dialect renders `ctx + off` loads/stores as struct fields
    /// (`ctx->fNN` / `(*ctx).fNN`) instead of `read_u*`/`write_*`. `false` keeps
    /// the raw helpers (the fail-closed form).
    pub ctx_struct: bool,
}

impl Renderer<'_> {
    fn name(&self, n: &str) -> String {
        self.names.get(n).cloned().unwrap_or_else(|| n.to_owned())
    }

    /// Unsigned type name for this dialect (`u32` / `uint32_t`).
    fn ty(&self, w: BitWidth) -> &'static str {
        if self.dialect.is_c_family() {
            cty(w)
        } else {
            uty(w)
        }
    }
    /// Signed type name (`i32` / `int32_t` / `s32`).
    fn sty(&self, w: BitWidth) -> &'static str {
        match self.dialect {
            Dialect::Rust => ity(w),
            Dialect::Pseudo => psty(w),
            _ => cty_signed(w),
        }
    }

    /// Cast `inner` (precedence `p`) to width `w` as a plain widen/truncate.
    /// Rust: `x as u32`. C: `(uint32_t)x`. Pseudo: elided -- the width is carried
    /// by context, so the cast is noise.
    fn widen(&self, inner: String, p: u8, w: BitWidth) -> (String, u8) {
        match self.dialect {
            Dialect::Pseudo => (inner, p),
            Dialect::Rust => (format!("{} as {}", paren(inner, p, 80), self.ty(w)), 80),
            _ => (format!("({}){}", self.ty(w), paren(inner, p, 85)), 85),
        }
    }

    fn lit(&self, w: BitWidth, v: u64) -> String {
        if let Some(n) = const_name(w, v, self.image_base) {
            return n.to_owned();
        }
        if let Some(n) = self.consts.get(&v) {
            return n.clone();
        }
        if v <= 9 {
            v.to_string()
        } else {
            format!("0x{v:x}")
        }
    }

    /// Render with a precedence; the caller parenthesises when it embeds a
    /// lower-precedence child.
    fn rx(&self, e: &Expr) -> (String, u8) {
        if let Some(name) = self.temps.get(e) {
            return (name.clone(), ATOM);
        }
        self.body(e)
    }

    /// A child, parenthesised when its precedence is below `need`.
    fn child(&self, e: &Expr, need: u8) -> String {
        let (s, p) = self.rx(e);
        if p < need { format!("({s})") } else { s }
    }

    /// The definition of a named temp: never substitute the root for its own name.
    pub fn define(&self, e: &Expr) -> String {
        self.body(e).0
    }

    /// A VM flag-word predicate (`vm_flags_*(K, x) & bit >> shift`), collapsed
    /// to the comparison it actually is -- but ONLY when exhaustively proven.
    /// The predicate is a pure function of its one non-constant operand `x`, so
    /// over `x`'s whole (<=16-bit) domain it either is a clean indicator of a
    /// comparison or it is not; if it is, emit `(x CMP C) [as T] [* scale]`,
    /// else fall through to the literal rendering. Never emits an unproven form.
    fn flag_predicate(&self, e: &Expr) -> Option<(String, u8)> {
        // The one `vm_flags_*` intrinsic in `e` (bail on zero or several), and
        // its single non-constant operand `x` -- the value the predicate tests.
        let mut flags = Vec::new();
        collect_vm_flags(e, &mut flags);
        let [intrin] = flags[..] else { return None };
        let ExprKind::Intrinsic { arguments, .. } = &intrin.kind else {
            return None;
        };
        let nonconst: Vec<&Expr> = arguments.iter().filter(|a| cst(a).is_none()).collect();
        let [x] = nonconst[..] else { return None };
        let xw = x.width;
        if xw.bits() > 16 {
            return None;
        }
        let out_w = e.width;
        // Replace `x` everywhere with a probe register, then tabulate `e` over the
        // probe's whole domain -- structure-agnostic and exact. If any other free
        // value or an unfoldable intrinsic remains, eval errors and we bail.
        let probe = Expr::register(xw, "__flagprobe");
        let e2 = replace_subtree(e, x, &probe);
        let n = 1u64 << xw.bits();
        let mut table = Vec::with_capacity(n as usize);
        for v in 0..n {
            let env: crate::exec::Env = std::iter::once(("__flagprobe".to_owned(), v)).collect();
            table.push(crate::exec::eval(&e2, &env).ok()?);
        }
        let scale = *table.iter().find(|&&v| v != 0)?;
        if table.iter().any(|&v| v != 0 && v != scale) {
            return None; // not a two-valued indicator
        }
        let hits: Vec<u64> = (0..n).filter(|&i| table[i as usize] == scale).collect();
        let (op, c) = recognize_cmp(&hits, n)?;
        let xr = self.child(x, 10);
        let cname = self.lit(xw, c);
        let cond = format!("{xr} {op} {cname}");
        if scale == 1 {
            // A 0/1 indicator: widen the boolean to the result type.
            let (s, p) = self.widen(cond, 10, out_w);
            return Some((s, p));
        }
        // A scaled indicator reads as a conditional, not `(cond as T) * k` -- the
        // two are identical because `cond` is exactly 0 or 1.
        let k = self.lit(out_w, scale);
        let s = match self.dialect {
            Dialect::Rust => format!("if {cond} {{ {k} }} else {{ 0 }}"),
            _ => format!("{cond} ? {k} : 0"),
        };
        Some((s, COND))
    }

    fn body(&self, e: &Expr) -> (String, u8) {
        if let Some(pred) = self.flag_predicate(e) {
            return pred;
        }
        match &e.kind {
            ExprKind::Const { value } => (self.lit(e.width, *value), ATOM),
            ExprKind::Register { name } => (self.name(name), ATOM),
            ExprKind::Unary { op, value } => match op {
                // Rust `!` is bitwise-not on integers; C/pseudo spell it `~`.
                UnaryOp::Not => {
                    let tilde = if self.dialect == Dialect::Rust {
                        "!"
                    } else {
                        "~"
                    };
                    (format!("{tilde}{}", self.child(value, 90)), 90)
                }
                // Unsigned negation is defined mod 2^n in both Rust (wrapping) and C.
                UnaryOp::Neg if self.dialect == Dialect::Rust => {
                    (format!("{}.wrapping_neg()", self.child(value, ATOM)), ATOM)
                }
                UnaryOp::Neg => (format!("-{}", self.child(value, 90)), 90),
            },
            ExprKind::Binary { op, left, right } => self.binary(e.width, *op, left, right),
            ExprKind::Load { address } => match self.field_sugar(address) {
                Some(f) => (f, ATOM),
                None => (
                    format!("read_u{}({})", e.width.bits(), self.child(address, 0)),
                    ATOM,
                ),
            },
            ExprKind::Intrinsic { name, arguments } => {
                let args: Vec<String> = arguments.iter().map(|a| self.child(a, 0)).collect();
                (format!("{}({})", intrinsic(name), args.join(", ")), ATOM)
            }
            ExprKind::Extract { value, lsb } => {
                let (vs, vp) = self.rx(value);
                let (inner, ip) = if *lsb == 0 {
                    (vs, vp)
                } else {
                    (format!("{} >> {lsb}", paren(vs, vp, 71)), 70)
                };
                self.widen(inner, ip, e.width)
            }
            ExprKind::Resize { op, value } => match op {
                ResizeOp::Truncate | ResizeOp::ZeroExtend => {
                    let (vs, vp) = self.rx(value);
                    self.widen(vs, vp, e.width)
                }
                ResizeOp::SignExtend => (self.sign_extend(value, e.width), 85),
            },
            ExprKind::Concat { high, low } => {
                let (hs, hp) = self.rx(high);
                let (ls, lp) = self.rx(low);
                let (h, _) = self.widen(hs, hp, e.width);
                let (l, _) = self.widen(ls, lp, e.width);
                (format!("({h} << {}) | ({l})", low.width.bits()), 40)
            }
        }
    }

    /// Sign-extend `value` to `w`: Rust casts through both signed widths, C stacks
    /// signed casts, pseudo keeps the signed markers.
    fn sign_extend(&self, value: &Expr, w: BitWidth) -> String {
        let v = self.child(value, 85);
        let (from, to) = (value.width, w);
        match self.dialect {
            Dialect::Rust => {
                format!(
                    "{v} as {} as {} as {}",
                    self.sty(from),
                    self.sty(to),
                    self.ty(w)
                )
            }
            Dialect::Pseudo => format!("({})({}){v}", self.sty(to), self.sty(from)),
            _ => format!("({})({})({}){v}", self.ty(w), self.sty(to), self.sty(from)),
        }
    }

    fn binary(&self, w: BitWidth, op: BinaryOp, l: &Expr, r: &Expr) -> (String, u8) {
        if self.dialect == Dialect::Rust {
            return self.binary_rust(w, op, l, r);
        }
        // C-family and pseudo: unsigned arithmetic wraps by definition, so plain
        // infix is already faithful -- no wrapping verbs needed.
        let infix = |sym: &str, prec: u8| -> (String, u8) {
            (
                format!("{} {sym} {}", self.child(l, prec), self.child(r, prec + 1)),
                prec,
            )
        };
        match op {
            BinaryOp::Mul => infix("*", 75),
            BinaryOp::Add => infix("+", 72),
            BinaryOp::Sub => infix("-", 72),
            BinaryOp::Shl => infix("<<", 70),
            BinaryOp::Shr => infix(">>", 70),
            BinaryOp::And => infix("&", 60),
            BinaryOp::Xor => infix("^", 50),
            BinaryOp::Or => infix("|", 40),
            BinaryOp::Sar => {
                // Arithmetic shift: shift the signed view; context reads it back.
                let a = self.child(l, 85);
                let b = self.child(r, 71);
                (format!("(({}){a}) >> {b}", self.sty(w)), 70)
            }
            BinaryOp::Rol | BinaryOp::Ror => {
                let f = if op == BinaryOp::Rol { "rotl" } else { "rotr" };
                let a = self.child(l, 0);
                let b = self.child(r, 0);
                (format!("{f}{}({a}, {b})", w.bits()), ATOM)
            }
        }
    }

    /// Faithful Rust: every op the VM relies on to wrap gets an explicit
    /// `.wrapping_*`, so debug and release agree.
    fn binary_rust(&self, w: BitWidth, op: BinaryOp, l: &Expr, r: &Expr) -> (String, u8) {
        let as_w = |s: String, need_as: bool| {
            if need_as {
                format!("{s} as {}", uty(w))
            } else {
                s
            }
        };
        let method = |name: &str| -> (String, u8) {
            let a = self.child(l, ATOM);
            let b = self.child(r, 0);
            (format!("{a}.{name}({b})"), ATOM)
        };
        match op {
            BinaryOp::Add => method("wrapping_add"),
            BinaryOp::Sub => method("wrapping_sub"),
            BinaryOp::Mul => method("wrapping_mul"),
            BinaryOp::Shl => {
                let a = self.child(l, ATOM);
                let b = self.child(r, 0);
                (format!("{a}.wrapping_shl({b} as u32)"), ATOM)
            }
            BinaryOp::Xor => (format!("{} ^ {}", self.child(l, 50), self.child(r, 51)), 50),
            BinaryOp::And => (format!("{} & {}", self.child(l, 60), self.child(r, 61)), 60),
            BinaryOp::Or => (format!("{} | {}", self.child(l, 40), self.child(r, 41)), 40),
            BinaryOp::Shr => {
                let s = format!("{} >> {}", self.child(l, 70), self.child(r, 0));
                (s, 70)
            }
            BinaryOp::Sar => {
                let a = self.child(l, 80);
                let b = self.child(r, 0);
                (as_w(format!("({a} as {}) >> {b}", ity(w)), true), 80)
            }
            BinaryOp::Rol | BinaryOp::Ror => {
                let f = if op == BinaryOp::Rol {
                    "rotate_left"
                } else {
                    "rotate_right"
                };
                let a = self.child(l, ATOM);
                let b = self.child(r, 0);
                (format!("{a}.{f}({b} as u32)"), ATOM)
            }
        }
    }

    /// Struct-field access for `ctx + off` when `ctx` is the inferred context
    /// pointer: `read_u64(ctx + 0x98)` -> `ctx.f98` / `ctx->f98` / `(*ctx).f98`.
    /// Pseudo always sugars (it is not compiled). A compilable dialect sugars only
    /// when a sound `Ctx` struct was synthesized (`ctx_struct`); otherwise it keeps
    /// `read_u*`/`write_*`, the fail-closed form.
    fn field_sugar(&self, address: &Expr) -> Option<String> {
        let (r, off) = base_offset(address)?;
        if self.name(r) != "ctx" {
            return None;
        }
        match self.dialect {
            Dialect::Pseudo => Some(format!("ctx.f{off:x}")),
            Dialect::Rust if self.ctx_struct => Some(format!("(*ctx).f{off:x}")),
            _ if self.ctx_struct => Some(format!("ctx->f{off:x}")),
            _ => None,
        }
    }

    /// A loop/branch condition, parenthesised for the C-style `if (...)`/`while
    /// (...)` of every dialect but Rust.
    fn guard(&self, c: &Cond) -> String {
        if self.dialect == Dialect::Rust {
            self.cond(c)
        } else {
            format!("({})", self.cond(c))
        }
    }

    fn cond(&self, c: &Cond) -> String {
        match c {
            Cond::Cmp { op, a, b } => {
                format!("{} {} {}", self.child(a, 10), op.c_op(), self.child(b, 10))
            }
            Cond::Opaque { desc, args } => {
                let a: Vec<String> = args.iter().map(|e| self.child(e, 0)).collect();
                format!("{desc}({})", a.join(", "))
            }
        }
    }

    /// Per-statement common-subexpression elimination, then the statement.
    ///
    /// Any non-trivial subexpression that occurs more than once inside this one
    /// statement is bound to a local emitted immediately before it, and the
    /// repeats become that name. Sound because every use sits in the same
    /// statement: no store executes between the binding and the uses, so
    /// collapsing a repeated load to a single evaluation is value-preserving.
    /// Never hoists across statements or across a store.
    fn stmt(&self, s: &Stmt, out: &mut String, depth: usize) {
        let mut tops: Vec<&Expr> = Vec::new();
        stmt_top_exprs(s, &mut tops);
        let binds = if tops.is_empty() {
            Vec::new()
        } else {
            self.plan_cse(&tops)
        };
        if binds.is_empty() {
            self.stmt_leaf(s, out, depth);
            return;
        }
        let mut overlay = self.temps.clone();
        for (nm, def) in &binds {
            overlay.insert(def.clone(), nm.clone());
        }
        let r2 = Renderer {
            dialect: self.dialect,
            names: self.names,
            image_base: self.image_base,
            temps: &overlay,
            consts: self.consts,
            ctx_struct: self.ctx_struct,
        };
        let pad = "    ".repeat(depth);
        for (nm, def) in &binds {
            let rhs = r2.define(def);
            let line = match self.dialect {
                Dialect::Rust => format!("{pad}let {nm} = {rhs};\n"),
                Dialect::Pseudo => format!("{pad}{nm} = {rhs}\n"),
                _ if self.dialect.auto_temps() => format!("{pad}auto {nm} = {rhs};\n"),
                _ => format!("{pad}{} {nm} = {rhs};\n", cty(def.width)),
            };
            out.push_str(&line);
        }
        r2.stmt_leaf(s, out, depth);
    }

    /// The CSE bindings for one statement: maximal, non-trivial, repeated
    /// subexpressions, in first-appearance order, with names that don't collide.
    fn plan_cse(&self, tops: &[&Expr]) -> Vec<(String, Expr)> {
        let mut counts: HashMap<Expr, usize> = HashMap::new();
        let mut order: Vec<Expr> = Vec::new();
        for e in tops {
            each_subexpr(e, &mut |se| {
                let c = counts.entry(se.clone()).or_insert(0);
                *c += 1;
                if *c == 1 {
                    order.push(se.clone());
                }
            });
        }
        let repeated: Vec<Expr> = order
            .into_iter()
            .filter(|e| counts[e] >= 2 && cse_worthy(e))
            .collect();
        // Keep only the maximal candidates -- drop any nested inside another.
        let maximal: Vec<Expr> = repeated
            .iter()
            .filter(|e| !repeated.iter().any(|o| proper_contains(o, e)))
            .cloned()
            .collect();
        let mut used: std::collections::HashSet<String> = self
            .names
            .values()
            .cloned()
            .chain(self.temps.values().cloned())
            .collect();
        let mut tn = 0usize;
        let mut out = Vec::new();
        for e in maximal {
            let base = self.cse_name(&e, &mut tn);
            let mut nm = base.clone();
            let mut k = 2;
            while !used.insert(nm.clone()) {
                nm = format!("{base}_{k}");
                k += 1;
            }
            out.push((nm, e));
        }
        out
    }

    /// A readable name for a CSE temp: `vNN` for a context field load
    /// `ctx + 0xNN`, else a fresh `tN`.
    fn cse_name(&self, e: &Expr, tn: &mut usize) -> String {
        if let ExprKind::Load { address } = &e.kind
            && let Some((r, off)) = base_offset(address)
            && self.name(r) == "ctx"
        {
            return format!("v{off:x}");
        }
        let n = *tn;
        *tn += 1;
        format!("t{n}")
    }

    fn stmt_leaf(&self, s: &Stmt, out: &mut String, depth: usize) {
        let pad = "    ".repeat(depth);
        // Pseudo-code drops statement terminators; every real dialect keeps them.
        let semi = if self.dialect == Dialect::Pseudo {
            ""
        } else {
            ";"
        };
        match s {
            Stmt::Assign(n, e) => {
                out.push_str(&format!(
                    "{pad}{} = {}{semi}\n",
                    self.name(n),
                    self.body(e).0
                ));
            }
            Stmt::Store(st) => {
                let val = self.body(&st.value).0;
                if let Some(f) = self.field_sugar(&st.address) {
                    out.push_str(&format!("{pad}{f} = {val}{semi}\n"));
                } else {
                    out.push_str(&format!(
                        "{pad}write_u{}({}, {val}){semi}\n",
                        st.value.width.bits(),
                        self.child(&st.address, 0),
                    ));
                }
            }
            Stmt::If { cond, then, els } => {
                out.push_str(&format!("{pad}if {} {{\n", self.guard(cond)));
                self.stmts(then, out, depth + 1);
                if els.is_empty() {
                    out.push_str(&format!("{pad}}}\n"));
                } else {
                    out.push_str(&format!("{pad}}} else {{\n"));
                    self.stmts(els, out, depth + 1);
                    out.push_str(&format!("{pad}}}\n"));
                }
            }
            Stmt::Loop(body) => {
                let head = if self.dialect == Dialect::Rust {
                    "loop {"
                } else {
                    "for (;;) {"
                };
                out.push_str(&format!("{pad}{head}\n"));
                self.stmts(body, out, depth + 1);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::While { cond, body } => {
                out.push_str(&format!("{pad}while {} {{\n", self.guard(cond)));
                self.stmts(body, out, depth + 1);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::DoWhile { body, cond } if self.dialect.is_c_family() => {
                out.push_str(&format!("{pad}do {{\n"));
                self.stmts(body, out, depth + 1);
                out.push_str(&format!("{pad}}} while ({});\n", self.cond(cond)));
            }
            Stmt::DoWhile { body, cond } => {
                // Rust/pseudo: no do-while, so a loop with a trailing break.
                let head = if self.dialect == Dialect::Rust {
                    "loop {"
                } else {
                    "for (;;) {"
                };
                out.push_str(&format!("{pad}{head}\n"));
                self.stmts(body, out, depth + 1);
                out.push_str(&format!(
                    "{pad}    if !({}) {{ break; }}\n",
                    self.cond(cond)
                ));
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Return(vals) => {
                let v = vals
                    .first()
                    .map_or_else(|| "0".to_owned(), |(_, e)| self.body(e).0);
                out.push_str(&format!("{pad}return {v}{semi}\n"));
            }
            Stmt::Break => out.push_str(&format!("{pad}break{semi}\n")),
            Stmt::Continue => out.push_str(&format!("{pad}continue{semi}\n")),
            Stmt::Comment(c) => out.push_str(&format!("{pad}// {c}\n")),
            Stmt::Label(n) => out.push_str(&format!("{pad}// label {n:?}\n")),
            Stmt::Goto(n) => out.push_str(&format!("{pad}// goto {n:?}\n")),
            Stmt::IfGoto { cond, target } => {
                out.push_str(&format!(
                    "{pad}if {} {{ /* goto {target:?} */ }}\n",
                    self.guard(cond)
                ));
            }
        }
    }

    fn stmts(&self, list: &[Stmt], out: &mut String, depth: usize) {
        for s in list {
            self.stmt(s, out, depth);
        }
    }
}

/// Role-name recurring XOR-mask constants: a value used as an `^` operand more
/// than once (and wider than a byte, so trivial toggles stay literal) is a
/// decode key. The most-used is `DECODE_KEY`, the next `DECODE_KEY2`, ...
fn classify_constants(
    temps: &[(String, Expr)],
    prologue: &[Stmt],
    body: &[Stmt],
) -> BTreeMap<u64, String> {
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    let mut visit = |e: &Expr| xor_masks(e, &mut counts);
    for (_, def) in temps {
        each_subexpr(def, &mut visit);
    }
    for s in prologue.iter().chain(body) {
        stmt_exprs(s, &mut visit);
    }
    let mut ranked: Vec<(u64, usize)> = counts
        .into_iter()
        .filter(|(v, c)| *v > 0xff && *c >= 2)
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out = BTreeMap::new();
    for (i, (v, _)) in ranked.into_iter().enumerate() {
        out.insert(
            v,
            if i == 0 {
                "DECODE_KEY".to_owned()
            } else {
                format!("DECODE_KEY{}", i + 1)
            },
        );
    }
    out
}

/// `reg + const` as `(reg, const)` -- a struct-field address shape.
fn base_offset(e: &Expr) -> Option<(&str, u64)> {
    if let ExprKind::Binary {
        op: BinaryOp::Add,
        left,
        right,
    } = &e.kind
        && let (ExprKind::Register { name }, ExprKind::Const { value }) = (&left.kind, &right.kind)
    {
        return Some((name, *value));
    }
    None
}

/// A bare argument display name (`a12`): an SSA value safe to re-role as `ctx`.
fn is_bare_arg(n: &str) -> bool {
    n.len() >= 2 && n.starts_with('a') && n[1..].bytes().all(|b| b.is_ascii_digit())
}

/// The argument register that behaves like a struct/context pointer: the base of
/// the most *distinct* `reg + const` field accesses (at least two). Returns its
/// internal name. Faithful to rename -- a structural fact, not a guess about the
/// field's meaning; only a register whose current display is a bare argument is
/// promoted, so nothing already carrying a role is clobbered.
fn infer_struct_base(
    names: &BTreeMap<String, String>,
    temps: &[(String, Expr)],
    prologue: &[Stmt],
    body: &[Stmt],
) -> Option<String> {
    let mut offs: BTreeMap<String, std::collections::BTreeSet<u64>> = BTreeMap::new();
    let mut visit = |e: &Expr| {
        if let Some((r, off)) = base_offset(e) {
            let display = names.get(r).map_or(r, String::as_str);
            if is_bare_arg(display) {
                offs.entry(r.to_owned()).or_default().insert(off);
            }
        }
    };
    for (_, def) in temps {
        each_subexpr(def, &mut visit);
    }
    for s in prologue.iter().chain(body) {
        stmt_exprs(s, &mut visit);
    }
    offs.into_iter()
        .filter(|(_, set)| set.len() >= 2)
        .max_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| b.0.cmp(&a.0)))
        .map(|(r, _)| r)
}

fn width_bytes(w: BitWidth) -> u64 {
    u64::from(w.bits()) / 8
}

/// Apply `f` to every `Store` statement, recursing through compound statements.
fn each_store(list: &[Stmt], f: &mut impl FnMut(&crate::ir::Store)) {
    for s in list {
        match s {
            Stmt::Store(st) => f(st),
            Stmt::If { then, els, .. } => {
                each_store(then, f);
                each_store(els, f);
            }
            Stmt::Loop(b) | Stmt::While { body: b, .. } | Stmt::DoWhile { body: b, .. } => {
                each_store(b, f);
            }
            _ => {}
        }
    }
}

/// A sound `Ctx` struct layout for context base `b`, or `None` to fall back to
/// raw `read_u*`/`write_*`. The layout is the sorted `(offset, width)` of every
/// field, and it is only returned when it is provably safe to model `b` as a
/// struct pointer:
///
/// - **Every** occurrence of `b` is the base of a `b + const` address that is
///   directly a load or store (never bare, never arithmetic, never an intrinsic
///   argument) -- so promoting `b` to `Ctx *` leaves no use that still needs the
///   integer address. Checked by counting: one `b` register per field access.
/// - No offset is accessed at two different widths (one field, one type).
/// - No two fields overlap.
///
/// Any violation returns `None`; the caller then keeps the faithful helpers.
fn ctx_struct_layout(
    b: &str,
    temps: &[(String, Expr)],
    prologue: &[Stmt],
    body: &[Stmt],
) -> Option<Vec<(u64, BitWidth)>> {
    let mut fields: BTreeMap<u64, BitWidth> = BTreeMap::new();
    let mut conflict = false;
    let mut n_ctx = 0usize;
    let mut n_use = 0usize;
    let mut record = |off: u64, w: BitWidth, fields: &mut BTreeMap<u64, BitWidth>| {
        if let Some(prev) = fields.insert(off, w)
            && prev != w
        {
            conflict = true;
        }
    };

    // Walk 1: count `b` registers and collect load fields.
    {
        let mut visit = |e: &Expr| match &e.kind {
            ExprKind::Register { name } if name == b => n_ctx += 1,
            ExprKind::Load { address } => {
                if let Some((r, off)) = base_offset(address)
                    && r == b
                {
                    record(off, e.width, &mut fields);
                    n_use += 1;
                }
            }
            _ => {}
        };
        for (_, def) in temps {
            each_subexpr(def, &mut visit);
        }
        for s in prologue.iter().chain(body) {
            stmt_exprs(s, &mut visit);
        }
    }
    // Walk 2: collect store fields (their `b` register is already in `n_ctx`).
    {
        let mut visit = |st: &crate::ir::Store| {
            if let Some((r, off)) = base_offset(&st.address)
                && r == b
            {
                record(off, st.value.width, &mut fields);
                n_use += 1;
            }
        };
        each_store(prologue, &mut visit);
        each_store(body, &mut visit);
    }

    if conflict || fields.is_empty() || n_ctx != n_use {
        return None;
    }
    // Reject overlapping fields.
    let sorted: Vec<(u64, BitWidth)> = fields.into_iter().collect();
    let mut end = 0u64;
    for (off, w) in &sorted {
        if *off < end {
            return None;
        }
        end = off + width_bytes(*w);
    }
    Some(sorted)
}

/// The synthesized `Ctx` struct declaration for `d`: a packed struct with a
/// `fNN` field at each accessed offset (explicit `_padN` between them so the
/// layout is exact) plus a compile-time offset assertion per field, so a layout
/// mismatch breaks the build rather than mis-mapping silently.
fn emit_ctx_struct(d: Dialect, fields: &[(u64, BitWidth)]) -> String {
    let field_name = |off: u64| format!("f{off:x}");
    let mut members = String::new();
    let mut cursor = 0u64;
    let mut padn = 0;
    for (off, w) in fields {
        if *off > cursor {
            let gap = off - cursor;
            match d {
                Dialect::Rust => members += &format!("    _pad{padn}: [u8; {gap}],\n"),
                _ => members += &format!("    uint8_t _pad{padn}[{gap}];\n"),
            }
            padn += 1;
        }
        match d {
            Dialect::Rust => members += &format!("    {}: {},\n", field_name(*off), uty(*w)),
            _ => members += &format!("    {} {};\n", cty(*w), field_name(*off)),
        }
        cursor = off + width_bytes(*w);
    }

    let mut out = String::new();
    match d {
        Dialect::Rust => {
            out += &format!("#[repr(C, packed)]\nstruct Ctx {{\n{members}}}\n");
            for (off, _) in fields {
                out += &format!(
                    "const _: () = assert!(core::mem::offset_of!(Ctx, {}) == 0x{off:x});\n",
                    field_name(*off)
                );
            }
        }
        _ => {
            let open = if d == Dialect::Cpp {
                "struct Ctx {"
            } else {
                "typedef struct {"
            };
            let close = if d == Dialect::Cpp { "};" } else { "} Ctx;" };
            out += &format!("#pragma pack(push, 1)\n{open}\n{members}{close}\n#pragma pack(pop)\n");
            // C23/C++ spell it `static_assert`; C99/C11 use the `_Static_assert`
            // keyword (no <assert.h> needed). `offsetof` comes from the prelude's
            // <stddef.h>/<cstddef>.
            let kw = if d == Dialect::C99 {
                "_Static_assert"
            } else {
                "static_assert"
            };
            for (off, _) in fields {
                out += &format!(
                    "{kw}(offsetof(Ctx, {}) == 0x{off:x}, \"ctx layout\");\n",
                    field_name(*off)
                );
            }
        }
    }
    out
}

fn xor_masks(e: &Expr, counts: &mut BTreeMap<u64, usize>) {
    if let ExprKind::Binary {
        op: BinaryOp::Xor,
        left,
        right,
    } = &e.kind
    {
        for side in [left, right] {
            if let ExprKind::Const { value } = side.kind {
                *counts.entry(value).or_default() += 1;
            }
        }
    }
}

fn each_subexpr(e: &Expr, f: &mut impl FnMut(&Expr)) {
    f(e);
    match &e.kind {
        ExprKind::Unary { value, .. }
        | ExprKind::Load { address: value }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => each_subexpr(value, f),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => {
            each_subexpr(left, f);
            each_subexpr(right, f);
        }
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().for_each(|a| each_subexpr(a, f)),
        ExprKind::Const { .. } | ExprKind::Register { .. } => {}
    }
}

fn stmt_exprs(s: &Stmt, f: &mut impl FnMut(&Expr)) {
    match s {
        Stmt::Assign(_, e) => each_subexpr(e, f),
        Stmt::Store(st) => {
            each_subexpr(&st.address, f);
            each_subexpr(&st.value, f);
        }
        Stmt::If { cond, then, els } => {
            cond.exprs().into_iter().for_each(|e| each_subexpr(e, f));
            then.iter().chain(els).for_each(|s| stmt_exprs(s, f));
        }
        Stmt::Loop(b) => b.iter().for_each(|s| stmt_exprs(s, f)),
        Stmt::While { cond, body: b } | Stmt::DoWhile { body: b, cond } => {
            cond.exprs().into_iter().for_each(|e| each_subexpr(e, f));
            b.iter().for_each(|s| stmt_exprs(s, f));
        }
        Stmt::Return(vals) => vals.iter().for_each(|(_, e)| each_subexpr(e, f)),
        Stmt::IfGoto { cond, .. } => cond.exprs().into_iter().for_each(|e| each_subexpr(e, f)),
        _ => {}
    }
}

/// The top-level expressions a leaf statement carries (`Assign`/`Store`/
/// `Return`). Compound statements carry none here -- their inner leaves are
/// reached by recursion, each getting its own per-statement CSE.
fn stmt_top_exprs<'a>(s: &'a Stmt, out: &mut Vec<&'a Expr>) {
    match s {
        Stmt::Assign(_, e) => out.push(e),
        Stmt::Store(st) => {
            out.push(&st.address);
            out.push(&st.value);
        }
        Stmt::Return(vals) => vals.iter().for_each(|(_, e)| out.push(e)),
        _ => {}
    }
}

/// `true` when `needle` occurs strictly below `hay` (not as `hay` itself).
fn proper_contains(hay: &Expr, needle: &Expr) -> bool {
    if hay == needle {
        return false;
    }
    let mut found = false;
    each_subexpr(hay, &mut |se| {
        if se == needle {
            found = true;
        }
    });
    found
}

/// Worth binding to a name: not a bare register or constant, and more than a
/// couple of nodes -- a load or an arithmetic tree, not a lone cast.
fn cse_worthy(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => false,
        _ => expr_size(e) >= 3,
    }
}

fn expr_size(e: &Expr) -> usize {
    let mut n = 0usize;
    each_subexpr(e, &mut |_| n += 1);
    n
}

fn collect_vm_flags<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let ExprKind::Intrinsic { name, .. } = &e.kind
        && name.starts_with("vm_flags_")
    {
        out.push(e);
    }
    match &e.kind {
        ExprKind::Unary { value, .. }
        | ExprKind::Load { address: value }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => collect_vm_flags(value, out),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => {
            collect_vm_flags(left, out);
            collect_vm_flags(right, out);
        }
        ExprKind::Intrinsic { arguments, .. } => {
            for a in arguments {
                collect_vm_flags(a, out);
            }
        }
        ExprKind::Const { .. } | ExprKind::Register { .. } => {}
    }
}

/// Rebuild `e` with every occurrence of `target` replaced by `repl` (same width,
/// so no constructor fails).
fn replace_subtree(e: &Expr, target: &Expr, repl: &Expr) -> Expr {
    if e == target {
        return repl.clone();
    }
    let rc = |c: &Expr| replace_subtree(c, target, repl);
    match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Unary { op, value } => Expr::unary(*op, rc(value)),
        ExprKind::Binary { op, left, right } => {
            Expr::binary(*op, rc(left), rc(right)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Load { address } => Expr::load(e.width, rc(address)),
        ExprKind::Intrinsic { name, arguments } => {
            Expr::intrinsic(e.width, name.clone(), arguments.iter().map(rc).collect())
        }
        ExprKind::Extract { value, lsb } => {
            Expr::extract(e.width, *lsb, rc(value)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Concat { high, low } => {
            Expr::concat(rc(high), rc(low)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Resize { op, value } => {
            Expr::resize(*op, e.width, rc(value)).unwrap_or_else(|_| e.clone())
        }
    }
}

fn cst(e: &Expr) -> Option<u64> {
    match e.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// The comparison whose indicator set over `0..n` is exactly `hits` (sorted),
/// or `None` when the set is not a single clean comparison.
fn recognize_cmp(hits: &[u64], n: u64) -> Option<(&'static str, u64)> {
    if hits.is_empty() || hits.len() as u64 == n {
        return None;
    }
    if hits.len() == 1 {
        return Some(("==", hits[0]));
    }
    if hits.len() as u64 == n - 1 {
        let present: std::collections::HashSet<u64> = hits.iter().copied().collect();
        return Some(("!=", (0..n).find(|v| !present.contains(v))?));
    }
    let (first, last) = (hits[0], *hits.last()?);
    if hits.windows(2).all(|w| w[1] == w[0] + 1) {
        if first == 0 {
            return Some(("<=", last));
        }
        if last == n - 1 {
            return Some((">=", first));
        }
    }
    None
}

fn intrinsic(name: &str) -> String {
    let base = match name.rsplit_once('_') {
        Some((head, tail)) if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) => head,
        _ => name,
    };
    match base {
        "read_timestamp_counter" => "rdtsc".to_owned(),
        "x86_unsigned_multiply_w32" => "umul32".to_owned(),
        "x86_unsigned_divide_w32" => "udiv32".to_owned(),
        other => other.to_owned(),
    }
}

/// The shared Rust prelude: typedef-free helpers the body uses. Emitted once.
/// Kept as the Rust default; [`preamble`] selects it per dialect.
pub fn prelude() -> &'static str {
    "// ---- prelude (generated) ----\n\
     const INT3: u8 = 0xCC;\n\
     #[inline] unsafe fn read_u8(a: u64)  -> u8  { *(a as *const u8) }\n\
     #[inline] unsafe fn read_u16(a: u64) -> u16 { *(a as *const u16) }\n\
     #[inline] unsafe fn read_u32(a: u64) -> u32 { *(a as *const u32) }\n\
     #[inline] unsafe fn read_u64(a: u64) -> u64 { *(a as *const u64) }\n\
     #[inline] unsafe fn write_u8(a: u64, v: u8)   { *(a as *mut u8)  = v; }\n\
     #[inline] unsafe fn write_u32(a: u64, v: u32) { *(a as *mut u32) = v; }\n\
     #[inline] unsafe fn write_u64(a: u64, v: u64) { *(a as *mut u64) = v; }\n"
}

/// The generated helper prelude for `d`: `read_*`/`write_*`, `INT3`, rotates.
#[must_use]
pub fn preamble(d: Dialect) -> String {
    match d {
        Dialect::Rust => prelude().to_owned(),
        Dialect::Pseudo => "// pseudo-code: widths and widening casts elided.\n\
             // read_uN(a) = load N bits at a; rotlN/rotrN = rotate; INT3 = 0xCC.\n"
            .to_owned(),
        _ => {
            let header = if d == Dialect::Cpp {
                "#include <cstdint>\n#include <cstddef>"
            } else {
                "#include <stdint.h>\n#include <stddef.h>"
            };
            let mut s = format!("// ---- prelude (generated) ----\n{header}\n#define INT3 0xCC\n");
            for b in [8u32, 16, 32, 64] {
                s += &format!(
                    "static inline uint{b}_t read_u{b}(uint64_t a){{ return *(uint{b}_t *)(uintptr_t)a; }}\n"
                );
            }
            for b in [8u32, 16, 32, 64] {
                s += &format!(
                    "static inline void write_u{b}(uint64_t a, uint{b}_t v){{ *(uint{b}_t *)(uintptr_t)a = v; }}\n"
                );
            }
            for b in [8u32, 16, 32, 64] {
                let m = b - 1;
                s += &format!(
                    "static inline uint{b}_t rotl{b}(uint{b}_t x, uint{b}_t n){{ return (uint{b}_t)((x << (n & {m})) | (x >> (({b} - n) & {m}))); }}\n"
                );
                s += &format!(
                    "static inline uint{b}_t rotr{b}(uint{b}_t x, uint{b}_t n){{ return (uint{b}_t)((x >> (n & {m})) | (x << (({b} - n) & {m}))); }}\n"
                );
            }
            s
        }
    }
}

/// Short width type (`u32`) to the dialect's spelling for a signature/return.
fn sig_ty(d: Dialect, t: &str) -> String {
    if d == Dialect::Pseudo {
        return String::new();
    }
    if !d.is_c_family() {
        return t.to_owned(); // Rust keeps u8/u64.
    }
    match t {
        "u8" => "uint8_t",
        "u16" => "uint16_t",
        "u32" => "uint32_t",
        "u64" => "uint64_t",
        other => other, // void, etc.
    }
    .to_owned()
}

/// Render one recovered function in the dialect selected by [`dialect`] (Rust
/// when unset), reusing the structures the C path already built.
#[allow(clippy::too_many_arguments)]
pub fn render_function(
    name: &str,
    c_sig: &[String],
    ret_ty: &str,
    decl: &BTreeMap<String, BitWidth>,
    temps_order: &[(String, Expr)],
    prologue: &[Stmt],
    body: &[Stmt],
    names: &BTreeMap<String, String>,
    temps_map: &HashMap<Expr, String>,
    symbols: &Symbols,
) -> String {
    let d = Dialect::parse(dialect().as_deref().unwrap_or("")).unwrap_or(Dialect::Rust);
    let consts = classify_constants(temps_order, prologue, body);
    let image_base = symbols.image_base();

    // Stage B: the argument that is a struct/context pointer becomes `ctx`. Only
    // in the readable path -- the C cross-check already ran on the original names.
    let mut names_owned = names.clone();
    // Display name the signature shows for the context arg, if any (`a14` -> `ctx`).
    let mut ctx_sig: Option<(String, String)> = None;
    // A sound `Ctx` struct layout for the compilable dialects (pseudo sugars
    // fields without a declared type, so it needs none).
    let mut ctx_fields: Option<Vec<(u64, BitWidth)>> = None;
    if let Some(b) = infer_struct_base(names, temps_order, prologue, body) {
        let old = names.get(&b).cloned().unwrap_or_else(|| b.clone());
        if d != Dialect::Pseudo {
            ctx_fields = ctx_struct_layout(&b, temps_order, prologue, body);
        }
        names_owned.insert(b, "ctx".to_owned());
        ctx_sig = Some((old, "ctx".to_owned()));
    }
    let names = &names_owned;
    let ctx_struct = ctx_fields.is_some();
    let r = Renderer {
        dialect: d,
        names,
        image_base,
        temps: temps_map,
        consts: &consts,
        ctx_struct,
    };

    // Signature: `u64 a1` -> `a1: u64` (Rust) / `uint64_t a1` (C) / `a1` (pseudo),
    // with the context pointer shown under its inferred name.
    let args: Vec<String> = c_sig
        .iter()
        .map(|a| match a.rsplit_once(' ') {
            Some((ty, raw)) => {
                let nm = match &ctx_sig {
                    Some((old, new)) if old == raw => new.as_str(),
                    _ => raw,
                };
                if nm == "ctx" && ctx_struct {
                    // Modeled as a struct pointer; field accesses carry the offsets.
                    match d {
                        Dialect::Rust => "ctx: *mut Ctx".to_owned(),
                        _ => "Ctx *ctx".to_owned(),
                    }
                } else {
                    match d {
                        Dialect::Rust => format!("{nm}: {ty}"),
                        Dialect::Pseudo => nm.to_owned(),
                        _ => format!("{} {nm}", sig_ty(d, ty)),
                    }
                }
            }
            None => a.clone(),
        })
        .collect();

    let mut out = String::new();

    // Per-function constant legend, with real values.
    let decl_const = |nm: &str, v: u64| -> String {
        match d {
            Dialect::Rust => format!("const {nm}: u64 = 0x{v:x};\n"),
            Dialect::C99 => format!("#define {nm} 0x{v:x}ULL\n"),
            Dialect::C23 | Dialect::Cpp => format!("constexpr uint64_t {nm} = 0x{v:x};\n"),
            Dialect::Pseudo => format!("// {nm} = 0x{v:x}\n"),
        }
    };
    if image_base != 0 {
        out.push_str(&decl_const("IMAGE_BASE", image_base));
    }
    for (v, nm) in &consts {
        out.push_str(&decl_const(nm, *v));
    }
    if image_base != 0 || !consts.is_empty() {
        out.push('\n');
    }

    // Synthesized context struct (compilable dialects, when a sound layout exists).
    if let Some(fields) = &ctx_fields {
        out.push_str(&emit_ctx_struct(d, fields));
        out.push('\n');
    }

    // Header.
    match d {
        Dialect::Rust => {
            let ret = if ret_ty == "void" {
                String::new()
            } else {
                format!(" -> {ret_ty}")
            };
            out.push_str(&format!("unsafe fn {name}({}){ret} {{\n", args.join(", ")));
        }
        Dialect::Pseudo => out.push_str(&format!("{name}({}):\n", args.join(", "))),
        _ => out.push_str(&format!(
            "{} {name}({}) {{\n",
            sig_ty(d, ret_ty),
            args.join(", ")
        )),
    }

    // Mutable locals the body assigns across branches.
    for (v, w) in decl {
        let nm = names.get(v).unwrap_or(v);
        let line = match d {
            Dialect::Rust => format!("    let mut {nm}: {} = 0;\n", uty(*w)),
            Dialect::Pseudo => format!("    {nm} = 0\n"),
            _ => format!("    {} {nm} = 0;\n", cty(*w)),
        };
        out.push_str(&line);
    }
    if !decl.is_empty() {
        out.push('\n');
    }

    // Shared subtrees, smaller first (each references only names above it).
    for (nm, def) in temps_order {
        let rhs = r.define(def);
        let line = match d {
            Dialect::Rust => format!("    let {nm} = {rhs};\n"),
            Dialect::Pseudo => format!("    {nm} = {rhs}\n"),
            _ if d.auto_temps() => format!("    auto {nm} = {rhs};\n"),
            _ => format!("    {} {nm} = {rhs};\n", cty(def.width)),
        };
        out.push_str(&line);
    }
    if !temps_order.is_empty() {
        out.push('\n');
    }

    r.stmts(prologue, &mut out, 1);
    r.stmts(body, &mut out, 1);
    out.push_str(if d == Dialect::Pseudo { "" } else { "}\n" });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::{BinaryOp, Expr, UnaryOp};

    fn render(e: &Expr, image_base: u64) -> String {
        render_as(Dialect::Rust, e, image_base)
    }
    fn render_as(dialect: Dialect, e: &Expr, image_base: u64) -> String {
        let names = BTreeMap::new();
        let temps = HashMap::new();
        let consts = BTreeMap::new();
        Renderer {
            dialect,
            names: &names,
            image_base,
            temps: &temps,
            consts: &consts,
            ctx_struct: false,
        }
        .body(e)
        .0
    }
    fn reg(n: &str) -> Expr {
        Expr::register(BitWidth::W64, n)
    }

    #[test]
    fn arithmetic_is_explicitly_wrapping() {
        assert_eq!(
            render(&Expr::binary(BinaryOp::Add, reg("x"), reg("y")).unwrap(), 0),
            "x.wrapping_add(y)"
        );
        assert_eq!(
            render(&Expr::binary(BinaryOp::Mul, reg("x"), reg("y")).unwrap(), 0),
            "x.wrapping_mul(y)"
        );
        // Bitwise ops never overflow -> plain infix, not a wrapping method.
        assert_eq!(
            render(&Expr::binary(BinaryOp::Xor, reg("x"), reg("y")).unwrap(), 0),
            "x ^ y"
        );
    }

    #[test]
    fn recognize_cmp_characterises_indicator_sets() {
        // The flag-idiom collapse only fires when the exhaustive table is a
        // clean indicator; recognize_cmp names which comparison.
        assert_eq!(recognize_cmp(&[0xCC], 256), Some(("==", 0xCC)));
        let all_but_one: Vec<u64> = (0..256).filter(|&v| v != 7).collect();
        assert_eq!(recognize_cmp(&all_but_one, 256), Some(("!=", 7)));
        let low: Vec<u64> = (0..10).collect();
        assert_eq!(recognize_cmp(&low, 256), Some(("<=", 9)));
        let high: Vec<u64> = (0x80..256).collect();
        assert_eq!(recognize_cmp(&high, 256), Some((">=", 0x80)));
        // Not a single clean comparison -> no collapse.
        assert_eq!(recognize_cmp(&[1, 3, 5], 256), None);
        assert_eq!(recognize_cmp(&[], 256), None);
    }

    #[test]
    fn not_loads_and_named_constants() {
        assert_eq!(render(&Expr::unary(UnaryOp::Not, reg("x")), 0), "!x");
        assert_eq!(
            render(&Expr::load(BitWidth::W32, reg("a")), 0),
            "read_u32(a)"
        );
        assert_eq!(render(&Expr::constant(BitWidth::W8, 0xCC), 0), "INT3");
        assert_eq!(
            render(&Expr::constant(BitWidth::W64, 0x1_4000_0000), 0x1_4000_0000),
            "IMAGE_BASE"
        );
    }

    #[test]
    fn dialects_render_the_same_tree_differently() {
        // Unsigned add: Rust needs a wrapping verb; C/pseudo wrap by definition.
        let add = Expr::binary(BinaryOp::Add, reg("a"), reg("b")).unwrap();
        assert_eq!(render_as(Dialect::Rust, &add, 0), "a.wrapping_add(b)");
        assert_eq!(render_as(Dialect::C99, &add, 0), "a + b");
        assert_eq!(render_as(Dialect::Pseudo, &add, 0), "a + b");

        // Bitwise-not: Rust `!`, the C family and pseudo `~`.
        let not = Expr::unary(UnaryOp::Not, reg("x"));
        assert_eq!(render_as(Dialect::Rust, &not, 0), "!x");
        assert_eq!(render_as(Dialect::Cpp, &not, 0), "~x");

        // A widening truncate: explicit cast in Rust/C, elided in pseudo.
        let trunc = Expr::resize(ResizeOp::Truncate, BitWidth::W32, reg("x")).unwrap();
        assert_eq!(render_as(Dialect::Rust, &trunc, 0), "x as u32");
        assert_eq!(render_as(Dialect::C99, &trunc, 0), "(uint32_t)x");
        assert_eq!(render_as(Dialect::Pseudo, &trunc, 0), "x");
    }

    #[test]
    fn struct_base_is_the_argument_with_the_most_fields() {
        let field = |base: &str, off: u64, w: BitWidth| {
            Expr::load(
                w,
                Expr::binary(BinaryOp::Add, reg(base), Expr::constant(BitWidth::W64, off)).unwrap(),
            )
        };
        // a14 is the base of two distinct offsets; a2 of only one.
        let temps = vec![
            ("t0".to_owned(), field("a14", 0x98, BitWidth::W64)),
            ("t1".to_owned(), field("a14", 0xa0, BitWidth::W8)),
            ("t2".to_owned(), field("a2", 0x8, BitWidth::W64)),
        ];
        let names = BTreeMap::new();
        assert_eq!(
            infer_struct_base(&names, &temps, &[], &[]).as_deref(),
            Some("a14")
        );
        // A single field access is not enough to call something a struct pointer.
        let one = vec![("t0".to_owned(), field("a2", 0x8, BitWidth::W64))];
        assert_eq!(infer_struct_base(&names, &one, &[], &[]), None);
    }

    fn render_stmt(dialect: Dialect, names: &BTreeMap<String, String>, s: &Stmt) -> String {
        let temps = HashMap::new();
        let consts = BTreeMap::new();
        let mut out = String::new();
        Renderer {
            dialect,
            names,
            image_base: 0,
            temps: &temps,
            consts: &consts,
            ctx_struct: false,
        }
        .stmt(s, &mut out, 1);
        out
    }
    fn ctx_field(off: u64, w: BitWidth) -> Expr {
        Expr::load(
            w,
            Expr::binary(
                BinaryOp::Add,
                reg("a14"),
                Expr::constant(BitWidth::W64, off),
            )
            .unwrap(),
        )
    }
    fn ctx_names() -> BTreeMap<String, String> {
        let mut names = BTreeMap::new();
        names.insert("a14".to_owned(), "ctx".to_owned());
        names
    }

    #[test]
    fn per_statement_cse_factors_a_repeated_load() {
        // r = read_u64(ctx+0x98) + read_u64(ctx+0x98)
        let fld = ctx_field(0x98, BitWidth::W64);
        let val = Expr::binary(BinaryOp::Add, fld.clone(), fld).unwrap();
        let out = render_stmt(
            Dialect::Rust,
            &ctx_names(),
            &Stmt::Assign("r".to_owned(), val),
        );
        // The load is evaluated once, in a `let v98`, and used twice by name.
        assert_eq!(
            out.matches("read_u64(").count(),
            1,
            "load evaluated once:\n{out}"
        );
        assert!(
            out.contains("let v98 = read_u64("),
            "binding present:\n{out}"
        );
        assert_eq!(out.matches("v98").count(), 3, "one def + two uses:\n{out}");
    }

    #[test]
    fn cse_never_hoists_across_a_store() {
        // A store, then a read of the same field in the next statement. The read
        // appears once in each statement -- no repeat WITHIN either -- so nothing
        // is factored across the store between them.
        let fld = ctx_field(0x98, BitWidth::W64);
        let store = Stmt::Store(crate::ir::Store {
            address: Expr::binary(
                BinaryOp::Add,
                reg("a14"),
                Expr::constant(BitWidth::W64, 0xa0),
            )
            .unwrap(),
            value: Expr::register(BitWidth::W8, "a0"),
        });
        let read_then = Stmt::Assign("r".to_owned(), fld);
        let names = ctx_names();
        let temps = HashMap::new();
        let consts = BTreeMap::new();
        let mut out = String::new();
        Renderer {
            dialect: Dialect::Rust,
            names: &names,
            image_base: 0,
            temps: &temps,
            consts: &consts,
            ctx_struct: false,
        }
        .stmts(&[store, read_then], &mut out, 1);
        assert!(
            !out.contains("let v98"),
            "no binding across statements:\n{out}"
        );
        assert_eq!(
            out.matches("read_u64(").count(),
            1,
            "the one read stays inline:\n{out}"
        );
    }

    // A two-valued indicator `(x == 0xCC)` scaled by 2, built from the VM's own
    // flag-word intrinsic so `flag_predicate` recognises it exhaustively.
    fn flag_scaled(scaled: bool) -> Expr {
        // The folder evaluates `vm_flags_*` only at W64 (see expr::simplify_intrinsic).
        let sub = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_sub_w8".to_owned(),
            vec![reg_w8("b"), Expr::constant(BitWidth::W8, 0xCC)],
        );
        // ZF (bit 6) is set iff b == 0xCC.
        let zf = Expr::binary(
            BinaryOp::And,
            Expr::binary(BinaryOp::Shr, sub, Expr::constant(BitWidth::W64, 6)).unwrap(),
            Expr::constant(BitWidth::W64, 1),
        )
        .unwrap();
        if scaled {
            Expr::binary(BinaryOp::Mul, zf, Expr::constant(BitWidth::W64, 2)).unwrap()
        } else {
            zf
        }
    }
    fn reg_w8(n: &str) -> Expr {
        Expr::register(BitWidth::W8, n)
    }

    #[test]
    fn scaled_indicator_renders_as_a_conditional() {
        let two = flag_scaled(true);
        assert_eq!(
            render_as(Dialect::Rust, &two, 0),
            "if b == INT3 { 2 } else { 0 }"
        );
        assert_eq!(render_as(Dialect::C99, &two, 0), "b == INT3 ? 2 : 0");
        assert_eq!(render_as(Dialect::Pseudo, &two, 0), "b == INT3 ? 2 : 0");
        // A plain 0/1 indicator (no scale) stays a widened boolean, not a ternary.
        let one = flag_scaled(false);
        assert_eq!(render_as(Dialect::Rust, &one, 0), "(b == INT3) as u64");
    }

    fn render_ctx(dialect: Dialect, ctx_struct: bool, e: &Expr) -> String {
        let names = ctx_names();
        let temps = HashMap::new();
        let consts = BTreeMap::new();
        Renderer {
            dialect,
            names: &names,
            image_base: 0,
            temps: &temps,
            consts: &consts,
            ctx_struct,
        }
        .body(e)
        .0
    }

    #[test]
    fn ctx_access_becomes_a_struct_field() {
        let f98 = ctx_field(0x98, BitWidth::W64);
        assert_eq!(render_ctx(Dialect::C99, true, &f98), "ctx->f98");
        assert_eq!(render_ctx(Dialect::Rust, true, &f98), "(*ctx).f98");
        assert_eq!(
            render_ctx(Dialect::Cpp, true, &ctx_field(0xa0, BitWidth::W8)),
            "ctx->fa0"
        );
        // Without a synthesized struct, a compilable dialect keeps the helper.
        assert_eq!(
            render_ctx(Dialect::C99, false, &f98),
            "read_u64(ctx + 0x98)"
        );
    }

    #[test]
    fn non_ctx_pointer_load_stays_a_helper() {
        // A loaded value used as an address is not a ctx field, even with a struct.
        let load = Expr::load(BitWidth::W8, reg("v98"));
        assert_eq!(render_ctx(Dialect::C99, true, &load), "read_u8(v98)");
    }

    #[test]
    fn ctx_struct_decl_pins_offsets_with_assertions() {
        let fields = vec![(0x98u64, BitWidth::W64), (0xa0u64, BitWidth::W8)];
        let c = emit_ctx_struct(Dialect::C99, &fields);
        assert!(c.contains("uint8_t _pad0[152];"), "pad to 0x98:\n{c}"); // 0x98 == 152
        assert!(c.contains("uint64_t f98;"), "{c}");
        assert!(c.contains("uint8_t fa0;"), "{c}");
        assert!(
            c.contains("_Static_assert(offsetof(Ctx, f98) == 0x98"),
            "offset assert:\n{c}"
        );
        assert!(
            c.contains("_Static_assert(offsetof(Ctx, fa0) == 0xa0"),
            "{c}"
        );
        let r = emit_ctx_struct(Dialect::Rust, &fields);
        assert!(r.contains("#[repr(C, packed)]"), "{r}");
        assert!(r.contains("offset_of!(Ctx, f98) == 0x98"), "{r}");
    }

    #[test]
    fn layout_is_sound_or_absent() {
        // Two distinct, non-overlapping fields through a14 -> a sound layout.
        let temps = vec![("v98".to_owned(), ctx_field(0x98, BitWidth::W64))];
        let store = Stmt::Store(crate::ir::Store {
            address: Expr::binary(
                BinaryOp::Add,
                reg("a14"),
                Expr::constant(BitWidth::W64, 0xa0),
            )
            .unwrap(),
            value: Expr::register(BitWidth::W8, "a0"),
        });
        assert_eq!(
            ctx_struct_layout("a14", &temps, &[], &[store]),
            Some(vec![(0x98, BitWidth::W64), (0xa0, BitWidth::W8)])
        );
        // A bare deref of a14 is not a field access -> fall back (no struct).
        let bare = Stmt::Assign("r".to_owned(), Expr::load(BitWidth::W8, reg("a14")));
        assert_eq!(ctx_struct_layout("a14", &temps, &[], &[bare]), None);
        // A W64 at 0x98 covers 0x98..0x9f, so a W32 at 0x9c overlaps -> fall back.
        let overlap = vec![
            ("v98".to_owned(), ctx_field(0x98, BitWidth::W64)),
            ("v9c".to_owned(), ctx_field(0x9c, BitWidth::W32)),
        ];
        assert_eq!(ctx_struct_layout("a14", &overlap, &[], &[]), None);
    }
}
