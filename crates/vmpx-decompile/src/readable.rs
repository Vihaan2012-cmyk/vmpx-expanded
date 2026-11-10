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

pub struct RustRenderer<'a> {
    pub names: &'a BTreeMap<String, String>,
    pub image_base: u64,
    pub temps: &'a HashMap<Expr, String>,
}

impl RustRenderer<'_> {
    fn name(&self, n: &str) -> String {
        self.names.get(n).cloned().unwrap_or_else(|| n.to_owned())
    }

    fn lit(&self, w: BitWidth, v: u64) -> String {
        if let Some(n) = const_name(w, v, self.image_base) {
            return n.to_owned();
        }
        if v <= 9 { v.to_string() } else { format!("0x{v:x}") }
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
    fn flag_predicate(&self, e: &Expr) -> Option<String> {
        // The one `vm_flags_*` intrinsic in `e` (bail on zero or several), and
        // its single non-constant operand `x` -- the value the predicate tests.
        let mut flags = Vec::new();
        collect_vm_flags(e, &mut flags);
        let [intrin] = flags[..] else { return None };
        let ExprKind::Intrinsic { arguments, .. } = &intrin.kind else { return None };
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
        let mut s = format!("({xr} {op} {cname}) as {}", uty(out_w));
        if scale != 1 {
            s = format!("(({s}).wrapping_mul({scale}))");
        }
        Some(s)
    }

    fn body(&self, e: &Expr) -> (String, u8) {
        if let Some(pred) = self.flag_predicate(e) {
            return (pred, ATOM);
        }
        match &e.kind {
            ExprKind::Const { value } => (self.lit(e.width, *value), ATOM),
            ExprKind::Register { name } => (self.name(name), ATOM),
            ExprKind::Unary { op, value } => match op {
                UnaryOp::Not => (format!("!{}", self.child(value, 90)), 90),
                UnaryOp::Neg => (format!("{}.wrapping_neg()", self.child(value, ATOM)), ATOM),
            },
            ExprKind::Binary { op, left, right } => self.binary(e.width, *op, left, right),
            ExprKind::Load { address } => {
                (format!("read_u{}({})", e.width.bits(), self.child(address, 0)), ATOM)
            }
            ExprKind::Intrinsic { name, arguments } => {
                let args: Vec<String> = arguments.iter().map(|a| self.child(a, 0)).collect();
                (format!("{}({})", intrinsic(name), args.join(", ")), ATOM)
            }
            ExprKind::Extract { value, lsb } => {
                let v = self.child(value, 70);
                let shifted = if *lsb == 0 { v } else { format!("({v} >> {lsb})") };
                (format!("{shifted} as {}", uty(e.width)), 80)
            }
            ExprKind::Resize { op, value } => {
                let v = self.child(value, 80);
                match op {
                    ResizeOp::Truncate | ResizeOp::ZeroExtend => {
                        (format!("{v} as {}", uty(e.width)), 80)
                    }
                    ResizeOp::SignExtend => (
                        format!("{v} as {} as {} as {}", ity(value.width), ity(e.width), uty(e.width)),
                        80,
                    ),
                }
            }
            ExprKind::Concat { high, low } => {
                let n = uty(e.width);
                let h = self.child(high, ATOM);
                let l = self.child(low, 80);
                (
                    format!("(({h} as {n}) << {}) | ({l} as {n})", low.width.bits()),
                    40,
                )
            }
        }
    }

    fn binary(&self, w: BitWidth, op: BinaryOp, l: &Expr, r: &Expr) -> (String, u8) {
        let as_w = |s: String, need_as: bool| if need_as { format!("{s} as {}", uty(w)) } else { s };
        // Arithmetic the VM relies on to wrap -> explicit wrapping methods, so
        // the output is faithful in both debug and release builds.
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
            // Bitwise ops and logical shift never overflow -> infix.
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
                let f = if op == BinaryOp::Rol { "rotate_left" } else { "rotate_right" };
                let a = self.child(l, ATOM);
                let b = self.child(r, 0);
                (format!("{a}.{f}({b} as u32)"), ATOM)
            }
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

    fn stmt(&self, s: &Stmt, out: &mut String, depth: usize) {
        let pad = "    ".repeat(depth);
        match s {
            Stmt::Assign(n, e) => {
                out.push_str(&format!("{pad}{} = {};\n", self.name(n), self.body(e).0));
            }
            Stmt::Store(st) => {
                out.push_str(&format!(
                    "{pad}write_u{}({}, {});\n",
                    st.value.width.bits(),
                    self.child(&st.address, 0),
                    self.body(&st.value).0
                ));
            }
            Stmt::If { cond, then, els } => {
                out.push_str(&format!("{pad}if {} {{\n", self.cond(cond)));
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
                out.push_str(&format!("{pad}loop {{\n"));
                self.stmts(body, out, depth + 1);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::While { cond, body } => {
                out.push_str(&format!("{pad}while {} {{\n", self.cond(cond)));
                self.stmts(body, out, depth + 1);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::DoWhile { body, cond } => {
                out.push_str(&format!("{pad}loop {{\n"));
                self.stmts(body, out, depth + 1);
                out.push_str(&format!("{pad}    if !({}) {{ break; }}\n", self.cond(cond)));
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Return(vals) => {
                let v = vals.first().map_or_else(|| "0".to_owned(), |(_, e)| self.body(e).0);
                out.push_str(&format!("{pad}return {v};\n"));
            }
            Stmt::Break => out.push_str(&format!("{pad}break;\n")),
            Stmt::Continue => out.push_str(&format!("{pad}continue;\n")),
            Stmt::Comment(c) => out.push_str(&format!("{pad}// {c}\n")),
            Stmt::Label(n) => out.push_str(&format!("{pad}// label {n:?}\n")),
            Stmt::Goto(n) => out.push_str(&format!("{pad}// goto {n:?}\n")),
            Stmt::IfGoto { cond, target } => {
                out.push_str(&format!("{pad}if {} {{ /* goto {target:?} */ }}\n", self.cond(cond)));
            }
        }
    }

    fn stmts(&self, list: &[Stmt], out: &mut String, depth: usize) {
        for s in list {
            self.stmt(s, out, depth);
        }
    }
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
        ExprKind::Binary { left, right, .. } | ExprKind::Concat { high: left, low: right } => {
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

/// The shared prelude: typedef-free Rust helpers the body uses. Emitted once.
pub fn prelude() -> &'static str {
    "// ---- prelude (generated) ----\n\
     const IMAGE_BASE: u64 = 0;  // set per image\n\
     const INT3: u8 = 0xCC;\n\
     #[inline] unsafe fn read_u8(a: u64)  -> u8  { *(a as *const u8) }\n\
     #[inline] unsafe fn read_u16(a: u64) -> u16 { *(a as *const u16) }\n\
     #[inline] unsafe fn read_u32(a: u64) -> u32 { *(a as *const u32) }\n\
     #[inline] unsafe fn read_u64(a: u64) -> u64 { *(a as *const u64) }\n\
     #[inline] unsafe fn write_u8(a: u64, v: u8)   { *(a as *mut u8)  = v; }\n\
     #[inline] unsafe fn write_u32(a: u64, v: u32) { *(a as *mut u32) = v; }\n\
     #[inline] unsafe fn write_u64(a: u64, v: u64) { *(a as *mut u64) = v; }\n"
}

/// Render one function as Rust, reusing the structures the C path already built.
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
    let r = RustRenderer { names, image_base: symbols.image_base(), temps: temps_map };
    // `u64 a1` -> `a1: u64`.
    let args: Vec<String> = c_sig
        .iter()
        .map(|a| match a.rsplit_once(' ') {
            Some((ty, nm)) => format!("{nm}: {ty}"),
            None => a.clone(),
        })
        .collect();
    let ret = if ret_ty == "void" { String::new() } else { format!(" -> {ret_ty}") };

    let mut out = String::new();
    out.push_str(&format!("unsafe fn {name}({}){ret} {{\n", args.join(", ")));
    // Mutable locals the body assigns across branches.
    for (v, w) in decl {
        out.push_str(&format!("    let mut {}: {} = 0;\n", names.get(v).unwrap_or(v), uty(*w)));
    }
    if !decl.is_empty() {
        out.push('\n');
    }
    // Shared subtrees, smaller first (each references only names above it).
    for (nm, def) in temps_order {
        out.push_str(&format!("    let {nm} = {};\n", r.define(def)));
    }
    if !temps_order.is_empty() {
        out.push('\n');
    }
    r.stmts(prologue, &mut out, 1);
    r.stmts(body, &mut out, 1);
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::{BinaryOp, Expr, UnaryOp};

    fn render(e: &Expr, image_base: u64) -> String {
        let names = BTreeMap::new();
        let temps = HashMap::new();
        RustRenderer { names: &names, image_base, temps: &temps }.body(e).0
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
        assert_eq!(render(&Expr::load(BitWidth::W32, reg("a")), 0), "read_u32(a)");
        assert_eq!(render(&Expr::constant(BitWidth::W8, 0xCC), 0), "INT3");
        assert_eq!(
            render(&Expr::constant(BitWidth::W64, 0x1_4000_0000), 0x1_4000_0000),
            "IMAGE_BASE"
        );
    }
}
