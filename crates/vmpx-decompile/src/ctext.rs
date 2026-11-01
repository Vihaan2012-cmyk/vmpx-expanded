//! Evaluates the rendered C text with real C semantics (integer promotion,
//! usual arithmetic conversions, shift/overflow UB detection) so the renderer is
//! checked against the IR it printed from, independent of the printer's own
//! precedence and cast decisions.

use std::collections::HashMap;

use crate::exec::mem_read;

#[derive(Clone, Copy, Debug)]
struct V {
    val: i128,
    bits: u32,
    signed: bool,
}

fn wrap(val: i128, bits: u32, signed: bool) -> V {
    let m = 1_i128 << bits;
    let mut r = val.rem_euclid(m);
    if signed && r >= m / 2 {
        r -= m;
    }
    V { val: r, bits, signed }
}

fn promote(x: V) -> V {
    if x.bits < 32 { V { val: x.val, bits: 32, signed: true } } else { x }
}

fn common(a: V, b: V) -> (u32, bool) {
    let (a, b) = (promote(a), promote(b));
    let bits = a.bits.max(b.bits);
    let signed = match (a.bits == bits, b.bits == bits) {
        (true, true) => a.signed && b.signed,
        (true, false) => a.signed,
        (false, _) => b.signed,
    };
    (bits, signed)
}

struct P<'a> {
    toks: Vec<String>,
    pos: usize,
    vars: &'a HashMap<String, (u64, u32)>,
}

fn tokenize(s: &str) -> Vec<String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_alphanumeric() || ch == '_' {
            let st = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(c[st..i].iter().collect());
        } else {
            let two: String = c[i..(i + 2).min(c.len())].iter().collect();
            if ["<<", ">>", "<=", ">=", "==", "!="].contains(&two.as_str()) {
                out.push(two);
                i += 2;
            } else {
                out.push(ch.to_string());
                i += 1;
            }
        }
    }
    out
}

fn ty_of(t: &str) -> Option<(u32, bool)> {
    Some(match t {
        "u8" => (8, false),
        "u16" => (16, false),
        "u32" => (32, false),
        "u64" => (64, false),
        "i8" => (8, true),
        "i16" => (16, true),
        "i32" => (32, true),
        "i64" => (64, true),
        _ => return None,
    })
}

impl P<'_> {
    fn peek(&self) -> Option<&str> {
        self.toks.get(self.pos).map(String::as_str)
    }
    fn next(&mut self) -> Option<String> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }
    fn eat(&mut self, s: &str) -> Result<(), String> {
        match self.next() {
            Some(t) if t == s => Ok(()),
            other => Err(format!("expected `{s}`, found {other:?}")),
        }
    }

    fn binary(&mut self, min: u8) -> Result<V, String> {
        let mut l = self.unary()?;
        loop {
            let Some(op) = self.peek().map(str::to_owned) else { break };
            let prec = match op.as_str() {
                "|" => 1,
                "^" => 2,
                "&" => 3,
                "==" | "!=" => 4,
                "<" | "<=" | ">" | ">=" => 5,
                "<<" | ">>" => 6,
                "+" | "-" => 7,
                "*" => 8,
                _ => break,
            };
            if prec < min {
                break;
            }
            self.pos += 1;
            let r = self.binary(prec + 1)?;
            l = apply(&op, l, r)?;
        }
        Ok(l)
    }

    fn unary(&mut self) -> Result<V, String> {
        let t = self.next().ok_or("unexpected end")?;
        match t.as_str() {
            "~" => {
                let x = promote(self.unary()?);
                Ok(wrap(!x.val, x.bits, x.signed))
            }
            "-" => {
                let x = promote(self.unary()?);
                let r = wrap(-x.val, x.bits, x.signed);
                if x.signed && -x.val != r.val {
                    return Err("signed overflow in negation".into());
                }
                Ok(r)
            }
            "*" => {
                self.eat("(")?;
                let ty = self.next().ok_or("type")?;
                self.eat("*")?;
                self.eat(")")?;
                let (bits, _) = ty_of(&ty).ok_or("deref type")?;
                let a = self.unary()?;
                Ok(wrap(i128::from(mem_read(a.val.rem_euclid(1 << 64) as u64, bw(bits))), bits, false))
            }
            "(" => {
                // cast or parenthesized expression
                if let Some(ty) = self.peek().and_then(ty_of) {
                    if self.toks.get(self.pos + 1).map(String::as_str) == Some(")") {
                        self.pos += 2;
                        let x = self.unary()?;
                        return Ok(wrap(x.val, ty.0, ty.1));
                    }
                }
                let v = self.binary(1)?;
                self.eat(")")?;
                Ok(v)
            }
            _ => {
                if t.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                    return number(&t);
                }
                if self.peek() == Some("(") {
                    // rolN / rorN
                    self.pos += 1;
                    let a = self.binary(1)?;
                    self.eat(",")?;
                    let n = self.binary(1)?;
                    self.eat(")")?;
                    let (kind, w) = t.split_at(3);
                    let bits: u32 = w.parse().map_err(|_| format!("unsupported call `{t}`"))?;
                    if kind != "rol" && kind != "ror" {
                        return Err(format!("unsupported call `{t}`"));
                    }
                    let x = wrap(a.val, bits, false).val as u128;
                    let c = (n.val.rem_euclid(i128::from(bits))) as u32;
                    let m = (1_u128 << bits) - 1;
                    let r = if c == 0 {
                        x
                    } else if kind == "rol" {
                        ((x << c) | (x >> (bits - c))) & m
                    } else {
                        ((x >> c) | (x << (bits - c))) & m
                    };
                    return Ok(wrap(r as i128, bits, false));
                }
                let (v, bits) = self.vars.get(&t).copied().ok_or_else(|| format!("unknown identifier `{t}`"))?;
                Ok(wrap(i128::from(v), bits, false))
            }
        }
    }
}

fn bw(bits: u32) -> vmpx_vm::BitWidth {
    vmpx_vm::BitWidth::from_bits(bits).expect("width")
}

fn number(t: &str) -> Result<V, String> {
    let (body, suffix) = {
        let lower = t.to_ascii_lowercase();
        let s = lower.trim_end_matches(['u', 'l']).len();
        (t[..s].to_owned(), t[s..].to_ascii_lowercase())
    };
    let (val, hex) = if let Some(h) = body.strip_prefix("0x") {
        (i128::from_str_radix(h, 16).map_err(|e| e.to_string())?, true)
    } else {
        (body.parse::<i128>().map_err(|e| e.to_string())?, false)
    };
    let unsigned = suffix.contains('u');
    let long = suffix.contains('l');
    // C literal typing: int, unsigned int (hex or u), long long, unsigned long long
    let candidates: &[(u32, bool)] = &[(32, true), (32, false), (64, true), (64, false)];
    for &(bits, signed) in candidates {
        if long && bits < 64 {
            continue;
        }
        if unsigned && signed {
            continue;
        }
        if !hex && !unsigned && !signed {
            continue; // decimal without u never becomes unsigned
        }
        let max = if signed { (1_i128 << (bits - 1)) - 1 } else { (1_i128 << bits) - 1 };
        if val <= max {
            return Ok(V { val, bits, signed });
        }
    }
    Err(format!("literal `{t}` does not fit"))
}

fn apply(op: &str, a: V, b: V) -> Result<V, String> {
    match op {
        "<<" | ">>" => {
            let l = promote(a);
            let c = promote(b).val;
            if c < 0 || c >= i128::from(l.bits) {
                return Err(format!("shift count {c} out of range for {}-bit type (UB)", l.bits));
            }
            let r = if op == "<<" {
                if l.signed && (l.val < 0 || (l.val << c) >= (1_i128 << (l.bits - 1))) {
                    return Err("signed left shift overflow (UB)".into());
                }
                l.val << c
            } else {
                l.val >> c
            };
            Ok(wrap(r, l.bits, l.signed))
        }
        "==" | "!=" | "<" | "<=" | ">" | ">=" => {
            let (bits, signed) = common(a, b);
            let (x, y) = (wrap(a.val, bits, signed).val, wrap(b.val, bits, signed).val);
            let r = match op {
                "==" => x == y,
                "!=" => x != y,
                "<" => x < y,
                "<=" => x <= y,
                ">" => x > y,
                _ => x >= y,
            };
            Ok(V { val: i128::from(r), bits: 32, signed: true })
        }
        _ => {
            let (bits, signed) = common(a, b);
            let (x, y) = (wrap(a.val, bits, signed).val, wrap(b.val, bits, signed).val);
            let raw = match op {
                "+" => x + y,
                "-" => x - y,
                "*" => x * y,
                "&" => x & y,
                "|" => x | y,
                "^" => x ^ y,
                _ => return Err(format!("unsupported operator {op}")),
            };
            let r = wrap(raw, bits, signed);
            if signed && matches!(op, "+" | "-" | "*") && raw != r.val {
                return Err("signed overflow (UB)".into());
            }
            Ok(r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(t: &str, vars: &[(&str, u64, u32)]) -> Result<u64, String> {
        let m = vars.iter().map(|(n, v, b)| ((*n).to_owned(), (*v, *b))).collect();
        eval(t, &m)
    }

    #[test]
    fn c_precedence_and_promotion() {
        // `a >> n & 63` parses as `(a >> n) & 63`, not `a >> (n & 63)`
        assert_eq!(ev("a >> 1 & 63", &[("a", 0x1ff, 64)]).unwrap(), 0xff & 63);
        // u16 + u16 promotes to int; the cast truncates
        assert_eq!(ev("(u16)(a + b)", &[("a", 0xffff, 16), ("b", 2, 16)]).unwrap(), 1);
        // shifting by >= width is flagged as UB
        assert!(ev("a << 32", &[("a", 1, 32)]).is_err());
        // signed overflow via promoted multiply is flagged
        assert!(ev("a * b", &[("a", 0xffff, 16), ("b", 0xffff, 16)]).is_err());
        assert_eq!(ev("rol32(a, 8)", &[("a", 0x1234_5678, 32)]).unwrap(), 0x3456_7812);
    }

    #[test]
    fn mba_xor_folds() {
        use vmpx_vm::{BinaryOp, BitWidth, Expr, UnaryOp};
        let a = Expr::register(BitWidth::W32, "a");
        let b = Expr::register(BitWidth::W32, "b");
        let nand = Expr::unary(UnaryOp::Not, Expr::binary(BinaryOp::And, a.clone(), b.clone()).unwrap());
        let or = Expr::unary(
            UnaryOp::Not,
            Expr::binary(BinaryOp::And, Expr::unary(UnaryOp::Not, a.clone()), Expr::unary(UnaryOp::Not, b.clone())).unwrap(),
        );
        let xor = Expr::binary(BinaryOp::And, or, nand).unwrap();
        let folded = crate::xu::mba_fold(&xor);
        assert_eq!(folded, Expr::binary(BinaryOp::Xor, a, b).unwrap());
    }
}

/// Evaluates `text` as a C expression; variables are `(value, declared bits)`
/// (all declared unsigned). Returns the value truncated to its own type.
pub fn eval(text: &str, vars: &HashMap<String, (u64, u32)>) -> Result<u64, String> {
    let mut p = P { toks: tokenize(text), pos: 0, vars };
    let v = p.binary(1)?;
    if p.pos != p.toks.len() {
        return Err(format!("trailing tokens after position {}", p.pos));
    }
    Ok(wrap(v.val, v.bits, false).val as u64)
}
