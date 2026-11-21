//! Symbolic recovery of a VMProtect keyed-transform decode codec.
//!
//! Where vmpx-vm's `dispatch_extract` pattern-matcher rejects a handler's decode
//! because every step is in a mutation-saturated / MBA form, this traces the
//! entry's bootstrap->dispatch path (following opaque-predicate conditionals),
//! composes the per-instruction semantics symbolically (`lift_native_block` ->
//! `Expr` -> substitute -> simplify, which folds the MBA away), and peels the
//! resulting handler-register expression into the pipeline's own
//! [`KeyedTransformCodec`].
//!
//! This is the library form of the `symdecode` diagnostic. A recovered codec is an
//! *unproven* candidate (it assumes the followed conditional arms), so callers that
//! feed it into role-assignment must label it accordingly. Because `vmpx-mutation`
//! depends on `vmpx-vm` (not the reverse), the recovery lives here and produces the
//! `vmpx-vm` codec type directly — `dispatch_extract` cannot call back into the
//! symbolic lifter without a dependency cycle, so the integration is this function
//! invoked as a gated fallback at the layer that drives bootstrap recovery.

use std::collections::BTreeMap;

use iced_x86::FlowControl;
use vmpx_disasm::{trace_direct_jump_fragments, DirectJumpTraceLimits, Gpr};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    BinaryOp, BitWidth, Expr, ExprKind, KeyOperation, KeyedTransformCodec, Transform,
    TransformChain, TransformOp, UnaryOp,
};

use crate::lift_native_block;

fn as_const(e: &Expr) -> Option<u64> {
    match &e.kind {
        ExprKind::Const { value } => Some(*value),
        _ => None,
    }
}

fn as_reg(e: &Expr) -> Option<&str> {
    match &e.kind {
        ExprKind::Register { name } => Some(name.as_str()),
        _ => None,
    }
}

fn contains_load(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Load { .. } => true,
        ExprKind::Unary { value, .. }
        | ExprKind::Resize { value, .. }
        | ExprKind::Extract { value, .. } => contains_load(value),
        ExprKind::Binary { left, right, .. } => contains_load(left) || contains_load(right),
        ExprKind::Concat { high, low } => contains_load(high) || contains_load(low),
        _ => false,
    }
}

/// Whether `name` appears as a `Register` leaf anywhere in `e`. Used to tell the
/// rolling-key operand (references the key register) from the token displacement
/// (never does) when both sides of the key-mix carry bytecode loads.
fn contains_register(e: &Expr, name: &str) -> bool {
    match &e.kind {
        ExprKind::Register { name: n } => n == name,
        ExprKind::Unary { value, .. }
        | ExprKind::Resize { value, .. }
        | ExprKind::Extract { value, .. } => contains_register(value, name),
        ExprKind::Load { address } => contains_register(address, name),
        ExprKind::Binary { left, right, .. } => {
            contains_register(left, name) || contains_register(right, name)
        }
        ExprKind::Concat { high, low } => {
            contains_register(high, name) || contains_register(low, name)
        }
        ExprKind::Intrinsic { arguments, .. } => {
            arguments.iter().any(|a| contains_register(a, name))
        }
        ExprKind::Const { .. } => false,
    }
}

const FLAG_INPUTS: [&str; 6] = ["cf_in", "pf_in", "af_in", "zf_in", "sf_in", "of_in"];

/// Fold the flag-input registers to zero. VMProtect's mutated arithmetic threads
/// carry/flag inputs (`adc`/`sbb`, carry-add decode) that are guaranteed 0 on a
/// clean run (the VM's own mutation sets them so) — the same `cf = 0` assumption
/// `VMPX_MUTATED_ARITH` applies in the pattern-matcher, here at the expression
/// level so the carry terms (`+ ZeroExtend(cf_in)`) collapse and the decode reads
/// as the clean keyed transform.
fn zero_flags(e: &Expr) -> Expr {
    match &e.kind {
        ExprKind::Register { name } if FLAG_INPUTS.contains(&name.as_str()) => {
            Expr::constant(e.width, 0)
        }
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Load { address } => Expr::load(e.width, zero_flags(address)),
        ExprKind::Unary { op, value } => Expr::unary(*op, zero_flags(value)),
        ExprKind::Binary { op, left, right } => {
            Expr::binary(*op, zero_flags(left), zero_flags(right)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Resize { op, value } => {
            Expr::resize(*op, e.width, zero_flags(value)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Extract { value, lsb } => {
            Expr::extract(e.width, *lsb, zero_flags(value)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Concat { high, low } => {
            Expr::concat(zero_flags(high), zero_flags(low)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Intrinsic { .. } => e.clone(),
    }
}

fn binop_keyop(op: BinaryOp) -> Option<KeyOperation> {
    match op {
        BinaryOp::Add => Some(KeyOperation::Add),
        BinaryOp::Sub => Some(KeyOperation::Sub),
        BinaryOp::Xor => Some(KeyOperation::Xor),
        _ => None,
    }
}

fn binop_transform(op: BinaryOp) -> Option<TransformOp> {
    match op {
        BinaryOp::Add => Some(TransformOp::Add),
        BinaryOp::Sub => Some(TransformOp::Sub),
        BinaryOp::Xor => Some(TransformOp::Xor),
        BinaryOp::Rol => Some(TransformOp::Rol),
        BinaryOp::Ror => Some(TransformOp::Ror),
        _ => None,
    }
}

/// Strip width casts (`Truncate`/`ZeroExtend`/`SignExtend`, `Extract@0`) to reach
/// the core expression — the MBA lifter wraps every sub-op in them.
fn strip_casts(e: &Expr) -> &Expr {
    match &e.kind {
        ExprKind::Resize { value, .. } => strip_casts(value),
        ExprKind::Extract { value, lsb, .. } if *lsb == 0 => strip_casts(value),
        _ => e,
    }
}

/// Flatten an `Or`-tree into its leaf terms (stripping casts between the Ors).
fn flatten_or<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match &strip_casts(e).kind {
        ExprKind::Binary { op: BinaryOp::Or, left, right } => {
            flatten_or(left, out);
            flatten_or(right, out);
        }
        _ => out.push(strip_casts(e)),
    }
}

/// Parse one byte-move term `((inner >> shr) & 0xff) << shl` → `(shr, shl, inner)`.
/// `& 0xff` is optional (the top byte omits it); `shr`/`shl` default to 0.
fn parse_byte_term(e: &Expr) -> Option<(u32, u32, &Expr)> {
    let e = strip_casts(e);
    let (shl, e) = match &e.kind {
        ExprKind::Binary { op: BinaryOp::Shl, left, right } => {
            (u32::try_from(as_const(right)?).ok()?, strip_casts(left))
        }
        _ => (0, e),
    };
    let e = match &e.kind {
        ExprKind::Binary { op: BinaryOp::And, left, right } if as_const(right) == Some(0xff) => {
            strip_casts(left)
        }
        _ => e,
    };
    let (shr, inner) = match &e.kind {
        ExprKind::Binary { op: BinaryOp::Shr, left, right } => {
            (u32::try_from(as_const(right)?).ok()?, strip_casts(left))
        }
        _ => (0, e),
    };
    Some((shr, shl, inner))
}

/// Recognize a 32-bit byte swap: `Or` of the four byte-moves of a common inner
/// expression in reversed byte order. Returns the inner expression.
fn match_bswap32(e: &Expr) -> Option<&Expr> {
    let mut terms = Vec::new();
    flatten_or(e, &mut terms);
    if terms.len() != 4 {
        return None;
    }
    let parsed: Vec<(u32, u32, &Expr)> =
        terms.iter().map(|t| parse_byte_term(t)).collect::<Option<_>>()?;
    let inner = strip_casts(parsed[0].2);
    if !parsed.iter().all(|(_, _, i)| strip_casts(i) == inner) {
        return None;
    }
    let mut moves: Vec<(u32, u32)> = parsed.iter().map(|(a, b, _)| (*a, *b)).collect();
    moves.sort_unstable();
    if moves == [(0, 24), (8, 16), (16, 8), (24, 0)] {
        Some(inner)
    } else {
        None
    }
}

#[derive(Default)]
struct Peeled {
    /// Transforms in outermost-first order (reversed before building the chain).
    transforms: Vec<Transform>,
    key_op: Option<KeyOperation>,
    key_on_left: bool,
    width: Option<BitWidth>,
}

/// Peel a decode-displacement `Expr` (anchor already stripped) into codec pieces.
/// Returns `false` if it uses an operation the keyed-transform codec can't model
/// (variable shift, multiply, and/or, a two-operand mix, etc.).
fn peel(e: &Expr, p: &mut Peeled, key_in: &str) -> bool {
    match &e.kind {
        ExprKind::Resize { value, .. } => {
            let w = value.width;
            p.width.get_or_insert(w);
            peel(value, p, key_in)
        }
        ExprKind::Extract { value, lsb, .. } if *lsb == 0 => {
            let w = e.width;
            p.width.get_or_insert(w);
            peel(value, p, key_in)
        }
        ExprKind::Unary { op, value } => {
            let top = match op {
                UnaryOp::Not => TransformOp::Not,
                UnaryOp::Neg => TransformOp::Neg,
            };
            p.transforms.push(Transform { op: top, width: e.width, operand: 0 });
            peel(value, p, key_in)
        }
        ExprKind::Binary { op, left, right } => {
            // Constant transform: `token <op> c`. (Check const first so an `Add 0`
            // left by flag-folding is consumed as an identity transform.)
            if let Some(top) = binop_transform(*op) {
                if let Some(c) = as_const(right) {
                    p.transforms.push(Transform { op: top, width: e.width, operand: c });
                    return peel(left, p, key_in);
                }
                if let Some(c) = as_const(left) {
                    if matches!(op, BinaryOp::Add | BinaryOp::Xor) {
                        p.transforms.push(Transform { op: top, width: e.width, operand: c });
                        return peel(right, p, key_in);
                    }
                }
            }
            // Byte swap: VMProtect decodes often `bswap` mid-chain, which the MBA
            // lifter expands into a 4-byte `Or`-tree. Recompose it to one transform.
            if matches!(op, BinaryOp::Or) {
                if let Some(inner) = match_bswap32(e) {
                    p.transforms.push(Transform { op: TransformOp::Bswap, width: e.width, operand: 0 });
                    return peel(inner, p, key_in);
                }
            }
            // Key-mix: one side references the rolling-key register, the other is the
            // token displacement spine (which never references the key). This cleanly
            // separates them even when BOTH carry bytecode loads — the key is a prior
            // handler's derived rolling key, so it still contains `load[vip]`.
            if p.key_op.is_none() {
                let lk = contains_register(left, key_in);
                let rk = contains_register(right, key_in);
                if lk ^ rk {
                    let Some(k) = binop_keyop(*op) else {
                        return false;
                    };
                    let token_side: &Expr = if lk { right } else { left };
                    p.key_op = Some(k);
                    p.key_on_left = lk;
                    p.width.get_or_insert(e.width);
                    return peel(token_side, p, key_in);
                }
            }
            false
        }
        // Reached the token (bytecode load, or a degenerate input/const bottom).
        ExprKind::Load { .. } | ExprKind::Register { .. } | ExprKind::Const { .. } => true,
        _ => false,
    }
}

/// `handler = handler_in + displacement`; return the displacement sub-expression.
fn split_anchor<'a>(handler: &'a Expr, handler_in: &str) -> Option<&'a Expr> {
    if let ExprKind::Binary { op: BinaryOp::Add, left, right } = &handler.kind {
        if as_reg(left) == Some(handler_in) {
            return Some(&**right);
        }
        if as_reg(right) == Some(handler_in) {
            return Some(&**left);
        }
    }
    None
}

/// Count `Register` leaves matching `name`.
fn count_register(e: &Expr, name: &str) -> usize {
    match &e.kind {
        ExprKind::Register { name: n } => usize::from(n == name),
        ExprKind::Unary { value, .. }
        | ExprKind::Resize { value, .. }
        | ExprKind::Extract { value, .. } => count_register(value, name),
        ExprKind::Load { address } => count_register(address, name),
        ExprKind::Binary { left, right, .. } => {
            count_register(left, name) + count_register(right, name)
        }
        ExprKind::Concat { high, low } => count_register(high, name) + count_register(low, name),
        ExprKind::Intrinsic { arguments, .. } => {
            arguments.iter().map(|a| count_register(a, name)).sum()
        }
        ExprKind::Const { .. } => 0,
    }
}

/// Recover the rolling-key update operation from the key register's composed
/// expression: `key_after = key_in <op> decoded`. Handles the direct form and a
/// uniform associative spine (`(key_in ^ A) ^ B` folds to `key_in ^ (A ^ B)`), but
/// only when the key register appears exactly once — a multi-term mutation-composed
/// update (two decodes xored into the key) is genuinely ambiguous and returns None.
fn key_update(key_expr: &Expr, key_in: &str) -> Option<KeyOperation> {
    let ExprKind::Binary { op, left, right } = &key_expr.kind else {
        return None;
    };
    let kop = binop_keyop(*op)?;
    let lk = contains_register(left, key_in);
    let rk = contains_register(right, key_in);
    if !(lk ^ rk) {
        return None;
    }
    let key_side: &Expr = if lk { left } else { right };
    if as_reg(key_side) == Some(key_in) {
        return Some(kop); // key_after = key_in OP decoded (direct)
    }
    // Nested, but only safe to flatten for associative/commutative ops with a single
    // key occurrence — then key_after = key_in OP (everything else).
    if matches!(op, BinaryOp::Add | BinaryOp::Xor) && count_register(key_expr, key_in) == 1 {
        return key_update(key_side, key_in).filter(|u| *u == kop);
    }
    None
}

/// Build a [`KeyedTransformCodec`] from a composed register state: the handler
/// register's decode expression plus the rolling-key register's update. Pure;
/// unit-tested without a PE.
fn codec_from_regs(
    regs: &BTreeMap<Gpr, Expr>,
    handler_reg: Gpr,
    key_reg: Gpr,
) -> Option<KeyedTransformCodec> {
    let key_in = key_reg.input_name();
    // Fold flag inputs to zero (the clean-run cf=0 assumption) then re-simplify, so
    // the mutated carry terms collapse before peeling.
    let handler = zero_flags(&regs.get(&handler_reg)?.simplify()).simplify();
    let disp = split_anchor(&handler, &handler_reg.input_name())?;
    let mut p = Peeled::default();
    if !peel(disp, &mut p, &key_in) {
        return None;
    }
    if p.transforms.is_empty() && p.key_op.is_none() {
        return None; // nothing recovered beyond the raw token
    }
    let width = p.width.unwrap_or(BitWidth::W32);
    // Decode requires every transform at the codec width; force it (the chain runs
    // at the displacement width even where 64-bit register nodes appear).
    let transforms: Vec<Transform> = p
        .transforms
        .iter()
        .rev()
        .map(|t| Transform { width, operand: width.truncate(t.operand), op: t.op })
        .collect();
    let update = regs
        .get(&key_reg)
        .and_then(|k| key_update(&zero_flags(&k.simplify()).simplify(), &key_in));
    Some(KeyedTransformCodec {
        width,
        input_key_operation: p.key_op,
        input_key_on_left: p.key_on_left,
        transforms: TransformChain { transforms },
        update_key_operation: update,
    })
}

/// Compose the symbolic register state along the entry's dispatch path, stopping
/// once `handler_reg` first becomes token-derived (isolating one handler's decode).
fn compose_path(pe: &PeImage, entry: Rva, handler_reg: Gpr) -> Option<BTreeMap<Gpr, Expr>> {
    let mut limits = DirectJumpTraceLimits::default();
    limits.follow_conditional_fallthrough = true;
    limits.max_conditional_continuations = 128;
    let trace = trace_direct_jump_fragments(pe, entry, limits).ok()?;

    let mut regs: BTreeMap<Gpr, Expr> =
        Gpr::ALL.iter().map(|&g| (g, Expr::register(BitWidth::W64, g.input_name()))).collect();

    for ins in trace.instructions() {
        if ins.instruction.flow_control() != FlowControl::Next {
            continue;
        }
        let Ok(sem) = lift_native_block(&ins.bytes, u64::from(ins.rva.0)) else {
            continue;
        };
        let repl: BTreeMap<String, Expr> =
            Gpr::ALL.iter().map(|&g| (g.input_name(), regs[&g].clone())).collect();
        let mut next = regs.clone();
        for (g, e) in &sem.regs {
            if let Ok(c) = e.substitute(&repl) {
                next.insert(*g, c.simplify());
            }
        }
        regs = next;
        if contains_load(&regs[&handler_reg]) {
            break;
        }
    }
    Some(regs)
}

/// Recover the keyed-transform decode codec for `handler_reg` at a VM dispatch
/// entry, with `key_reg` as the rolling-key register. Returns `None` when the
/// handler never becomes token-derived or the decode uses an operation the codec
/// cannot model. The result is an unproven candidate (follows conditional arms).
#[must_use]
pub fn extract_symbolic_codec(
    pe: &PeImage,
    entry: Rva,
    handler_reg: Gpr,
    key_reg: Gpr,
) -> Option<KeyedTransformCodec> {
    let regs = compose_path(pe, entry, handler_reg)?;
    codec_from_regs(&regs, handler_reg, key_reg)
}

/// Like [`extract_symbolic_codec`] but auto-selects the rolling-key register from
/// `key_candidates`. The path is composed once (it does not depend on the key);
/// the true key register is the one whose decode exhibits a key-mix
/// (`input_key_operation` is `Some`) — the token displacement never references the
/// key, so a wrong key yields no mix or no codec. Returns the codec and the chosen
/// key register. This is the robust token/key disambiguation for the fallback.
#[must_use]
pub fn extract_symbolic_codec_auto(
    pe: &PeImage,
    entry: Rva,
    handler_reg: Gpr,
    key_candidates: &[Gpr],
) -> Option<(KeyedTransformCodec, Gpr)> {
    let regs = compose_path(pe, entry, handler_reg)?;
    let mut keyless: Option<(KeyedTransformCodec, Gpr)> = None;
    for &key in key_candidates {
        if key == handler_reg {
            continue;
        }
        if let Some(codec) = codec_from_regs(&regs, handler_reg, key) {
            if codec.input_key_operation.is_some() {
                return Some((codec, key)); // a real key-mix fired: this is the key
            }
            keyless.get_or_insert((codec, key));
        }
    }
    keyless
}

/// Map an iced-x86 GPR to the symbolic lifter's `Gpr` by name (`RDI` -> `rdi`).
fn gpr_from_iced(register: iced_x86::Register) -> Option<Gpr> {
    let name = format!("{register:?}").to_lowercase();
    Gpr::ALL.iter().copied().find(|g| g.input_name().trim_end_matches("_in") == name)
}

/// Install the symbolic-decode fallback into `vmpx-vm`'s role-assignment seam.
/// Call once at startup from a binary that depends on both crates; `vmpx-vm`
/// consults it when bootstrap role-assignment fails and `VMPX_SYMBOLIC_DECODE` is
/// set, surfacing the recovered decode codec as a diagnostic.
pub fn install_symbolic_codec_provider() {
    vmpx_vm::set_symbolic_codec_provider(Box::new(|pe, entry, handler, keys| {
        let handler_gpr = gpr_from_iced(handler)?;
        let key_gprs: Vec<Gpr> = keys.iter().filter_map(|&r| gpr_from_iced(r)).collect();
        extract_symbolic_codec_auto(pe, entry, handler_gpr, &key_gprs).map(|(codec, _)| codec)
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::ResizeOp;

    fn gpr(name: &str) -> Gpr {
        Gpr::ALL.iter().copied().find(|g| g.input_name() == name).expect("gpr")
    }

    // Build `handler_in + sext64( not( (load[rsi_in] ^ rbp_in) + 0x1234 ) )` — the
    // validated VMProtect decode shape (key-mix, then +const, then a Not), and check
    // the recovered codec decodes identically to the reference computation.
    #[test]
    fn recovers_keyed_transform_codec_and_round_trips() {
        let handler = gpr("rdi_in");
        let key = gpr("rbp_in");

        let token = Expr::load(BitWidth::W32, Expr::register(BitWidth::W64, "rsi_in"));
        let key_in = Expr::register(BitWidth::W32, "rbp_in");
        let mix = Expr::binary(BinaryOp::Xor, token, key_in).unwrap();
        let add = Expr::binary(BinaryOp::Add, mix, Expr::constant(BitWidth::W32, 0x1234)).unwrap();
        let notv = Expr::unary(UnaryOp::Not, add);
        let sext = Expr::resize(ResizeOp::SignExtend, BitWidth::W64, notv).unwrap();
        let handler_expr =
            Expr::binary(BinaryOp::Add, Expr::register(BitWidth::W64, "rdi_in"), sext).unwrap();

        // key_after = rbp_in ^ decoded
        let key_update_expr = Expr::binary(
            BinaryOp::Xor,
            Expr::register(BitWidth::W32, "rbp_in"),
            Expr::register(BitWidth::W32, "decoded"),
        )
        .unwrap();

        let mut regs = BTreeMap::new();
        regs.insert(handler, handler_expr);
        regs.insert(key, key_update_expr);

        let codec = codec_from_regs(&regs, handler, key).expect("codec recovered");

        assert_eq!(codec.width, BitWidth::W32);
        assert_eq!(codec.input_key_operation, Some(KeyOperation::Xor));
        assert!(!codec.input_key_on_left, "key is on the right (token ^ key)");
        assert_eq!(codec.update_key_operation, Some(KeyOperation::Xor));
        // apply order: key-mix first, then +0x1234, then Not
        assert_eq!(codec.transforms.transforms.len(), 2);
        assert_eq!(codec.transforms.transforms[0].op, TransformOp::Add);
        assert_eq!(codec.transforms.transforms[0].operand, 0x1234);
        assert_eq!(codec.transforms.transforms[1].op, TransformOp::Not);

        // Round-trip: decode must equal the reference MBA-free computation.
        for (tok, k) in [(0xABu64, 0xCDu64), (0x1234_5678, 0xDEAD_BEEF), (0, 0)] {
            let reference = {
                let mixed = (tok as u32) ^ (k as u32);
                let added = mixed.wrapping_add(0x1234);
                !added
            } as u64;
            let got = codec.decode(tok, k).expect("decode").decoded;
            assert_eq!(got, reference, "tok={tok:#x} key={k:#x}");
        }
    }

    #[test]
    fn key_update_handles_direct_and_nested_associative() {
        // direct: key_after = rbp_in ^ decoded
        let direct = Expr::binary(
            BinaryOp::Xor,
            Expr::register(BitWidth::W32, "rbp_in"),
            Expr::register(BitWidth::W32, "decoded"),
        )
        .unwrap();
        assert_eq!(key_update(&direct, "rbp_in"), Some(KeyOperation::Xor));

        // nested associative, single key occurrence: (rbp_in ^ A) ^ B -> Xor
        let a = Expr::constant(BitWidth::W32, 0x1111);
        let b = Expr::register(BitWidth::W32, "decoded");
        let inner = Expr::binary(BinaryOp::Xor, Expr::register(BitWidth::W32, "rbp_in"), a).unwrap();
        let nested = Expr::binary(BinaryOp::Xor, inner, b).unwrap();
        assert_eq!(key_update(&nested, "rbp_in"), Some(KeyOperation::Xor));

        // multi-term (key appears twice) is ambiguous -> None
        let twice = Expr::binary(
            BinaryOp::Xor,
            Expr::register(BitWidth::W32, "rbp_in"),
            Expr::register(BitWidth::W32, "rbp_in"),
        )
        .unwrap();
        assert_eq!(key_update(&twice, "rbp_in"), None);
    }

    fn byte_move(x: &Expr, shr: u32, shl: u32) -> Expr {
        let mut e = x.clone();
        if shr > 0 {
            e = Expr::binary(BinaryOp::Shr, e, Expr::constant(BitWidth::W32, shr.into())).unwrap();
        }
        e = Expr::binary(BinaryOp::And, e, Expr::constant(BitWidth::W32, 0xff)).unwrap();
        if shl > 0 {
            e = Expr::binary(BinaryOp::Shl, e, Expr::constant(BitWidth::W32, shl.into())).unwrap();
        }
        e
    }
    fn bswap32(x: &Expr) -> Expr {
        let a = byte_move(x, 0, 24);
        let b = byte_move(x, 8, 16);
        let c = byte_move(x, 16, 8);
        let d = byte_move(x, 24, 0);
        let ab = Expr::binary(BinaryOp::Or, a, b).unwrap();
        let abc = Expr::binary(BinaryOp::Or, ab, c).unwrap();
        Expr::binary(BinaryOp::Or, abc, d).unwrap()
    }

    // The MBA-expanded byte swap must be recomposed to a single Bswap transform
    // (VMProtect's per-instance codecs, e.g. instance 0xc, use bswap in the chain).
    #[test]
    fn recovers_bswap_in_the_chain() {
        let handler = gpr("rdi_in");
        let key = gpr("rbp_in");
        let token = Expr::load(BitWidth::W32, Expr::register(BitWidth::W64, "rsi_in"));
        let mix = Expr::binary(BinaryOp::Xor, token, Expr::register(BitWidth::W32, "rbp_in")).unwrap();
        let swapped = bswap32(&mix);
        let sext = Expr::resize(ResizeOp::SignExtend, BitWidth::W64, swapped).unwrap();
        let handler_expr =
            Expr::binary(BinaryOp::Add, Expr::register(BitWidth::W64, "rdi_in"), sext).unwrap();
        let mut regs = BTreeMap::new();
        regs.insert(handler, handler_expr);
        let codec = codec_from_regs(&regs, handler, key).expect("codec with bswap");
        assert_eq!(codec.input_key_operation, Some(KeyOperation::Xor));
        assert_eq!(codec.transforms.transforms.len(), 1);
        assert_eq!(codec.transforms.transforms[0].op, TransformOp::Bswap);
        // round-trip: decode(token,key) == bswap32(token ^ key)
        for (t, k) in [(0x11223344u64, 0xAABBCCDDu64), (1, 0), (0xDEADBEEF, 0xFF00FF00)] {
            let mixed = (t as u32) ^ (k as u32);
            let want = mixed.swap_bytes() as u64;
            assert_eq!(codec.decode(t, k).unwrap().decoded, want, "t={t:#x} k={k:#x}");
        }
    }

    // A decode using a non-modellable op (variable multiply) must be rejected, not
    // silently mis-recovered.
    #[test]
    fn rejects_unmodellable_decode() {
        let handler = gpr("rdi_in");
        let key = gpr("rbp_in");
        let token = Expr::load(BitWidth::W32, Expr::register(BitWidth::W64, "rsi_in"));
        // token * token — not a reversible keyed transform
        let mul = Expr::binary(BinaryOp::Mul, token.clone(), token).unwrap();
        let sext = Expr::resize(ResizeOp::SignExtend, BitWidth::W64, mul).unwrap();
        let handler_expr =
            Expr::binary(BinaryOp::Add, Expr::register(BitWidth::W64, "rdi_in"), sext).unwrap();
        let mut regs = BTreeMap::new();
        regs.insert(handler, handler_expr);
        assert!(codec_from_regs(&regs, handler, key).is_none());
    }
}
