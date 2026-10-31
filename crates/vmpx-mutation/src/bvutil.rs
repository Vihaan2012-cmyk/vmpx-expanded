//! Small structural helpers over `vmpx_vm::Expr` shared by the synthesis foundation.

use std::collections::{BTreeMap, HashMap, HashSet};

use vmpx_vm::{BitWidth, Expr, ExprError, ExprKind};

/// Visits every distinct node once (shared sub-DAGs are not re-walked).
pub fn walk(root: &Expr, visit: &mut dyn FnMut(&Expr)) {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut stack: Vec<&Expr> = vec![root];
    while let Some(e) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(e) as usize) {
            continue;
        }
        visit(e);
        match &e.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => {}
            ExprKind::Unary { value, .. }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => stack.push(value),
            ExprKind::Load { address } => stack.push(address),
            ExprKind::Binary { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            ExprKind::Concat { high, low } => {
                stack.push(high);
                stack.push(low);
            }
            ExprKind::Intrinsic { arguments, .. } => stack.extend(arguments.iter()),
        }
    }
}

/// Free registers with widths; `None` when one name appears at two widths.
pub fn registers(root: &Expr) -> Option<BTreeMap<String, BitWidth>> {
    let mut out = BTreeMap::new();
    let mut clash = false;
    walk(root, &mut |e| {
        if let ExprKind::Register { name } = &e.kind
            && *out.entry(name.clone()).or_insert(e.width) != e.width
        {
            clash = true;
        }
    });
    (!clash).then_some(out)
}

/// Rebuilds `root` bottom-up, offering each `Register` and (address-rebuilt)
/// `Load` node to `leaf`; `Some(e)` replaces the node. Memoised per node.
pub fn rebuild(
    root: &Expr,
    leaf: &mut dyn FnMut(&Expr) -> Option<Expr>,
) -> Result<Expr, ExprError> {
    go(root, leaf, &mut HashMap::new())
}

fn go(
    e: &Expr,
    leaf: &mut dyn FnMut(&Expr) -> Option<Expr>,
    memo: &mut HashMap<usize, Expr>,
) -> Result<Expr, ExprError> {
    let key = std::ptr::from_ref(e) as usize;
    if let Some(hit) = memo.get(&key) {
        return Ok(hit.clone());
    }
    let out = match &e.kind {
        ExprKind::Const { .. } => e.clone(),
        ExprKind::Register { .. } => leaf(e).unwrap_or_else(|| e.clone()),
        ExprKind::Load { address } => {
            let load = Expr::load(e.width, go(address, leaf, memo)?);
            leaf(&load).unwrap_or(load)
        }
        ExprKind::Unary { op, value } => Expr::unary(*op, go(value, leaf, memo)?),
        ExprKind::Binary { op, left, right } => {
            Expr::binary(*op, go(left, leaf, memo)?, go(right, leaf, memo)?)?
        }
        ExprKind::Extract { value, lsb } => Expr::extract(e.width, *lsb, go(value, leaf, memo)?)?,
        ExprKind::Concat { high, low } => {
            Expr::concat(go(high, leaf, memo)?, go(low, leaf, memo)?)?
        }
        ExprKind::Resize { op, value } => Expr::resize(*op, e.width, go(value, leaf, memo)?)?,
        ExprKind::Intrinsic { name, arguments } => {
            let args = arguments
                .iter()
                .map(|a| go(a, leaf, memo))
                .collect::<Result<Vec<_>, _>>()?;
            Expr::intrinsic(e.width, name.clone(), args)
        }
    };
    memo.insert(key, out.clone());
    Ok(out)
}

/// Concrete evaluation: registers from `env`, loads from little-endian `memory`.
/// `None` if anything stays symbolic (missing register/byte, opaque intrinsic).
pub fn eval_concrete(
    root: &Expr,
    env: &BTreeMap<String, u64>,
    memory: &BTreeMap<u64, u8>,
) -> Option<u64> {
    let folded = rebuild(root, &mut |e| match &e.kind {
        ExprKind::Register { name } => env.get(name).map(|v| Expr::constant(e.width, *v)),
        ExprKind::Load { address } => {
            let ExprKind::Const { value: base } = address.kind else {
                return None;
            };
            let mut v = 0_u64;
            for i in 0..u64::from(e.width.bytes()) {
                let byte = *memory.get(&base.wrapping_add(i))?;
                v |= u64::from(byte) << (8 * i);
            }
            Some(Expr::constant(e.width, v))
        }
        _ => None,
    })
    .ok()?
    .simplify();
    match folded.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

pub const fn mask(width: BitWidth) -> u64 {
    u64::MAX >> (64 - width.bits())
}
