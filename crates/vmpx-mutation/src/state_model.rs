//! Handler state-model: a VM handler as a pure state-in -> state-out bitvector
//! function with explicit side-effect outputs.
//!
//! Inputs are named slots (VM registers, flags, virtual-stack slots relative to
//! the entry VSP). Memory-in is an implicit array read through
//! `ExprKind::Load`; every `Load` observes the **pre-handler** memory (the
//! builder must forward any in-handler store to a later load itself). Outputs
//! are only the slots the handler changes (absent = preserved), the VSP delta,
//! and an ordered list of memory writes. Expressions are `vmpx_vm::Expr`.
//!
//! Fail-closed: [`HandlerModel::validate`] rejects expressions that read an
//! undeclared slot; [`HandlerModel::apply`] errors on a missing input, an
//! unresolved load or an opaque intrinsic instead of guessing.

use std::collections::BTreeMap;
use std::fmt;

use vmpx_vm::{BitWidth, Expr, ExprError};

use crate::bvutil::{eval_concrete, mask, registers};
use crate::linear_mba::simplify_linear;

/// A state location. Flags are modelled as `W8` values holding 0 or 1.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Slot {
    Reg(String),
    Flag(String),
    /// Virtual-stack slot, byte offset from the VSP at handler entry.
    Stack(i64),
}

impl Slot {
    /// Name of the `Expr::Register` that stands for this slot's input value.
    #[must_use]
    pub fn var_name(&self) -> String {
        match self {
            Self::Reg(n) => format!("r:{n}"),
            Self::Flag(n) => format!("f:{n}"),
            Self::Stack(o) => format!("s:{o}"),
        }
    }

    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        let (kind, rest) = name.split_once(':')?;
        match kind {
            "r" => Some(Self::Reg(rest.to_owned())),
            "f" => Some(Self::Flag(rest.to_owned())),
            "s" => rest.parse().ok().map(Self::Stack),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum ModelError {
    /// Expression reads a register that is not a declared input slot.
    UndeclaredInput(String),
    WidthMismatch(Slot),
    BadAddressWidth,
    MissingInput(Slot),
    /// Evaluation left something symbolic (unresolved load, intrinsic, ...).
    NotConcrete(String),
    Expr(ExprError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndeclaredInput(n) => write!(f, "expression reads undeclared input `{n}`"),
            Self::WidthMismatch(s) => write!(f, "width mismatch on {s:?}"),
            Self::BadAddressWidth => f.write_str("memory address must be 64-bit"),
            Self::MissingInput(s) => write!(f, "no concrete value for input {s:?}"),
            Self::NotConcrete(w) => write!(f, "not concretely evaluable: {w}"),
            Self::Expr(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<ExprError> for ModelError {
    fn from(e: ExprError) -> Self {
        Self::Expr(e)
    }
}

/// One memory store: `mem[addr] = value` (little-endian, `value.width` bytes).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemWrite {
    pub addr: Expr,
    pub value: Expr,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HandlerModel {
    pub name: String,
    inputs: BTreeMap<Slot, BitWidth>,
    outputs: BTreeMap<Slot, Expr>,
    writes: Vec<MemWrite>,
    sp_delta: i64,
}

/// Concrete pre-state: slot values plus sparse byte memory.
#[derive(Clone, Debug, Default)]
pub struct ConcreteState {
    pub values: BTreeMap<Slot, u64>,
    pub memory: BTreeMap<u64, u8>,
}

/// Concrete effect of one handler execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConcreteOut {
    pub outputs: BTreeMap<Slot, u64>,
    pub sp_delta: i64,
    /// `(address, value, width)` in program order.
    pub writes: Vec<(u64, u64, BitWidth)>,
}

impl HandlerModel {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_input(mut self, slot: Slot, width: BitWidth) -> Self {
        self.inputs.insert(slot, width);
        self
    }

    /// The input value of a declared slot as an expression.
    #[must_use]
    pub fn input(&self, slot: &Slot) -> Option<Expr> {
        self.inputs
            .get(slot)
            .map(|w| Expr::register(*w, slot.var_name()))
    }

    /// # Errors
    ///
    /// [`ModelError::WidthMismatch`] if `slot` is a declared input of another width.
    pub fn set_output(&mut self, slot: Slot, value: Expr) -> Result<(), ModelError> {
        if self.inputs.get(&slot).is_some_and(|w| *w != value.width) {
            return Err(ModelError::WidthMismatch(slot));
        }
        self.outputs.insert(slot, value);
        Ok(())
    }

    /// # Errors
    ///
    /// [`ModelError::BadAddressWidth`] unless `addr` is 64-bit.
    pub fn push_write(&mut self, addr: Expr, value: Expr) -> Result<(), ModelError> {
        if addr.width != BitWidth::W64 {
            return Err(ModelError::BadAddressWidth);
        }
        self.writes.push(MemWrite { addr, value });
        Ok(())
    }

    pub fn set_sp_delta(&mut self, delta: i64) {
        self.sp_delta = delta;
    }

    #[must_use]
    pub const fn sp_delta(&self) -> i64 {
        self.sp_delta
    }

    #[must_use]
    pub const fn inputs(&self) -> &BTreeMap<Slot, BitWidth> {
        &self.inputs
    }

    #[must_use]
    pub const fn outputs(&self) -> &BTreeMap<Slot, Expr> {
        &self.outputs
    }

    #[must_use]
    pub fn writes(&self) -> &[MemWrite] {
        &self.writes
    }

    /// Every register read by any output/write must be a declared input slot of
    /// the same width.
    ///
    /// # Errors
    ///
    /// [`ModelError::UndeclaredInput`] / [`ModelError::WidthMismatch`].
    pub fn validate(&self) -> Result<(), ModelError> {
        let exprs = self
            .outputs
            .values()
            .chain(self.writes.iter().flat_map(|w| [&w.addr, &w.value]));
        for e in exprs {
            let regs = registers(e).ok_or_else(|| ModelError::NotConcrete("width clash".into()))?;
            for (name, width) in regs {
                let slot =
                    Slot::parse(&name).ok_or_else(|| ModelError::UndeclaredInput(name.clone()))?;
                match self.inputs.get(&slot) {
                    None => return Err(ModelError::UndeclaredInput(name)),
                    Some(w) if *w != width => return Err(ModelError::WidthMismatch(slot)),
                    Some(_) => {}
                }
            }
        }
        Ok(())
    }

    /// Runs the handler on a concrete state.
    ///
    /// # Errors
    ///
    /// [`ModelError::MissingInput`] or [`ModelError::NotConcrete`]; never a guess.
    pub fn apply(&self, state: &ConcreteState) -> Result<ConcreteOut, ModelError> {
        self.validate()?;
        let mut env = BTreeMap::new();
        for (slot, width) in &self.inputs {
            let v = state
                .values
                .get(slot)
                .ok_or_else(|| ModelError::MissingInput(slot.clone()))?;
            env.insert(slot.var_name(), *v & mask(*width));
        }
        let ev = |e: &Expr, what: String| {
            eval_concrete(e, &env, &state.memory).ok_or(ModelError::NotConcrete(what))
        };
        let outputs = self
            .outputs
            .iter()
            .map(|(s, e)| Ok((s.clone(), ev(e, format!("{s:?}"))?)))
            .collect::<Result<_, ModelError>>()?;
        let writes = self
            .writes
            .iter()
            .map(|w| {
                Ok((
                    ev(&w.addr, "write address".into())?,
                    ev(&w.value, "write value".into())?,
                    w.value.width,
                ))
            })
            .collect::<Result<_, ModelError>>()?;
        Ok(ConcreteOut {
            outputs,
            sp_delta: self.sp_delta,
            writes,
        })
    }

    /// Copy with every output/write expression `Expr`-simplified and, where it is
    /// a linear MBA, replaced by its signature-proven canonical form. Anything
    /// outside the linear class is kept as-is.
    #[must_use]
    pub fn simplify_mba(&self) -> Self {
        let s = |e: &Expr| {
            let e = e.simplify();
            simplify_linear(&e).map_or(e, |r| r.expr)
        };
        let mut out = self.clone();
        out.outputs = self.outputs.iter().map(|(k, e)| (k.clone(), s(e))).collect();
        out.writes = self
            .writes
            .iter()
            .map(|w| MemWrite {
                addr: s(&w.addr),
                value: s(&w.value),
            })
            .collect();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::BinaryOp;

    const W: BitWidth = BitWidth::W64;

    fn b(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }

    /// Stack-VM `add`: pops s:0, s:8; pushes the sum at s:8; VSP += 8.
    fn vadd(mba: bool) -> HandlerModel {
        let mut m = HandlerModel::new("vadd")
            .with_input(Slot::Stack(0), W)
            .with_input(Slot::Stack(8), W);
        let (x, y) = (m.input(&Slot::Stack(0)).unwrap(), m.input(&Slot::Stack(8)).unwrap());
        let sum = if mba {
            b(
                BinaryOp::Add,
                b(BinaryOp::Xor, x.clone(), y.clone()),
                b(BinaryOp::Mul, Expr::constant(W, 2), b(BinaryOp::And, x, y)),
            )
        } else {
            b(BinaryOp::Add, x, y)
        };
        m.set_output(Slot::Stack(8), sum).unwrap();
        m.set_sp_delta(8);
        m
    }

    fn state(vals: &[(Slot, u64)]) -> ConcreteState {
        ConcreteState {
            values: vals.iter().cloned().collect(),
            memory: BTreeMap::new(),
        }
    }

    #[test]
    fn simple_handler_round_trips_to_reference() {
        let m = vadd(false);
        m.validate().unwrap();
        let out = m
            .apply(&state(&[(Slot::Stack(0), 40), (Slot::Stack(8), u64::MAX - 1)]))
            .unwrap();
        assert_eq!(out.outputs[&Slot::Stack(8)], 40_u64.wrapping_add(u64::MAX - 1));
        assert_eq!(out.sp_delta, 8);
        assert!(out.writes.is_empty());
        assert_eq!(Slot::parse(&Slot::Stack(-16).var_name()), Some(Slot::Stack(-16)));
    }

    #[test]
    fn mba_mutated_handler_simplifies_to_the_same_function() {
        let (plain, mutated) = (vadd(false), vadd(true));
        let simp = mutated.simplify_mba();
        assert!(
            simp.outputs()[&Slot::Stack(8)].node_count_bounded(100)
                < mutated.outputs()[&Slot::Stack(8)].node_count_bounded(100)
        );
        for (a, c) in [(0, 0), (1, u64::MAX), (0xDEAD_BEEF, 0x1234_5678_9ABC), (u64::MAX, u64::MAX)] {
            let st = state(&[(Slot::Stack(0), a), (Slot::Stack(8), c)]);
            let want = plain.apply(&st).unwrap();
            assert_eq!(mutated.apply(&st).unwrap(), want);
            assert_eq!(simp.apply(&st).unwrap(), want);
        }
    }

    #[test]
    fn store_and_load_effects_are_explicit_outputs() {
        // vstore: mem[s:0] = s:8 ; VSP += 16.  vload: s:0 = mem[s:0].
        let mut st = HandlerModel::new("vstore")
            .with_input(Slot::Stack(0), W)
            .with_input(Slot::Stack(8), W);
        st.push_write(st.input(&Slot::Stack(0)).unwrap(), st.input(&Slot::Stack(8)).unwrap())
            .unwrap();
        st.set_sp_delta(16);
        let out = st
            .apply(&state(&[(Slot::Stack(0), 0x1000), (Slot::Stack(8), 7)]))
            .unwrap();
        assert_eq!(out.writes, vec![(0x1000, 7, W)]);

        let mut ld = HandlerModel::new("vload").with_input(Slot::Stack(0), W);
        let load = Expr::load(W, ld.input(&Slot::Stack(0)).unwrap());
        ld.set_output(Slot::Stack(0), load).unwrap();
        let mut s = state(&[(Slot::Stack(0), 0x2000)]);
        for i in 0..8_u64 {
            s.memory.insert(0x2000 + i, (i + 1) as u8);
        }
        assert_eq!(
            ld.apply(&s).unwrap().outputs[&Slot::Stack(0)],
            0x0807_0605_0403_0201
        );
        s.memory.remove(&0x2003);
        assert!(matches!(ld.apply(&s), Err(ModelError::NotConcrete(_))));
    }

    #[test]
    fn fails_closed_on_undeclared_missing_and_opaque() {
        let mut m = HandlerModel::new("bad");
        m.set_output(Slot::Reg("rax".into()), Expr::register(W, "r:rbx")).unwrap();
        assert!(matches!(m.validate(), Err(ModelError::UndeclaredInput(_))));

        let mut m = HandlerModel::new("mismatch").with_input(Slot::Reg("rax".into()), W);
        assert!(matches!(
            m.set_output(Slot::Reg("rax".into()), Expr::constant(BitWidth::W8, 1)),
            Err(ModelError::WidthMismatch(_))
        ));

        let m = vadd(false);
        assert!(matches!(
            m.apply(&state(&[(Slot::Stack(0), 1)])),
            Err(ModelError::MissingInput(_))
        ));

        let mut f = HandlerModel::new("flags").with_input(Slot::Stack(0), W);
        let opaque = Expr::intrinsic(BitWidth::W8, "pf", vec![f.input(&Slot::Stack(0)).unwrap()]);
        f.set_output(Slot::Flag("pf".into()), opaque).unwrap();
        assert!(matches!(
            f.apply(&state(&[(Slot::Stack(0), 3)])),
            Err(ModelError::NotConcrete(_))
        ));
    }
}
