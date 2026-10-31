//! Cross-build handler dictionary: recognize a handler already solved in
//! another build, even when mutated or its registers relabeled.
//!
//! Two cheap keys nominate a *candidate*; neither is ever trusted:
//! * Key A, [`io_fingerprint`]: hash of the model's outputs on a fixed seeded
//!   input battery (robust to reorder/rename/MBA mutation).
//! * Key B, canonical hash: general-MBA simplify, role-normalize register
//!   slots, canonically order commutative operands, hash (exact hit only).
//!
//! A candidate is returned as [`Lookup::Recognized`] only after the
//! [`PromotionGate`] machinery proves the *original* query equals the stored
//! entry (structural / signature theorem / optional SMT backend). Refuted
//! candidates are hash collisions: logged, never recognized. Anything the gate
//! cannot prove stays [`Lookup::Unconfirmed`] (Observed at best).

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind};

use crate::bvutil::{eval_concrete, mask, registers, walk};
use crate::gate::{EquivalenceBackend, NoSolver, PromotionGate, Tier};
use crate::general_mba::{certify_general, simplify_general};
use crate::sampling::{next_rand, sample_assignments};
use crate::state_model::{ConcreteState, HandlerModel, Slot};

const BATTERY_SEED: u64 = 0x5EED_D1C7_10AA_0001;
const BATTERY_RANDOM: usize = 16;
const GATE_RANDOM: usize = 32;
const MAX_INPUTS: usize = 16;
const EMIT_BUDGET: usize = 1 << 15;
const MAX_CONFIRMS: usize = 8;
const MAX_COLLISION_LOG: usize = 256;
const MAX_BUILDS_PER_ENTRY: usize = 256;
const DEFAULT_CAPACITY: usize = 1 << 16;

struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for x in b {
            self.0 = (self.0 ^ u64::from(*x)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn u(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn s(&mut self, s: &str) {
        self.u(s.len() as u64);
        self.bytes(s.as_bytes());
    }
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    /// Every free variable renamed by first use (expression-level key).
    Alpha,
    /// Only `r:` variables renamed; stack/flag slots are fixed roles.
    Align,
    /// Names taken literally (already role-normalized models).
    Literal,
}

struct Canon {
    mode: Mode,
    idx: BTreeMap<String, u32>,
    order: Vec<String>,
    shapes: HashMap<usize, u64>,
    budget: usize,
}

const fn commutative(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Mul | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or
    )
}

fn kids(e: &Expr) -> Vec<&Expr> {
    match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => vec![],
        ExprKind::Unary { value, .. }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => vec![value],
        ExprKind::Load { address } => vec![address],
        ExprKind::Binary { left, right, .. } => vec![left, right],
        ExprKind::Concat { high, low } => vec![high, low],
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().collect(),
    }
}

fn is_commutative(e: &Expr) -> bool {
    matches!(&e.kind, ExprKind::Binary { op, .. } if commutative(*op))
}

fn head(h: &mut Fnv, e: &Expr) {
    h.u(u64::from(e.width.bits()));
    match &e.kind {
        ExprKind::Const { value } => {
            h.bytes(b"c");
            h.u(*value & mask(e.width));
        }
        ExprKind::Register { .. } => h.bytes(b"v"),
        ExprKind::Unary { op, .. } => {
            h.bytes(b"u");
            h.s(&format!("{op:?}"));
        }
        ExprKind::Binary { op, .. } => {
            h.bytes(b"b");
            h.s(&format!("{op:?}"));
        }
        ExprKind::Load { .. } => h.bytes(b"l"),
        ExprKind::Intrinsic { name, arguments } => {
            h.bytes(b"i");
            h.s(name);
            h.u(arguments.len() as u64);
        }
        ExprKind::Extract { lsb, .. } => {
            h.bytes(b"x");
            h.u(u64::from(*lsb));
        }
        ExprKind::Concat { .. } => h.bytes(b"k"),
        ExprKind::Resize { op, .. } => {
            h.bytes(b"z");
            h.s(&format!("{op:?}"));
        }
    }
}

impl Canon {
    fn new(mode: Mode) -> Self {
        Self {
            mode,
            idx: BTreeMap::new(),
            order: Vec::new(),
            shapes: HashMap::new(),
            budget: EMIT_BUDGET,
        }
    }

    fn renamed(&self, var: &str) -> bool {
        match self.mode {
            Mode::Alpha => true,
            Mode::Align => var.starts_with("r:"),
            Mode::Literal => false,
        }
    }

    fn touch(&mut self, var: &str) -> u32 {
        if let Some(i) = self.idx.get(var) {
            return *i;
        }
        let i = self.order.len() as u32;
        self.idx.insert(var.to_owned(), i);
        self.order.push(var.to_owned());
        i
    }

    /// Name-blind (for renamed vars) Merkle hash used only to order operands.
    fn shape(&mut self, e: &Expr) -> u64 {
        let key = std::ptr::from_ref(e) as usize;
        if let Some(s) = self.shapes.get(&key) {
            return *s;
        }
        let mut h = Fnv::new();
        head(&mut h, e);
        if let ExprKind::Register { name } = &e.kind {
            if !self.renamed(name) {
                h.s(name);
            }
        } else {
            let mut ks: Vec<u64> = kids(e).into_iter().map(|k| self.shape(k)).collect();
            if is_commutative(e) {
                ks.sort_unstable();
            }
            for k in ks {
                h.u(k);
            }
        }
        let s = h.finish();
        self.shapes.insert(key, s);
        s
    }

    /// Emits the canonical token stream; `false` when over budget.
    fn emit(&mut self, e: &Expr, h: &mut Fnv) -> bool {
        if self.budget == 0 {
            return false;
        }
        self.budget -= 1;
        head(h, e);
        if let ExprKind::Register { name } = &e.kind {
            if self.renamed(name) {
                let i = self.touch(name);
                h.u(u64::from(i));
            } else {
                h.s(name);
            }
            return true;
        }
        let mut ks = kids(e);
        if is_commutative(e) && ks.len() == 2 && self.shape(ks[1]) < self.shape(ks[0]) {
            ks.swap(0, 1);
        }
        ks.into_iter().all(|k| self.emit(k, h))
    }

    fn hash_expr(&mut self, e: &Expr) -> Option<u64> {
        let mut h = Fnv::new();
        self.emit(e, &mut h).then(|| h.finish())
    }
}

/// Key B for a single expression: general-MBA simplify, alpha-rename free
/// variables by first use, canonically order commutative operands, hash.
/// `None` when the expression is too large to canonicalize.
#[must_use]
pub fn canonical_hash(e: &Expr) -> Option<u64> {
    let s = simplify_general(&e.simplify()).expr;
    Canon::new(Mode::Alpha).hash_expr(&s)
}

fn model_exprs(m: &HandlerModel) -> impl Iterator<Item = &Expr> {
    m.outputs()
        .values()
        .chain(m.writes().iter().flat_map(|w| [&w.addr, &w.value]))
}

fn pseudo_byte(addr: u64) -> u8 {
    let mut s = addr;
    next_rand(&mut s) as u8
}

/// Deterministic memory covering every load the battery point reaches.
fn fill_memory(m: &HandlerModel, env: &BTreeMap<String, u64>) -> BTreeMap<u64, u8> {
    let mut loads: Vec<Expr> = Vec::new();
    for e in model_exprs(m) {
        walk(e, &mut |n| {
            if matches!(n.kind, ExprKind::Load { .. }) && loads.len() < 64 {
                loads.push(n.clone());
            }
        });
    }
    let mut mem = BTreeMap::new();
    for _ in 0..=loads.len().min(8) {
        let mut grew = false;
        for l in &loads {
            let ExprKind::Load { address } = &l.kind else {
                continue;
            };
            if let Some(a) = eval_concrete(address, env, &mem) {
                for i in 0..u64::from(l.width.bytes()) {
                    if let Entry::Vacant(v) = mem.entry(a.wrapping_add(i)) {
                        v.insert(pseudo_byte(a.wrapping_add(i)));
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            break;
        }
    }
    mem
}

fn input_vars(m: &HandlerModel) -> BTreeMap<String, BitWidth> {
    m.inputs().iter().map(|(s, w)| (s.var_name(), *w)).collect()
}

/// Key A: hash of the handler's concrete effect (outputs, `sp_delta`, writes)
/// over a fixed deterministic battery ({0,1,-1,min,max} cross/sweeps plus
/// seeded random). Slot names are hashed, so compare role-normalized models.
/// `None` on any modeling gap (unevaluable point, too many inputs).
#[must_use]
pub fn io_fingerprint(m: &HandlerModel) -> Option<u64> {
    if m.inputs().len() > MAX_INPUTS {
        return None;
    }
    let mut h = Fnv::new();
    for (s, w) in m.inputs() {
        h.s(&s.var_name());
        h.u(u64::from(w.bits()));
    }
    for env in sample_assignments(&input_vars(m), BATTERY_RANDOM, BATTERY_SEED) {
        let mut st = ConcreteState {
            memory: fill_memory(m, &env),
            ..ConcreteState::default()
        };
        for (n, v) in &env {
            st.values.insert(Slot::parse(n)?, *v);
        }
        let out = m.apply(&st).ok()?;
        for (s, v) in &out.outputs {
            h.s(&s.var_name());
            h.u(*v);
        }
        h.u(out.sp_delta as u64);
        h.u(out.writes.len() as u64);
        for (a, v, w) in &out.writes {
            h.u(*a);
            h.u(*v);
            h.u(u64::from(w.bits()));
        }
    }
    Some(h.finish())
}

fn model_canon_hash(m: &HandlerModel) -> Option<u64> {
    let mut c = Canon::new(Mode::Literal);
    let mut h = Fnv::new();
    for (s, w) in m.inputs() {
        h.s(&s.var_name());
        h.u(u64::from(w.bits()));
    }
    h.u(m.sp_delta() as u64);
    for (s, e) in m.outputs() {
        h.s(&s.var_name());
        h.u(c.hash_expr(e)?);
    }
    h.u(m.writes().len() as u64);
    for w in m.writes() {
        h.u(c.hash_expr(&w.addr)?);
        h.u(c.hash_expr(&w.value)?);
    }
    Some(h.finish())
}

fn map_model(m: &HandlerModel, f: &dyn Fn(&Expr) -> Option<Expr>) -> Option<HandlerModel> {
    let mut o = HandlerModel::new(m.name.clone());
    for (s, w) in m.inputs() {
        o = o.with_input(s.clone(), *w);
    }
    for (s, e) in m.outputs() {
        o.set_output(s.clone(), f(e)?).ok()?;
    }
    for w in m.writes() {
        o.push_write(f(&w.addr)?, f(&w.value)?).ok()?;
    }
    o.set_sp_delta(m.sp_delta());
    Some(o)
}

/// `reg_map`: original register name (no `r:`) -> role-normalized name.
fn rename_model(m: &HandlerModel, reg_map: &BTreeMap<String, String>) -> Option<HandlerModel> {
    let slot = |s: &Slot| match s {
        Slot::Reg(n) => Slot::Reg(reg_map.get(n).cloned().unwrap_or_else(|| n.clone())),
        other => other.clone(),
    };
    let sub = |e: &Expr| -> Option<Expr> {
        let mut rep = BTreeMap::new();
        for (name, w) in registers(e)? {
            if let Some(Slot::Reg(n)) = Slot::parse(&name)
                && let Some(new) = reg_map.get(&n)
            {
                rep.insert(name, Expr::register(w, Slot::Reg(new.clone()).var_name()));
            }
        }
        if rep.is_empty() {
            Some(e.clone())
        } else {
            e.substitute(&rep).ok()
        }
    };
    let mut o = HandlerModel::new(m.name.clone());
    for (s, w) in m.inputs() {
        o = o.with_input(slot(s), *w);
    }
    for (s, e) in m.outputs() {
        o.set_output(slot(s), sub(e)?).ok()?;
    }
    for w in m.writes() {
        o.push_write(sub(&w.addr)?, sub(&w.value)?).ok()?;
    }
    o.set_sp_delta(m.sp_delta());
    Some(o)
}

/// A query handler in role-normalized form. `aligned` is the caller's model
/// untouched except for register relabeling (what the gate must be proven
/// against); `simplified` is its simplification (same slots); `keyed` is the
/// pruned simplified form that both keys and stored entries are built from.
struct Prepared {
    aligned: HandlerModel,
    simplified: HandlerModel,
    keyed: HandlerModel,
    alignment: BTreeMap<Slot, Slot>,
}

fn prepare(m: &HandlerModel) -> Option<Prepared> {
    m.validate().ok()?;
    let simp = map_model(m, &|e| Some(simplify_general(&e.simplify()).expr))?;

    let mut pruned = HandlerModel::new(m.name.clone());
    let live: BTreeMap<&Slot, &Expr> = simp
        .outputs()
        .iter()
        .filter(|(s, e)| **e != Expr::register(e.width, s.var_name()))
        .collect();
    let mut used: BTreeMap<String, BitWidth> = BTreeMap::new();
    for e in live.values().copied().chain(simp.writes().iter().flat_map(|w| [&w.addr, &w.value])) {
        used.extend(registers(e)?);
    }
    for (n, w) in &used {
        pruned = pruned.with_input(Slot::parse(n)?, *w);
    }
    for (s, e) in &live {
        pruned.set_output((*s).clone(), (*e).clone()).ok()?;
    }
    for w in simp.writes() {
        pruned.push_write(w.addr.clone(), w.value.clone()).ok()?;
    }
    pruned.set_sp_delta(simp.sp_delta());

    // Role alignment of `r:` slots by first use in a name-blind canonical walk.
    let mut c = Canon::new(Mode::Align);
    let mut outs: Vec<(Slot, &Expr)> = pruned.outputs().iter().map(|(s, e)| (s.clone(), e)).collect();
    let mut keyed_outs = Vec::with_capacity(outs.len());
    for (s, e) in outs.drain(..) {
        let k = match &s {
            Slot::Stack(o) => (0_u8, *o, 0_u64, String::new()),
            Slot::Flag(n) => (1, 0, 0, n.clone()),
            Slot::Reg(n) => (2, i64::from(e.width.bits()), c.shape(e), n.clone()),
        };
        keyed_outs.push((k, s, e));
    }
    keyed_outs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut sink = Fnv::new();
    for (_, s, e) in &keyed_outs {
        if matches!(s, Slot::Reg(_)) {
            c.touch(&s.var_name());
        }
        if !c.emit(e, &mut sink) {
            return None;
        }
    }
    for w in pruned.writes() {
        if !c.emit(&w.addr, &mut sink) || !c.emit(&w.value, &mut sink) {
            return None;
        }
    }
    let mut all_regs: BTreeSet<String> = BTreeSet::new();
    let mut add = |s: &Slot| {
        if let Slot::Reg(n) = s {
            all_regs.insert(n.clone());
        }
    };
    m.inputs().keys().for_each(&mut add);
    m.outputs().keys().for_each(&mut add);
    for e in model_exprs(m) {
        for name in registers(e)?.keys() {
            Slot::parse(name).iter().for_each(&mut add);
        }
    }
    let mut ordered: Vec<String> = c
        .order
        .iter()
        .filter_map(|v| match Slot::parse(v) {
            Some(Slot::Reg(n)) => Some(n),
            _ => None,
        })
        .collect();
    let rest: Vec<String> = all_regs.iter().filter(|n| !ordered.contains(n)).cloned().collect();
    ordered.extend(rest);
    let reg_map: BTreeMap<String, String> = ordered
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, format!("#{i}")))
        .collect();
    let alignment = reg_map
        .iter()
        .map(|(o, n)| (Slot::Reg(o.clone()), Slot::Reg(n.clone())))
        .collect();

    Some(Prepared {
        aligned: rename_model(m, &reg_map)?,
        simplified: rename_model(&simp, &reg_map)?,
        keyed: rename_model(&pruned, &reg_map)?,
        alignment,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Key {
    CanonicalHash,
    IoFingerprint,
}

#[derive(Clone, Debug)]
pub struct DictEntry {
    /// Role-normalized, simplified solved handler.
    pub model: HandlerModel,
    pub io_fingerprint: u64,
    pub canonical_hash: u64,
    pub builds: BTreeSet<String>,
    /// Observed or Proven: confidence in the stored handler itself.
    pub status: Tier,
}

impl DictEntry {
    #[must_use]
    pub fn widths(&self) -> BTreeMap<Slot, BitWidth> {
        self.model.inputs().clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Lookup {
    /// The gate proved the query equals the stored entry (equivalence is
    /// `Proven`; `entry_status` is the stored entry's own tier).
    Recognized {
        entry: usize,
        via: Key,
        entry_status: Tier,
        /// Query register slot -> entry slot (stack/flag slots are identity).
        alignment: BTreeMap<Slot, Slot>,
    },
    /// Keys hit but equality is unproven (Observed/Unproven). Not a result.
    Unconfirmed { entry: usize, via: Key, tier: Tier },
    Miss,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Collision {
    pub entry: usize,
    pub via: Key,
}

#[derive(Debug, Eq, PartialEq)]
pub enum DictError {
    /// Only Observed/Proven handlers are stored.
    BadStatus(Tier),
    /// Modeling gap: no keys can be computed.
    Unmodelable,
    /// The stored form was refuted against the supplied handler.
    Inconsistent,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Inserted {
    pub index: usize,
    pub new: bool,
    pub status: Tier,
}

pub struct HandlerDictionary<B: EquivalenceBackend + Clone = NoSolver> {
    entries: Vec<DictEntry>,
    by_canon: HashMap<u64, Vec<usize>>,
    by_io: HashMap<u64, Vec<usize>>,
    backend: B,
    capacity: usize,
    collisions: Vec<Collision>,
    collision_count: u64,
}

impl Default for HandlerDictionary<NoSolver> {
    fn default() -> Self {
        Self::new(NoSolver)
    }
}

impl<B: EquivalenceBackend + Clone> HandlerDictionary<B> {
    #[must_use]
    pub fn new(backend: B) -> Self {
        Self {
            entries: Vec::new(),
            by_canon: HashMap::new(),
            by_io: HashMap::new(),
            backend,
            capacity: DEFAULT_CAPACITY,
            collisions: Vec::new(),
            collision_count: 0,
        }
    }

    #[must_use]
    pub fn with_capacity_limit(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entry(&self, i: usize) -> Option<&DictEntry> {
        self.entries.get(i)
    }

    #[must_use]
    pub const fn collision_count(&self) -> u64 {
        self.collision_count
    }

    /// Most recent collisions (bounded log).
    #[must_use]
    pub fn collisions(&self) -> &[Collision] {
        &self.collisions
    }

    /// Stores a solved handler. The stored status is capped by what the gate
    /// can re-prove about the normalized form against `model` itself.
    ///
    /// # Errors
    ///
    /// See [`DictError`].
    pub fn insert(
        &mut self,
        model: &HandlerModel,
        build: &str,
        status: Tier,
    ) -> Result<Inserted, DictError> {
        if !matches!(status, Tier::Observed | Tier::Proven) {
            return Err(DictError::BadStatus(status));
        }
        let p = prepare(model).ok_or(DictError::Unmodelable)?;
        let (Some(canon), Some(io)) = (model_canon_hash(&p.keyed), io_fingerprint(&p.keyed)) else {
            return Err(DictError::Unmodelable);
        };
        let verified = self.confirm(&p.keyed, &p);
        if verified == Tier::Refuted {
            return Err(DictError::Inconsistent);
        }
        self.insert_keyed(p.keyed, canon, io, build, status.min(verified.max(Tier::Observed)))
    }

    fn insert_keyed(
        &mut self,
        model: HandlerModel,
        canon: u64,
        io: u64,
        build: &str,
        status: Tier,
    ) -> Result<Inserted, DictError> {
        let dup = self
            .by_canon
            .get(&canon)
            .into_iter()
            .flatten()
            .copied()
            .find(|i| self.entries[*i].model == model);
        if let Some(i) = dup {
            let e = &mut self.entries[i];
            if e.builds.len() < MAX_BUILDS_PER_ENTRY {
                e.builds.insert(build.to_owned());
            }
            e.status = e.status.max(status);
            return Ok(Inserted { index: i, new: false, status: e.status });
        }
        if self.entries.len() >= self.capacity {
            return Err(DictError::Full);
        }
        let index = self.entries.len();
        self.entries.push(DictEntry {
            model,
            io_fingerprint: io,
            canonical_hash: canon,
            builds: BTreeSet::from([build.to_owned()]),
            status,
        });
        self.by_canon.entry(canon).or_default().push(index);
        self.by_io.entry(io).or_default().push(index);
        Ok(Inserted { index, new: true, status })
    }

    /// Recognizes `model` against stored entries. Hash hits only nominate
    /// candidates; each is gate-confirmed against the original query.
    pub fn lookup(&mut self, model: &HandlerModel, build: &str) -> Lookup {
        let Some(p) = prepare(model) else {
            return Lookup::Miss;
        };
        let mut cands: Vec<(usize, Key)> = Vec::new();
        let push = |ids: Option<&Vec<usize>>, key: Key, cands: &mut Vec<(usize, Key)>| {
            for i in ids.into_iter().flatten() {
                if !cands.iter().any(|(j, _)| j == i) {
                    cands.push((*i, key));
                }
            }
        };
        if let Some(k) = model_canon_hash(&p.keyed) {
            push(self.by_canon.get(&k), Key::CanonicalHash, &mut cands);
        }
        if let Some(k) = io_fingerprint(&p.keyed) {
            push(self.by_io.get(&k), Key::IoFingerprint, &mut cands);
        }
        cands.truncate(MAX_CONFIRMS);

        let mut best: Option<(usize, Key, Tier)> = None;
        for (i, via) in cands {
            match self.confirm(&self.entries[i].model, &p) {
                Tier::Proven => {
                    let e = &mut self.entries[i];
                    if e.builds.len() < MAX_BUILDS_PER_ENTRY {
                        e.builds.insert(build.to_owned());
                    }
                    return Lookup::Recognized {
                        entry: i,
                        via,
                        entry_status: e.status,
                        alignment: p.alignment,
                    };
                }
                Tier::Refuted => {
                    self.collision_count += 1;
                    if self.collisions.len() == MAX_COLLISION_LOG {
                        self.collisions.remove(0);
                    }
                    self.collisions.push(Collision { entry: i, via });
                }
                t => {
                    if best.is_none_or(|(_, _, bt)| t > bt) {
                        best = Some((i, via, t));
                    }
                }
            }
        }
        best.map_or(Lookup::Miss, |(entry, via, tier)| Lookup::Unconfirmed { entry, via, tier })
    }

    /// Tier of "stored `entry` == query", proven against the *original* query
    /// (directly, or via a certified simplification of it). Fresh gate per call
    /// so tests never accumulate.
    fn confirm(&self, entry: &HandlerModel, q: &Prepared) -> Tier {
        let (qa, qs) = (&q.aligned, &q.simplified);
        if entry.sp_delta() != qa.sp_delta() || entry.writes().len() != qa.writes().len() {
            return Tier::Refuted;
        }
        let mut vars = input_vars(entry);
        for (n, w) in input_vars(qa) {
            if *vars.entry(n).or_insert(w) != w {
                return Tier::Unproven;
            }
        }
        let mut gate = PromotionGate::with_backend(self.backend.clone());
        for t in sample_assignments(&vars, GATE_RANDOM, BATTERY_SEED) {
            gate.add_test(t);
        }
        let width_of = |s: &Slot| {
            entry
                .inputs()
                .get(s)
                .or_else(|| qa.inputs().get(s))
                .copied()
                .or_else(|| entry.outputs().get(s).or_else(|| qa.outputs().get(s)).map(|e| e.width))
        };
        let side = |m: &HandlerModel, s: &Slot| {
            m.outputs()
                .get(s)
                .cloned()
                .or_else(|| width_of(s).map(|w| Expr::register(w, s.var_name())))
        };
        let mut tier = Tier::Proven;
        let slots: BTreeSet<&Slot> = entry.outputs().keys().chain(qa.outputs().keys()).collect();
        for s in slots {
            let (Some(x), Some(y), Some(ys)) = (side(entry, s), side(qa, s), side(qs, s)) else {
                return Tier::Unproven;
            };
            tier = tier.min(prove(&mut gate, &x, &y, &ys));
        }
        for (x, y) in entry.writes().iter().zip(qa.writes()) {
            tier = tier.min(prove(&mut gate, &x.addr, &y.addr, &y.addr));
            tier = tier.min(prove(&mut gate, &x.value, &y.value, &y.value));
        }
        tier
    }
}

/// `x` (stored) vs `y` (original query), with `ys` a simplification of `y`.
fn prove<B: EquivalenceBackend>(gate: &mut PromotionGate<B>, x: &Expr, y: &Expr, ys: &Expr) -> Tier {
    let direct = gate.judge(x, y).tier;
    if matches!(direct, Tier::Proven | Tier::Refuted) || ys == y || !certify_general(y, ys) {
        return direct;
    }
    direct.max(gate.judge(x, ys).tier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::BinaryOp as Op;

    const W: BitWidth = BitWidth::W64;

    fn b(op: Op, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn k(v: u64) -> Expr {
        Expr::constant(W, v)
    }

    /// Stack-VM add: s:8 = s:0 + s:8; VSP += 8. `mba`: (x^y) + 2*(x&y).
    fn vadd(mba: bool) -> HandlerModel {
        let mut m = HandlerModel::new("vadd")
            .with_input(Slot::Stack(0), W)
            .with_input(Slot::Stack(8), W);
        let (x, y) = (m.input(&Slot::Stack(0)).unwrap(), m.input(&Slot::Stack(8)).unwrap());
        let sum = if mba {
            b(Op::Add, b(Op::Xor, x.clone(), y.clone()), b(Op::Mul, k(2), b(Op::And, x, y)))
        } else {
            b(Op::Add, x, y)
        };
        m.set_output(Slot::Stack(8), sum).unwrap();
        m.set_sp_delta(8);
        m
    }

    fn vsub() -> HandlerModel {
        let mut m = HandlerModel::new("vsub")
            .with_input(Slot::Stack(0), W)
            .with_input(Slot::Stack(8), W);
        let (x, y) = (m.input(&Slot::Stack(0)).unwrap(), m.input(&Slot::Stack(8)).unwrap());
        m.set_output(Slot::Stack(8), b(Op::Sub, x, y)).unwrap();
        m.set_sp_delta(8);
        m
    }

    /// reg handler `<dst> = <a> - 3*<b>` with caller-chosen register names.
    fn regsub(dst: &str, a: &str, bn: &str) -> HandlerModel {
        let (sa, sb) = (Slot::Reg(a.into()), Slot::Reg(bn.into()));
        let mut m = HandlerModel::new("regsub").with_input(sa.clone(), W).with_input(sb.clone(), W);
        let (x, y) = (m.input(&sa).unwrap(), m.input(&sb).unwrap());
        let m3 = b(Op::Mul, y, k(3));
        m.set_output(Slot::Reg(dst.into()), b(Op::Sub, x, m3)).unwrap();
        m
    }

    #[test]
    fn mutated_equivalent_handler_shares_fingerprint_and_is_recognized_without_resolving() {
        let (plain, mutated) = (vadd(false), vadd(true));
        assert_eq!(io_fingerprint(&plain), io_fingerprint(&mutated));
        assert!(io_fingerprint(&plain).is_some());
        assert_ne!(io_fingerprint(&plain), io_fingerprint(&vsub()));

        let mut d = HandlerDictionary::default();
        d.insert(&plain, "build-1", Tier::Proven).unwrap();
        match d.lookup(&mutated, "build-2") {
            Lookup::Recognized { entry, entry_status, .. } => {
                assert_eq!(entry, 0);
                assert_eq!(entry_status, Tier::Proven);
                assert_eq!(d.entry(0).unwrap().builds.len(), 2);
            }
            other => panic!("expected recognition, got {other:?}"),
        }
    }

    #[test]
    fn io_key_alone_nominates_and_gate_confirms() {
        let (plain, mutated) = (vadd(false), vadd(true));
        let p = prepare(&plain).unwrap();
        let mut d = HandlerDictionary::default();
        // Forged canonical hash: Key B misses, Key A must carry the lookup.
        let io = io_fingerprint(&p.keyed).unwrap();
        d.insert_keyed(p.keyed, 0xDEAD, io, "b1", Tier::Proven).unwrap();
        match d.lookup(&mutated, "b2") {
            Lookup::Recognized { via, .. } => assert_eq!(via, Key::IoFingerprint),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn forced_collision_is_rejected_by_the_gate() {
        let (add, sub) = (vadd(false), vsub());
        let (pa, ps) = (prepare(&add).unwrap(), prepare(&sub).unwrap());
        assert_ne!(model_canon_hash(&pa.keyed), model_canon_hash(&ps.keyed));
        let mut d = HandlerDictionary::default();
        // Store `add` under `sub`'s exact keys.
        let (canon, io) = (model_canon_hash(&ps.keyed).unwrap(), io_fingerprint(&ps.keyed).unwrap());
        d.insert_keyed(pa.keyed, canon, io, "b1", Tier::Proven).unwrap();
        assert_eq!(d.lookup(&sub, "b2"), Lookup::Miss);
        assert_eq!(d.collision_count(), 1);
        assert!(!d.collisions().is_empty());
        assert_eq!(d.entry(0).unwrap().builds.len(), 1);
    }

    #[test]
    fn canonical_hash_hit_recognizes_renamed_register_variant() {
        let mut d = HandlerDictionary::default();
        d.insert(&regsub("rax", "rax", "rbx"), "b1", Tier::Proven).unwrap();
        // Different register names, different sort order.
        let variant = regsub("zz", "zz", "aa");
        match d.lookup(&variant, "b2") {
            Lookup::Recognized { via, alignment, .. } => {
                assert_eq!(via, Key::CanonicalHash);
                assert_eq!(alignment[&Slot::Reg("zz".into())], Slot::Reg("#0".into()));
                assert_eq!(alignment[&Slot::Reg("aa".into())], Slot::Reg("#1".into()));
            }
            other => panic!("{other:?}"),
        }
        // Swapped operand roles are a different function.
        let swapped = {
            let (sa, sb) = (Slot::Reg("p".into()), Slot::Reg("q".into()));
            let mut m = HandlerModel::new("sw").with_input(sa.clone(), W).with_input(sb.clone(), W);
            let (x, y) = (m.input(&sa).unwrap(), m.input(&sb).unwrap());
            m.set_output(sa, b(Op::Sub, y, b(Op::Mul, x, k(3)))).unwrap();
            m
        };
        assert!(!matches!(d.lookup(&swapped, "b3"), Lookup::Recognized { .. }));
    }

    #[test]
    fn expression_canonical_hash_is_alpha_and_commutation_invariant() {
        let v = |n: &str| Expr::register(W, n);
        let e1 = b(Op::Add, v("x"), b(Op::Mul, v("y"), k(3)));
        let e2 = b(Op::Add, b(Op::Mul, k(3), v("b")), v("a"));
        assert_eq!(canonical_hash(&e1), canonical_hash(&e2));
        assert_ne!(canonical_hash(&e1), canonical_hash(&b(Op::Sub, v("x"), b(Op::Mul, v("y"), k(3)))));
    }

    #[test]
    fn nonlinear_mutation_without_a_solver_stays_unconfirmed() {
        // x*y == (x|y)*(x&y) + (x&~y)*(~x&y): same function, outside the linear class.
        let mul = |mutated: bool| {
            let mut m = HandlerModel::new("vmul")
                .with_input(Slot::Stack(0), W)
                .with_input(Slot::Stack(8), W);
            let (x, y) = (m.input(&Slot::Stack(0)).unwrap(), m.input(&Slot::Stack(8)).unwrap());
            let e = if mutated {
                let nx = Expr::unary(vmpx_vm::UnaryOp::Not, x.clone());
                let ny = Expr::unary(vmpx_vm::UnaryOp::Not, y.clone());
                b(
                    Op::Add,
                    b(Op::Mul, b(Op::Or, x.clone(), y.clone()), b(Op::And, x.clone(), y.clone())),
                    b(Op::Mul, b(Op::And, x, ny), b(Op::And, nx, y)),
                )
            } else {
                b(Op::Mul, x, y)
            };
            m.set_output(Slot::Stack(8), e).unwrap();
            m.set_sp_delta(8);
            m
        };
        assert_eq!(io_fingerprint(&mul(false)), io_fingerprint(&mul(true)));
        let mut d = HandlerDictionary::default();
        d.insert(&mul(false), "b1", Tier::Proven).unwrap();
        match d.lookup(&mul(true), "b2") {
            Lookup::Recognized { .. } => panic!("must not be recognized without a proof"),
            Lookup::Unconfirmed { tier, .. } => assert_eq!(tier, Tier::Observed),
            Lookup::Miss => panic!("fingerprint should nominate"),
        }
    }

    #[test]
    fn insert_is_fail_closed_and_dedups() {
        let mut d = HandlerDictionary::default().with_capacity_limit(1);
        assert_eq!(d.insert(&vadd(false), "b", Tier::Refuted), Err(DictError::BadStatus(Tier::Refuted)));
        let a = d.insert(&vadd(false), "b1", Tier::Observed).unwrap();
        assert!(a.new);
        let again = d.insert(&vadd(true), "b2", Tier::Proven).unwrap();
        assert!(!again.new);
        assert_eq!(again.status, Tier::Proven);
        assert_eq!(d.insert(&vsub(), "b3", Tier::Proven), Err(DictError::Full));
        assert_eq!(d.len(), 1);

        let mut opaque = HandlerModel::new("op").with_input(Slot::Stack(0), W);
        let i = Expr::intrinsic(BitWidth::W8, "pf", vec![opaque.input(&Slot::Stack(0)).unwrap()]);
        opaque.set_output(Slot::Flag("pf".into()), i).unwrap();
        assert_eq!(d.insert(&opaque, "b", Tier::Proven), Err(DictError::Unmodelable));
        assert_eq!(io_fingerprint(&opaque), None);
    }
}
