//! Hand-off from a dynamic run to the static pipeline.
//!
//! Every record here is tagged [`Observed`]: it reflects ONE concrete execution
//! path, not a proof. The static side must label or quarantine these facts and
//! must NEVER treat them as sound output. The provenance marker makes that
//! obligation impossible to lose — a fact cannot be built without it.

/// Provenance marker carried by every observed record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Observed;

/// How an import was identified on the observed path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolOrOrdinal {
    Name(String),
    Ordinal(u16),
}

/// An import stub seen resolving/called during the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedImport {
    pub provenance: Observed,
    /// Stub sentinel VA or IAT slot the guest reached.
    pub stub_va_or_slot: u64,
    pub module: String,
    pub symbol: SymbolOrOrdinal,
}

/// A global whose concrete bytes were observed at a virtual address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedGlobal {
    pub provenance: Observed,
    pub va: u64,
    pub bytes: Vec<u8>,
}

/// An observed back-edge trip count for a loop head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservedLoopDepth {
    pub provenance: Observed,
    pub loop_head: u64,
    pub depth: u64,
}

/// Accumulated facts from one run. Collector methods stamp [`Observed`] so a
/// caller cannot record an un-provenanced fact; later WPs fill this from hooks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObservedFacts {
    pub imports: Vec<ObservedImport>,
    pub globals: Vec<ObservedGlobal>,
    pub loop_depths: Vec<ObservedLoopDepth>,
    pub oep: Option<u64>,
}

impl ObservedFacts {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_import(
        &mut self,
        stub_va_or_slot: u64,
        module: impl Into<String>,
        symbol: SymbolOrOrdinal,
    ) {
        self.imports.push(ObservedImport {
            provenance: Observed,
            stub_va_or_slot,
            module: module.into(),
            symbol,
        });
    }

    pub fn record_global(&mut self, va: u64, bytes: Vec<u8>) {
        self.globals.push(ObservedGlobal { provenance: Observed, va, bytes });
    }

    pub fn record_loop_depth(&mut self, loop_head: u64, depth: u64) {
        self.loop_depths.push(ObservedLoopDepth { provenance: Observed, loop_head, depth });
    }

    pub fn set_oep(&mut self, oep: u64) {
        self.oep = Some(oep);
    }
}
