//! SP7: emulation-first dynamic-assist engine (foundation).
//!
//! # Architecture
//!
//! The dynamic-unpack-and-observe flow runs a packed/protected sample's code
//! inside a *software* x64 interpreter, observes its runtime behaviour, dumps
//! the unpacked image and hands it to the static devirtualizer.
//!
//! **Containment rationale.** The sample never executes on real hardware: every
//! instruction is decoded (iced-x86) and applied to an in-process [`Cpu`] and
//! [`Memory`] model. There is no host pointer the guest can reach, no syscall
//! it can issue and no native code path it can take, so there is no escape
//! surface by construction. Anything the interpreter cannot model exactly stops
//! the run ([`Stop::Unsupported`]) instead of being guessed.
//!
//! **Soundness rules.** An instruction executes only with fully modeled
//! semantics. Flags the architecture leaves undefined are tracked in
//! [`Cpu::undef_flags`]; reading one stops the run. A faulting instruction
//! leaves registers, flags, memory and rip untouched.
//!
//! **Observation seam.** [`Emulator::trace`] records every retired instruction
//! (optionally with effective address / value / VSP when [`ObserveConfig`] is
//! enabled). A [`Hook`] sees each memory access and control-flow transfer
//! *before* it commits; it may stop the run, or patch [`Cpu`]/[`Memory`] in
//! place and redirect `rip` (the Layer-2 API-stub pattern) and resume. Facts a
//! run surfaces for the static side are collected as [`ObservedFacts`], tagged
//! [`Observed`] so they are never mistaken for sound output.
//!
//! # Layers
//!
//! 1. PE loader ([`loader`]): map sections of a `vmpx-pe` image with section
//!    permissions, build a stack, TEB/PEB stubs and an import table pointing at
//!    stub thunks. **Implemented.**
//! 2. API-stub hooks: resolve thunk addresses in `Hook::on_flow`, emulate the
//!    needed Win32 surface (memory, loader, sync) and return into the guest.
//! 3. Instruction coverage: widen the subset (string ops, SSE, flags-reading
//!    ops, 32-bit mode) one modeled instruction at a time.
//! 4. Dump interface: detect OEP / unpack completion (write-then-execute,
//!    `run_until`), snapshot the mapped image and feed it to `vmpx-pe`
//!    rebuild and then the static devirtualizer (`vmpx-vm`/`vmpx-vmlift`).
//! 5. VM observation: hook the VM dispatcher to record handler traces.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod alu;
mod cpu;
mod detonate;
mod differential;
mod dump;
mod emu;
mod loader;
mod memory;
mod observe;
mod vm_observe;
mod winapi;

pub use alu::{Alu, Flags, Shift};
pub use detonate::{DetonateError, DetonateOptions, DetonationResult, detonate};
pub use dump::{DumpImage, DumpRegion, OepHook, capture_globals, dump_image};
pub use cpu::{AF, CF, Cpu, OF, PF, SF, ZF};
pub use differential::{
    DiffRecord, DiffReport, Divergence, Incomplete, Limits, Outcome, OutputRegion, RunError,
    RunResult, Target, differential, edge_values, input_set, run_function,
};
pub use emu::{
    Emulator, FlowEvent, FlowKind, Hook, HookAction, MemEvent, ObserveConfig, Stop, TraceEntry,
};
pub use loader::{LoadError, LoadOptions, LoadedImage, load};
pub use memory::{Access, MapError, MemFault, Memory, PAGE_SIZE, Perm};
pub use observe::{
    Observed, ObservedFacts, ObservedGlobal, ObservedImport, ObservedLoopDepth, SymbolOrOrdinal,
};
pub use vm_observe::{VmObservations, VmObserveHook};
pub use winapi::WinApiHook;
