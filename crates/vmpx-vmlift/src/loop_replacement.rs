//! Native replacement for a recovered VM function whose body is a loop.
//!
//! [`crate::build_replacement`] lowers a single straight-line exit state. A guard
//! whose recovered CFG contains a back edge (an integrity/checksum loop) cannot
//! be expressed that way, so this module lowers the structured [`RNode`] tree to
//! the project IR WITH control flow: the loop-carried cells become spill slots
//! (phis), each block's expressions lower through the shared scratch machinery,
//! and `Fork`/`BackEdge`/`Exit` become `BranchIf`/`Jump`/epilogue transfers.
//!
//! Soundness: every expression is `fold`ed (checksum/image concretization) by the
//! caller's closure before lowering, exactly as the flat path folds each leg. A
//! `Fork` whose selector folds to a constant collapses to the one arm it selects,
//! so only genuinely loop-variant control flow survives as a real branch.

use std::collections::BTreeMap;

use vmpx_vm::{BitWidth, Expr, ExprKind, X64NativeRegister};

use vmpx_recompile::{
    Address, BinarySource, CodeAddress, Condition, Operation, PhysicalRegister, RegisterAssignment,
    SpillSlot, StackFrame, VirtualRegister, Width, compile,
};

use crate::replacement::{
    ReplacementConfig, Trust, canonical_register, lower_fragment_with, physical_for,
};
use crate::rolled::{RExit, RNode, RStore};
use crate::ExpressionInput;

fn width_of(width: BitWidth) -> Width {
    match width {
        BitWidth::W8 => Width::W8,
        BitWidth::W16 => Width::W16,
        BitWidth::W32 => Width::W32,
        _ => Width::W64,
    }
}

fn constant_value(expr: &Expr) -> Option<u64> {
    match expr.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// Shared assembly state threaded through the recursive walk.
struct Ctx<'a> {
    fold: &'a dyn Fn(&Expr) -> Result<Expr, String>,
    config: &'a ReplacementConfig,
    /// Entry-register inputs plus one input per phi, each bound to a spill slot.
    inputs: Vec<ExpressionInput>,
    /// Phi name -> its persistent spill slot, in declaration order.
    phi_slots: Vec<(String, SpillSlot)>,
    /// Output register -> its result slot, written by whichever exit runs.
    result_slots: BTreeMap<X64NativeRegister, SpillSlot>,
    /// Base of the small "park" region: transient slots that hold a value across
    /// a sibling fragment's lowering (a store's value while its address is built,
    /// a fork selector across its compare chain, a back edge's next values).
    /// Reused across recursion depth because an outer park is always dead once a
    /// branch into an arm is taken.
    park_base: u16,
    /// First slot a lowered fragment may use for its own temporaries (above the
    /// persistent frame and the park region).
    shared_end: u16,
    operations: Vec<Operation>,
    canonical: BTreeMap<PhysicalRegister, VirtualRegister>,
    assignments: Vec<RegisterAssignment>,
    next_register: u32,
    trust: Trust,
    next_label: u32,
    epilogue: vmpx_recompile::Label,
    header: vmpx_recompile::Label,
    /// Set when the tree has a region (staged-continuation) exit. The slot holds
    /// the computed continuation target; the region epilogue transfers through it.
    region_epilogue: Option<vmpx_recompile::Label>,
    continuation_slot: SpillSlot,
}

impl Ctx<'_> {
    fn fresh_label(&mut self) -> vmpx_recompile::Label {
        let label = vmpx_recompile::Label(self.next_label);
        self.next_label += 1;
        label
    }

    /// Lower `expr` (after folding) into the shared namespaces, appending its
    /// operations and returning the register that holds its value.
    fn lower(&mut self, expr: &Expr) -> Result<VirtualRegister, String> {
        let folded = (self.fold)(expr)?;
        let mut window_base = self.shared_end;
        let (ops, result) = lower_fragment_with(
            &folded,
            &self.inputs,
            &[],
            self.config,
            &mut self.canonical,
            &mut self.assignments,
            &mut self.next_register,
            &mut window_base,
            &self.shared_end,
            &mut self.trust,
            false,
        )
        .map_err(|error| error.to_string())?;
        self.operations.extend(ops);
        Ok(result)
    }

    fn scratch(&mut self, physical: PhysicalRegister) -> VirtualRegister {
        canonical_register(
            physical,
            &mut self.canonical,
            &mut self.assignments,
            &mut self.next_register,
        )
    }

    fn emit(&mut self, node: &RNode) -> Result<(), String> {
        match node {
            RNode::Stores { stores, next } => {
                for store in stores {
                    self.emit_store(store)?;
                }
                self.emit(next)
            }
            RNode::Fork { selector, arms, .. } => self.emit_fork(selector, arms),
            RNode::Exit(exit) => self.emit_exit(exit),
            RNode::BackEdge { next } => self.emit_back_edge(next),
            RNode::Loop { .. } => Err("nested loop not yet supported".to_owned()),
            RNode::Dead(_) => {
                // An arm the roller proved cannot reach a function exit (it spins
                // on a repeated block). It must never run on a live path; emit a
                // self-trap so a model error faults rather than returns a value.
                let trap = self.fresh_label();
                self.operations.push(Operation::Label(trap));
                self.operations.push(Operation::Jump(trap));
                Ok(())
            }
        }
    }

    fn emit_store(&mut self, store: &RStore) -> Result<(), String> {
        let value = self.lower(&store.value)?;
        // The value register is R10/R11-family; park it so computing the address
        // (same scratch) cannot clobber it.
        let value_slot = SpillSlot(self.park_base);
        self.operations.push(Operation::SpillStore {
            slot: value_slot,
            source: value,
        });
        let address = self.lower(&store.address)?;
        let reloaded = self.scratch(PhysicalRegister::R9);
        self.operations.push(Operation::SpillLoad {
            destination: VirtualRegister::new(reloaded.id, width_of(store.width)),
            slot: value_slot,
        });
        self.operations.push(Operation::Store {
            address: Address::new(address, 0),
            source: VirtualRegister::new(reloaded.id, width_of(store.width)),
        });
        Ok(())
    }

    fn emit_fork(&mut self, selector: &Expr, arms: &[(u64, RNode)]) -> Result<(), String> {
        let folded = (self.fold)(selector)?;
        // A selector that folded to a constant picks exactly one arm; emit only it.
        if let Some(value) = constant_value(&folded) {
            let arm = arms
                .iter()
                .find(|(key, _)| *key == value)
                .map(|(_, arm)| arm)
                .or_else(|| arms.last().map(|(_, arm)| arm))
                .ok_or_else(|| "fork with no arms".to_owned())?;
            return self.emit(arm);
        }
        if arms.is_empty() {
            return Err("fork with no arms".to_owned());
        }
        // Symbolic selector: compare against each arm key and branch. Park the
        // selector in a slot so each arm's lowering cannot clobber it.
        let selector_register = self.lower(&folded)?;
        let selector_slot = SpillSlot(self.park_base + 1);
        self.operations.push(Operation::SpillStore {
            slot: selector_slot,
            source: selector_register,
        });
        let labels: Vec<_> = arms.iter().map(|_| self.fresh_label()).collect();
        for (index, (value, _)) in arms.iter().enumerate() {
            if index + 1 == arms.len() {
                // Last arm is the fall-through default.
                self.operations.push(Operation::Jump(labels[index]));
                break;
            }
            let selector = self.scratch(PhysicalRegister::R10);
            self.operations.push(Operation::SpillLoad {
                destination: selector,
                slot: selector_slot,
            });
            let immediate = self.scratch(PhysicalRegister::R8);
            self.operations.push(Operation::MoveImmediate {
                destination: immediate,
                value: *value,
            });
            self.operations.push(Operation::Compare {
                left: selector,
                right: BinarySource::Register(immediate),
            });
            self.operations.push(Operation::BranchIf {
                condition: Condition::Equal,
                target: labels[index],
            });
        }
        for (index, (_, arm)) in arms.iter().enumerate() {
            self.operations.push(Operation::Label(labels[index]));
            self.emit(arm)?;
        }
        Ok(())
    }

    fn emit_exit(&mut self, exit: &RExit) -> Result<(), String> {
        // A region exit tail-transfers to a computed continuation rather than
        // returning to the caller, so it hands that target its computed register
        // values and need not honour callee-save. The target is staged in its own
        // slot and the region epilogue returns through it (see the build below).
        let tail_transfer = exit.region;
        // Compute each output register's value and park it in that register's
        // shared result slot; the epilogue loads them all after the loop.
        let registers: Vec<(X64NativeRegister, Expr)> = exit
            .registers
            .iter()
            .map(|(register, expr)| (*register, expr.clone()))
            .collect();
        for (register, expr) in &registers {
            // A nonvolatile register the body never clobbers keeps its entry
            // value, so an exit that leaves it equal to entry needs no work --
            // but only when it returns to the caller; a tail transfer hands every
            // value on, so nothing is skipped.
            if !tail_transfer
                && physical_for(register).is_win64_nonvolatile()
                && is_entry_value(expr, register)
            {
                continue;
            }
            let value = self.lower(expr)?;
            let slot = *self
                .result_slots
                .get(register)
                .ok_or_else(|| format!("exit writes {register:?} with no result slot"))?;
            self.operations.push(Operation::SpillStore {
                slot,
                source: value,
            });
        }
        if exit.region {
            let target = exit
                .continuation
                .as_ref()
                .ok_or_else(|| "region exit without a continuation target".to_owned())?;
            let region_epilogue = self
                .region_epilogue
                .ok_or_else(|| "region exit but no region epilogue was reserved".to_owned())?;
            let value = self.lower(target)?;
            self.operations.push(Operation::SpillStore {
                slot: self.continuation_slot,
                source: value,
            });
            self.operations.push(Operation::Jump(region_epilogue));
        } else {
            self.operations.push(Operation::Jump(self.epilogue));
        }
        Ok(())
    }

    fn emit_back_edge(&mut self, next: &[Expr]) -> Result<(), String> {
        if next.len() != self.phi_slots.len() {
            return Err(format!(
                "back edge carries {} value(s) for {} phi(s)",
                next.len(),
                self.phi_slots.len()
            ));
        }
        // Parallel assignment: every next value is a function of the CURRENT phis,
        // so compute them all into temporaries first, then copy into the phi slots.
        let mut temps = Vec::with_capacity(next.len());
        for expr in next {
            let value = self.lower(expr)?;
            let temp = SpillSlot(self.park_base + 2 + u16::try_from(temps.len()).unwrap_or(0));
            self.operations.push(Operation::SpillStore {
                slot: temp,
                source: value,
            });
            temps.push(temp);
        }
        for (index, (_, phi_slot)) in self.phi_slots.clone().iter().enumerate() {
            let carrier = self.scratch(PhysicalRegister::R10);
            self.operations.push(Operation::SpillLoad {
                destination: carrier,
                slot: temps[index],
            });
            self.operations.push(Operation::SpillStore {
                slot: *phi_slot,
                source: carrier,
            });
        }
        self.operations.push(Operation::Jump(self.header));
        Ok(())
    }
}

/// True when `expr` is exactly the entry value of `register` (its symbolic name).
fn is_entry_value(expr: &Expr, register: &X64NativeRegister) -> bool {
    match &expr.kind {
        ExprKind::Register { name } => name == &register.symbolic_name(),
        _ => false,
    }
}

/// Collect every output register any exit in the tree writes.
fn collect_exit_registers(node: &RNode, out: &mut BTreeMap<X64NativeRegister, ()>) {
    match node {
        RNode::Exit(exit) => {
            for (register, expr) in &exit.registers {
                // A region exit tail-transfers, so it writes every register (it
                // keeps nothing for the caller); a normal exit may skip a
                // nonvolatile it leaves at its entry value. Must match emit_exit.
                if exit.region
                    || !(physical_for(register).is_win64_nonvolatile()
                        && is_entry_value(expr, register))
                {
                    out.insert(*register, ());
                }
            }
        }
        RNode::Stores { next, .. } => collect_exit_registers(next, out),
        RNode::Fork { arms, .. } => {
            for (_, arm) in arms {
                collect_exit_registers(arm, out);
            }
        }
        RNode::Loop { body, .. } => collect_exit_registers(body, out),
        RNode::BackEdge { .. } | RNode::Dead(_) => {}
    }
}

/// Whether any exit in the tree is a region (staged-continuation) exit, which
/// needs a continuation slot and the transfer epilogue reserved up front.
fn tree_has_region_exit(node: &RNode) -> bool {
    match node {
        RNode::Stores { next, .. } => tree_has_region_exit(next),
        RNode::Fork { arms, .. } => arms.iter().any(|(_, arm)| tree_has_region_exit(arm)),
        RNode::Exit(exit) => exit.region,
        RNode::Loop { body, .. } => tree_has_region_exit(body),
        RNode::BackEdge { .. } | RNode::Dead(_) => false,
    }
}

/// Every symbol any expression in the tree reads, so the prologue knows which
/// synthetic entry inputs (the flag word, the return address) to capture.
fn collect_tree_symbols(node: &RNode, out: &mut std::collections::BTreeSet<String>) {
    match node {
        RNode::Stores { stores, next } => {
            for store in stores {
                out.extend(store.address.dependencies());
                out.extend(store.value.dependencies());
            }
            collect_tree_symbols(next, out);
        }
        RNode::Fork { selector, arms, .. } => {
            out.extend(selector.dependencies());
            for (_, arm) in arms {
                collect_tree_symbols(arm, out);
            }
        }
        RNode::Exit(exit) => {
            for expr in exit.registers.values() {
                out.extend(expr.dependencies());
            }
            if let Some(continuation) = &exit.continuation {
                out.extend(continuation.dependencies());
            }
        }
        RNode::Loop { phis, body, .. } => {
            for phi in phis {
                out.extend(phi.init.dependencies());
            }
            collect_tree_symbols(body, out);
        }
        RNode::BackEdge { next } => {
            for expr in next {
                out.extend(expr.dependencies());
            }
        }
        RNode::Dead(_) => {}
    }
}

/// Build native code for a recovered VM function with genuine control flow: a
/// loop (back edge), input-dependent branches, or both. Loop-carried cells
/// become spill-slot phis; `Fork` becomes a compare-and-branch; `Exit` sets the
/// result slots and jumps the shared epilogue. A function whose branches all
/// collapse after folding is handled by the simpler flat emitter instead; this
/// is for the ones that do not. `fold` concretizes each expression exactly as
/// the flat path does.
///
/// # Errors
///
/// Fails when the tree uses a construct this emitter does not yet cover (region
/// exits, nested loops) or when an expression cannot be lowered.
pub fn build_control_flow_function(
    root: &RNode,
    config: &ReplacementConfig,
    fold: &dyn Fn(&Expr) -> Result<Expr, String>,
) -> Result<vmpx_recompile::Function, String> {
    // Peel any leading pre-loop stores. The remainder is either a loop or, for a
    // branchy non-loop function, the body itself (no phis, no back edge).
    let mut leading: Vec<&RStore> = Vec::new();
    let mut cursor = root;
    while let RNode::Stores { stores, next } = cursor {
        leading.extend(stores.iter());
        cursor = next;
    }
    const NO_PHIS: &[crate::rolled::RPhi] = &[];
    let (phis, body): (&[crate::rolled::RPhi], &RNode) = match cursor {
        RNode::Loop { phis, body, .. } => (phis.as_slice(), body.as_ref()),
        other => (NO_PHIS, other),
    };

    // A function may read synthetic entry inputs beyond its GP registers -- the
    // entry flag word and the caller's return address. Capture only the ones the
    // tree actually references, appended after the GP inputs (their captures read
    // flags / the stack and so must run once every GP register is spilled).
    let mut referenced = std::collections::BTreeSet::<String>::new();
    collect_tree_symbols(cursor, &mut referenced);
    for store in &leading {
        referenced.extend(store.address.dependencies());
        referenced.extend(store.value.dependencies());
    }
    #[derive(Clone, Copy)]
    enum Cap {
        Reg(PhysicalRegister),
        Flags,
        ReturnAddr,
    }
    let mut capture_inputs: Vec<(String, Cap)> = config
        .inputs
        .iter()
        .map(|(name, reg)| (name.clone(), Cap::Reg(*reg)))
        .collect();
    if referenced.contains(crate::replacement::NATIVE_RFLAGS_INITIAL) {
        capture_inputs.push((crate::replacement::NATIVE_RFLAGS_INITIAL.to_owned(), Cap::Flags));
    }
    if referenced.contains(crate::replacement::NATIVE_RETURN_ADDRESS_INITIAL) {
        capture_inputs.push((crate::replacement::NATIVE_RETURN_ADDRESS_INITIAL.to_owned(), Cap::ReturnAddr));
    }

    // Slot layout: [inputs | phis | results | park | scratch window].
    let input_count = u16::try_from(capture_inputs.len()).map_err(|_| "too many inputs".to_owned())?;
    let mut inputs: Vec<ExpressionInput> = Vec::new();
    for (index, (name, _)) in capture_inputs.iter().enumerate() {
        inputs.push(ExpressionInput::slot(
            name.clone(),
            BitWidth::W64,
            SpillSlot(u16::try_from(index).unwrap_or(0)),
        ));
    }
    let mut phi_slots: Vec<(String, SpillSlot)> = Vec::new();
    for (index, phi) in phis.iter().enumerate() {
        let slot = SpillSlot(input_count + u16::try_from(index).unwrap_or(0));
        inputs.push(ExpressionInput::slot(phi.name.clone(), phi.width, slot));
        phi_slots.push((phi.name.clone(), slot));
    }
    let phi_count = u16::try_from(phis.len()).map_err(|_| "too many phis".to_owned())?;

    let mut result_registers = BTreeMap::new();
    collect_exit_registers(cursor, &mut result_registers);
    let mut result_slots = BTreeMap::new();
    for (index, register) in result_registers.keys().enumerate() {
        let slot = SpillSlot(input_count + phi_count + u16::try_from(index).unwrap_or(0));
        result_slots.insert(*register, slot);
    }
    let result_count = u16::try_from(result_slots.len()).map_err(|_| "too many results".to_owned())?;
    // A region (staged-continuation) exit stages its computed target in a slot of
    // its own, placed right after the result slots; the region epilogue returns
    // through it. Reserved only when the tree actually has such an exit.
    let has_region = tree_has_region_exit(cursor);
    let continuation_slot = SpillSlot(input_count + phi_count + result_count);
    let continuation_count = u16::from(has_region);
    let park_base = input_count + phi_count + result_count + continuation_count;
    // Park region: [0] store value, [1] fork selector, [2..2+phis] back-edge temps.
    let park_size = 2 + phi_count;
    let shared_end = park_base + park_size;

    let mut canonical: BTreeMap<PhysicalRegister, VirtualRegister> = BTreeMap::new();
    let mut assignments: Vec<RegisterAssignment> = Vec::new();
    let mut next_register = 0_u32;

    // Prologue: capture each input into its slot. GP registers spill directly;
    // the flag word and return address are materialised through a carrier (Rax,
    // already spilled above) with `ReadFlags` / an entry-stack-pointer load.
    let mut prologue: Vec<Operation> = Vec::new();
    for (index, (_, cap)) in capture_inputs.iter().enumerate() {
        let slot = SpillSlot(u16::try_from(index).unwrap_or(0));
        match cap {
            Cap::Reg(register) => {
                let source =
                    canonical_register(*register, &mut canonical, &mut assignments, &mut next_register);
                prologue.push(Operation::SpillStore { slot, source });
            }
            Cap::Flags => {
                let carrier = canonical_register(
                    PhysicalRegister::Rax,
                    &mut canonical,
                    &mut assignments,
                    &mut next_register,
                );
                prologue.push(Operation::ReadFlags { destination: carrier });
                prologue.push(Operation::SpillStore { slot, source: carrier });
            }
            Cap::ReturnAddr => {
                let carrier = canonical_register(
                    PhysicalRegister::Rax,
                    &mut canonical,
                    &mut assignments,
                    &mut next_register,
                );
                prologue.push(Operation::EntryStackPointer { destination: carrier });
                prologue.push(Operation::Load {
                    destination: carrier,
                    address: Address::new(carrier, 0),
                });
                prologue.push(Operation::SpillStore { slot, source: carrier });
            }
        }
    }

    let header = vmpx_recompile::Label(0);
    let epilogue = vmpx_recompile::Label(1);
    let region_epilogue = has_region.then(|| vmpx_recompile::Label(2));
    let mut ctx = Ctx {
        fold,
        config,
        inputs,
        phi_slots,
        result_slots,
        park_base,
        shared_end,
        operations: prologue,
        canonical,
        assignments,
        next_register,
        trust: Trust::default(),
        next_label: if has_region { 3 } else { 2 },
        epilogue,
        header,
        region_epilogue,
        continuation_slot,
    };

    // Pre-loop stores run once, before the header.
    for store in &leading {
        ctx.emit_store(store)?;
    }
    // Initialise each phi slot from its entry value.
    let phi_inits: Vec<(SpillSlot, Expr)> = phis
        .iter()
        .enumerate()
        .map(|(index, phi)| (SpillSlot(input_count + u16::try_from(index).unwrap_or(0)), phi.init.clone()))
        .collect();
    for (slot, init) in &phi_inits {
        let value = ctx.lower(init)?;
        ctx.operations.push(Operation::SpillStore { slot: *slot, source: value });
    }
    // Loop body.
    ctx.operations.push(Operation::Label(header));
    ctx.emit(body)?;
    // Epilogue: load each result slot into its register, then return.
    ctx.operations.push(Operation::Label(epilogue));
    let result_slots = ctx.result_slots.clone();
    for (register, slot) in &result_slots {
        let destination = ctx.scratch(physical_for(register));
        ctx.operations.push(Operation::SpillLoad { destination, slot: *slot });
    }
    ctx.operations.push(Operation::Return);

    // Region epilogue: load the result registers, then transfer through the
    // staged continuation slot instead of returning. `ReturnRetaining` tears the
    // frame down with a `lea` (not pops), leaving RSP pointing at the staged
    // target, so the following `ret` jumps to it. The retained byte count is
    // patched once the frame size is final.
    let mut retain_index: Option<usize> = None;
    if let Some(region_epilogue) = region_epilogue {
        ctx.operations.push(Operation::Label(region_epilogue));
        let result_slots = ctx.result_slots.clone();
        for (register, slot) in &result_slots {
            let destination = ctx.scratch(physical_for(register));
            ctx.operations.push(Operation::SpillLoad { destination, slot: *slot });
        }
        retain_index = Some(ctx.operations.len());
        ctx.operations.push(Operation::ReturnRetaining { bytes: 0 });
    }

    // Size the frame to cover every spill slot the body references.
    let mut operations = ctx.operations;
    let max_slot = operations
        .iter_mut()
        .flat_map(|operation| crate::replacement::operation_slots_mut(operation))
        .map(|slot| slot.0)
        .max()
        .unwrap_or(0);
    let stack_allocation = (u32::from(max_slot) + 1) * 8;
    // Patch the retained byte count: popping slot k (which sits k*8 above the
    // body RSP) needs `stack_allocation - k*8` retained so the `lea` lands RSP on
    // it. No preserved pushes here, so that is the whole total.
    if let Some(index) = retain_index {
        let offset = u32::from(continuation_slot.0) * 8;
        let bytes = stack_allocation
            .checked_sub(offset)
            .filter(|bytes| *bytes >= 8)
            .ok_or_else(|| "continuation slot outside the frame".to_owned())?;
        operations[index] = Operation::ReturnRetaining { bytes };
    }
    let frame = StackFrame {
        stack_allocation,
        preserve: Vec::new(),
        call_ready: false,
        preinitialized_slots: Vec::new(),
        tail_transfer: has_region,
    };
    let function = vmpx_recompile::Function {
        operations,
        assignments: ctx.assignments,
        frame,
    };
    Ok(function)
}

/// Build and compile a control-flow replacement at `address`. The Function is
/// position-independent; the compile is the only VA-dependent step, so compact
/// placement builds the Function once and compiles it at the packed offset.
///
/// # Errors
///
/// Fails when [`build_control_flow_function`] fails or the compile fails.
pub fn build_control_flow_replacement(
    root: &RNode,
    config: &ReplacementConfig,
    address: CodeAddress,
    fold: &dyn Fn(&Expr) -> Result<Expr, String>,
) -> Result<Vec<u8>, String> {
    let function = build_control_flow_function(root, config, fold)?;
    let compiled = compile(&function, address).map_err(|error| format!("compile: {error}"))?;
    Ok(compiled.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rolled::{RExit, RPhi};
    use std::collections::BTreeMap;
    use vmpx_pe::Va;
    use vmpx_vm::BinaryOp;

    fn empty_config() -> ReplacementConfig {
        ReplacementConfig {
            inputs: Vec::new(),
            max_expression_nodes: 65_536,
            max_spill_slots: 1024,
            import_calls: BTreeMap::new(),
            continuation: None,
            effects: Vec::new(),
            effect_calls: Vec::new(),
            rflags: None,
            symbol_rewrites: BTreeMap::new(),
        }
    }

    /// `for (c = 0; c != 10; c += 1); return c;` lowered to native code.
    #[test]
    fn counting_loop_compiles_to_native() {
        let counter = Expr::register(BitWidth::W64, "c");
        let phi = RPhi {
            name: "c".to_owned(),
            width: BitWidth::W64,
            init: Expr::constant(BitWidth::W64, 0),
        };
        let exit = RNode::Exit(RExit {
            registers: BTreeMap::from([(X64NativeRegister::Rax, counter.clone())]),
            region: false,
            continuation: None,
        });
        let back = RNode::BackEdge {
            next: vec![Expr::binary(BinaryOp::Add, counter.clone(), Expr::constant(BitWidth::W64, 1)).unwrap()],
        };
        // selector == 10 -> exit; otherwise (fall-through) -> back edge.
        let fork = RNode::Fork {
            selector: counter,
            env: false,
            block: Va(0),
            arms: vec![(10, exit), (0, back)],
        };
        let root = RNode::Loop {
            header: Va(0),
            phis: vec![phi],
            body: Box::new(fork),
        };
        let fold = |expr: &Expr| -> Result<Expr, String> { Ok(expr.clone()) };
        let bytes = build_control_flow_replacement(&root, &empty_config(), CodeAddress(0x1000), &fold)
            .expect("loop should compile");
        assert!(!bytes.is_empty(), "emitted code must be non-empty");
    }

    /// A loop whose exit is a region (staged-continuation) exit: it tail-transfers
    /// to a computed target rather than returning. Previously refused; it must now
    /// compile, staging the target and transferring through it.
    #[test]
    fn a_region_exit_in_a_loop_compiles() {
        let counter = Expr::register(BitWidth::W64, "c");
        let phi = RPhi {
            name: "c".to_owned(),
            width: BitWidth::W64,
            init: Expr::constant(BitWidth::W64, 0),
        };
        // On exit, hand Rax = c and tail-transfer to the computed target (here the
        // counter itself stands in for a recovered continuation address).
        let exit = RNode::Exit(RExit {
            registers: BTreeMap::from([(X64NativeRegister::Rax, counter.clone())]),
            region: true,
            continuation: Some(counter.clone()),
        });
        let back = RNode::BackEdge {
            next: vec![
                Expr::binary(BinaryOp::Add, counter.clone(), Expr::constant(BitWidth::W64, 1))
                    .unwrap(),
            ],
        };
        let fork = RNode::Fork {
            selector: counter,
            env: false,
            block: Va(0),
            arms: vec![(10, exit), (0, back)],
        };
        let root = RNode::Loop {
            header: Va(0),
            phis: vec![phi],
            body: Box::new(fork),
        };
        let fold = |expr: &Expr| -> Result<Expr, String> { Ok(expr.clone()) };
        let bytes = build_control_flow_replacement(&root, &empty_config(), CodeAddress(0x1000), &fold)
            .expect("region-exit loop should compile");
        assert!(!bytes.is_empty(), "region exit must emit native code");
    }
}
