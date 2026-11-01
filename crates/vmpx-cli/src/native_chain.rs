//! Follows a protected function through the native segments VMProtect leaves
//! between VM regions.
//!
//! VMProtect does not virtualize SSE: the VM exits, the original instruction
//! runs natively, and a `push seed; call entry` re-enters the VM. The unroll sees
//! each such exit as a region exit whose `ret` target is a constant code address.
//! This module lifts that native segment (with the SSE slice), applies it to the
//! state the VM exited with, and continues into the next VM entry, composing the
//! regions by substitution. Nothing is executed.
//!
//! Only a segment that leaves the stack as it found it, writes no memory and ends
//! in `push imm; call <VM entry>` (or a `ret`) is followed; anything else keeps the
//! old behavior of counting the region exit as an excluded arm.

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind};
use vmpx_disasm::{LiftOptions, lift_block_with};
use vmpx_mutation::cfg_recovery::{CfgOptions, VmAlgebra};
use vmpx_pe::{PeImage, Rva};
use vmpx_verify::differential::{apply_native_segment, compose_into_next, prepare_recovered, same_exit};
use vmpx_vm::ExprKind;
use vmpx_vmlift::{ExitSemantics, UnrolledFunction, UnrolledPath};

const MAX_DEPTH: usize = 24;
const MAX_SEGMENT_INSTRUCTIONS: usize = 64;
const MAX_COMPOSITES: usize = 16;

#[derive(Default)]
struct Tally {
    gate_arms: usize,
    followed_segments: usize,
    forks: usize,
    decided: usize,
    pruned: usize,
    blocks: usize,
    notes: Vec<String>,
}

fn is_push_imm(i: &Instruction) -> bool {
    i.mnemonic() == Mnemonic::Push
        && matches!(
            i.op0_kind(),
            OpKind::Immediate8
                | OpKind::Immediate16
                | OpKind::Immediate32
                | OpKind::Immediate64
                | OpKind::Immediate8to16
                | OpKind::Immediate8to32
                | OpKind::Immediate8to64
                | OpKind::Immediate32to64
        )
}

enum Tail {
    /// `push imm; call entry`: the VM is re-entered at this RVA.
    Reenter(Rva),
    Return,
}

fn decode_segment(pe: &PeImage<'_>, target_va: u64) -> Result<(Vec<Instruction>, Tail), String> {
    let base = pe.optional.image_base.0;
    let rva = u32::try_from(target_va.checked_sub(base).ok_or("segment below the image base")?)
        .map_err(|_| "segment outside the image".to_owned())?;
    let off = pe
        .rva_to_file_offset(Rva(rva))
        .map_err(|e| format!("segment {rva:#x}: {e}"))?
        .0 as usize;
    let data = pe.data();
    let slice = data
        .get(off..(off + MAX_SEGMENT_INSTRUCTIONS * 15).min(data.len()))
        .ok_or("segment bytes outside the image")?;
    let mut decoder = Decoder::with_ip(64, slice, target_va, DecoderOptions::NONE);
    let mut body: Vec<Instruction> = Vec::new();
    for _ in 0..MAX_SEGMENT_INSTRUCTIONS {
        let ins = decoder.decode();
        if ins.is_invalid() {
            return Err("undecodable bytes in the native segment".to_owned());
        }
        match ins.flow_control() {
            FlowControl::Next => body.push(ins),
            FlowControl::Return if ins.mnemonic() == Mnemonic::Ret && ins.op_count() == 0 => {
                return Ok((body, Tail::Return));
            }
            FlowControl::Call => {
                let Some(push) = body.pop().filter(is_push_imm) else {
                    return Err("native segment ends in a call that is not `push imm; call`".to_owned());
                };
                let rva = u32::try_from(push.ip().wrapping_sub(base))
                    .map_err(|_| "re-entry stub outside the image".to_owned())?;
                return Ok((body, Tail::Reenter(Rva(rva))));
            }
            other => return Err(format!("native segment ends in unmodeled control flow {other:?}")),
        }
    }
    Err("native segment is too long".to_owned())
}

fn region_target(path: &UnrolledPath) -> Option<u64> {
    let region = path.region.as_ref()?;
    if region.staged_words != 1 {
        return None;
    }
    match region.target.simplify().kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

fn region(
    pe: &PeImage<'_>,
    rva: Rva,
    depth: usize,
    limits: (usize, usize, usize),
    tally: &mut Tally,
) -> Result<Vec<ExitSemantics>, String> {
    if depth > MAX_DEPTH {
        return Err("VM/native chain is too deep".to_owned());
    }
    let (instruction_limit, max_blocks, max_paths) = limits;
    let base = crate::decoded_splice::unroll_entry_cfg(pe, rva, instruction_limit, max_blocks, max_paths)?;
    region_of(pe, &base, depth, limits, tally)
}

fn region_of(
    pe: &PeImage<'_>,
    base: &UnrolledFunction,
    depth: usize,
    limits: (usize, usize, usize),
    tally: &mut Tally,
) -> Result<Vec<ExitSemantics>, String> {
    tally.forks += base.forks;
    tally.decided += base.branches_decided;
    tally.pruned += base.spin_arms_pruned;
    let mut exits: Vec<ExitSemantics> = Vec::new();
    for (index, path) in base.paths.iter().enumerate() {
        tally.blocks += path.blocks_executed;
        let (prepared, _) = match prepare_recovered(&path.semantics, &path.store_epochs, base.entry_sp_offset) {
            Ok(p) => p,
            Err(e) => {
                if path.region_exit {
                    tally.gate_arms += 1;
                    continue;
                }
                return Err(format!("path {index}: {e}"));
            }
        };
        if !path.region_exit {
            push_unique(&mut exits, prepared);
            continue;
        }
        let Some(target) = region_target(path) else {
            if std::env::var_os("VMPX_DIFF_TRACE").is_some() {
                eprintln!(
                    "[chain] region exit {index} not followable: region={:?} stores={}",
                    path.region.as_ref().map(|r| (r.staged_words, format!("{:?}", r.target.simplify()))),
                    path.semantics.effects.stores.len()
                );
            }
            tally.gate_arms += 1;
            continue;
        };
        match follow(pe, target, &prepared, depth, limits, tally) {
            Ok(followed) => {
                tally.followed_segments += 1;
                for exit in followed {
                    push_unique(&mut exits, exit);
                }
            }
            Err(e) => {
                tally.notes.push(format!("region exit to {target:#x} not followed: {e}"));
                tally.gate_arms += 1;
            }
        }
        if exits.len() > MAX_COMPOSITES {
            return Err("too many distinct composite exits".to_owned());
        }
    }
    Ok(exits)
}

fn push_unique(exits: &mut Vec<ExitSemantics>, exit: ExitSemantics) {
    if !exits.iter().any(|e| same_exit(e, &exit)) {
        exits.push(exit);
    }
}

fn follow(
    pe: &PeImage<'_>,
    target_va: u64,
    start: &ExitSemantics,
    depth: usize,
    limits: (usize, usize, usize),
    tally: &mut Tally,
) -> Result<Vec<ExitSemantics>, String> {
    let (body, tail) = decode_segment(pe, target_va)?;
    let options: LiftOptions = CfgOptions::PROVEN_FP.into();
    let lifted = lift_block_with(&VmAlgebra, &body, options).map_err(|e| e.to_string())?;
    let after = apply_native_segment(start, &lifted)?;
    match tail {
        Tail::Return => Ok(vec![after]),
        Tail::Reenter(rva) => {
            let nested = region(pe, rva, depth + 1, limits, tally)?;
            nested.iter().map(|next| compose_into_next(&after, next)).collect()
        }
    }
}

/// [`crate::decoded_splice::unroll_entry_cfg`] that also follows constant-target
/// region exits through their native segments into the VM regions that follow.
pub(crate) fn unroll_chained(
    pe: &PeImage<'_>,
    rva: Rva,
    instruction_limit: usize,
    max_blocks: usize,
    max_paths: usize,
) -> Result<UnrolledFunction, String> {
    let mut tally = Tally::default();
    let limits = (instruction_limit, max_blocks, max_paths);
    let first = crate::decoded_splice::unroll_entry_cfg(pe, rva, instruction_limit, max_blocks, max_paths)?;
    if first.paths.iter().all(|p| !p.region_exit || p.region.is_none()) {
        return Ok(first);
    }
    let exits = region_of(pe, &first, 0, limits, &mut tally)?;
    let mut paths = Vec::new();
    for semantics in exits {
        paths.push(UnrolledPath {
            conditions: Vec::new(),
            semantics,
            store_epochs: Vec::new(),
            rflags_is_entry_flags: false,
            return_pop: 0,
            blocks_executed: tally.blocks,
            region_exit: false,
            region: None,
            env_forks: 0,
        });
    }
    if std::env::var_os("VMPX_DIFF_TRACE").is_some() {
        eprintln!(
            "[chain] {} composite exit(s), {} native segment(s) followed, {} gate arm(s), notes={:?}",
            paths.len(),
            tally.followed_segments,
            tally.gate_arms,
            tally.notes
        );
    }
    if let Some(template) = paths.first().cloned() {
        for _ in 0..tally.gate_arms {
            let mut gate = template.clone();
            gate.region_exit = true;
            paths.push(gate);
        }
    }
    Ok(UnrolledFunction {
        paths,
        branches_decided: tally.decided,
        forks: tally.forks,
        spin_arms_pruned: tally.pruned,
        entry_sp_offset: first.entry_sp_offset,
        notes: tally.notes,
    })
}
