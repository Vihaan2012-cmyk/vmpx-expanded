//! End-to-end native demutation: recover each function's CFG, lift blocks to
//! `Expr`, demutate every output register expression, and report before/after.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use vmpx_mutation::{NativeCfg, ProblemClass, Terminator, recover_native_cfg_pe};
use vmpx_pe::{PeImage, Rva};
use vmpx_tournament::Elo;
use vmpx_vm::Expr;

use crate::pipeline::{Tier, demutate_verified};

const MIN_NODES: usize = 4;
const MAX_NODES: usize = 2000;

#[derive(Clone, Debug, Default)]
pub struct BinaryReport {
    pub functions: usize,
    /// Functions whose CFG could not be recovered at all.
    pub functions_failed: usize,
    pub blocks: usize,
    pub blocks_lifted: usize,
    pub instructions_lifted: usize,
    pub exprs_total: usize,
    pub exprs_attempted: usize,
    pub exprs_simplified: usize,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub sound_count: usize,
    pub experimental_count: usize,
    /// Block terminators/lifts that failed, grouped by short cause.
    pub unsupported: BTreeMap<String, usize>,
    pub errors: Vec<String>,
}

fn cause(reason: &str) -> String {
    let head = reason.split_whitespace().next().unwrap_or("?");
    let mnem = reason
        .split('`')
        .nth(1)
        .and_then(|t| t.split_whitespace().next());
    match mnem {
        Some(m) => match reason.rsplit_once("`: ") {
            Some((_, why)) => format!("{head} {m}: {}", why.chars().take(60).collect::<String>()),
            None => format!("{head} {m}"),
        },
        None => head.to_owned(),
    }
}

fn nodes(e: &Expr) -> usize {
    e.node_count_bounded(MAX_NODES)
}

fn best_demutation(e: &Expr, elo: &Elo) -> Option<(Expr, Tier)> {
    ProblemClass::RANKED
        .iter()
        .filter_map(|&c| demutate_verified(e, c, elo))
        .min_by_key(|d| nodes(&d.expr))
        .map(|d| (d.expr, d.tier))
}

impl BinaryReport {
    /// Demutates every lifted register expression of an already-recovered CFG.
    pub fn add_cfg(&mut self, cfg: &NativeCfg, elo: &Elo) {
        self.functions += 1;
        for b in cfg.blocks.values() {
            self.blocks += 1;
            if let Terminator::Unsupported(why) = &b.terminator {
                *self.unsupported.entry(cause(why)).or_default() += 1;
            }
            let Some(sem) = &b.semantics else { continue };
            self.blocks_lifted += 1;
            self.instructions_lifted += sem.instruction_count;
            for gpr in &sem.written {
                let Some(e) = sem.regs.get(gpr) else { continue };
                let n = nodes(e);
                self.exprs_total += 1;
                self.nodes_before += n;
                if !(MIN_NODES..MAX_NODES).contains(&n) {
                    self.nodes_after += n;
                    continue;
                }
                self.exprs_attempted += 1;
                match best_demutation(e, elo) {
                    Some((out, tier)) if nodes(&out) < n => {
                        self.exprs_simplified += 1;
                        self.nodes_after += nodes(&out);
                        match tier {
                            Tier::Sound => self.sound_count += 1,
                            Tier::Experimental => self.experimental_count += 1,
                        }
                    }
                    _ => self.nodes_after += n,
                }
            }
        }
    }

    pub fn merge(&mut self, o: &Self) {
        self.functions += o.functions;
        self.functions_failed += o.functions_failed;
        self.blocks += o.blocks;
        self.blocks_lifted += o.blocks_lifted;
        self.instructions_lifted += o.instructions_lifted;
        self.exprs_total += o.exprs_total;
        self.exprs_attempted += o.exprs_attempted;
        self.exprs_simplified += o.exprs_simplified;
        self.nodes_before += o.nodes_before;
        self.nodes_after += o.nodes_after;
        self.sound_count += o.sound_count;
        self.experimental_count += o.experimental_count;
        for (k, v) in &o.unsupported {
            *self.unsupported.entry(k.clone()).or_default() += v;
        }
        self.errors.extend(o.errors.iter().cloned());
    }

    #[must_use]
    pub fn render(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "functions: {} ({} CFG-failed)\nblocks: {} ({} lifted, {} instructions)",
            self.functions, self.functions_failed, self.blocks, self.blocks_lifted,
            self.instructions_lifted
        );
        let _ = writeln!(
            s,
            "exprs: {} total, {} attempted, {} simplified (sound {}, experimental {})\nnodes: {} -> {}",
            self.exprs_total, self.exprs_attempted, self.exprs_simplified, self.sound_count,
            self.experimental_count, self.nodes_before, self.nodes_after
        );
        if self.unsupported.is_empty() {
            s.push_str("lift failures: none\n");
        } else {
            s.push_str("lift failures (block terminators/bodies):\n");
            for (k, v) in &self.unsupported {
                let _ = writeln!(s, "  {k}: {v}");
            }
        }
        for e in &self.errors {
            let _ = writeln!(s, "error: {e}");
        }
        s
    }
}

/// Recovers the function at `entry`, lifts its blocks and demutates every
/// output-register expression. Never panics on unliftable code; failures are
/// recorded in the report.
#[must_use]
pub fn demutate_binary(pe: &PeImage<'_>, entry: Rva, elo: &Elo) -> BinaryReport {
    let mut r = BinaryReport::default();
    match recover_native_cfg_pe(pe, entry) {
        Ok(cfg) => r.add_cfg(&cfg, elo),
        Err(e) => {
            r.functions_failed = 1;
            r.errors.push(format!("{entry}: {e}"));
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::recover_native_cfg;

    // mov rax,rcx; xor rax,rdx; mov r8,rcx; and r8,rdx; add r8,r8; add rax,r8; ret
    // rax = (x ^ y) + 2*(x & y)
    const OBF_ADD: &[u8] = &[
        0x48, 0x89, 0xC8, 0x48, 0x31, 0xD0, 0x49, 0x89, 0xC8, 0x49, 0x21, 0xD0, 0x4D, 0x01, 0xC0,
        0x4C, 0x01, 0xC0, 0xC3,
    ];
    // test ecx,ecx; je +5; mov eax,1; ret
    const IF_CODE: &[u8] = &[0x85, 0xC9, 0x74, 0x05, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    // mov rax,rcx; call rax; ret
    const CALL_CODE: &[u8] = &[0x48, 0x89, 0xC8, 0xFF, 0xD0, 0xC3];

    fn run(code: &[u8]) -> BinaryReport {
        let cfg = recover_native_cfg(code, 0x1000, 0x1000).expect("cfg");
        let mut r = BinaryReport::default();
        r.add_cfg(&cfg, &Elo::new());
        r
    }

    #[test]
    fn counts_functions_and_blocks() {
        let r = run(IF_CODE);
        assert_eq!((r.functions, r.blocks, r.blocks_lifted), (1, 3, 3));
        assert!(r.unsupported.is_empty());
    }

    #[test]
    fn obfuscated_expression_is_simplified() {
        let r = run(OBF_ADD);
        assert!(r.exprs_simplified >= 1, "{}", r.render());
        assert!(r.nodes_after < r.nodes_before);
    }

    #[test]
    fn unsupported_instruction_is_reported_not_panicked() {
        let r = run(CALL_CODE);
        assert_eq!(r.functions, 1);
        assert!(!r.unsupported.is_empty());
        assert!(r.render().contains("lift failures"));
    }

    #[test]
    fn real_dll_function_and_bad_entry() {
        let Ok(data) = std::fs::read("E:/VMPX/testcorpus/corpus/unitops.dll") else {
            return;
        };
        let pe = PeImage::parse(&data).expect("pe");
        let elo = Elo::new();
        let ok = demutate_binary(&pe, Rva(0x10F0), &elo);
        assert_eq!((ok.functions, ok.blocks), (1, 1));
        let bad = demutate_binary(&pe, Rva(0), &elo);
        assert_eq!(bad.functions_failed, 1);
    }
}
