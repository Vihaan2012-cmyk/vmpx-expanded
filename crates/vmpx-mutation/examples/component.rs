// How many functions does a full PE actually need replaced?
//
// The two mutated natives are not self-contained: their recovered CFGs end in
// `Call { target: Direct(..) }` into `.vmp2`, so a replacement for them needs
// replacements for their callees, recursively. This walks that component --
// breadth-first over direct call and tail-call targets from the given seeds --
// and reports the size of the job instead of guessing at it.
//
// Per function it records whether control flow closed (`is_complete`) and
// whether every body lifted (`all_bodies_lifted`), because those are the two
// different things a replacement needs.
//
// Usage: component IMAGE RVA [RVA ...]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use vmpx_mutation::{CallTarget, CfgOptions, NativeCfg, Terminator, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};

/// Direct call / tail-call targets of this CFG: the functions a replacement for
/// it would also have to stand in for.
fn callees(cfg: &NativeCfg) -> BTreeSet<u64> {
    let mut out = BTreeSet::new();
    for block in cfg.blocks.values() {
        let target = match &block.terminator {
            Terminator::Call { target, .. } | Terminator::TailCall { target } => target,
            _ => continue,
        };
        // An Import target is a real DLL import, which a replacement keeps as
        // an import rather than reimplementing.
        if let CallTarget::Direct(va) = target {
            out.insert(*va);
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();
    let options = CfgOptions {
        model_calls: true,
        ..CfgOptions::PROVEN
    };
    let section = |rva: u32| {
        pe.sections
            .iter()
            .find(|s| s.contains_mapped_rva(Rva(rva)))
            .map_or("(unmapped)", |s| s.name.as_str())
    };

    let mut queue: VecDeque<u32> = args[2..]
        .iter()
        .filter_map(|a| u32::from_str_radix(a.trim_start_matches("0x"), 16).ok())
        .collect();
    let seeds: BTreeSet<u32> = queue.iter().copied().collect();
    let mut seen: BTreeSet<u32> = seeds.clone();
    let mut done: BTreeMap<u32, (usize, usize, bool, bool)> = BTreeMap::new();
    let mut refused: BTreeMap<u32, String> = BTreeMap::new();
    let mut by_section: BTreeMap<&str, usize> = BTreeMap::new();
    // A component can be large; stop rather than run away, and say so.
    const MAX: usize = 4096;

    while let Some(rva) = queue.pop_front() {
        if done.len() + refused.len() >= MAX {
            println!("STOPPED at {MAX} functions; the component is larger than this bound");
            break;
        }
        *by_section.entry(section(rva)).or_default() += 1;
        match recover_native_cfg_pe_with(&pe, Rva(rva), options) {
            Err(error) => {
                refused.insert(rva, error.to_string());
            }
            Ok(cfg) => {
                let lifted = cfg.blocks.values().filter(|b| b.semantics.is_some()).count();
                done.insert(
                    rva,
                    (
                        cfg.blocks.len(),
                        lifted,
                        cfg.is_complete(),
                        cfg.all_bodies_lifted(),
                    ),
                );
                for va in callees(&cfg) {
                    if let Some(next) = va.checked_sub(base).and_then(|v| u32::try_from(v).ok())
                        && seen.insert(next)
                    {
                        queue.push_back(next);
                    }
                }
            }
        }
    }

    let complete = done.values().filter(|(_, _, c, _)| *c).count();
    let all_lifted = done.values().filter(|(_, _, _, l)| *l).count();
    let blocks: usize = done.values().map(|(b, _, _, _)| *b).sum();
    let lifted: usize = done.values().map(|(_, l, _, _)| *l).sum();

    println!("seeds: {}", seeds.len());
    println!("component reached: {} function(s)", seen.len());
    println!("  recovered            : {}", done.len());
    println!("  refused outright     : {}", refused.len());
    println!("  control flow complete: {complete} of {}", done.len());
    println!("  all bodies lifted    : {all_lifted} of {}", done.len());
    println!("  blocks               : {blocks} ({lifted} lifted)");
    println!("\nfunctions by section:");
    for (name, count) in &by_section {
        println!("  {name:>10}: {count}");
    }
    if !refused.is_empty() {
        println!("\nrefusals:");
        for (rva, why) in refused.iter().take(10) {
            println!("  {rva:#x}: {why}");
        }
    }
    let incomplete: Vec<_> = done
        .iter()
        .filter(|(_, (_, _, c, l))| !c || !l)
        .take(20)
        .collect();
    if !incomplete.is_empty() {
        println!("\nfunctions still missing control flow or semantics:");
        for (rva, (b, l, c, al)) in incomplete {
            println!("  {rva:#x}: {b} block(s), {l} lifted, complete={c}, all_bodies={al}");
        }
    }
}
