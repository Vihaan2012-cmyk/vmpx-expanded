// Demonstrates the environment-gate fork on a real protected image: recovers
// the CFG of the given VM entries (static only, nothing is run), lists every
// branch gated on a host-environment source, and shows that BOTH arms of each
// labeled fork entered the CFG, with the code reachable only behind them.
//
// Usage: env_gate_demo IMAGE ENTRY_RVA [ENTRY_RVA...]   (hex, e.g. 0x1201a5)
// env XPL_ENTRY_TIMEOUT_SECS overrides the per-entry wall budget (default 300).

use std::collections::{BTreeMap, BTreeSet};

use vmpx_pe::{PeImage, Rva, Va};
use vmpx_vm::{
    EnvironmentGateRoute, EntryProfileRecoveryLimits, VmCfgAssumption, VmCfgLimits,
    VmDiscoveryLimits, analyze_vm_entry_at, recover_vm_cfg, set_branch_refusal_diagnostics,
    take_environment_gate_forks,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("read image");
    let pe = PeImage::parse(&bytes).expect("parse image");
    let secs = std::env::var("XPL_ENTRY_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300);
    for token in args[2..].iter().filter(|a| a.starts_with("0x")) {
        let rva = u32::from_str_radix(token.trim_start_matches("0x"), 16).expect("hex rva");
        println!("=== entry {rva:#x}");
        let Some(candidate) = analyze_vm_entry_at(&pe, Rva(rva), VmDiscoveryLimits::default())
        else {
            println!("  no VM entry candidate");
            continue;
        };
        let limits = VmCfgLimits {
            entry: EntryProfileRecoveryLimits {
                instruction_limit: 512,
                ..EntryProfileRecoveryLimits::default()
            },
            max_wall_time: Some(std::time::Duration::from_secs(secs)),
            ..VmCfgLimits::default()
        };
        set_branch_refusal_diagnostics(true);
        let cfg = match recover_vm_cfg(&pe, &candidate, limits) {
            Ok(cfg) => cfg,
            Err(error) => {
                println!("  cfg recovery failed: {error}");
                continue;
            }
        };
        let forks = take_environment_gate_forks();
        println!(
            "  blocks={} edges={} stalls={} complete={} limit_hit={:?}",
            cfg.blocks.len(),
            cfg.edges.len(),
            cfg.stalls.len(),
            cfg.all_paths_reach_vm_exit(),
            cfg.limit_hit
        );
        println!(
            "  OverApproximatedEnvironmentGate label present: {}",
            cfg.assumptions
                .contains(&VmCfgAssumption::OverApproximatedEnvironmentGate)
        );
        let mut by_route: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
        for fork in &forks {
            by_route
                .entry(format!("{:?}", fork.route))
                .or_default()
                .insert(fork.vip.0);
        }
        for (route, vips) in &by_route {
            println!("  environment-sourced branch sites, route {route}: {}", vips.len());
        }
        let by_address: BTreeMap<Va, Va> = cfg
            .blocks
            .iter()
            .filter_map(|(key, block)| Some((block.address?, *key)))
            .collect();
        // One line per distinct gate site (a site is re-evaluated per path state).
        let mut sites: BTreeMap<u64, (BTreeSet<String>, BTreeSet<String>, BTreeSet<u64>)> =
            BTreeMap::new();
        for fork in forks.iter().filter(|f| f.route != EnvironmentGateRoute::Refused) {
            let site = sites
                .entry(fork.vip.0)
                .or_insert_with(|| (BTreeSet::new(), BTreeSet::new(), BTreeSet::new()));
            site.0.insert(format!("{:?}", fork.route));
            site.1.extend(fork.sources.iter().map(|s| format!("{s:?}")));
            site.2.extend(fork.targets.iter().copied());
        }
        for (vip, (route, sources, targets)) in &sites {
            println!(
                "  GATE vip {vip:#x} routes {route:?} sources {sources:?} -> {} arms",
                targets.len()
            );
            for target in targets {
                let Some(key) = by_address.get(&Va(*target)) else {
                    println!("    arm {target:#x}: NOT in CFG");
                    continue;
                };
                let block = &cfg.blocks[key];
                // Code reachable only through this arm: BFS over recovered edges.
                let mut seen = BTreeSet::from([*key]);
                let mut queue = vec![*key];
                while let Some(at) = queue.pop() {
                    for edge in cfg.edges.iter().filter(|e| e.from == at) {
                        if seen.insert(edge.to) {
                            queue.push(edge.to);
                        }
                    }
                }
                let instructions: usize = seen
                    .iter()
                    .filter_map(|k| cfg.blocks.get(k))
                    .map(|b| b.instructions.len())
                    .sum();
                println!(
                    "    arm {target:#x}: block {key} ({} VM instrs, exit {:?}); reachable behind it: {} blocks / {instructions} VM instrs",
                    block.instructions.len(),
                    block.exit,
                    seen.len(),
                );
            }
        }
        let remaining = cfg
            .stalls
            .iter()
            .filter(|s| forks.iter().any(|f| f.route == EnvironmentGateRoute::Refused && f.vip == s.vip))
            .count();
        println!("  environment-sourced stalls (refused, unbounded/dynamic-only): {remaining}");
    }
}
