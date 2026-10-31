//! Certified optimizing recompilation.
//!
//! Every [`Pass`] must `certify` before it runs. A certificate is either a
//! structural invariant the pass enforces by construction, or a rewrite rule
//! proven equivalent by [`expr::verify_rule`] (with the counterexample kept in
//! the [`Rejection`] when it is not). A pass whose certificate fails is skipped,
//! never run. After each pass the result is re-validated by the backend
//! validator and checked to preserve the volatile instruction sequence; a pass
//! that breaks either is reverted. Whatever is not provably safe stays as is.

mod analysis;
pub mod expr;
mod passes;

use crate::{
    CodeAddress, CompileError, CompiledFunction, Function, Operation, compile, validate_function,
};
use std::collections::BTreeSet;

pub use expr::{Counterexample, ProofMethod, Rule, Verdict, verify_rule};
pub use passes::{
    AlgebraicRules, Alias, ConstantFold, CopyPropagation, DeadDefinitionElimination,
    DeadStoreElimination, alias,
};

/// One operation plus the side-effect flag the passes honour.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Instr {
    pub op: Operation,
    /// Volatile / side-effecting: never edited, removed, or stepped over.
    pub volatile: bool,
}

#[derive(Clone, Debug, Default)]
pub struct OptimizeOptions {
    /// Indices into `Function::operations` of volatile operations.
    pub volatile: BTreeSet<usize>,
    /// Let dead-store elimination step over same-base provably-disjoint
    /// accesses. Off by default: such an access could fault and a handler would
    /// observe the store, so precise faults are preserved unless opted out.
    pub assume_fault_free_disjoint_access: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Certificate {
    /// Enforced by the pass's own guards; stated so it can be audited.
    Structural { invariant: &'static str },
    /// A rewrite rule proven for all widths up to 64.
    Rule {
        name: &'static str,
        method: ProofMethod,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Rejection {
    Refuted {
        rule: &'static str,
        counterexample: Counterexample,
    },
    Unproven {
        rule: &'static str,
    },
}

/// Certifies a rule set: all rules proven, or the first failure.
///
/// # Errors
/// [`Rejection`] naming the first refuted (with counterexample) or unproven rule.
pub fn certify_rules(rules: &[Rule]) -> Result<Vec<Certificate>, Rejection> {
    rules
        .iter()
        .map(|rule| match verify_rule(rule) {
            Verdict::Proven(method) => Ok(Certificate::Rule {
                name: rule.name,
                method,
            }),
            Verdict::Refuted(counterexample) => Err(Rejection::Refuted {
                rule: rule.name,
                counterexample,
            }),
            Verdict::Unknown => Err(Rejection::Unproven { rule: rule.name }),
        })
        .collect()
}

pub trait Pass {
    fn name(&self) -> &'static str;
    /// # Errors
    /// [`Rejection`] if any rule the pass relies on is not proven.
    fn certify(&self) -> Result<Vec<Certificate>, Rejection>;
    /// Rewrites `program` in place and returns the number of rewrites.
    fn run(&self, program: &mut Vec<Instr>, options: &OptimizeOptions) -> usize;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PassStatus {
    Applied,
    /// Certificate failed; the pass never ran.
    Rejected(Rejection),
    /// Ran, but its output was rejected by validation or the volatile check.
    Reverted(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PassReport {
    pub name: &'static str,
    pub certificates: Vec<Certificate>,
    pub rewrites: usize,
    pub status: PassStatus,
}

#[derive(Clone, Debug, Default)]
pub struct OptimizationReport {
    pub passes: Vec<PassReport>,
    pub operations_before: usize,
    pub operations_after: usize,
    pub rounds: usize,
    /// Set when the input itself failed validation; nothing was optimized.
    pub input_error: Option<String>,
}

impl OptimizationReport {
    #[must_use]
    pub fn total_rewrites(&self) -> usize {
        self.passes.iter().map(|p| p.rewrites).sum()
    }

    #[must_use]
    pub fn pass(&self, name: &str) -> Option<&PassReport> {
        self.passes.iter().find(|p| p.name == name)
    }
}

pub struct Optimizer {
    passes: Vec<Box<dyn Pass>>,
}

const MAX_ROUNDS: usize = 8;

impl Optimizer {
    #[must_use]
    pub fn new(passes: Vec<Box<dyn Pass>>) -> Self {
        Self { passes }
    }

    #[must_use]
    pub fn standard() -> Self {
        Self::new(vec![
            Box::new(ConstantFold),
            Box::new(AlgebraicRules::standard()),
            Box::new(CopyPropagation),
            Box::new(DeadDefinitionElimination),
            Box::new(DeadStoreElimination),
        ])
    }

    #[must_use]
    pub fn run(
        &self,
        function: &Function,
        options: &OptimizeOptions,
    ) -> (Function, OptimizationReport) {
        let mut report = OptimizationReport {
            operations_before: function.operations.len(),
            operations_after: function.operations.len(),
            ..OptimizationReport::default()
        };
        if let Err(error) = validate_function(function) {
            report.input_error = Some(error.to_string());
            return (function.clone(), report);
        }
        let mut program: Vec<Instr> = function
            .operations
            .iter()
            .enumerate()
            .map(|(index, op)| Instr {
                op: op.clone(),
                volatile: options.volatile.contains(&index),
            })
            .collect();

        let mut live: Vec<bool> = Vec::new();
        for pass in &self.passes {
            match pass.certify() {
                Ok(certificates) => {
                    live.push(true);
                    report.passes.push(PassReport {
                        name: pass.name(),
                        certificates,
                        rewrites: 0,
                        status: PassStatus::Applied,
                    });
                }
                Err(rejection) => {
                    live.push(false);
                    report.passes.push(PassReport {
                        name: pass.name(),
                        certificates: Vec::new(),
                        rewrites: 0,
                        status: PassStatus::Rejected(rejection),
                    });
                }
            }
        }

        for round in 0..MAX_ROUNDS {
            let mut changed = false;
            for (slot, pass) in self.passes.iter().enumerate() {
                if !live[slot] {
                    continue;
                }
                let mut candidate = program.clone();
                let rewrites = pass.run(&mut candidate, options);
                if rewrites == 0 {
                    continue;
                }
                match check_candidate(function, &program, &candidate) {
                    Ok(()) => {
                        program = candidate;
                        report.passes[slot].rewrites += rewrites;
                        changed = true;
                    }
                    Err(reason) => {
                        report.passes[slot].status = PassStatus::Reverted(reason);
                        live[slot] = false;
                    }
                }
            }
            report.rounds = round + 1;
            if !changed {
                break;
            }
        }
        report.operations_after = program.len();
        (rebuild(function, &program), report)
    }
}

fn rebuild(template: &Function, program: &[Instr]) -> Function {
    Function {
        operations: program.iter().map(|i| i.op.clone()).collect(),
        assignments: template.assignments.clone(),
        frame: template.frame.clone(),
    }
}

fn volatile_sequence(program: &[Instr]) -> Vec<&Operation> {
    program
        .iter()
        .filter(|i| i.volatile)
        .map(|i| &i.op)
        .collect()
}

fn check_candidate(template: &Function, before: &[Instr], after: &[Instr]) -> Result<(), String> {
    if volatile_sequence(before) != volatile_sequence(after) {
        return Err("volatile instruction sequence changed".to_owned());
    }
    validate_function(&rebuild(template, after)).map_err(|error| error.to_string())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmissionTier {
    /// The optimized function was encoded to machine code.
    Optimized,
    /// The optimized function could not be emitted; the original was.
    FellBack { reason: String },
}

#[derive(Clone, Debug)]
pub struct OptimizedCompilation {
    pub compiled: CompiledFunction,
    /// The function whose code is in `compiled`.
    pub emitted: Function,
    pub report: OptimizationReport,
    pub tier: EmissionTier,
}

/// Optimizes then encodes. Falls back to encoding the original unchanged if the
/// optimized function does not encode; the report says which happened.
///
/// # Errors
/// The backend error for the original function when it cannot be compiled.
pub fn compile_optimized(
    function: &Function,
    address: CodeAddress,
    options: &OptimizeOptions,
) -> Result<OptimizedCompilation, CompileError> {
    compile_optimized_with(&Optimizer::standard(), function, address, options)
}

/// [`compile_optimized`] with a caller-supplied pass pipeline.
///
/// # Errors
/// The backend error for the original function when it cannot be compiled.
pub fn compile_optimized_with(
    optimizer: &Optimizer,
    function: &Function,
    address: CodeAddress,
    options: &OptimizeOptions,
) -> Result<OptimizedCompilation, CompileError> {
    let (optimized, report) = optimizer.run(function, options);
    match compile(&optimized, address) {
        Ok(compiled) => Ok(OptimizedCompilation {
            compiled,
            emitted: optimized,
            report,
            tier: EmissionTier::Optimized,
        }),
        Err(error) => Ok(OptimizedCompilation {
            compiled: compile(function, address)?,
            emitted: function.clone(),
            report,
            tier: EmissionTier::FellBack {
                reason: error.to_string(),
            },
        }),
    }
}
