use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;

use crate::{Expr, ExprError, ExprKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymbolicDomainLimits {
    pub max_variables: usize,
    pub max_values_per_variable: usize,
    pub max_combinations: usize,
    pub max_expression_nodes: usize,
}

impl Default for SymbolicDomainLimits {
    fn default() -> Self {
        Self {
            max_variables: 8,
            max_values_per_variable: 256,
            max_combinations: 4_096,
            max_expression_nodes: 200_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SymbolicConcreteAlternative {
    pub value: u64,
    pub assignments: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SymbolicDomainReport {
    pub schema_version: u32,
    pub dependencies: Vec<String>,
    pub unresolved_dependencies: Vec<String>,
    pub combinations_evaluated: usize,
    pub unique_values: Vec<u64>,
    pub alternatives: Vec<SymbolicConcreteAlternative>,
    pub complete: bool,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum SymbolicDomainError {
    #[error("expression contains {actual} nodes, exceeding limit {limit}")]
    ExpressionLimit { actual: usize, limit: usize },

    #[error("expression has {actual} dependencies, exceeding limit {limit}")]
    VariableLimit { actual: usize, limit: usize },

    #[error("domain for {name} is empty")]
    EmptyDomain { name: String },

    #[error("domain for {name} disappeared during validation")]
    MissingDomain { name: String },

    #[error("domain for {name} has {actual} values, exceeding limit {limit}")]
    DomainValueLimit {
        name: String,
        actual: usize,
        limit: usize,
    },

    #[error("domain value {index} for {name} is not a constant")]
    NonConstantDomainValue { name: String, index: usize },

    #[error(
        "domain value {index} for {name} has width {actual:?}, expected expression dependency width {expected:?}"
    )]
    DomainWidthMismatch {
        name: String,
        index: usize,
        expected: crate::BitWidth,
        actual: crate::BitWidth,
    },

    #[error("domain combination count exceeds limit {limit}")]
    CombinationLimit { limit: usize },

    #[error("substitution failed: {0}")]
    Expression(#[from] ExprError),

    #[error("expression remained symbolic after substituting one complete domain assignment")]
    NonConcreteResult,
}

fn dependency_widths(expression: &Expr) -> BTreeMap<String, crate::BitWidth> {
    let mut output = BTreeMap::new();
    let mut pending = vec![expression];
    while let Some(current) = pending.pop() {
        match &current.kind {
            ExprKind::Register { name } => {
                output.entry(name.clone()).or_insert(current.width);
            }
            ExprKind::Const { .. } => {}
            ExprKind::Unary { value, .. }
            | ExprKind::Load { address: value }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => pending.push(value),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat {
                high: left,
                low: right,
            } => {
                pending.push(right);
                pending.push(left);
            }
            ExprKind::Intrinsic { arguments, .. } => pending.extend(arguments),
        }
    }
    output
}

fn domain_constants(
    name: &str,
    expected_width: crate::BitWidth,
    values: &[Expr],
    limits: SymbolicDomainLimits,
) -> Result<Vec<Expr>, SymbolicDomainError> {
    if values.is_empty() {
        return Err(SymbolicDomainError::EmptyDomain {
            name: name.to_owned(),
        });
    }
    if values.len() > limits.max_values_per_variable {
        return Err(SymbolicDomainError::DomainValueLimit {
            name: name.to_owned(),
            actual: values.len(),
            limit: limits.max_values_per_variable,
        });
    }
    for (index, value) in values.iter().enumerate() {
        if value.width != expected_width {
            return Err(SymbolicDomainError::DomainWidthMismatch {
                name: name.to_owned(),
                index,
                expected: expected_width,
                actual: value.width,
            });
        }
        if !matches!(value.kind, ExprKind::Const { .. }) {
            return Err(SymbolicDomainError::NonConstantDomainValue {
                name: name.to_owned(),
                index,
            });
        }
    }
    Ok(values.to_vec())
}

fn evaluate_assignment(
    expression: &Expr,
    replacements: &BTreeMap<String, Expr>,
) -> Result<u64, SymbolicDomainError> {
    let evaluated = expression.substitute(replacements)?;
    match evaluated.kind {
        ExprKind::Const { value } => Ok(value),
        _ => Err(SymbolicDomainError::NonConcreteResult),
    }
}

struct Enumeration<'a> {
    expression: &'a Expr,
    variables: &'a [(String, Vec<Expr>)],
    limits: SymbolicDomainLimits,
    replacements: BTreeMap<String, Expr>,
    alternatives: Vec<SymbolicConcreteAlternative>,
    unique_values: BTreeSet<u64>,
}

impl Enumeration<'_> {
    fn walk(&mut self, index: usize) -> Result<(), SymbolicDomainError> {
        if self.alternatives.len() >= self.limits.max_combinations {
            return Err(SymbolicDomainError::CombinationLimit {
                limit: self.limits.max_combinations,
            });
        }
        if index == self.variables.len() {
            let value = evaluate_assignment(self.expression, &self.replacements)?;
            let assignments = self
                .replacements
                .iter()
                .map(|(name, expression)| {
                    let ExprKind::Const { value } = expression.kind else {
                        unreachable!("validated domain assignments are constants")
                    };
                    (name.clone(), value)
                })
                .collect();
            self.unique_values.insert(value);
            self.alternatives
                .push(SymbolicConcreteAlternative { value, assignments });
            return Ok(());
        }
        let (name, values) = &self.variables[index];
        for value in values {
            self.replacements.insert(name.clone(), value.clone());
            self.walk(index + 1)?;
        }
        self.replacements.remove(name);
        Ok(())
    }
}

/// Enumerates a symbolic value over explicit, finite constant domains.
/// Missing dependencies produce an incomplete report with no guessed values.
///
/// # Errors
///
/// Rejects malformed domains, resource-limit violations, width mismatches, and
/// expressions that remain symbolic after a complete assignment.
pub fn enumerate_symbolic_value(
    expression: &Expr,
    domains: &BTreeMap<String, Vec<Expr>>,
    limits: SymbolicDomainLimits,
) -> Result<SymbolicDomainReport, SymbolicDomainError> {
    let actual_nodes = expression.node_count_bounded(limits.max_expression_nodes);
    if actual_nodes > limits.max_expression_nodes {
        return Err(SymbolicDomainError::ExpressionLimit {
            actual: actual_nodes,
            limit: limits.max_expression_nodes,
        });
    }
    let dependency_widths = dependency_widths(expression);
    if dependency_widths.len() > limits.max_variables {
        return Err(SymbolicDomainError::VariableLimit {
            actual: dependency_widths.len(),
            limit: limits.max_variables,
        });
    }
    let dependencies: Vec<_> = dependency_widths.keys().cloned().collect();
    let unresolved_dependencies: Vec<_> = dependency_widths
        .keys()
        .filter(|name| !domains.contains_key(*name))
        .cloned()
        .collect();
    if !unresolved_dependencies.is_empty() {
        return Ok(SymbolicDomainReport {
            schema_version: 1,
            dependencies,
            unresolved_dependencies,
            combinations_evaluated: 0,
            unique_values: Vec::new(),
            alternatives: Vec::new(),
            complete: false,
        });
    }

    let mut variables = Vec::new();
    let mut combination_count = 1_usize;
    for (name, width) in dependency_widths {
        let domain = domains
            .get(&name)
            .ok_or_else(|| SymbolicDomainError::MissingDomain { name: name.clone() })?;
        let values = domain_constants(&name, width, domain, limits)?;
        combination_count = combination_count.checked_mul(values.len()).ok_or(
            SymbolicDomainError::CombinationLimit {
                limit: limits.max_combinations,
            },
        )?;
        if combination_count > limits.max_combinations {
            return Err(SymbolicDomainError::CombinationLimit {
                limit: limits.max_combinations,
            });
        }
        variables.push((name, values));
    }

    let mut enumeration = Enumeration {
        expression,
        variables: &variables,
        limits,
        replacements: BTreeMap::new(),
        alternatives: Vec::with_capacity(combination_count),
        unique_values: BTreeSet::new(),
    };
    enumeration.walk(0)?;
    Ok(SymbolicDomainReport {
        schema_version: 1,
        dependencies,
        unresolved_dependencies: Vec::new(),
        combinations_evaluated: enumeration.alternatives.len(),
        unique_values: enumeration.unique_values.into_iter().collect(),
        alternatives: enumeration.alternatives,
        complete: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryOp, BitWidth};

    #[test]
    fn enumerates_and_deduplicates_finite_symbolic_targets() {
        let selector = Expr::register(BitWidth::W64, "selector");
        let low_bit = Expr::binary(BinaryOp::And, selector, Expr::constant(BitWidth::W64, 1))
            .expect("same widths");
        let offset = Expr::binary(BinaryOp::Mul, low_bit, Expr::constant(BitWidth::W64, 0x100))
            .expect("same widths");
        let target = Expr::binary(
            BinaryOp::Add,
            Expr::constant(BitWidth::W64, 0x0014_0000_1000),
            offset,
        )
        .expect("same widths");
        let domains = BTreeMap::from([(
            "selector".to_owned(),
            (0_u64..=3)
                .map(|value| Expr::constant(BitWidth::W64, value))
                .collect(),
        )]);
        let report = enumerate_symbolic_value(&target, &domains, SymbolicDomainLimits::default())
            .expect("finite domain must enumerate");
        assert!(report.complete);
        assert_eq!(report.combinations_evaluated, 4);
        assert_eq!(
            report.unique_values,
            vec![0x0014_0000_1000, 0x0014_0000_1100]
        );
    }

    #[test]
    fn missing_domains_remain_explicitly_unresolved() {
        let expression = Expr::register(BitWidth::W32, "unknown_input");
        let report = enumerate_symbolic_value(
            &expression,
            &BTreeMap::new(),
            SymbolicDomainLimits::default(),
        )
        .expect("missing domain is a report state");
        assert!(!report.complete);
        assert_eq!(report.unresolved_dependencies, vec!["unknown_input"]);
        assert!(report.unique_values.is_empty());
    }

    #[test]
    fn combination_limits_are_checked_before_enumeration() {
        let left = Expr::register(BitWidth::W8, "left");
        let right = Expr::register(BitWidth::W8, "right");
        let expression = Expr::binary(BinaryOp::Xor, left, right).expect("same widths");
        let values = (0_u64..4)
            .map(|value| Expr::constant(BitWidth::W8, value))
            .collect::<Vec<_>>();
        let error = enumerate_symbolic_value(
            &expression,
            &BTreeMap::from([
                ("left".to_owned(), values.clone()),
                ("right".to_owned(), values),
            ]),
            SymbolicDomainLimits {
                max_combinations: 8,
                ..SymbolicDomainLimits::default()
            },
        )
        .expect_err("sixteen combinations exceed the limit");
        assert_eq!(error, SymbolicDomainError::CombinationLimit { limit: 8 });
    }
}
