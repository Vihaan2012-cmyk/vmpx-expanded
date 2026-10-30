//! SP6 Wave-3: the solver tournament wired together. Synthetic benchmark,
//! tournament runner (input-test scored, ELO ranked) and the `demutate` pipeline.

pub mod bench;
pub mod pipeline;
pub mod registry;
pub mod tournament;

pub use pipeline::{Demutated, Tier, demutate, demutate_verified};

use vmpx_vm::{BinaryOp, Expr, ExprKind, UnaryOp};

/// Compact infix rendering for reports.
#[must_use]
pub fn fmt_expr(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Const { value } => value.to_string(),
        ExprKind::Register { name } => name.clone(),
        ExprKind::Unary { op, value } => match op {
            UnaryOp::Not => format!("~{}", fmt_expr(value)),
            UnaryOp::Neg => format!("-{}", fmt_expr(value)),
        },
        ExprKind::Binary { op, left, right } => {
            let s = match op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Xor => "^",
                BinaryOp::And => "&",
                BinaryOp::Or => "|",
                BinaryOp::Shl => "<<",
                BinaryOp::Shr => ">>",
                BinaryOp::Sar => ">>s",
                BinaryOp::Rol => "rol",
                BinaryOp::Ror => "ror",
            };
            format!("({} {s} {})", fmt_expr(left), fmt_expr(right))
        }
        _ => format!("{e:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bench::{BENCH_CLASSES, generate};
    use vmpx_mutation::{BitWidth, ProblemClass};
    use vmpx_tournament::Elo;
    use vmpx_verify::input_testing::InputTester;

    #[test]
    fn benchmark_pairs_are_equivalent_and_deterministic() {
        let a = generate(7, 6, BitWidth::W8);
        let b = generate(7, 6, BitWidth::W8);
        assert_eq!(a.len(), b.len());
        assert!(a.len() >= BENCH_CLASSES.len() * 4);
        let tester = InputTester::default();
        for (p, q) in a.iter().zip(&b) {
            assert_eq!(p.obfuscated, q.obfuscated);
            assert_ne!(p.obfuscated, p.clean);
            let r = tester.test(&p.clean, &p.obfuscated);
            assert_eq!(r.equal, r.total, "{:?}: {}", p.class, fmt_expr(&p.obfuscated));
        }
    }

    #[test]
    fn tournament_produces_leaderboard() {
        let cfg = tournament::Config {
            per_class: 4,
            gate: false,
            ..tournament::Config::default()
        };
        let report = tournament::run(&cfg);
        assert!(report.winner(ProblemClass::LinearMba).is_some());
        assert!(report.render().contains("Winner per class"));
    }

    #[test]
    fn demutate_reduces_known_obfuscation() {
        let w = BitWidth::W8;
        let x = Expr::register(w, "x");
        let y = Expr::register(w, "y");
        let b = |op, l: Expr, r: Expr| Expr {
            width: w,
            kind: ExprKind::Binary {
                op,
                left: vmpx_vm::ExprRef::new(l),
                right: vmpx_vm::ExprRef::new(r),
            },
        };
        let two = Expr::constant(w, 2);
        let obf = b(
            BinaryOp::Add,
            b(BinaryOp::Xor, x.clone(), y.clone()),
            b(BinaryOp::Mul, two, b(BinaryOp::And, x.clone(), y.clone())),
        );
        let clean = Expr::binary(BinaryOp::Add, x, y).unwrap();
        let elo = Elo::new();
        let out = demutate_verified(&obf, ProblemClass::LinearMba, &elo).expect("simplified");
        assert!(out.expr.node_count_bounded(1000) < obf.node_count_bounded(1000));
        let r = InputTester::default().test(&clean, &out.expr);
        assert_eq!(r.equal, r.total);
        assert_eq!(out.tier, Tier::Sound);
    }
}
