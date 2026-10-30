use std::process::ExitCode;

use vmpx_demutate::tournament::{self, Config};
use vmpx_demutate::{Tier, demutate_verified, fmt_expr};
use vmpx_mutation::{BitWidth, Expr, ProblemClass};
use vmpx_vm::{BinaryOp, ExprKind, ExprRef, UnaryOp};

fn parse_cfg(args: &[String]) -> Result<Config, String> {
    let mut cfg = Config::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--seed" => cfg.seed = val()?.parse().map_err(|e| format!("{e}"))?,
            "--per-class" => cfg.per_class = val()?.parse().map_err(|e| format!("{e}"))?,
            "--width" => {
                cfg.width = match val()?.as_str() {
                    "8" => BitWidth::W8,
                    "16" => BitWidth::W16,
                    "32" => BitWidth::W32,
                    "64" => BitWidth::W64,
                    w => return Err(format!("unsupported width {w}")),
                }
            }
            "--no-gate" => cfg.gate = false,
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(cfg)
}

fn raw(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr {
        width: l.width,
        kind: ExprKind::Binary {
            op,
            left: ExprRef::new(l),
            right: ExprRef::new(r),
        },
    }
}

fn demo(cfg: &Config) {
    let report = tournament::run(&Config { gate: false, ..*cfg });
    let w = cfg.width;
    let (x, y) = (Expr::register(w, "x"), Expr::register(w, "y"));
    let k = |v| Expr::constant(w, v);
    let not = |e: Expr| Expr {
        width: e.width,
        kind: ExprKind::Unary {
            op: UnaryOp::Not,
            value: ExprRef::new(e),
        },
    };
    let samples = [
        (
            "x+y via xor/and",
            ProblemClass::LinearMba,
            raw(
                BinaryOp::Add,
                raw(BinaryOp::Xor, x.clone(), y.clone()),
                raw(BinaryOp::Mul, k(2), raw(BinaryOp::And, x.clone(), y.clone())),
            ),
        ),
        (
            "~~x ^ 0",
            ProblemClass::Substitution,
            raw(BinaryOp::Xor, not(not(x.clone())), k(0)),
        ),
        (
            "x | (y & 0)",
            ProblemClass::DeadCode,
            raw(
                BinaryOp::Or,
                x.clone(),
                raw(BinaryOp::And, y.clone(), k(0)),
            ),
        ),
        (
            "x*y via and/or",
            ProblemClass::NonlinearMba,
            raw(
                BinaryOp::Add,
                raw(
                    BinaryOp::Mul,
                    raw(BinaryOp::And, x.clone(), y.clone()),
                    raw(BinaryOp::Or, x.clone(), y.clone()),
                ),
                raw(
                    BinaryOp::Mul,
                    raw(BinaryOp::And, x.clone(), not(y.clone())),
                    raw(BinaryOp::And, not(x.clone()), y.clone()),
                ),
            ),
        ),
        (
            "(x*(x+1)) & 1",
            ProblemClass::OpaquePredicate,
            raw(
                BinaryOp::And,
                raw(BinaryOp::Mul, x.clone(), raw(BinaryOp::Add, x, k(1))),
                k(1),
            ),
        ),
    ];
    for (name, class, e) in samples {
        println!("[{name}] class {class:?}");
        println!("  before: {}", fmt_expr(&e));
        match demutate_verified(&e, class, &report.elo) {
            Some(d) => println!(
                "  after:  {}  [{}]",
                fmt_expr(&d.expr),
                match d.tier {
                    Tier::Sound => "sound: gate-proven",
                    Tier::Experimental => "experimental: input-tested only",
                }
            ),
            None => println!("  after:  (no simplification found)"),
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        eprintln!("usage: vmpx-demutate <tournament|demo> [--seed N] [--per-class N] [--width 8|16|32|64] [--no-gate]");
        return ExitCode::from(2);
    };
    let cfg = match parse_cfg(rest) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match cmd.as_str() {
        "tournament" => print!("{}", tournament::run(&cfg).render()),
        "demo" => demo(&cfg),
        other => {
            eprintln!("unknown command {other}");
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}
