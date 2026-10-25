// Proves the native backend can emit and validate a real function end to end.
use vmpx_recompile::{
    BinaryOp, BinarySource, CodeAddress, Function, Operation, PhysicalRegister, RegisterAssignment,
    StackFrame, VirtualRegister, compile,
};

fn main() {
    let r0 = VirtualRegister::qword(0);
    let r1 = VirtualRegister::qword(1);
    let function = Function {
        operations: vec![
            Operation::MoveImmediate {
                destination: r0,
                value: 0x1000,
            },
            Operation::MoveImmediate {
                destination: r1,
                value: 0x234,
            },
            Operation::Binary {
                operation: BinaryOp::Add,
                destination: r0,
                source: BinarySource::Register(r1),
            },
            Operation::Return,
        ],
        assignments: vec![
            RegisterAssignment::new(r0, PhysicalRegister::Rax),
            RegisterAssignment::new(r1, PhysicalRegister::Rcx),
        ],
        frame: StackFrame::default(),
    };
    match compile(&function, CodeAddress(0x1_4000_1000)) {
        Ok(compiled) => println!(
            "compiled {} bytes: {}",
            compiled.bytes.len(),
            compiled
                .bytes
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ),
        Err(e) => println!("compile failed: {e}"),
    }
}
