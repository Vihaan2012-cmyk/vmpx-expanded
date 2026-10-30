//! Synthetic VMProtect-shaped VM fixture generator (test support only).
//!
//! Emits a small x64 PE32+ whose entry point is a `push seed; call bootstrap`
//! stub into a hand-assembled VM: full native-state save, cursor capture,
//! stack reservation, a keyed relative-dispatch tail, five handler families
//! (context write, context read, constant push, virtual branch, VM exit) and
//! an encoded bytecode stream that scatters the saved frame into a context,
//! rebuilds a restore frame and exits to a known native RVA.
//!
//! The VM is NOT `VMProtect`. Handler shapes are the minimal ones the current
//! recognizers accept; see `synthetic_pipeline.rs` for what this can and
//! cannot validate.

#![allow(dead_code)]

use iced_x86::code_asm::{
    AsmRegister64, CodeAssembler, CodeLabel, al, bpl, byte_ptr, dword_ptr, eax, ebp, ptr,
    qword_ptr, r8, r9, r10, r10d, r11, r12, r13, r14, r15, rax, rbp, rbx, rcx, rdi, rdx, rsi, rsp,
};
use iced_x86::{BlockEncoderOptions, IcedError};
use vmpx_vm::{
    BitWidth, KeyOperation, KeyedTransformCodec, Transform, TransformChain, TransformOp,
};

pub const IMAGE_BASE: u64 = 0x1_4000_0000;
pub const TEXT_RVA: u32 = 0x1000;
/// RAX forced by the linear program's restore frame instead of the saved one.
pub const FORCED_RAX: u64 = 0x539;
/// RAX forced on the two branches of the branching program.
pub const BRANCH_RAX: [u64; 2] = [0xA1, 0xB2];
/// Context byte-offset of native frame slot `i` is `CONTEXT_BASE + 8 * i`.
pub const CONTEXT_BASE: u8 = 0x10;
/// Number of qword slots in the saved native frame (15 GPRs, RFLAGS, zero
/// sentinel, call return address, entry immediate).
pub const FRAME_SLOTS: usize = 19;
const FILE_ALIGN: usize = 0x200;
const SECTION_ALIGN: usize = 0x1000;
/// File offset of `AddressOfEntryPoint` in generated images.
pub const ENTRY_POINT_FIELD_OFFSET: usize = 0x84 + 20 + 16;

/// File offset of a `.text` RVA in generated images.
pub fn file_offset(rva: u32) -> usize {
    FILE_ALIGN + usize::try_from(rva - TEXT_RVA).expect("small")
}

const DISPATCH_ADD: u32 = 0x7638_338D;
const DISPATCH_XOR: u32 = 0x76DF_F93A;
const CONST_ADD: i32 = 0x0A0B_0C0D;

/// Native frame slot indices, in decoder order (slot 0 is the VSP top).
pub mod slot {
    pub const SENTINEL: usize = 0;
    pub const RFLAGS: usize = 1;
    /// Slot of the GPR pushed `push_index`-th in the order
    /// rax,rbx,rcx,rdx,rbp,rsi,rdi,r8..r15.
    pub const fn gpr(push_index: usize) -> usize {
        16 - push_index
    }
    pub const RETURN_ADDRESS: usize = 17;
    pub const ENTRY_IMMEDIATE: usize = 18;
    pub const RCX_PUSH_INDEX: usize = 2;
    pub const RDX_PUSH_INDEX: usize = 3;
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Write(u8),
    Read(u8),
    Push(u64),
    PushOep(usize),
    Branch,
    Exit,
}

#[derive(Clone, Copy, Debug)]
enum Handler {
    Write,
    Read,
    Push,
    Branch,
    Exit,
}

impl Op {
    fn handler(self) -> Handler {
        match self {
            Self::Write(_) => Handler::Write,
            Self::Read(_) => Handler::Read,
            Self::Push(_) | Self::PushOep(_) => Handler::Push,
            Self::Branch => Handler::Branch,
            Self::Exit => Handler::Exit,
        }
    }

    fn operand_width(self) -> Option<BitWidth> {
        match self {
            Self::Write(_) | Self::Read(_) => Some(BitWidth::W8),
            Self::Push(_) | Self::PushOep(_) => Some(BitWidth::W64),
            Self::Branch | Self::Exit => None,
        }
    }

    fn is_terminal(self) -> bool {
        matches!(self, Self::Branch | Self::Exit)
    }
}

type Block = Vec<Op>;

#[derive(Clone, Copy, Debug)]
struct Layout {
    entry: u64,
    bootstrap: u64,
    anchor: u64,
    write: u64,
    read: u64,
    push: u64,
    branch: u64,
    exit: u64,
    oeps: [u64; 2],
    bytecode: u64,
}

impl Layout {
    fn handler(&self, handler: Handler) -> u64 {
        match handler {
            Handler::Write => self.write,
            Handler::Read => self.read,
            Handler::Push => self.push,
            Handler::Branch => self.branch,
            Handler::Exit => self.exit,
        }
    }
}

/// Everything a test needs to know about a generated image.
#[derive(Clone, Debug)]
pub struct SyntheticVm {
    pub image: Vec<u8>,
    pub image_base: u64,
    pub entry_rva: u32,
    pub seed: u32,
    /// RVA of the bootstrap stack-reservation (`sub rsp, imm`).
    pub bootstrap_rva: u32,
    /// RVAs of the two native continuations the bytecode can exit to.
    pub oep_rvas: [u32; 2],
    pub bytecode_rva: u32,
    /// Absolute VA of the first token of every bytecode block.
    pub block_vas: Vec<u64>,
    pub write_handler_rva: u32,
    pub read_handler_rva: u32,
    pub push_handler_rva: u32,
    pub branch_handler_rva: u32,
    pub exit_handler_rva: u32,
}

fn w8_codec() -> KeyedTransformCodec {
    let transform = |op, operand| Transform {
        op,
        width: BitWidth::W8,
        operand,
    };
    KeyedTransformCodec {
        width: BitWidth::W8,
        input_key_operation: Some(KeyOperation::Xor),
        input_key_on_left: false,
        transforms: TransformChain {
            transforms: vec![
                transform(TransformOp::Sub, 0x6D),
                transform(TransformOp::Rol, 1),
                transform(TransformOp::Neg, 0),
            ],
        },
        update_key_operation: Some(KeyOperation::Xor),
    }
}

fn w64_codec() -> KeyedTransformCodec {
    let transform = |op, operand| Transform {
        op,
        width: BitWidth::W64,
        operand,
    };
    KeyedTransformCodec {
        width: BitWidth::W64,
        input_key_operation: Some(KeyOperation::Xor),
        input_key_on_left: false,
        transforms: TransformChain {
            transforms: vec![
                transform(TransformOp::Sub, 1),
                transform(TransformOp::Bswap, 0),
                transform(TransformOp::Add, u64::from(CONST_ADD.cast_unsigned())),
                transform(TransformOp::Neg, 0),
            ],
        },
        update_key_operation: Some(KeyOperation::Xor),
    }
}

fn w32_dispatch_codec() -> KeyedTransformCodec {
    let transform = |op, operand| Transform {
        op,
        width: BitWidth::W32,
        operand,
    };
    KeyedTransformCodec {
        width: BitWidth::W32,
        input_key_operation: Some(KeyOperation::Xor),
        input_key_on_left: false,
        transforms: TransformChain {
            transforms: vec![
                transform(TransformOp::Ror, 1),
                transform(TransformOp::Neg, 0),
                transform(TransformOp::Add, u64::from(DISPATCH_ADD)),
                transform(TransformOp::Xor, u64::from(DISPATCH_XOR)),
            ],
        },
        update_key_operation: Some(KeyOperation::Xor),
    }
}

/// Encodes `decoded` so `codec.decode(encoded, key)` yields it, then verifies.
fn encode(codec: &KeyedTransformCodec, decoded: u64, key: u64) -> (u64, u64) {
    let mixed = codec
        .transforms
        .inverse()
        .apply(decoded)
        .expect("invertible chain");
    let encoded = mixed ^ codec.width.truncate(key);
    let check = codec.decode(encoded, key).expect("codec decodes");
    assert_eq!(check.decoded, codec.width.truncate(decoded));
    (encoded, check.key_after)
}

fn emit_dispatch(asm: &mut CodeAssembler) -> Result<(), IcedError> {
    asm.mov(r10d, dword_ptr(r15))?;
    asm.add(r15, 4)?;
    asm.xor(r10d, ebp)?;
    asm.ror(r10d, 1)?;
    asm.neg(r10d)?;
    asm.add(r10d, DISPATCH_ADD)?;
    asm.xor(r10d, DISPATCH_XOR)?;
    asm.push(rbp)?;
    asm.xor(dword_ptr(rsp), r10d)?;
    asm.pop(rbp)?;
    asm.movsxd(r10, r10d)?;
    asm.add(r14, r10)?;
    asm.jmp(r14)
}

fn emit_w8_operand(asm: &mut CodeAssembler) -> Result<(), IcedError> {
    asm.movzx(eax, byte_ptr(r15))?;
    asm.inc(r15)?;
    asm.xor(al, bpl)?;
    asm.sub(al, 0x6D)?;
    asm.rol(al, 1)?;
    asm.neg(al)?;
    asm.push(rbp)?;
    asm.xor(byte_ptr(rsp), al)?;
    asm.pop(rbp)
}

fn emit_w64_operand(asm: &mut CodeAssembler) -> Result<(), IcedError> {
    asm.mov(rax, qword_ptr(r15))?;
    asm.add(r15, 8)?;
    asm.xor(rax, rbp)?;
    asm.sub(rax, 1)?;
    asm.bswap(rax)?;
    asm.add(rax, CONST_ADD)?;
    asm.neg(rax)?;
    asm.push(rbp)?;
    asm.xor(qword_ptr(rsp), rax)?;
    asm.pop(rbp)
}

fn bind(asm: &mut CodeAssembler, label: &mut CodeLabel) -> Result<(), IcedError> {
    asm.set_label(label)
}

#[allow(clippy::too_many_lines)]
fn assemble(seed: u32, bytecode: &[u8]) -> Result<(Vec<u8>, Layout), IcedError> {
    let mut asm = CodeAssembler::new(64)?;
    let mut entry = asm.create_label();
    let mut boot = asm.create_label();
    let mut reserve = asm.create_label();
    let mut anchor = asm.create_label();
    let mut write = asm.create_label();
    let mut read = asm.create_label();
    let mut push = asm.create_label();
    let mut branch = asm.create_label();
    let mut exit = asm.create_label();
    let mut oep0 = asm.create_label();
    let mut oep1 = asm.create_label();
    let mut data = asm.create_label();
    let anchor_ref = anchor;
    let boot_ref = boot;

    bind(&mut asm, &mut entry)?;
    asm.push(seed.cast_signed())?;
    asm.call(boot_ref)?;
    asm.ret()?;

    bind(&mut asm, &mut boot)?;
    let saved: [AsmRegister64; 15] = [
        rax, rbx, rcx, rdx, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15,
    ];
    for register in saved {
        asm.push(register)?;
    }
    asm.pushfq()?;
    asm.xor(r10d, r10d)?;
    asm.push(r10)?;
    asm.mov(rax, qword_ptr(rsp + 144))?;
    asm.mov(r11, rsp)?;
    bind(&mut asm, &mut reserve)?;
    asm.sub(rsp, 0x180)?;
    asm.and(rsp, -16)?;
    asm.mov(r15, IMAGE_BASE)?;
    asm.add(r15, rax)?;
    asm.mov(rbp, r15)?;
    asm.lea(r14, ptr(anchor_ref))?;
    emit_dispatch(&mut asm)?;

    bind(&mut asm, &mut anchor)?;
    asm.int3()?;

    bind(&mut asm, &mut write)?;
    emit_w8_operand(&mut asm)?;
    asm.mov(rcx, qword_ptr(r11))?;
    asm.add(r11, 8)?;
    asm.mov(qword_ptr(rsp + rax), rcx)?;
    emit_dispatch(&mut asm)?;

    bind(&mut asm, &mut read)?;
    emit_w8_operand(&mut asm)?;
    asm.mov(rcx, qword_ptr(rsp + rax))?;
    asm.sub(r11, 8)?;
    asm.mov(qword_ptr(r11), rcx)?;
    emit_dispatch(&mut asm)?;

    bind(&mut asm, &mut push)?;
    emit_w64_operand(&mut asm)?;
    asm.sub(r11, 8)?;
    asm.mov(qword_ptr(r11), rax)?;
    emit_dispatch(&mut asm)?;

    bind(&mut asm, &mut branch)?;
    asm.mov(rax, qword_ptr(r11))?;
    asm.add(r11, 8)?;
    asm.mov(r15, rax)?;
    asm.mov(rbp, r15)?;
    asm.lea(r14, ptr(anchor_ref))?;
    emit_dispatch(&mut asm)?;

    bind(&mut asm, &mut exit)?;
    asm.mov(rsp, r11)?;
    for register in saved {
        asm.pop(register)?;
    }
    asm.popfq()?;
    asm.ret()?;

    bind(&mut asm, &mut oep0)?;
    asm.mov(eax, 7)?;
    asm.ret()?;
    bind(&mut asm, &mut oep1)?;
    asm.mov(eax, 9)?;
    asm.ret()?;

    bind(&mut asm, &mut data)?;
    asm.db(bytecode)?;

    let result = asm.assemble_options(
        IMAGE_BASE + u64::from(TEXT_RVA),
        BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
    )?;
    let layout = Layout {
        entry: result.label_ip(&entry)?,
        bootstrap: result.label_ip(&reserve)?,
        anchor: result.label_ip(&anchor)?,
        write: result.label_ip(&write)?,
        read: result.label_ip(&read)?,
        push: result.label_ip(&push)?,
        branch: result.label_ip(&branch)?,
        exit: result.label_ip(&exit)?,
        oeps: [result.label_ip(&oep0)?, result.label_ip(&oep1)?],
        bytecode: result.label_ip(&data)?,
    };
    Ok((result.inner.code_buffer, layout))
}

fn ctx(slot: usize) -> u8 {
    CONTEXT_BASE + u8::try_from(slot * 8).expect("context offset fits")
}

fn scatter() -> Block {
    (0..FRAME_SLOTS).map(|i| Op::Write(ctx(i))).collect()
}

/// Restore frame, highest address first: return address, RFLAGS, r15..rax.
/// RCX is replaced by the entry immediate and RAX by `forced_rax`.
fn restore_frame(oep: usize, forced_rax: u64) -> Block {
    let mut ops = vec![Op::PushOep(oep), Op::Read(ctx(slot::RFLAGS))];
    for push_index in (0..15).rev() {
        ops.push(match push_index {
            0 => Op::Push(forced_rax),
            slot::RCX_PUSH_INDEX => Op::Read(ctx(slot::ENTRY_IMMEDIATE)),
            _ => Op::Read(ctx(slot::gpr(push_index))),
        });
    }
    ops.push(Op::Exit);
    ops
}

fn linear_program() -> Vec<Block> {
    let mut block = scatter();
    block.extend(restore_frame(0, FORCED_RAX));
    vec![block]
}

/// Branch on the saved RDX: the virtual branch pops its target from the VM
/// stack, so the successor block is selected by the native RDX at entry.
fn branching_program() -> Vec<Block> {
    let mut dispatcher = scatter();
    dispatcher.push(Op::Read(ctx(slot::gpr(slot::RDX_PUSH_INDEX))));
    dispatcher.push(Op::Branch);
    vec![
        dispatcher,
        restore_frame(0, BRANCH_RAX[0]),
        restore_frame(1, BRANCH_RAX[1]),
    ]
}

/// Scatter, push a constant target, then a virtual branch to a restore-and-exit
/// block. The branch target is the pushed constant, not a native input.
fn const_branch_program(target: u64) -> Vec<Block> {
    let mut dispatcher = scatter();
    dispatcher.push(Op::Push(target));
    dispatcher.push(Op::Branch);
    vec![dispatcher, restore_frame(0, FORCED_RAX)]
}

fn block_len(block: &[Op]) -> usize {
    4 + block
        .iter()
        .filter(|op| !op.is_terminal())
        .map(|op| {
            let operand = op
                .operand_width()
                .map_or(0, |width| usize::try_from(width.bytes()).expect("small"));
            operand + 4
        })
        .sum::<usize>()
}

fn encode_blocks(blocks: &[Block], layout: &Layout) -> (Vec<u8>, Vec<u64>) {
    let dispatch = w32_dispatch_codec();
    let token = |bytes: &mut Vec<u8>, base: u64, target: u64, key: &mut u64| {
        let displacement = i32::try_from(i128::from(target) - i128::from(base))
            .expect("handler displacement fits")
            .cast_unsigned();
        let (encoded, key_after) = encode(&dispatch, u64::from(displacement), *key);
        bytes.extend_from_slice(&u32::try_from(encoded).expect("w32").to_le_bytes());
        *key = key_after;
    };
    let mut bytes = Vec::new();
    let mut starts = Vec::new();
    for block in blocks {
        let start = layout.bytecode + u64::try_from(bytes.len()).expect("small");
        starts.push(start);
        let mut key = start;
        let mut base = layout.anchor;
        token(
            &mut bytes,
            base,
            layout.handler(block[0].handler()),
            &mut key,
        );
        base = layout.handler(block[0].handler());
        for (index, op) in block.iter().enumerate() {
            if op.is_terminal() {
                break;
            }
            match *op {
                Op::Write(operand) | Op::Read(operand) => {
                    let (encoded, key_after) = encode(&w8_codec(), u64::from(operand), key);
                    bytes.push(u8::try_from(encoded).expect("w8"));
                    key = key_after;
                }
                Op::Push(_) | Op::PushOep(_) => {
                    let operand = match *op {
                        Op::Push(value) => value,
                        Op::PushOep(which) => layout.oeps[which],
                        _ => unreachable!(),
                    };
                    let (encoded, key_after) = encode(&w64_codec(), operand, key);
                    bytes.extend_from_slice(&encoded.to_le_bytes());
                    key = key_after;
                }
                Op::Branch | Op::Exit => unreachable!(),
            }
            let next = layout.handler(block[index + 1].handler());
            token(&mut bytes, base, next, &mut key);
            base = next;
        }
    }
    (bytes, starts)
}

fn align(value: usize, to: usize) -> usize {
    value.div_ceil(to) * to
}

fn wrap_pe(text: &[u8], entry_rva: u32) -> Vec<u8> {
    let raw_size = align(text.len(), FILE_ALIGN);
    let virtual_size = text.len();
    let image_size = align(
        usize::try_from(TEXT_RVA).expect("small") + virtual_size,
        SECTION_ALIGN,
    );
    let small = |value: usize| u32::try_from(value).expect("small");
    let mut data = vec![0_u8; FILE_ALIGN + raw_size];
    let put16 = |data: &mut [u8], at: usize, value: u16| {
        data[at..at + 2].copy_from_slice(&value.to_le_bytes());
    };
    let put32 = |data: &mut [u8], at: usize, value: u32| {
        data[at..at + 4].copy_from_slice(&value.to_le_bytes());
    };
    put16(&mut data, 0, 0x5A4D);
    put32(&mut data, 0x3C, 0x80);
    put32(&mut data, 0x80, 0x0000_4550);
    let coff = 0x84;
    put16(&mut data, coff, 0x8664);
    put16(&mut data, coff + 2, 1);
    put16(&mut data, coff + 16, 0xF0);
    put16(&mut data, coff + 18, 0x2022);
    let optional = coff + 20;
    put16(&mut data, optional, 0x20B);
    put32(&mut data, optional + 4, small(raw_size));
    put32(&mut data, optional + 16, entry_rva);
    put32(&mut data, optional + 20, TEXT_RVA);
    data[optional + 24..optional + 32].copy_from_slice(&IMAGE_BASE.to_le_bytes());
    put32(&mut data, optional + 32, small(SECTION_ALIGN));
    put32(&mut data, optional + 36, small(FILE_ALIGN));
    put32(&mut data, optional + 56, small(image_size));
    put32(&mut data, optional + 60, small(FILE_ALIGN));
    put16(&mut data, optional + 68, 3);
    put32(&mut data, optional + 108, 16);
    let section = optional + 0xF0;
    data[section..section + 5].copy_from_slice(b".text");
    put32(&mut data, section + 8, small(virtual_size));
    put32(&mut data, section + 12, TEXT_RVA);
    put32(&mut data, section + 16, small(raw_size));
    put32(&mut data, section + 20, small(FILE_ALIGN));
    put32(&mut data, section + 36, 0x6000_0020);
    data[FILE_ALIGN..FILE_ALIGN + text.len()].copy_from_slice(text);
    data
}

fn rva(va: u64) -> u32 {
    u32::try_from(va - IMAGE_BASE).expect("RVA fits")
}

fn build(blocks: &[Block]) -> SyntheticVm {
    let placeholder = vec![0_u8; blocks.iter().map(|block| block_len(block)).sum()];
    let (_, first) = assemble(TEXT_RVA, &placeholder).expect("layout pass");
    let seed = rva(first.bytecode);
    let (_, layout) = assemble(seed, &placeholder).expect("seed pass");
    let (bytecode, block_vas) = encode_blocks(blocks, &layout);
    assert_eq!(bytecode.len(), placeholder.len());
    let (text, final_layout) = assemble(seed, &bytecode).expect("final pass");
    assert_eq!(final_layout.bytecode, layout.bytecode);
    SyntheticVm {
        image: wrap_pe(&text, rva(final_layout.entry)),
        image_base: IMAGE_BASE,
        entry_rva: rva(final_layout.entry),
        seed,
        bootstrap_rva: rva(final_layout.bootstrap),
        oep_rvas: [rva(final_layout.oeps[0]), rva(final_layout.oeps[1])],
        bytecode_rva: rva(final_layout.bytecode),
        block_vas,
        write_handler_rva: rva(final_layout.write),
        read_handler_rva: rva(final_layout.read),
        push_handler_rva: rva(final_layout.push),
        branch_handler_rva: rva(final_layout.branch),
        exit_handler_rva: rva(final_layout.exit),
    }
}

/// One straight-line program: scatter the frame, rebuild it, exit to
/// `oep_rvas[0]` with RAX forced to [`FORCED_RAX`] and RCX to the entry seed.
pub fn build_linear() -> SyntheticVm {
    build(&linear_program())
}

/// Scatter, then a virtual branch whose target is the saved native RDX. Block
/// `1` exits to `oep_rvas[0]` (RAX [`BRANCH_RAX`]`[0]`), block `2` to
/// `oep_rvas[1]` (RAX `[1]`). `block_vas[1..]` are the RDX values selecting them.
pub fn build_branching() -> SyntheticVm {
    build(&branching_program())
}

/// A virtual branch to a constant. With `target` unset it is block `1`
/// (`block_vas[1]`); otherwise the given VA, which may be invalid.
pub fn build_const_branch(target: Option<u64>) -> SyntheticVm {
    let placeholder = build(&const_branch_program(0));
    build(&const_branch_program(
        target.unwrap_or(placeholder.block_vas[1]),
    ))
}
