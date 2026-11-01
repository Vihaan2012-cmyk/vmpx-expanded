//! Full emulator state checkpoint: CPU + paged memory to a single file, so a
//! costly init (e.g. the ~27B-step `.xpl` startup) is run once and restored in
//! seconds per experiment.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::cpu::Cpu;
use crate::memory::Memory;

const MAGIC: &[u8; 8] = b"VMPXSNP1";

/// Write `cpu` + `mem` to `path`.
pub fn save(cpu: &Cpu, mem: &Memory, path: &Path) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    cpu.save(&mut w)?;
    mem.save(&mut w)?;
    w.flush()
}

/// Restore a `(Cpu, Memory)` written by [`save`].
pub fn load(path: &Path) -> std::io::Result<(Cpu, Memory)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a vmpx snapshot",
        ));
    }
    let cpu = Cpu::load(&mut r)?;
    let mem = Memory::load(&mut r)?;
    Ok((cpu, mem))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Perm, PAGE_SIZE};

    #[test]
    fn round_trip_preserves_cpu_and_memory() {
        let dir = std::env::temp_dir();
        let path = dir.join("vmpx_snap_test.bin");

        let mut cpu = Cpu {
            regs: [0; 16],
            rip: 0x1400,
            rflags: 0x246,
            undef_flags: 0,
            gs_base: 0x7fff_0000,
            xmm: [0; 16],
            mxcsr: 0x1f80,
        };
        for (i, r) in cpu.regs.iter_mut().enumerate() {
            *r = 0xdead_0000 + i as u64;
        }
        cpu.xmm[3] = 0x1122_3344_5566_7788_9900_aabb_ccdd_eeff;

        let mut mem = Memory::new();
        let a = 0x10_0000u64;
        mem.map(a, PAGE_SIZE, Perm::RW).unwrap();
        mem.map(a + PAGE_SIZE * 4, PAGE_SIZE, Perm::RX).unwrap();
        mem.write_uint(a, 8, 0xcafe_babe_dead_beef).unwrap();
        mem.write_uint(a + 0x20, 4, 0x1234_5678).unwrap();

        save(&cpu, &mem, &path).unwrap();
        let (cpu2, mut mem2) = load(&path).unwrap();

        assert_eq!(cpu, cpu2);
        assert_eq!(mem2.read_uint(a, 8).unwrap(), 0xcafe_babe_dead_beef);
        assert_eq!(mem2.read_uint(a + 0x20, 4).unwrap(), 0x1234_5678);
        // RX page restored with its perm (reads ok, writes fault).
        assert!(mem2.write_uint(a + PAGE_SIZE * 4, 8, 1).is_err());

        let _ = std::fs::remove_file(&path);
    }
}
