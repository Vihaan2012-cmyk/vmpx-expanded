use std::collections::BTreeMap;

pub const PAGE_SIZE: u64 = 4096;
const PAGE_MASK: u64 = PAGE_SIZE - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Perm(pub u8);

impl Perm {
    pub const R: Perm = Perm(1);
    pub const W: Perm = Perm(2);
    pub const X: Perm = Perm(4);
    pub const RW: Perm = Perm(3);
    pub const RX: Perm = Perm(5);
    pub const RWX: Perm = Perm(7);

    #[must_use]
    pub fn contains(self, other: Perm) -> bool {
        self.0 & other.0 == other.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    Exec,
}

impl Access {
    fn perm(self) -> Perm {
        match self {
            Access::Read => Perm::R,
            Access::Write => Perm::W,
            Access::Exec => Perm::X,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemFault {
    pub addr: u64,
    pub access: Access,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError {
    Unaligned,
    Empty,
    Overlap,
    Overflow,
}

struct Page {
    data: Box<[u8]>,
    perm: Perm,
}

/// Sparse paged guest memory; unmapped addresses fault.
#[derive(Default)]
pub struct Memory {
    pages: BTreeMap<u64, Page>,
}

impl Memory {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Map zeroed pages over `[addr, addr+len)`; both must be page aligned.
    ///
    /// # Errors
    /// On misalignment, empty/overflowing range or overlap with a mapped page.
    pub fn map(&mut self, addr: u64, len: u64, perm: Perm) -> Result<(), MapError> {
        if addr & PAGE_MASK != 0 || len & PAGE_MASK != 0 {
            return Err(MapError::Unaligned);
        }
        if len == 0 {
            return Err(MapError::Empty);
        }
        let end = addr.checked_add(len).ok_or(MapError::Overflow)?;
        if self.pages.range(addr..end).next().is_some() {
            return Err(MapError::Overlap);
        }
        let mut a = addr;
        while a < end {
            let data = vec![0u8; PAGE_SIZE as usize].into_boxed_slice();
            self.pages.insert(a, Page { data, perm });
            a += PAGE_SIZE;
        }
        Ok(())
    }

    fn check(&self, addr: u64, len: usize, access: Access) -> Result<(), MemFault> {
        let fault = |addr| MemFault { addr, access };
        let end = addr.checked_add(len as u64).ok_or_else(|| fault(addr))?;
        let need = access.perm();
        let mut a = addr;
        while a < end {
            let base = a & !PAGE_MASK;
            match self.pages.get(&base) {
                Some(p) if p.perm.contains(need) => {}
                _ => return Err(fault(a)),
            }
            match base.checked_add(PAGE_SIZE) {
                Some(n) => a = n,
                None => break,
            }
        }
        Ok(())
    }

    fn copy_out(&self, addr: u64, buf: &mut [u8]) {
        let mut a = addr;
        let mut done = 0;
        while done < buf.len() {
            let base = a & !PAGE_MASK;
            let off = (a - base) as usize;
            let n = (PAGE_SIZE as usize - off).min(buf.len() - done);
            let page = &self.pages[&base];
            buf[done..done + n].copy_from_slice(&page.data[off..off + n]);
            done += n;
            a = a.wrapping_add(n as u64);
        }
    }

    fn copy_in(&mut self, addr: u64, data: &[u8]) {
        let mut a = addr;
        let mut done = 0;
        while done < data.len() {
            let base = a & !PAGE_MASK;
            let off = (a - base) as usize;
            let n = (PAGE_SIZE as usize - off).min(data.len() - done);
            let page = self.pages.get_mut(&base).expect("checked");
            page.data[off..off + n].copy_from_slice(&data[done..done + n]);
            done += n;
            a = a.wrapping_add(n as u64);
        }
    }

    /// # Errors
    /// Faults if any byte is unmapped or not readable.
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        self.check(addr, buf.len(), Access::Read)?;
        self.copy_out(addr, buf);
        Ok(())
    }

    /// All-or-nothing write.
    ///
    /// # Errors
    /// Faults if any byte is unmapped or not writable; nothing is written then.
    pub fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        self.check(addr, data.len(), Access::Write)?;
        self.copy_in(addr, data);
        Ok(())
    }

    /// Loader write: ignores page permissions but requires mapped pages.
    ///
    /// # Errors
    /// Faults if any byte is unmapped.
    pub fn poke(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        let fault = |addr| MemFault { addr, access: Access::Write };
        let end = addr.checked_add(data.len() as u64).ok_or_else(|| fault(addr))?;
        let mut a = addr;
        while a < end {
            let base = a & !PAGE_MASK;
            if !self.pages.contains_key(&base) {
                return Err(fault(a));
            }
            match base.checked_add(PAGE_SIZE) {
                Some(n) => a = n,
                None => break,
            }
        }
        self.copy_in(addr, data);
        Ok(())
    }

    /// Copy the executable prefix at `addr` (up to `buf.len()` bytes).
    ///
    /// # Errors
    /// Faults when the first byte is not executable.
    pub fn fetch(&self, addr: u64, buf: &mut [u8]) -> Result<usize, MemFault> {
        let mut n = buf.len();
        while n > 0 {
            if self.check(addr, n, Access::Exec).is_ok() {
                self.copy_out(addr, &mut buf[..n]);
                return Ok(n);
            }
            n -= 1;
        }
        Err(MemFault { addr, access: Access::Exec })
    }

    /// Little-endian read of 1, 2, 4 or 8 bytes.
    ///
    /// # Errors
    /// Faults like [`Memory::read`].
    pub fn read_uint(&self, addr: u64, bytes: usize) -> Result<u64, MemFault> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b[..bytes])?;
        Ok(u64::from_le_bytes(b))
    }

    /// Little-endian write of the low `bytes` bytes of `val`.
    ///
    /// # Errors
    /// Faults like [`Memory::write`].
    pub fn write_uint(&mut self, addr: u64, bytes: usize, val: u64) -> Result<(), MemFault> {
        self.write(addr, &val.to_le_bytes()[..bytes])
    }

    /// Enumerate mapped pages in ascending address order as `(base, perm,
    /// bytes)`. Read-only view used to snapshot the guest image.
    pub fn pages(&self) -> impl Iterator<Item = (u64, Perm, &[u8])> {
        self.pages.iter().map(|(&base, p)| (base, p.perm, &p.data[..]))
    }
}
