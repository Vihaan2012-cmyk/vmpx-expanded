use std::cell::Cell;
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
    /// Set once a cached code block was built from this page: any later write
    /// to it bumps [`Memory::code_gen`] so cached blocks revalidate.
    code: bool,
}

/// Direct-mapped page lookup cache size (entries).
const TLB_SIZE: usize = 256;
/// Tag that matches no page base (bases are page aligned, so never all-ones).
const TLB_EMPTY: u64 = u64::MAX;

/// Sparse paged guest memory; unmapped addresses fault.
///
/// Pages live in an append-only slab (`slab`); `index` maps base -> slab slot
/// in address order. Pages are never unmapped, so a cached `(base, slot)` in
/// `tlb` can never go stale; permissions are read from the slab on each
/// access, so a reprotect is seen immediately.
pub struct Memory {
    index: BTreeMap<u64, u32>,
    slab: Vec<Page>,
    tlb: [Cell<(u64, u32)>; TLB_SIZE],
    xgen: u64,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            index: BTreeMap::new(),
            slab: Vec::new(),
            tlb: std::array::from_fn(|_| Cell::new((TLB_EMPTY, 0))),
            xgen: 1,
        }
    }
}

impl Memory {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize every mapped page (base, perm byte, 4096 data bytes) for a
    /// snapshot. Pages are written in address order.
    pub fn save(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&(self.index.len() as u64).to_le_bytes())?;
        for (&base, &slot) in &self.index {
            let page = &self.slab[slot as usize];
            w.write_all(&base.to_le_bytes())?;
            w.write_all(&[page.perm.0])?;
            w.write_all(&page.data)?;
        }
        Ok(())
    }

    /// Rebuild paged memory from a [`Memory::save`] stream.
    pub fn load(r: &mut impl std::io::Read) -> std::io::Result<Memory> {
        let mut m = Memory::new();
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let n = u64::from_le_bytes(b8);
        for _ in 0..n {
            r.read_exact(&mut b8)?;
            let base = u64::from_le_bytes(b8);
            let mut pb = [0u8; 1];
            r.read_exact(&mut pb)?;
            let mut data = vec![0u8; PAGE_SIZE as usize];
            r.read_exact(&mut data)?;
            m.index.insert(base, m.slab.len() as u32);
            m.slab.push(Page { data: data.into_boxed_slice(), perm: Perm(pb[0]), code: false });
        }
        Ok(m)
    }

    #[inline(always)]
    fn slot(&self, base: u64) -> Option<usize> {
        let (tag, idx) = self.tlb[(base >> 12) as usize & (TLB_SIZE - 1)].get();
        if tag == base {
            return Some(idx as usize);
        }
        self.slot_miss(base)
    }

    #[cold]
    #[inline(never)]
    fn slot_miss(&self, base: u64) -> Option<usize> {
        let idx = *self.index.get(&base)?;
        self.tlb[(base >> 12) as usize & (TLB_SIZE - 1)].set((base, idx));
        Some(idx as usize)
    }

    #[inline(always)]
    fn page(&self, base: u64) -> Option<&Page> {
        self.slot(base).map(|i| &self.slab[i])
    }

    #[inline(always)]
    fn page_mut(&mut self, base: u64) -> Option<&mut Page> {
        self.slot(base).map(|i| &mut self.slab[i])
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
        if self.index.range(addr..end).next().is_some() {
            return Err(MapError::Overlap);
        }
        self.xgen += 1;
        let mut a = addr;
        while a < end {
            let data = vec![0u8; PAGE_SIZE as usize].into_boxed_slice();
            self.index.insert(a, self.slab.len() as u32);
            self.slab.push(Page { data, perm, code: false });
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
            match self.page(base) {
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
            let page = self.page(base).expect("checked");
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
            let page = self.page_mut(base).expect("checked");
            page.data[off..off + n].copy_from_slice(&data[done..done + n]);
            if page.code {
                self.xgen += 1;
            }
            done += n;
            a = a.wrapping_add(n as u64);
        }
    }

    /// # Errors
    /// Faults if any byte is unmapped or not readable.
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        let off = (addr & PAGE_MASK) as usize;
        if !buf.is_empty() && off + buf.len() <= PAGE_SIZE as usize {
            return match self.page(addr & !PAGE_MASK) {
                Some(p) if p.perm.contains(Perm::R) => {
                    buf.copy_from_slice(&p.data[off..off + buf.len()]);
                    Ok(())
                }
                _ => Err(MemFault { addr, access: Access::Read }),
            };
        }
        self.check(addr, buf.len(), Access::Read)?;
        self.copy_out(addr, buf);
        Ok(())
    }

    /// All-or-nothing write.
    ///
    /// # Errors
    /// Faults if any byte is unmapped or not writable; nothing is written then.
    pub fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), MemFault> {
        let off = (addr & PAGE_MASK) as usize;
        if !data.is_empty() && off + data.len() <= PAGE_SIZE as usize {
            return match self.page_mut(addr & !PAGE_MASK) {
                Some(p) if p.perm.contains(Perm::W) => {
                    p.data[off..off + data.len()].copy_from_slice(data);
                    if p.code {
                        self.xgen += 1;
                    }
                    Ok(())
                }
                _ => Err(MemFault { addr, access: Access::Write }),
            };
        }
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
            if !self.index.contains_key(&base) {
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

    /// The 15 bytes at `addr` straight from the page, when they lie within one
    /// executable page (the common instruction-fetch case). `None` otherwise;
    /// the caller then uses [`Memory::fetch`], which handles page crossings and
    /// short executable prefixes identically.
    #[inline]
    #[must_use]
    pub fn fetch15(&self, addr: u64) -> Option<&[u8]> {
        let off = (addr & PAGE_MASK) as usize;
        if off + 15 > PAGE_SIZE as usize {
            return None;
        }
        let p = self.page(addr & !PAGE_MASK)?;
        p.perm.contains(Perm::X).then(|| &p.data[off..off + 15])
    }

    /// Little-endian read of 1, 2, 4 or 8 bytes.
    ///
    /// # Errors
    /// Faults like [`Memory::read`].
    #[inline(always)]
    pub fn read_uint(&self, addr: u64, bytes: usize) -> Result<u64, MemFault> {
        let off = (addr & PAGE_MASK) as usize;
        // Hit: one 8-byte load from a readable page, masked to `bytes`.
        if off + 8 <= PAGE_SIZE as usize && matches!(bytes, 1 | 2 | 4 | 8) {
            if let Some(p) = self.page(addr & !PAGE_MASK)
                && p.perm.contains(Perm::R)
                && let Some(w) = p.data.get(off..off + 8)
            {
                let v = u64::from_le_bytes(w.try_into().unwrap_or([0; 8]));
                return Ok(v & (u64::MAX >> (64 - 8 * bytes)));
            }
        }
        self.read_uint_slow(addr, bytes)
    }

    #[inline(never)]
    fn read_uint_slow(&self, addr: u64, bytes: usize) -> Result<u64, MemFault> {
        let off = (addr & PAGE_MASK) as usize;
        if matches!(bytes, 1 | 2 | 4 | 8) && off + bytes <= PAGE_SIZE as usize {
            return match self.page(addr & !PAGE_MASK) {
                Some(p) if p.perm.contains(Perm::R) => {
                    let d = &p.data[off..];
                    Ok(match bytes {
                        1 => u64::from(d[0]),
                        2 => u64::from(u16::from_le_bytes([d[0], d[1]])),
                        4 => u64::from(u32::from_le_bytes([d[0], d[1], d[2], d[3]])),
                        _ => u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]),
                    })
                }
                _ => Err(MemFault { addr, access: Access::Read }),
            };
        }
        let mut b = [0u8; 8];
        self.read(addr, &mut b[..bytes])?;
        Ok(u64::from_le_bytes(b))
    }

    /// Little-endian write of the low `bytes` bytes of `val`.
    ///
    /// # Errors
    /// Faults like [`Memory::write`].
    #[inline(always)]
    pub fn write_uint(&mut self, addr: u64, bytes: usize, val: u64) -> Result<(), MemFault> {
        let off = (addr & PAGE_MASK) as usize;
        // Hit: read-modify-write of one 8-byte window of a writable page.
        if off + 8 <= PAGE_SIZE as usize && matches!(bytes, 1 | 2 | 4 | 8) {
            if let Some(p) = self.page_mut(addr & !PAGE_MASK)
                && p.perm.contains(Perm::W)
                && let Some(w) = p.data.get_mut(off..off + 8)
            {
                let m = u64::MAX >> (64 - 8 * bytes);
                let old = u64::from_le_bytes((&*w).try_into().unwrap_or([0; 8]));
                w.copy_from_slice(&((old & !m) | (val & m)).to_le_bytes());
                if p.code {
                    self.xgen += 1;
                }
                return Ok(());
            }
        }
        self.write_uint_slow(addr, bytes, val)
    }

    #[inline(never)]
    fn write_uint_slow(&mut self, addr: u64, bytes: usize, val: u64) -> Result<(), MemFault> {
        let off = (addr & PAGE_MASK) as usize;
        if matches!(bytes, 1 | 2 | 4 | 8) && off + bytes <= PAGE_SIZE as usize {
            return match self.page_mut(addr & !PAGE_MASK) {
                Some(p) if p.perm.contains(Perm::W) => {
                    let d = &mut p.data[off..];
                    match bytes {
                        1 => d[0] = val as u8,
                        2 => d[..2].copy_from_slice(&(val as u16).to_le_bytes()),
                        4 => d[..4].copy_from_slice(&(val as u32).to_le_bytes()),
                        _ => d[..8].copy_from_slice(&val.to_le_bytes()),
                    }
                    if p.code {
                        self.xgen += 1;
                    }
                    Ok(())
                }
                _ => Err(MemFault { addr, access: Access::Write }),
            };
        }
        self.write(addr, &val.to_le_bytes()[..bytes])
    }

    /// Enumerate mapped pages in ascending address order as `(base, perm,
    /// bytes)`. Read-only view used to snapshot the guest image.
    pub fn pages(&self) -> impl Iterator<Item = (u64, Perm, &[u8])> {
        self.index.iter().map(|(&base, &i)| {
            let p = &self.slab[i as usize];
            (base, p.perm, &p.data[..])
        })
    }

    /// Permission of the page containing `addr`, or `None` if unmapped.
    #[must_use]
    pub fn perm_at(&self, addr: u64) -> Option<Perm> {
        self.page(addr & !PAGE_MASK).map(|p| p.perm)
    }

    /// Generation counter bumped by every change that can invalidate cached
    /// decoded code: a write to a page code was cached from, a loader poke, a
    /// reprotect or a new mapping.
    #[inline]
    #[must_use]
    pub fn code_gen(&self) -> u64 {
        self.xgen
    }

    /// The bytes from `addr` to the end of its page, when that page is
    /// executable; marks the page as holding cached code so later writes to it
    /// bump [`Memory::code_gen`].
    pub fn code_bytes(&mut self, addr: u64) -> Option<&[u8]> {
        let slot = self.slot(addr & !PAGE_MASK)?;
        let p = &mut self.slab[slot];
        if !p.perm.contains(Perm::X) {
            return None;
        }
        p.code = true;
        Some(&p.data[(addr & PAGE_MASK) as usize..])
    }

    /// `len` executable bytes at `addr` within one page, if any.
    #[inline]
    #[must_use]
    pub fn exec_slice(&self, addr: u64, len: usize) -> Option<&[u8]> {
        let off = (addr & PAGE_MASK) as usize;
        if off + len > PAGE_SIZE as usize {
            return None;
        }
        let p = self.page(addr & !PAGE_MASK)?;
        p.perm.contains(Perm::X).then(|| &p.data[off..off + len])
    }

    /// Change the permission of every page overlapping `[addr, addr+len)` to
    /// `perm` (an NT reprotect). The range is rounded out to page boundaries.
    ///
    /// # Errors
    /// Faults (access `Write`) if any page in the range is unmapped; pages
    /// before the gap keep the new permission (NT reprotect is not atomic).
    pub fn protect(&mut self, addr: u64, len: u64, perm: Perm) -> Result<(), MemFault> {
        let fault = |addr| MemFault { addr, access: Access::Write };
        if len == 0 {
            return Ok(());
        }
        let base = addr & !PAGE_MASK;
        let end = addr.checked_add(len).ok_or_else(|| fault(addr))?;
        self.xgen += 1;
        let mut a = base;
        while a < end {
            match self.page_mut(a) {
                Some(p) => p.perm = perm,
                None => return Err(fault(a)),
            }
            match a.checked_add(PAGE_SIZE) {
                Some(n) => a = n,
                None => break,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u64 = 0x4000;

    /// Reprotecting a page the lookup cache already holds takes effect at once.
    #[test]
    fn protect_is_seen_through_the_page_cache() {
        let mut m = Memory::new();
        m.map(A, PAGE_SIZE, Perm::RW).unwrap();
        m.write_uint(A, 8, 0x1122_3344_5566_7788).unwrap();
        assert_eq!(m.read_uint(A, 8).unwrap(), 0x1122_3344_5566_7788);
        m.protect(A, PAGE_SIZE, Perm::R).unwrap();
        assert_eq!(m.write_uint(A, 8, 0).unwrap_err(), MemFault { addr: A, access: Access::Write });
        assert_eq!(m.fetch15(A), None, "not executable");
        m.protect(A, PAGE_SIZE, Perm::RX).unwrap();
        assert_eq!(m.fetch15(A).map(<[u8]>::len), Some(15));
    }

    /// A page-crossing access faults on the first bad byte, and only commits
    /// when every page allows it; in-page and slow paths agree.
    #[test]
    fn page_crossing_access_is_all_or_nothing() {
        let mut m = Memory::new();
        m.map(A, PAGE_SIZE, Perm::RW).unwrap();
        m.map(A + PAGE_SIZE * 2, PAGE_SIZE, Perm::RW).unwrap();
        let edge = A + PAGE_SIZE - 4;
        assert_eq!(m.write_uint(edge, 8, u64::MAX).unwrap_err().addr, A + PAGE_SIZE);
        assert_eq!(m.read_uint(edge, 4).unwrap(), 0, "nothing was written");
        m.map(A + PAGE_SIZE, PAGE_SIZE, Perm::RW).unwrap();
        m.write_uint(edge, 8, 0x0807_0605_0403_0201).unwrap();
        assert_eq!(m.read_uint(edge, 8).unwrap(), 0x0807_0605_0403_0201);
        assert_eq!(m.read_uint(A + PAGE_SIZE, 4).unwrap(), 0x0807_0605);
        assert_eq!(m.read_uint(0x9000, 1).unwrap_err(), MemFault { addr: 0x9000, access: Access::Read });
        assert!(m.read(0x9000, &mut []).is_ok(), "empty read never faults");
    }

    /// Executable bytes within a page match the page-crossing `fetch` result.
    #[test]
    fn fetch15_agrees_with_fetch() {
        let mut m = Memory::new();
        m.map(A, PAGE_SIZE, Perm::RX).unwrap();
        let pat: Vec<u8> = (0..=255u8).cycle().take(PAGE_SIZE as usize).collect();
        m.poke(A, &pat).unwrap();
        for off in [0u64, 1, 100, PAGE_SIZE - 15] {
            let mut buf = [0u8; 15];
            assert_eq!(m.fetch(A + off, &mut buf).unwrap(), 15);
            assert_eq!(m.fetch15(A + off).unwrap(), &buf);
        }
        assert_eq!(m.fetch15(A + PAGE_SIZE - 14), None, "crossing the page end");
    }
}
