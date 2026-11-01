//! SPIKE environment assumption: the protector's runtime VM-context block.
//!
//! Every virtualized export in the corpus first loads a context pointer from an image global
//! that nothing in the contained init ever stores (it stays 0), then dereferences it, so the
//! export faults at a tiny address. The init DOES allocate and key a context block; we locate
//! that block (the lowest page written outside image/stack/TEB) and seed the global with it.
//! This is an ASSUMPTION, labelled wherever it is used: observed tier only.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use vmpx_emulate::{Access, Cpu, Hook, HookAction, Interest, MemEvent, Memory};
use vmpx_pe::PeImage;

use crate::harness::Session;

/// Records which guest pages are written outside the image, stack and TEB.
pub struct PageWatch {
    pub pages: Rc<RefCell<BTreeMap<u64, u64>>>,
    image: (u64, u64),
}

impl PageWatch {
    #[must_use]
    pub fn new(image: (u64, u64)) -> Self {
        Self { pages: Rc::new(RefCell::new(BTreeMap::new())), image }
    }
}

impl Hook for PageWatch {
    fn interest(&self) -> Interest {
        Interest { write_all: true, ..Interest::NONE }
    }
    fn on_mem(&mut self, ev: &MemEvent, _cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        let a = ev.addr;
        let skip = (0x10_0000_0000..0x10_1000_0000).contains(&a)
            || (0x30_0000_0000..0x30_0000_2000).contains(&a)
            || (self.image.0..self.image.1).contains(&a);
        if ev.access == Access::Write && !skip {
            *self.pages.borrow_mut().entry(a & !0xfff).or_insert(0) += 1;
        }
        HookAction::Continue
    }
}

/// The context page the init prepared.
///
/// # Errors
/// Init failure.
pub fn detect_ctx_page(pe: &PeImage<'_>) -> Result<Option<u64>, String> {
    let base = pe.optional.image_base.0;
    let w = PageWatch::new((base, base + u64::from(pe.optional.size_of_image)));
    let pages = Rc::clone(&w.pages);
    let _s = Session::new_with(pe, 50_000_000, vec![Box::new(w)])?;
    let p = pages.borrow().keys().next().copied();
    Ok(p)
}
