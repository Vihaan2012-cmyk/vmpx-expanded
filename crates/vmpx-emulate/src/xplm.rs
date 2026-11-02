//! Standalone model of the X-Plane host SDK (XPLM + XPWidgets) for the contained interpreter.
//!
//! Pure data-in/data-out: the integrator forwards an intercepted import call to
//! [`Xplm::dispatch`] with the Win64 register args and a [`GuestMem`] adapter, then applies the
//! returned [`Ret`] to `rax` (and `xmm0` when set) and resumes the guest. Registration stubs
//! RECORD the plugin's callbacks ([`Xplm::callbacks`]) so a host driver can invoke them later.
//! Functions without a stub are reported ([`Dispatch::Unlisted`], [`Xplm::unlisted`]), never
//! silently zeroed, so a fail-closed loop can find the next stub to add.
//!
//! Signatures are from the public XPLM SDK, from memory (not verified against headers).
#![allow(dead_code)] // consumed by the WinApiHook integration, not wired yet

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Minimal guest-memory interface; the integrator adapts `Memory` to it.
pub trait GuestMem {
    /// Fill `buf` from `addr`; false if any byte is unmapped.
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool;
    /// Write `data` at `addr`; false on fault.
    fn write(&mut self, addr: u64, data: &[u8]) -> bool;
    /// Allocate `len` bytes of persistent guest memory (for stubs that return a pointer).
    fn alloc(&mut self, _len: usize) -> Option<u64> {
        None
    }
}

/// Win64 call context: first four integer/pointer args and the callee-entry `rsp`
/// (`[rsp]` = return address; 5th arg at `rsp+0x28`). A hook firing on the `call`
/// instruction, before the return address is pushed, must pass `rsp - 8`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Call {
    pub args: [u64; 4],
    pub rsp: u64,
}

/// Value to place in `rax`, plus raw low bits for `xmm0` when the function returns a float/double.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ret {
    pub rax: u64,
    pub xmm0: Option<u64>,
}

impl Ret {
    pub const VOID: Ret = Ret { rax: 0, xmm0: None };
    #[must_use]
    pub fn int(v: i32) -> Ret {
        Ret { rax: u64::from(v as u32), xmm0: None }
    }
    #[must_use]
    pub fn ptr(v: u64) -> Ret {
        Ret { rax: v, xmm0: None }
    }
    #[must_use]
    pub fn f32(v: f32) -> Ret {
        Ret { rax: 0, xmm0: Some(u64::from(v.to_bits())) }
    }
    #[must_use]
    pub fn f64(v: f64) -> Ret {
        Ret { rax: 0, xmm0: Some(v.to_bits()) }
    }
}

/// Outcome of [`Xplm::dispatch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    /// The dll is not an XPLM/XPWidgets module; route elsewhere.
    NotXplm,
    /// Modeled; apply [`Ret`] and resume.
    Handled(Ret),
    /// XPLM function with no stub; recorded in [`Xplm::unlisted`]. Integrator should stop (fail closed).
    Unlisted,
}

/// Slot of an `XPLMRegisterDataAccessor` callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataSlot {
    ReadInt,
    WriteInt,
    ReadFloat,
    WriteFloat,
    ReadDouble,
    WriteDouble,
    ReadIntArray,
    WriteIntArray,
    ReadFloatArray,
    WriteFloatArray,
    ReadData,
    WriteData,
}

const DATA_SLOTS: [DataSlot; 12] = [
    DataSlot::ReadInt,
    DataSlot::WriteInt,
    DataSlot::ReadFloat,
    DataSlot::WriteFloat,
    DataSlot::ReadDouble,
    DataSlot::WriteDouble,
    DataSlot::ReadIntArray,
    DataSlot::WriteIntArray,
    DataSlot::ReadFloatArray,
    DataSlot::WriteFloatArray,
    DataSlot::ReadData,
    DataSlot::WriteData,
];

/// What kind of callback the plugin registered (the ABI the driver must call it with).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CallbackKind {
    /// `float f(float sinceLast, float sinceLoop, int counter, void* refcon)`
    FlightLoop,
    /// `int f(XPLMDrawingPhase, int isBefore, void* refcon)`; `aux` = phase, `name` = "before"/"after".
    Draw,
    /// `int f(char key, XPLMKeyFlags, char vkey, void* refcon)`
    KeySniffer,
    /// `int f(XPLMCommandRef, XPLMCommandPhase, void* refcon)`; `name` = command, `aux` = before flag.
    CommandHandler,
    /// Data accessor callback; `name` = dataref, `refcon` = read/write refcon for that slot.
    DataAccessor(DataSlot),
    /// `void f(void* menuRef, void* itemRef)`; `refcon` = menuRef, `aux` = itemRef, `name` = item.
    MenuItem,
    /// `void f(XPLMWindowID, void* refcon)`
    WindowDraw,
    /// `int f(XPLMWindowID, int x, int y, XPLMMouseStatus, void* refcon)`
    WindowMouseClick,
    /// `int f(XPLMWindowID, int x, int y, XPLMMouseStatus, void* refcon)` (right button)
    WindowRightClick,
    /// `void f(XPLMWindowID, char key, XPLMKeyFlags, char vkey, void* refcon, int losingFocus)`
    WindowKey,
    /// `XPLMCursorStatus f(XPLMWindowID, int x, int y, void* refcon)`
    WindowCursor,
    /// `int f(XPLMWindowID, int x, int y, int wheel, int clicks, void* refcon)`
    WindowMouseWheel,
    /// `int f(XPLMCameraPosition_t*, int isLosingControl, void* refcon)`
    Camera,
    /// `void f(XPLMObjectRef, void* refcon)`
    ObjectLoaded,
    /// `void f(const char* msg)`
    ErrorCallback,
    /// `void f(const char* feature, void* refcon)`
    FeatureEnumerator,
    /// `void f(int idx, int l, int t, int r, int b, void* refcon)`
    MonitorBounds,
}

/// A callback the plugin handed to the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedCallback {
    pub kind: CallbackKind,
    pub callback_va: u64,
    pub refcon: u64,
    /// Kind-specific secondary value, see [`CallbackKind`].
    pub aux: u64,
    pub name: Option<String>,
    /// Handle returned to the plugin for the registration (0 if none).
    pub handle: u64,
    /// False once the plugin unregistered/destroyed it.
    pub active: bool,
}

/// How a driven guest call returns its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetKind {
    Void,
    Int,
    Float,
    Double,
}

/// One host-to-plugin call in Win64 form: positional args 0-3 ride in rcx/rdx/r8/r9 (or
/// `xmm[i]` when `Some`, for float/double), args 4+ go on the stack after the shadow space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestCall {
    pub va: u64,
    pub ints: [u64; 4],
    pub xmm: [Option<u128>; 4],
    pub stack: Vec<u64>,
    pub ret: RetKind,
    pub label: String,
}

impl GuestCall {
    #[must_use]
    pub fn new(label: impl Into<String>, va: u64, ints: [u64; 4], ret: RetKind) -> Self {
        Self { va, ints, xmm: [None; 4], stack: Vec::new(), ret, label: label.into() }
    }
}

const TICK: f32 = 1.0 / 60.0;
const KEY_A: u64 = 0x61;
const VK_A: u64 = 0x41;
const KEY_DOWN_FLAG: u64 = 8;
const MOUSE_DOWN: u64 = 1;
const MOUSE_UP: u64 = 3;
const CURSOR_XY: u64 = 100;

fn f32_bits(v: f32) -> u128 {
    u128::from(v.to_bits())
}

impl RecordedCallback {
    /// Benign deterministic calls that exercise this callback; `scratch` is a zeroed guest
    /// buffer (>= 256 bytes) for out/in arrays. Repeating kinds (flight loop, draw) are
    /// driven every `iteration`; the rest only on iteration 0. Kinds the host never
    /// initiates a call for (object-loaded, error, feature, monitor) yield nothing.
    #[must_use]
    pub fn invocations(&self, iteration: u32, scratch: u64) -> Vec<GuestCall> {
        let (va, rc) = (self.callback_va, self.refcon);
        let tag = format!("{:?}@{va:#x}", self.kind);
        let call = |ints, ret| GuestCall::new(tag.clone(), va, ints, ret);
        let repeating = matches!(self.kind, CallbackKind::FlightLoop | CallbackKind::Draw);
        if iteration > 0 && !repeating {
            return Vec::new();
        }
        match self.kind {
            CallbackKind::FlightLoop => {
                let mut g = call([0, 0, u64::from(iteration) + 1, rc], RetKind::Float);
                g.xmm[0] = Some(f32_bits(TICK));
                g.xmm[1] = Some(f32_bits(TICK));
                vec![g]
            }
            CallbackKind::Draw => {
                let before = u64::from(self.name.as_deref() == Some("before"));
                vec![call([self.aux, before, rc, 0], RetKind::Int)]
            }
            CallbackKind::KeySniffer => {
                vec![call([KEY_A, KEY_DOWN_FLAG, VK_A, rc], RetKind::Int)]
            }
            CallbackKind::CommandHandler => [0, 2]
                .map(|phase| call([self.handle, phase, rc, 0], RetKind::Int))
                .to_vec(),
            CallbackKind::DataAccessor(slot) => data_accessor_calls(&call, slot, rc, scratch),
            CallbackKind::MenuItem => vec![call([rc, self.aux, 0, 0], RetKind::Void)],
            CallbackKind::WindowDraw => vec![call([self.handle, rc, 0, 0], RetKind::Void)],
            CallbackKind::WindowMouseClick | CallbackKind::WindowRightClick => {
                [MOUSE_DOWN, MOUSE_UP]
                    .map(|st| {
                        let mut g = call([self.handle, CURSOR_XY, CURSOR_XY, st], RetKind::Int);
                        g.stack = vec![rc];
                        g
                    })
                    .to_vec()
            }
            CallbackKind::WindowKey => {
                let mut g = call([self.handle, KEY_A, KEY_DOWN_FLAG, VK_A], RetKind::Void);
                g.stack = vec![rc, 0];
                vec![g]
            }
            CallbackKind::WindowCursor => {
                vec![call([self.handle, CURSOR_XY, CURSOR_XY, rc], RetKind::Int)]
            }
            CallbackKind::WindowMouseWheel => {
                let mut g = call([self.handle, CURSOR_XY, CURSOR_XY, 0], RetKind::Int);
                g.stack = vec![1, rc];
                vec![g]
            }
            CallbackKind::Camera => vec![call([scratch, 0, rc, 0], RetKind::Int)],
            CallbackKind::ObjectLoaded
            | CallbackKind::ErrorCallback
            | CallbackKind::FeatureEnumerator
            | CallbackKind::MonitorBounds => Vec::new(),
        }
    }
}

fn data_accessor_calls(
    call: &dyn Fn([u64; 4], RetKind) -> GuestCall,
    slot: DataSlot,
    rc: u64,
    scratch: u64,
) -> Vec<GuestCall> {
    use DataSlot::{
        ReadData, ReadDouble, ReadFloat, ReadFloatArray, ReadInt, ReadIntArray, WriteData,
        WriteDouble, WriteFloat, WriteFloatArray, WriteInt, WriteIntArray,
    };
    // Arrays and blobs: a NULL size query, then a bounded transfer.
    let arrays = |ret| vec![call([rc, 0, 0, 0], ret), call([rc, scratch, 0, 8], ret)];
    match slot {
        ReadInt => vec![call([rc, 0, 0, 0], RetKind::Int)],
        ReadFloat => vec![call([rc, 0, 0, 0], RetKind::Float)],
        ReadDouble => vec![call([rc, 0, 0, 0], RetKind::Double)],
        ReadIntArray | ReadFloatArray | ReadData => arrays(RetKind::Int),
        WriteInt => vec![call([rc, 1, 0, 0], RetKind::Void)],
        WriteFloat => {
            let mut g = call([rc, 0, 0, 0], RetKind::Void);
            g.xmm[1] = Some(f32_bits(1.0));
            vec![g]
        }
        WriteDouble => {
            let mut g = call([rc, 0, 0, 0], RetKind::Void);
            g.xmm[1] = Some(u128::from(1.0f64.to_bits()));
            vec![g]
        }
        WriteIntArray | WriteFloatArray | WriteData => {
            vec![call([rc, scratch, 0, 1], RetKind::Void)]
        }
    }
}

/// Argument/return accessor handed to stub handlers.
pub struct Ctx<'a> {
    call: &'a Call,
    mem: &'a mut dyn GuestMem,
}

impl Ctx<'_> {
    /// Integer/pointer arg `i` (0-3 registers, 4+ stack).
    fn a(&self, i: usize) -> u64 {
        if i < 4 {
            self.call.args[i]
        } else {
            self.r64(self.call.rsp.wrapping_add(0x28 + 8 * (i as u64 - 4))).unwrap_or(0)
        }
    }
    fn r32(&self, addr: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.mem.read(addr, &mut b).then(|| u32::from_le_bytes(b))
    }
    fn r64(&self, addr: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.mem.read(addr, &mut b).then(|| u64::from_le_bytes(b))
    }
    fn cstr(&self, mut addr: u64) -> String {
        let mut s = Vec::new();
        while addr != 0 && s.len() < 4096 {
            let mut b = [0u8; 1];
            if !self.mem.read(addr, &mut b) || b[0] == 0 {
                break;
            }
            s.push(b[0]);
            addr += 1;
        }
        String::from_utf8_lossy(&s).into_owned()
    }
    fn wbytes(&mut self, addr: u64, d: &[u8]) {
        if addr != 0 {
            let _ = self.mem.write(addr, d);
        }
    }
    fn w32(&mut self, addr: u64, v: u32) {
        self.wbytes(addr, &v.to_le_bytes());
    }
    fn wstr(&mut self, addr: u64, s: &str) {
        if addr != 0 {
            let mut v = s.as_bytes().to_vec();
            v.push(0);
            self.wbytes(addr, &v);
        }
    }
}

type Handler = fn(&mut Xplm, &mut Ctx) -> Ret;

const HANDLE_BASE: u64 = 0x7000_0000_0000;
const SYSTEM_PATH: &str = "C:\\X-Plane 12\\";
const PREFS_PATH: &str = "C:\\X-Plane 12\\Output\\preferences\\X-Plane.prf";
const NAV_NOT_FOUND: i32 = -1;
const NO_PLUGIN_ID: i32 = -1;
const MY_PLUGIN_ID: u64 = 1;
const KNOWN_FEATURES: [&str; 3] =
    ["XPLM_USE_NATIVE_PATHS", "XPLM_USE_NATIVE_WIDGET_WINDOWS", "XPLM_WANTS_REFLECTIONS"];

/// The XPLM/XPWidgets host model.
pub struct Xplm {
    table: HashMap<(String, String), Handler>,
    handles: BTreeMap<String, u64>,
    handle_names: BTreeMap<u64, String>,
    next_handle: u64,
    seq: u64,
    callbacks: Vec<RecordedCallback>,
    debug_strings: Vec<String>,
    spoken: Vec<String>,
    unlisted: BTreeMap<(String, String), u64>,
    unlisted_order: Vec<(String, String)>,
    calls: BTreeMap<String, u64>,
    dataref_types: BTreeMap<u64, u32>,
    dataref_access: BTreeMap<String, u64>,
    window_visible: BTreeMap<u64, bool>,
    menu_items: BTreeMap<u64, i32>,
    menu_handlers: BTreeMap<u64, (u64, u64)>,
    features: BTreeSet<String>,
    next_tex: u32,
    sep_addr: Option<u64>,
    /// Recorded sim/ dataref stream served on reads (observed tier). None on the
    /// ordinary path, so behavior is unchanged when no recording is loaded.
    replay: Option<crate::replay::Replay>,
}

impl Default for Xplm {
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical module stem (`"xplm"` / `"xpwidgets"`) for a dll name, or None.
fn stem(dll: &str) -> Option<&'static str> {
    let l = dll.to_ascii_lowercase();
    if l.starts_with("xpwidgets") {
        Some("xpwidgets")
    } else if l.starts_with("xplm") {
        Some("xplm")
    } else {
        None
    }
}

impl Xplm {
    #[must_use]
    pub fn new() -> Self {
        let mut x = Xplm {
            table: HashMap::new(),
            handles: BTreeMap::new(),
            handle_names: BTreeMap::new(),
            next_handle: HANDLE_BASE,
            seq: 0,
            callbacks: Vec::new(),
            debug_strings: Vec::new(),
            spoken: Vec::new(),
            unlisted: BTreeMap::new(),
            unlisted_order: Vec::new(),
            calls: BTreeMap::new(),
            dataref_types: BTreeMap::new(),
            dataref_access: BTreeMap::new(),
            window_visible: BTreeMap::new(),
            menu_items: BTreeMap::new(),
            menu_handlers: BTreeMap::new(),
            features: BTreeSet::new(),
            next_tex: 1000,
            sep_addr: None,
            replay: None,
        };
        for (name, h) in STUBS {
            x.register_stub("XPLM_64.dll", name, *h);
        }
        for (name, h) in WIDGET_STUBS {
            x.register_stub("XPWidgets_64.dll", name, *h);
        }
        x
    }

    /// Add or override a stub, keyed by (dll, function).
    pub fn register_stub(&mut self, dll: &str, func: &str, handler: Handler) {
        if let Some(s) = stem(dll) {
            self.table.insert((s.to_owned(), func.to_owned()), handler);
        }
    }

    /// True if `(dll, func)` has a stub.
    #[must_use]
    pub fn is_modeled(&self, dll: &str, func: &str) -> bool {
        stem(dll).is_some_and(|s| self.table.contains_key(&(s.to_owned(), func.to_owned())))
    }

    /// Number of modeled functions.
    #[must_use]
    pub fn modeled_count(&self) -> usize {
        self.table.len()
    }

    /// Handle an intercepted import call.
    pub fn dispatch(&mut self, dll: &str, func: &str, call: &Call, mem: &mut dyn GuestMem) -> Dispatch {
        let Some(s) = stem(dll) else { return Dispatch::NotXplm };
        let key = (s.to_owned(), func.to_owned());
        let Some(h) = self.table.get(&key).copied() else {
            let n = self.unlisted.entry(key.clone()).or_insert(0);
            if *n == 0 {
                self.unlisted_order.push(key);
            }
            *n += 1;
            return Dispatch::Unlisted;
        };
        *self.calls.entry(func.to_owned()).or_insert(0) += 1;
        let mut ctx = Ctx { call, mem };
        Dispatch::Handled(h(self, &mut ctx))
    }

    /// Every callback the plugin registered, in registration order.
    #[must_use]
    pub fn callbacks(&self) -> &[RecordedCallback] {
        &self.callbacks
    }

    /// Registered callbacks of one kind that are still active.
    pub fn active_callbacks(&self, kind: CallbackKind) -> impl Iterator<Item = &RecordedCallback> {
        self.callbacks.iter().filter(move |c| c.active && c.kind == kind)
    }

    /// Strings passed to `XPLMDebugString`.
    #[must_use]
    pub fn debug_strings(&self) -> &[String] {
        &self.debug_strings
    }

    /// Strings passed to `XPLMSpeakString`.
    #[must_use]
    pub fn spoken(&self) -> &[String] {
        &self.spoken
    }

    /// Unmodeled (dll stem, function) pairs in first-seen order, with call counts.
    #[must_use]
    pub fn unlisted(&self) -> Vec<(&(String, String), u64)> {
        self.unlisted_order.iter().map(|k| (k, self.unlisted[k])).collect()
    }

    /// Per-function call counts for modeled functions.
    #[must_use]
    pub fn call_counts(&self) -> &BTreeMap<String, u64> {
        &self.calls
    }

    /// Dataref name for a handle minted by `FindDataRef`/`RegisterDataAccessor`.
    #[must_use]
    pub fn dataref_name(&self, handle: u64) -> Option<&str> {
        self.handle_names.get(&handle).and_then(|n| n.strip_prefix("dr:"))
    }

    /// Command name for a handle minted by `FindCommand`/`CreateCommand`.
    #[must_use]
    pub fn command_name(&self, handle: u64) -> Option<&str> {
        self.handle_names.get(&handle).and_then(|n| n.strip_prefix("cmd:"))
    }

    /// Get/Set accesses per dataref name (what the plugin tried to read or write).
    #[must_use]
    pub fn dataref_accesses(&self) -> &BTreeMap<String, u64> {
        &self.dataref_access
    }

    fn handle(&mut self, key: &str) -> u64 {
        if let Some(h) = self.handles.get(key) {
            return *h;
        }
        self.next_handle += 0x10;
        self.handles.insert(key.to_owned(), self.next_handle);
        self.handle_names.insert(self.next_handle, key.to_owned());
        self.next_handle
    }

    fn fresh(&mut self, prefix: &str) -> u64 {
        self.seq += 1;
        let k = format!("{prefix}#{}", self.seq);
        self.handle(&k)
    }

    fn record(&mut self, kind: CallbackKind, va: u64, refcon: u64, aux: u64, name: Option<String>, handle: u64) {
        if va != 0 {
            self.callbacks.push(RecordedCallback { kind, callback_va: va, refcon, aux, name, handle, active: true });
        }
    }

    fn deactivate(&mut self, f: impl Fn(&RecordedCallback) -> bool) {
        for c in self.callbacks.iter_mut().filter(|c| c.active) {
            if f(c) {
                c.active = false;
            }
        }
    }

    fn touch_dataref(&mut self, handle: u64) {
        let n = self.handle_names.get(&handle).cloned().unwrap_or_else(|| format!("?{handle:#x}"));
        *self.dataref_access.entry(n).or_insert(0) += 1;
    }

    /// Loads a recorded sim/ dataref stream to serve on reads (observed tier).
    pub fn set_replay(&mut self, r: crate::replay::Replay) {
        self.replay = Some(r);
    }

    /// Recorded frame count (0 when no recording is loaded).
    #[must_use]
    pub fn replay_frames(&self) -> usize {
        self.replay.as_ref().map_or(0, crate::replay::Replay::n_frames)
    }

    /// Advances the recording one frame, so the next dataref reads reflect the
    /// next captured instant. No-op without a recording.
    pub fn replay_advance(&mut self) {
        if let Some(r) = self.replay.as_mut() {
            r.advance();
        }
    }
}

// ---- handler helpers --------------------------------------------------------------------------

fn void(_: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::VOID
}
fn zero(_: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::int(0)
}
fn one(_: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::int(1)
}
fn zero_f(_: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::f32(0.0)
}
fn zero_d(_: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::f64(0.0)
}
fn nav_not_found(_: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::int(NAV_NOT_FOUND)
}

fn get_data(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.touch_dataref(c.a(0));
    Ret::int(0)
}
fn get_datai(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let h = c.a(0);
    x.touch_dataref(h);
    if let Some(name) = x.dataref_name(h).map(str::to_owned) {
        if let Some(v) = x.replay.as_ref().and_then(|r| r.get_i32(&name)) {
            return Ret::int(v);
        }
    }
    Ret::int(0)
}
fn get_dataf(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let h = c.a(0);
    x.touch_dataref(h);
    if let Some(name) = x.dataref_name(h).map(str::to_owned) {
        if let Some(v) = x.replay.as_ref().and_then(|r| r.get_f32(&name)) {
            return Ret::f32(v);
        }
    }
    Ret::f32(0.0)
}
fn get_datad(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let h = c.a(0);
    x.touch_dataref(h);
    if let Some(name) = x.dataref_name(h).map(str::to_owned) {
        if let Some(v) = x.replay.as_ref().and_then(|r| r.get_f64(&name)) {
            return Ret::f64(v);
        }
    }
    Ret::f64(0.0)
}
fn set_data(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.touch_dataref(c.a(0));
    Ret::VOID
}

fn find_dataref(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let n = c.cstr(c.a(0));
    Ret::ptr(x.handle(&format!("dr:{n}")))
}
fn dataref_types(x: &mut Xplm, c: &mut Ctx) -> Ret {
    Ret::int(x.dataref_types.get(&c.a(0)).copied().unwrap_or(7) as i32)
}
fn register_accessor(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let name = c.cstr(c.a(0));
    let h = x.handle(&format!("dr:{name}"));
    x.dataref_types.insert(h, c.a(1) as u32);
    let (rrc, wrc) = (c.a(15), c.a(16));
    for (i, slot) in DATA_SLOTS.iter().enumerate() {
        let rc = if i % 2 == 0 { rrc } else { wrc };
        x.record(CallbackKind::DataAccessor(*slot), c.a(3 + i), rc, 0, Some(name.clone()), h);
    }
    Ret::ptr(h)
}
fn unregister_accessor(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let h = c.a(0);
    x.deactivate(|r| matches!(r.kind, CallbackKind::DataAccessor(_)) && r.handle == h);
    Ret::VOID
}

fn find_command(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let n = c.cstr(c.a(0));
    Ret::ptr(x.handle(&format!("cmd:{n}")))
}
fn register_cmd_handler(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let h = c.a(0);
    let name = x.command_name(h).map(str::to_owned);
    x.record(CallbackKind::CommandHandler, c.a(1), c.a(3), c.a(2), name, h);
    Ret::VOID
}
fn unregister_cmd_handler(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let (h, cb, rc) = (c.a(0), c.a(1), c.a(3));
    x.deactivate(|r| r.kind == CallbackKind::CommandHandler && r.handle == h && r.callback_va == cb && r.refcon == rc);
    Ret::VOID
}

fn register_flight_loop(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.record(CallbackKind::FlightLoop, c.a(0), c.a(2), 0, None, 0);
    Ret::VOID
}
fn unregister_flight_loop(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let (cb, rc) = (c.a(0), c.a(1));
    x.deactivate(|r| r.kind == CallbackKind::FlightLoop && r.callback_va == cb && r.refcon == rc);
    Ret::VOID
}
fn create_flight_loop(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let p = c.a(0);
    let h = x.fresh("flightloop");
    if let (Some(cb), Some(rc)) = (c.r64(p + 8), c.r64(p + 16)) {
        x.record(CallbackKind::FlightLoop, cb, rc, 0, None, h);
    }
    Ret::ptr(h)
}
fn destroy_flight_loop(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let h = c.a(0);
    x.deactivate(|r| r.kind == CallbackKind::FlightLoop && r.handle == h);
    Ret::VOID
}

fn register_draw(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let when = if c.a(2) as u32 != 0 { "before" } else { "after" };
    x.record(CallbackKind::Draw, c.a(0), c.a(3), c.a(1), Some(when.to_owned()), 0);
    Ret::int(1)
}
fn unregister_draw(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let (cb, rc) = (c.a(0), c.a(3));
    x.deactivate(|r| r.kind == CallbackKind::Draw && r.callback_va == cb && r.refcon == rc);
    Ret::int(1)
}
fn register_sniffer(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.record(CallbackKind::KeySniffer, c.a(0), c.a(2), c.a(1), None, 0);
    Ret::int(1)
}
fn unregister_sniffer(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let (cb, rc) = (c.a(0), c.a(2));
    x.deactivate(|r| r.kind == CallbackKind::KeySniffer && r.callback_va == cb && r.refcon == rc);
    Ret::int(1)
}

fn control_camera(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.record(CallbackKind::Camera, c.a(1), c.a(2), 0, None, 0);
    Ret::VOID
}
fn load_object_async(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.record(CallbackKind::ObjectLoaded, c.a(1), c.a(2), 0, None, 0);
    Ret::VOID
}
fn set_error_callback(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.record(CallbackKind::ErrorCallback, c.a(0), 0, 0, None, 0);
    Ret::VOID
}
fn enumerate_features(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.record(CallbackKind::FeatureEnumerator, c.a(0), c.a(1), 0, None, 0);
    Ret::VOID
}
fn all_monitor_bounds(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.record(CallbackKind::MonitorBounds, c.a(0), c.a(1), 0, None, 0);
    Ret::VOID
}

fn create_window(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let p = c.a(0);
    let h = x.fresh("window");
    let size = u64::from(c.r32(p).unwrap_or(0));
    let visible = c.r32(p + 20).unwrap_or(0) != 0;
    x.window_visible.insert(h, visible);
    let rc = if size >= 72 { c.r64(p + 64).unwrap_or(0) } else { 0 };
    for (off, kind) in [
        (24, CallbackKind::WindowDraw),
        (32, CallbackKind::WindowMouseClick),
        (40, CallbackKind::WindowKey),
        (48, CallbackKind::WindowCursor),
        (56, CallbackKind::WindowMouseWheel),
        (80, CallbackKind::WindowRightClick),
    ] {
        if size >= off + 8 {
            if let Some(cb) = c.r64(p + off) {
                x.record(kind, cb, rc, 0, None, h);
            }
        }
    }
    Ret::ptr(h)
}
fn destroy_window(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let h = c.a(0);
    x.deactivate(|r| {
        r.handle == h
            && matches!(
                r.kind,
                CallbackKind::WindowDraw
                    | CallbackKind::WindowMouseClick
                    | CallbackKind::WindowRightClick
                    | CallbackKind::WindowKey
                    | CallbackKind::WindowCursor
                    | CallbackKind::WindowMouseWheel
            )
    });
    x.window_visible.remove(&h);
    Ret::VOID
}
fn get_window_visible(x: &mut Xplm, c: &mut Ctx) -> Ret {
    Ret::int(i32::from(x.window_visible.get(&c.a(0)).copied().unwrap_or(false)))
}
fn set_window_visible(x: &mut Xplm, c: &mut Ctx) -> Ret {
    x.window_visible.insert(c.a(0), c.a(1) as u32 != 0);
    Ret::VOID
}
fn window_geometry(_: &mut Xplm, c: &mut Ctx) -> Ret {
    // left, top, right, bottom
    for (i, v) in [100u32, 700, 500, 100].into_iter().enumerate() {
        let p = c.a(1 + i);
        c.w32(p, v);
    }
    Ret::VOID
}
fn screen_size(_: &mut Xplm, c: &mut Ctx) -> Ret {
    let (w, h) = (c.a(0), c.a(1));
    c.w32(w, 1920);
    c.w32(h, 1080);
    Ret::VOID
}
fn screen_bounds(_: &mut Xplm, c: &mut Ctx) -> Ret {
    for (i, v) in [0u32, 1080, 1920, 0].into_iter().enumerate() {
        let p = c.a(i);
        c.w32(p, v);
    }
    Ret::VOID
}
fn mouse_location(_: &mut Xplm, c: &mut Ctx) -> Ret {
    let (px, py) = (c.a(0), c.a(1));
    c.w32(px, 0);
    c.w32(py, 0);
    Ret::VOID
}
fn font_dimensions(_: &mut Xplm, c: &mut Ctx) -> Ret {
    for (i, v) in [8u32, 12, 0].into_iter().enumerate() {
        let p = c.a(1 + i);
        c.w32(p, v);
    }
    Ret::VOID
}
fn measure_string(_: &mut Xplm, c: &mut Ctx) -> Ret {
    Ret::f32(c.a(2) as u32 as f32 * 8.0)
}
fn gen_textures(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let (p, n) = (c.a(0), c.a(1) as u32);
    for i in 0..n.min(256) {
        c.w32(p + u64::from(i) * 4, x.next_tex);
        x.next_tex += 1;
    }
    Ret::VOID
}
fn world_to_local(_: &mut Xplm, c: &mut Ctx) -> Ret {
    // (lat, lon, alt as doubles in xmm0-2), double* out x/y/z = args 3,4,5; also LocalToWorld
    for i in 3..6 {
        let p = c.a(i);
        c.wbytes(p, &0f64.to_le_bytes());
    }
    Ret::VOID
}

fn find_plugins_menu(x: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::ptr(x.handle("menu:plugins"))
}
fn find_aircraft_menu(x: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::ptr(x.handle("menu:aircraft"))
}
fn create_menu(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let h = x.fresh("menu");
    x.menu_handlers.insert(h, (c.a(3), c.a(4)));
    Ret::ptr(h)
}
fn append_menu_item(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let (menu, item_ref) = (c.a(0), c.a(2));
    let name = c.cstr(c.a(1));
    x.append_item(menu, name, item_ref)
}
fn append_menu_item_cmd(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let menu = c.a(0);
    let name = c.cstr(c.a(1));
    x.append_item(menu, name, 0)
}
fn append_separator(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let m = c.a(0);
    *x.menu_items.entry(m).or_insert(0) += 1;
    Ret::VOID
}
fn check_menu_state(_: &mut Xplm, c: &mut Ctx) -> Ret {
    let p = c.a(2);
    c.w32(p, 0);
    Ret::VOID
}
fn destroy_menu(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let m = c.a(0);
    x.deactivate(|r| r.kind == CallbackKind::MenuItem && r.handle == m);
    Ret::VOID
}

fn create_probe(x: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::ptr(x.fresh("probe"))
}
fn probe_terrain(_: &mut Xplm, c: &mut Ctx) -> Ret {
    // zero the XPLMProbeInfo_t body after structSize; result 0 = hit terrain
    let p = c.a(4);
    if let Some(sz) = c.r32(p) {
        let n = (sz as usize).clamp(4, 256) - 4;
        c.wbytes(p + 4, &vec![0u8; n]);
    }
    Ret::int(0)
}
fn create_instance(x: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::ptr(x.fresh("instance"))
}

fn debug_string(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let s = c.cstr(c.a(0));
    x.debug_strings.push(s);
    Ret::VOID
}
fn speak_string(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let s = c.cstr(c.a(0));
    x.spoken.push(s);
    Ret::VOID
}

fn get_versions(_: &mut Xplm, c: &mut Ctx) -> Ret {
    for (i, v) in [12140u32, 410, 1].into_iter().enumerate() {
        let p = c.a(i);
        c.w32(p, v);
    }
    Ret::VOID
}
fn get_my_id(_: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::ptr(MY_PLUGIN_ID)
}
fn find_plugin_by_sig(_: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::int(NO_PLUGIN_ID)
}
fn get_plugin_info(_: &mut Xplm, c: &mut Ctx) -> Ret {
    for (i, s) in ["Stub Plugin", "C:\\X-Plane 12\\Resources\\plugins\\stub\\win_x64\\stub.xpl", "stub.plugin", "stub"]
        .into_iter()
        .enumerate()
    {
        let p = c.a(1 + i);
        c.wstr(p, s);
    }
    Ret::VOID
}
fn is_plugin_enabled(_: &mut Xplm, c: &mut Ctx) -> Ret {
    Ret::int(i32::from(c.a(0) == MY_PLUGIN_ID))
}
fn has_feature(_: &mut Xplm, c: &mut Ctx) -> Ret {
    let n = c.cstr(c.a(0));
    Ret::int(i32::from(KNOWN_FEATURES.contains(&n.as_str())))
}
fn is_feature_enabled(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let n = c.cstr(c.a(0));
    Ret::int(i32::from(x.features.contains(&n)))
}
fn enable_feature(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let n = c.cstr(c.a(0));
    if c.a(1) as u32 != 0 {
        x.features.insert(n);
    } else {
        x.features.remove(&n);
    }
    Ret::VOID
}

fn get_system_path(_: &mut Xplm, c: &mut Ctx) -> Ret {
    let p = c.a(0);
    c.wstr(p, SYSTEM_PATH);
    Ret::VOID
}
fn get_prefs_path(_: &mut Xplm, c: &mut Ctx) -> Ret {
    let p = c.a(0);
    c.wstr(p, PREFS_PATH);
    Ret::VOID
}
fn get_dir_sep(x: &mut Xplm, c: &mut Ctx) -> Ret {
    if x.sep_addr.is_none() {
        x.sep_addr = c.mem.alloc(2).filter(|a| c.mem.write(*a, b"\\\0"));
    }
    Ret::ptr(x.sep_addr.unwrap_or(0))
}

fn nav_info(_: &mut Xplm, c: &mut Ctx) -> Ret {
    for i in 1..=6 {
        let p = c.a(i);
        c.w32(p, 0);
    }
    for i in 7..=9 {
        let p = c.a(i);
        c.wstr(p, "");
    }
    Ret::VOID
}
fn fms_info(_: &mut Xplm, c: &mut Ctx) -> Ret {
    for i in [1, 3, 4, 5, 6] {
        let p = c.a(i);
        c.w32(p, 0);
    }
    let id = c.a(2);
    c.wstr(id, "");
    Ret::VOID
}

fn create_command(x: &mut Xplm, c: &mut Ctx) -> Ret {
    find_command(x, c)
}

fn xp_create_widget(x: &mut Xplm, _: &mut Ctx) -> Ret {
    Ret::ptr(x.fresh("widget"))
}
fn xp_widget_window(x: &mut Xplm, c: &mut Ctx) -> Ret {
    let w = c.a(0);
    Ret::ptr(x.handle(&format!("widgetwin:{w:#x}")))
}

impl Xplm {
    fn append_item(&mut self, menu: u64, name: String, item_ref: u64) -> Ret {
        let idx = self.menu_items.entry(menu).or_insert(0);
        let i = *idx;
        *idx += 1;
        let (h, r) = self.menu_handlers.get(&menu).copied().unwrap_or((0, 0));
        self.record(CallbackKind::MenuItem, h, r, item_ref, Some(name), menu);
        Ret::int(i)
    }
}

// ---- stub table -------------------------------------------------------------------------------

const STUBS: &[(&str, Handler)] = &[
    // identity / features
    ("XPLMGetVersions", get_versions),
    ("XPLMGetMyID", get_my_id),
    ("XPLMCountPlugins", one),
    ("XPLMGetNthPlugin", get_my_id),
    ("XPLMFindPluginBySignature", find_plugin_by_sig),
    ("XPLMGetPluginInfo", get_plugin_info),
    ("XPLMIsPluginEnabled", is_plugin_enabled),
    ("XPLMEnablePlugin", is_plugin_enabled),
    ("XPLMDisablePlugin", void),
    ("XPLMSendMessageToPlugin", void),
    ("XPLMHasFeature", has_feature),
    ("XPLMIsFeatureEnabled", is_feature_enabled),
    ("XPLMEnableFeature", enable_feature),
    ("XPLMEnumerateFeatures", enumerate_features),
    ("XPLMSetErrorCallback", set_error_callback),
    // paths / diagnostics
    ("XPLMGetSystemPath", get_system_path),
    ("XPLMGetPrefsPath", get_prefs_path),
    ("XPLMGetDirectorySeparator", get_dir_sep),
    ("XPLMDebugString", debug_string),
    ("XPLMSpeakString", speak_string),
    // datarefs
    ("XPLMFindDataRef", find_dataref),
    ("XPLMCanWriteDataRef", one),
    ("XPLMGetDataRefTypes", dataref_types),
    ("XPLMGetDatai", get_datai),
    ("XPLMGetDataf", get_dataf),
    ("XPLMGetDatad", get_datad),
    ("XPLMGetDatavi", get_data),
    ("XPLMGetDatavf", get_data),
    ("XPLMGetDatab", get_data),
    ("XPLMSetDatai", set_data),
    ("XPLMSetDataf", set_data),
    ("XPLMSetDatad", set_data),
    ("XPLMSetDatavi", set_data),
    ("XPLMSetDatavf", set_data),
    ("XPLMSetDatab", set_data),
    ("XPLMRegisterDataAccessor", register_accessor),
    ("XPLMUnregisterDataAccessor", unregister_accessor),
    // commands
    ("XPLMFindCommand", find_command),
    ("XPLMCreateCommand", create_command),
    ("XPLMCommandBegin", void),
    ("XPLMCommandEnd", void),
    ("XPLMCommandOnce", void),
    ("XPLMRegisterCommandHandler", register_cmd_handler),
    ("XPLMUnregisterCommandHandler", unregister_cmd_handler),
    // flight loops / draw / keys / camera
    ("XPLMRegisterFlightLoopCallback", register_flight_loop),
    ("XPLMUnregisterFlightLoopCallback", unregister_flight_loop),
    ("XPLMCreateFlightLoop", create_flight_loop),
    ("XPLMDestroyFlightLoop", destroy_flight_loop),
    ("XPLMScheduleFlightLoop", void),
    ("XPLMRegisterDrawCallback", register_draw),
    ("XPLMUnregisterDrawCallback", unregister_draw),
    ("XPLMRegisterKeySniffer", register_sniffer),
    ("XPLMUnregisterKeySniffer", unregister_sniffer),
    ("XPLMControlCamera", control_camera),
    // windows / screen
    ("XPLMCreateWindowEx", create_window),
    ("XPLMDestroyWindow", destroy_window),
    ("XPLMGetScreenSize", screen_size),
    ("XPLMGetScreenBoundsGlobal", screen_bounds),
    ("XPLMGetAllMonitorBoundsGlobal", all_monitor_bounds),
    ("XPLMGetMouseLocationGlobal", mouse_location),
    ("XPLMGetWindowGeometry", window_geometry),
    ("XPLMGetWindowGeometryOS", window_geometry),
    ("XPLMSetWindowGeometry", void),
    ("XPLMSetWindowGeometryOS", void),
    ("XPLMGetWindowIsVisible", get_window_visible),
    ("XPLMSetWindowIsVisible", set_window_visible),
    ("XPLMWindowIsPoppedOut", zero),
    ("XPLMSetWindowResizingLimits", void),
    ("XPLMSetWindowPositioningMode", void),
    ("XPLMSetWindowTitle", void),
    ("XPLMTakeKeyboardFocus", void),
    ("XPLMHasKeyboardFocus", zero),
    ("XPLMBringWindowToFront", void),
    ("XPLMIsWindowInFront", one),
    // graphics
    ("XPLMSetGraphicsState", void),
    ("XPLMBindTexture2d", void),
    ("XPLMGenerateTextureNumbers", gen_textures),
    ("XPLMWorldToLocal", world_to_local),
    ("XPLMLocalToWorld", world_to_local),
    ("XPLMDrawTranslucentDarkBox", void),
    ("XPLMDrawString", void),
    ("XPLMGetFontDimensions", font_dimensions),
    ("XPLMMeasureString", measure_string),
    // instances / probes / objects
    ("XPLMCreateInstance", create_instance),
    ("XPLMDestroyInstance", void),
    ("XPLMInstanceSetPosition", void),
    ("XPLMCreateProbe", create_probe),
    ("XPLMDestroyProbe", void),
    ("XPLMProbeTerrainXYZ", probe_terrain),
    ("XPLMLoadObjectAsync", load_object_async),
    ("XPLMUnloadObject", void),
    // menus
    ("XPLMFindPluginsMenu", find_plugins_menu),
    ("XPLMFindAircraftMenu", find_aircraft_menu),
    ("XPLMCreateMenu", create_menu),
    ("XPLMDestroyMenu", destroy_menu),
    ("XPLMAppendMenuItem", append_menu_item),
    ("XPLMAppendMenuItemWithCommand", append_menu_item_cmd),
    ("XPLMAppendMenuSeparator", append_separator),
    ("XPLMSetMenuItemName", void),
    ("XPLMCheckMenuItem", void),
    ("XPLMCheckMenuItemState", check_menu_state),
    ("XPLMEnableMenuItem", void),
    ("XPLMRemoveMenuItem", void),
    // navigation / FMS: "nothing found"
    ("XPLMGetNextNavAid", nav_not_found),
    ("XPLMFindFirstNavAidOfType", nav_not_found),
    ("XPLMFindLastNavAidOfType", nav_not_found),
    ("XPLMFindNavAid", nav_not_found),
    ("XPLMGetNavAidInfo", nav_info),
    ("XPLMCountFMSEntries", zero),
    ("XPLMGetDestinationFMSEntry", zero),
    ("XPLMGetFMSEntryInfo", fms_info),
];

const WIDGET_STUBS: &[(&str, Handler)] = &[
    ("XPCreateWidget", xp_create_widget),
    ("XPGetWidgetGeometry", window_geometry),
    ("XPGetWidgetUnderlyingWindow", xp_widget_window),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Mock {
        b: BTreeMap<u64, u8>,
        bump: u64,
    }
    impl Mock {
        fn new() -> Self {
            Mock { b: BTreeMap::new(), bump: 0x9000_0000 }
        }
        fn put(&mut self, a: u64, d: &[u8]) {
            for (i, v) in d.iter().enumerate() {
                self.b.insert(a + i as u64, *v);
            }
        }
        fn cstr(&self, mut a: u64) -> String {
            let mut s = Vec::new();
            while let Some(&v) = self.b.get(&a) {
                if v == 0 {
                    break;
                }
                s.push(v);
                a += 1;
            }
            String::from_utf8(s).unwrap()
        }
        fn u32(&self, a: u64) -> u32 {
            let mut x = [0u8; 4];
            for (i, v) in x.iter_mut().enumerate() {
                *v = self.b[&(a + i as u64)];
            }
            u32::from_le_bytes(x)
        }
    }
    impl GuestMem for Mock {
        fn read(&self, a: u64, buf: &mut [u8]) -> bool {
            for (i, v) in buf.iter_mut().enumerate() {
                match self.b.get(&(a + i as u64)) {
                    Some(x) => *v = *x,
                    None => return false,
                }
            }
            true
        }
        fn write(&mut self, a: u64, d: &[u8]) -> bool {
            self.put(a, d);
            true
        }
        fn alloc(&mut self, len: usize) -> Option<u64> {
            let a = self.bump;
            self.bump += len as u64;
            Some(a)
        }
    }

    const DLL: &str = "XPLM_64.dll";

    fn call(x: &mut Xplm, m: &mut Mock, f: &str, args: [u64; 4]) -> Ret {
        match x.dispatch(DLL, f, &Call { args, rsp: 0 }, m) {
            Dispatch::Handled(r) => r,
            other => panic!("{f}: {other:?}"),
        }
    }

    #[test]
    fn registration_records_callback_and_returns_handle() {
        let (mut x, mut m) = (Xplm::new(), Mock::new());
        call(&mut x, &mut m, "XPLMRegisterFlightLoopCallback", [0x1111, 0, 0x2222, 0]);
        let r = call(&mut x, &mut m, "XPLMRegisterDrawCallback", [0x3333, 5, 1, 0x4444]);
        assert_eq!(r.rax, 1);
        // CreateFlightLoop: struct {size, phase, cb@8, refcon@16}
        m.put(0x100, &20u32.to_le_bytes());
        m.put(0x108, &0x5555u64.to_le_bytes());
        m.put(0x110, &0x6666u64.to_le_bytes());
        let h = call(&mut x, &mut m, "XPLMCreateFlightLoop", [0x100, 0, 0, 0]).rax;
        assert_ne!(h, 0);
        let cbs = x.callbacks();
        assert_eq!(cbs.len(), 3);
        assert_eq!((cbs[0].kind, cbs[0].callback_va, cbs[0].refcon), (CallbackKind::FlightLoop, 0x1111, 0x2222));
        assert_eq!((cbs[1].kind, cbs[1].callback_va, cbs[1].refcon, cbs[1].aux), (CallbackKind::Draw, 0x3333, 0x4444, 5));
        assert_eq!((cbs[2].callback_va, cbs[2].refcon, cbs[2].handle), (0x5555, 0x6666, h));
        call(&mut x, &mut m, "XPLMDestroyFlightLoop", [h, 0, 0, 0]);
        assert_eq!(x.active_callbacks(CallbackKind::FlightLoop).count(), 1);
    }

    #[test]
    fn data_accessor_records_slots_and_matches_find() {
        let (mut x, mut m) = (Xplm::new(), Mock::new());
        m.put(0x200, b"my/dataref\0");
        // stack args: rsp=0x1000, arg4 @ 0x1028 ... arg(3+i) for slots; slot 0 ReadInt = arg3 (reg)
        let rsp = 0x1000u64;
        let mut stack = vec![0u64; 13];
        stack[0] = 0xAAA0; // arg4 = WriteInt
        stack[11] = 0xBEEF; // arg15 readRefcon
        stack[12] = 0xCAFE; // arg16 writeRefcon
        for (i, v) in stack.iter().enumerate() {
            m.put(rsp + 0x28 + 8 * i as u64, &v.to_le_bytes());
        }
        let r = match x.dispatch(DLL, "XPLMRegisterDataAccessor", &Call { args: [0x200, 1, 1, 0xAAA1], rsp }, &mut m) {
            Dispatch::Handled(r) => r,
            d => panic!("{d:?}"),
        };
        let kinds: Vec<_> = x.callbacks().iter().map(|c| (c.kind, c.callback_va, c.refcon)).collect();
        assert_eq!(kinds, vec![
            (CallbackKind::DataAccessor(DataSlot::ReadInt), 0xAAA1, 0xBEEF),
            (CallbackKind::DataAccessor(DataSlot::WriteInt), 0xAAA0, 0xCAFE),
        ]);
        assert_eq!(x.callbacks()[0].name.as_deref(), Some("my/dataref"));
        let f = call(&mut x, &mut m, "XPLMFindDataRef", [0x200, 0, 0, 0]);
        assert_eq!(f.rax, r.rax);
        assert_eq!(x.dataref_name(f.rax), Some("my/dataref"));
        assert_eq!(call(&mut x, &mut m, "XPLMGetDataRefTypes", [f.rax, 0, 0, 0]).rax, 1);
    }

    #[test]
    fn find_dataref_mints_stable_handles_per_name() {
        let (mut x, mut m) = (Xplm::new(), Mock::new());
        m.put(0x300, b"a/b\0");
        m.put(0x310, b"c/d\0");
        let a1 = call(&mut x, &mut m, "XPLMFindDataRef", [0x300, 0, 0, 0]).rax;
        let b = call(&mut x, &mut m, "XPLMFindDataRef", [0x310, 0, 0, 0]).rax;
        let a2 = call(&mut x, &mut m, "XPLMFindDataRef", [0x300, 0, 0, 0]).rax;
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        assert_ne!(a1, 0);
        let c = call(&mut x, &mut m, "XPLMFindCommand", [0x300, 0, 0, 0]).rax;
        assert_ne!(c, a1);
        assert_eq!(x.command_name(c), Some("a/b"));
    }

    #[test]
    fn system_path_and_versions_written_to_guest() {
        let (mut x, mut m) = (Xplm::new(), Mock::new());
        call(&mut x, &mut m, "XPLMGetSystemPath", [0x400, 0, 0, 0]);
        assert_eq!(m.cstr(0x400), "C:\\X-Plane 12\\");
        call(&mut x, &mut m, "XPLMGetVersions", [0x500, 0x504, 0x508, 0]);
        assert_eq!((m.u32(0x500), m.u32(0x504), m.u32(0x508)), (12140, 410, 1));
        let sep = call(&mut x, &mut m, "XPLMGetDirectorySeparator", [0; 4]).rax;
        assert_eq!(m.cstr(sep), "\\");
        assert_eq!(call(&mut x, &mut m, "XPLMGetDirectorySeparator", [0; 4]).rax, sep);
    }

    #[test]
    fn debug_and_speak_strings_captured() {
        let (mut x, mut m) = (Xplm::new(), Mock::new());
        m.put(0x600, b"init failed: no datarefs\n\0");
        m.put(0x700, b"hello\0");
        call(&mut x, &mut m, "XPLMDebugString", [0x600, 0, 0, 0]);
        call(&mut x, &mut m, "XPLMSpeakString", [0x700, 0, 0, 0]);
        assert_eq!(x.debug_strings(), ["init failed: no datarefs\n"]);
        assert_eq!(x.spoken(), ["hello"]);
    }

    #[test]
    fn unlisted_function_is_reported_not_zeroed() {
        let (mut x, mut m) = (Xplm::new(), Mock::new());
        let c = Call::default();
        assert_eq!(x.dispatch(DLL, "XPLMNoSuchThing", &c, &mut m), Dispatch::Unlisted);
        assert_eq!(x.dispatch(DLL, "XPLMNoSuchThing", &c, &mut m), Dispatch::Unlisted);
        assert_eq!(x.dispatch("kernel32.dll", "Sleep", &c, &mut m), Dispatch::NotXplm);
        let u = x.unlisted();
        assert_eq!(u.len(), 1);
        assert_eq!((u[0].0.1.as_str(), u[0].1), ("XPLMNoSuchThing", 2));
    }

    #[test]
    fn window_menu_and_widgets() {
        let (mut x, mut m) = (Xplm::new(), Mock::new());
        // window struct: size 88, visible=1, draw@24, click@32, refcon@64
        m.put(0x800, &88u32.to_le_bytes());
        m.put(0x814, &1u32.to_le_bytes());
        m.put(0x818, &0xD0u64.to_le_bytes());
        m.put(0x820, &0xC0u64.to_le_bytes());
        m.put(0x840, &0xFEEDu64.to_le_bytes());
        let w = call(&mut x, &mut m, "XPLMCreateWindowEx", [0x800, 0, 0, 0]).rax;
        assert_eq!(call(&mut x, &mut m, "XPLMGetWindowIsVisible", [w, 0, 0, 0]).rax, 1);
        let kinds: Vec<_> = x.callbacks().iter().map(|c| (c.kind, c.callback_va, c.refcon)).collect();
        assert_eq!(kinds[0], (CallbackKind::WindowDraw, 0xD0, 0xFEED));
        assert_eq!(kinds[1], (CallbackKind::WindowMouseClick, 0xC0, 0xFEED));
        // menu with handler + item
        m.put(0x900, b"Item\0");
        let menu = call(&mut x, &mut m, "XPLMCreateMenu", [0x900, 0, 0, 0xABC]).rax;
        let idx = call(&mut x, &mut m, "XPLMAppendMenuItem", [menu, 0x900, 0x77, 0]);
        assert_eq!(idx.rax, 0);
        let mi = x.active_callbacks(CallbackKind::MenuItem).next().unwrap();
        assert_eq!((mi.callback_va, mi.aux, mi.name.as_deref()), (0xABC, 0x77, Some("Item")));
        // widgets
        let wd = match x.dispatch("XPWidgets_64.dll", "XPCreateWidget", &Call::default(), &mut m) {
            Dispatch::Handled(r) => r.rax,
            d => panic!("{d:?}"),
        };
        assert_ne!(wd, 0);
        assert!(x.is_modeled("XPWidgets_64.dll", "XPGetWidgetUnderlyingWindow"));
    }

    #[test]
    fn all_imported_functions_are_modeled() {
        const IMPORTS: &str = "XPLMControlCamera XPLMFindDataRef XPLMCanWriteDataRef XPLMGetDataRefTypes XPLMGetDatai XPLMSetDatai XPLMGetDataf XPLMSetDataf XPLMGetDatad XPLMSetDatad XPLMGetDatavi XPLMSetDatavi XPLMGetDatavf XPLMSetDatavf XPLMGetDatab XPLMSetDatab XPLMRegisterDataAccessor XPLMUnregisterDataAccessor XPLMRegisterDrawCallback XPLMUnregisterDrawCallback XPLMCreateWindowEx XPLMDestroyWindow XPLMGetScreenSize XPLMGetScreenBoundsGlobal XPLMGetAllMonitorBoundsGlobal XPLMGetMouseLocationGlobal XPLMGetWindowGeometry XPLMSetWindowGeometry XPLMGetWindowGeometryOS XPLMSetWindowGeometryOS XPLMGetWindowIsVisible XPLMSetWindowIsVisible XPLMWindowIsPoppedOut XPLMSetWindowResizingLimits XPLMSetWindowPositioningMode XPLMSetWindowTitle XPLMTakeKeyboardFocus XPLMHasKeyboardFocus XPLMBringWindowToFront XPLMIsWindowInFront XPLMRegisterKeySniffer XPLMUnregisterKeySniffer XPLMSetGraphicsState XPLMBindTexture2d XPLMGenerateTextureNumbers XPLMWorldToLocal XPLMLocalToWorld XPLMDrawTranslucentDarkBox XPLMDrawString XPLMGetFontDimensions XPLMMeasureString XPLMCreateInstance XPLMDestroyInstance XPLMInstanceSetPosition XPLMSetErrorCallback XPLMGetMyID XPLMCountPlugins XPLMGetNthPlugin XPLMFindPluginBySignature XPLMGetPluginInfo XPLMIsPluginEnabled XPLMEnablePlugin XPLMDisablePlugin XPLMSendMessageToPlugin XPLMHasFeature XPLMIsFeatureEnabled XPLMEnableFeature XPLMEnumerateFeatures XPLMFindPluginsMenu XPLMFindAircraftMenu XPLMCreateMenu XPLMDestroyMenu XPLMAppendMenuItem XPLMAppendMenuItemWithCommand XPLMAppendMenuSeparator XPLMSetMenuItemName XPLMCheckMenuItem XPLMCheckMenuItemState XPLMEnableMenuItem XPLMRemoveMenuItem XPLMGetNextNavAid XPLMFindFirstNavAidOfType XPLMFindLastNavAidOfType XPLMFindNavAid XPLMGetNavAidInfo XPLMCountFMSEntries XPLMGetDestinationFMSEntry XPLMGetFMSEntryInfo XPLMRegisterFlightLoopCallback XPLMUnregisterFlightLoopCallback XPLMCreateFlightLoop XPLMDestroyFlightLoop XPLMScheduleFlightLoop XPLMCreateProbe XPLMDestroyProbe XPLMProbeTerrainXYZ XPLMLoadObjectAsync XPLMUnloadObject XPLMGetSystemPath XPLMGetVersions XPLMDebugString XPLMSpeakString XPLMFindCommand XPLMCommandBegin XPLMCommandEnd XPLMCommandOnce XPLMCreateCommand XPLMRegisterCommandHandler XPLMUnregisterCommandHandler";
        let (mut x, mut m) = (Xplm::new(), Mock::new());
        let names: Vec<_> = IMPORTS.split_whitespace().collect();
        assert_eq!(names.len(), 109);
        for n in names {
            // zeroed args exercise the null-pointer guards in every stub
            let d = x.dispatch(DLL, n, &Call::default(), &mut m);
            assert!(matches!(d, Dispatch::Handled(_)), "{n} not modeled");
        }
        for n in ["XPCreateWidget", "XPGetWidgetGeometry", "XPGetWidgetUnderlyingWindow"] {
            assert!(x.is_modeled("XPWidgets_64.dll", n), "{n}");
        }
        assert!(x.unlisted().is_empty());
    }

    #[test]
    fn invocations_follow_each_callback_abi() {
        let rec = |kind, aux, name: Option<&str>| RecordedCallback {
            kind,
            callback_va: 0x10,
            refcon: 0x20,
            aux,
            name: name.map(str::to_owned),
            handle: 0x30,
            active: true,
        };
        let fl = rec(CallbackKind::FlightLoop, 0, None).invocations(2, 0x900);
        assert_eq!(fl.len(), 1);
        assert_eq!((fl[0].ints, fl[0].ret), ([0, 0, 3, 0x20], RetKind::Float));
        assert_eq!(fl[0].xmm[0], Some(u128::from((1.0f32 / 60.0).to_bits())));
        // Only repeating kinds run past the first round.
        assert!(rec(CallbackKind::CommandHandler, 0, None).invocations(1, 0x900).is_empty());
        let cmd = rec(CallbackKind::CommandHandler, 0, None).invocations(0, 0x900);
        assert_eq!(cmd.iter().map(|c| c.ints[1]).collect::<Vec<_>>(), [0, 2]);
        let draw = rec(CallbackKind::Draw, 5, Some("before")).invocations(1, 0x900);
        assert_eq!(draw[0].ints, [5, 1, 0x20, 0]);
        // The 5th/6th args of a wheel callback ride the stack.
        let wheel = rec(CallbackKind::WindowMouseWheel, 0, None).invocations(0, 0x900);
        assert_eq!(wheel[0].stack, [1, 0x20]);
        // Arrays: a NULL size query, then a bounded transfer; scalar writes carry the value in xmm1.
        let arr = rec(CallbackKind::DataAccessor(DataSlot::ReadFloatArray), 0, None).invocations(0, 0x900);
        assert_eq!((arr[0].ints[1], arr[1].ints[1], arr[1].ints[3]), (0, 0x900, 8));
        let wf = rec(CallbackKind::DataAccessor(DataSlot::WriteFloat), 0, None).invocations(0, 0x900);
        assert_eq!(wf[0].xmm[1], Some(u128::from(1.0f32.to_bits())));
        assert!(rec(CallbackKind::ErrorCallback, 0, None).invocations(0, 0x900).is_empty());
    }
}
