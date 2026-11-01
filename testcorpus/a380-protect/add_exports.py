#!/usr/bin/env python3
"""Append a synthetic PE export table to fbw_a380_systems.dll so VMProtect_Con's
name-based MapAddress lookup can reach internal (non-exported) functions.

Why: VMProtect_Con resolves <Procedure MapAddress="..."> ONLY against the PE
export directory (confirmed empirically -- raw VA/RVA hex/decimal and the
Rust v0 mangled symbol name all fail with "not found"; only real export
names like "XPluginStart" resolve). This DLL exports just 5 functions.
This script adds new export entries (name -> existing code RVA) for a chosen
address list, keeping the original 5 exports and all existing bytes
byte-identical. It only APPENDS a new section after the last one -- no
existing bytes move, no code changes, nothing is executed.

Usage: add_exports.py <in.dll> <out.dll> <addr_name_list.txt>
  addr_name_list.txt: one "<hex_va> <export_name>" per line. export_name
  must be a valid, unique C identifier-ish string (no spaces/nulls); if two
  input symbols would collide on name, caller must disambiguate first.
"""
import struct, sys, os

IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
IMAGE_SCN_MEM_READ = 0x40000000


def round_up(v, a):
    return (v + a - 1) // a * a


def read_cstr(data, off):
    end = data.index(b"\x00", off)
    return bytes(data[off:end])


def main():
    in_path, out_path, list_path = sys.argv[1], sys.argv[2], sys.argv[3]
    data = bytearray(open(in_path, "rb").read())

    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    coff_off = pe_off + 4
    machine, nsec, ts, ptr_sym, n_sym, opt_sz, chars = struct.unpack_from("<HHIIIHH", data, coff_off)
    opt_off = coff_off + 20
    magic = struct.unpack_from("<H", data, opt_off)[0]
    assert magic == 0x20B, "expected PE32+"
    imagebase = struct.unpack_from("<Q", data, opt_off + 24)[0]
    sec_align, file_align = struct.unpack_from("<II", data, opt_off + 32)
    datadir_off = opt_off + 112
    export_va, export_sz = struct.unpack_from("<II", data, datadir_off)
    sect_table_off = opt_off + opt_sz
    free_hdr_bytes = struct.unpack_from("<I", data, opt_off + 60)[0] - (sect_table_off + nsec * 40)
    assert free_hdr_bytes >= 40, f"no room for a new section header ({free_hdr_bytes} free bytes)"

    sections = []
    for i in range(nsec):
        name, vsz, vaddr, rawsz, rawptr = struct.unpack_from("<8sIIII", data, sect_table_off + i * 40)
        sections.append(dict(name=name, vsz=vsz, vaddr=vaddr, rawsz=rawsz, rawptr=rawptr))

    # --- read the existing export directory (5 real exports) to preserve them ---
    def rva_to_off(rva):
        for s in sections:
            if s["vaddr"] <= rva < s["vaddr"] + max(s["vsz"], s["rawsz"]):
                return s["rawptr"] + (rva - s["vaddr"])
        raise ValueError(f"rva {rva:#x} not in any section")

    exp_off = rva_to_off(export_va)
    (characteristics, edir_ts, maj, minr, name_rva, base, n_func, n_names,
     addr_func_rva, addr_names_rva, addr_ord_rva) = struct.unpack_from("<IIHHIIIIIII", data, exp_off)
    orig_dll_name = read_cstr(data, rva_to_off(name_rva))
    func_off = rva_to_off(addr_func_rva)
    names_off = rva_to_off(addr_names_rva)
    ord_off = rva_to_off(addr_ord_rva)
    orig_funcs = list(struct.unpack_from(f"<{n_func}I", data, func_off))
    orig_name_rvas = list(struct.unpack_from(f"<{n_names}I", data, names_off))
    orig_ordinals = list(struct.unpack_from(f"<{n_names}H", data, ord_off))
    orig_entries = []  # (name_bytes, func_rva)
    for nr, ordidx in zip(orig_name_rvas, orig_ordinals):
        orig_entries.append((read_cstr(data, rva_to_off(nr)), orig_funcs[ordidx]))

    # --- new entries from the address/name list ---
    new_entries = []
    seen = {nm for nm, _ in orig_entries}
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            addr_s, name = line.split(None, 1)
            va = int(addr_s, 16)
            rva = va - imagebase
            name_b = name.encode("ascii", "strict")
            if name_b in seen:
                raise ValueError(f"duplicate export name: {name}")
            seen.add(name_b)
            new_entries.append((name_b, rva))

    all_entries = orig_entries + new_entries
    total = len(all_entries)
    # AddressTable order: as-given (orig first, then new); ordinal i -> all_entries[i]
    # NamePointerTable/OrdinalTable: sorted by name, ascending (required for export lookup).
    order_by_name = sorted(range(total), key=lambda i: all_entries[i][0])

    last = sections[-1]
    new_va = round_up(last["vaddr"] + max(last["vsz"], last["rawsz"]), sec_align)
    file_end = round_up(len(data), file_align)

    # --- lay out the new export block ---
    struct_sz = 40
    func_tbl_off = struct_sz
    names_tbl_off = func_tbl_off + total * 4
    ord_tbl_off = names_tbl_off + total * 4
    strings_off = ord_tbl_off + total * 2

    blob = bytearray(strings_off)
    func_rvas = [e[1] for e in all_entries]
    struct.pack_into(f"<{total}I", blob, func_tbl_off, *func_rvas)

    name_rvas = []
    ordinals_sorted = []
    cur = strings_off
    name_offsets = {}
    for i in order_by_name:
        nm = all_entries[i][0]
        name_offsets[i] = cur
        blob.extend(nm)
        blob.append(0)
        cur += len(nm) + 1
    for i in order_by_name:
        name_rvas.append(new_va + name_offsets[i])
        ordinals_sorted.append(i)  # index into AddressTable == ordinal - base
    struct.pack_into(f"<{total}I", blob, names_tbl_off, *name_rvas)
    struct.pack_into(f"<{total}H", blob, ord_tbl_off, *ordinals_sorted)

    dll_name_off = len(blob)
    blob.extend(orig_dll_name)
    blob.append(0)

    struct.pack_into("<IIHHIIIIIII", blob, 0,
                      0, edir_ts, 0, 0,
                      new_va + dll_name_off,
                      base, total, total,
                      new_va + func_tbl_off, new_va + names_tbl_off, new_va + ord_tbl_off)

    blob_len = len(blob)
    rawsz = round_up(blob_len, file_align)
    blob.extend(b"\x00" * (rawsz - blob_len))

    # --- append section to file ---
    data.extend(b"\x00" * (file_end - len(data)))
    assert len(data) == file_end
    new_rawptr = file_end
    data.extend(blob)

    # --- new section header ---
    name_field = b".nexp\x00\x00\x00"
    sect_hdr = struct.pack("<8sIIIIIIHHI", name_field, blob_len, new_va, rawsz, new_rawptr,
                            0, 0, 0, 0, IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ)
    new_sect_off = sect_table_off + nsec * 40
    data[new_sect_off:new_sect_off + 40] = sect_hdr

    # --- patch COFF + optional headers ---
    struct.pack_into("<H", data, coff_off + 2, nsec + 1)  # NumberOfSections
    new_size_of_image = new_va + round_up(blob_len, sec_align)
    struct.pack_into("<I", data, opt_off + 56, new_size_of_image)  # SizeOfImage
    struct.pack_into("<I", data, opt_off + 64, 0)  # CheckSum -> 0 (unverified, loader-safe)
    struct.pack_into("<II", data, datadir_off, new_va, blob_len)  # Export data directory

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)

    print(f"OK: {len(orig_entries)} original + {len(new_entries)} new exports -> {total} total")
    print(f"new section .nexp VA={new_va:#x} size={blob_len} rawptr={new_rawptr:#x} rawsz={rawsz}")
    print(f"output: {out_path} ({len(data)} bytes, input was {len(open(in_path,'rb').read())} bytes)")


if __name__ == "__main__":
    main()
