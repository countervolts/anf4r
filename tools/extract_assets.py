#!/usr/bin/env python3
"""anf4r: extract libanf.so's embedded assets.

Obfuscation recovered from the loader at libanf.so:0x1c3904 (XREF uncompress@plt):
  key    = basename(name) with last extension removed   (the stem)
  stage1 = cipher[i] ^ key[i % len(key)]  for i in [0, stored_size)   # repeating XOR
  plain  = zlib.decompress(stage1)                                     # expected len = size
Table of 1278 entries (24 B: char* name, u32 offset, u32 stored, u64 size) in .data.rel.ro;
offsets are relative to the start of .data. No cryptographic key, no device binding.
"""
import sys, os, struct, zlib
sys.path.insert(0, os.path.dirname(__file__))
from elfmap import Elf

def main(lib, outdir):
    e = Elf(lib)
    data_base = e.off(e.sec['.data'][3])          # file offset of .data
    rows = [l.rstrip('\n').split('\t') for l in open(os.path.join(os.path.dirname(outdir) or '.', 'filetable.tsv'))] \
           if os.path.exists('vendor/filetable.tsv') else None
    # Re-read the table straight from the ELF so the extractor is self-contained.
    a = 0x1d2100
    def entry(va):
        o = e.off(va); _z, off, stored = struct.unpack_from("<QII", e.d, o); name_ptr = e.rel.get(va, 0); size, = struct.unpack_from('<Q', e.d, o+16)
        if not (0 < name_ptr < 0x60000): return None
        no = e.off(name_ptr)
        if no is None: return None
        end = e.d.index(b'\0', no); name = e.d[no:end].decode('latin1')
        if not name or '/' not in name and '.' not in name: return None
        return name, off, stored, size
    ents = []
    while entry(a): ents.append(entry(a)); a += 24
    ok = bad = 0
    for name, off, stored, size in ents:
        stem = os.path.basename(name).rsplit('.', 1)[0]
        key = stem.encode('latin1')
        ct = e.d[data_base+off: data_base+off+stored]
        pt = bytes(c ^ key[i % len(key)] for i, c in enumerate(ct))
        try:
            out = zlib.decompress(pt)
        except zlib.error as ex:
            print("FAIL", name, ex); bad += 1; continue
        if len(out) != size:
            print("SIZE", name, len(out), "!=", size); bad += 1; continue
        dst = os.path.join(outdir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, 'wb').write(out); ok += 1
    print(f"extracted {ok}/{len(ents)} assets to {outdir} ({bad} failed)")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
