#!/usr/bin/env python3
"""Scan an ELF file for SPIR-V modules, zlib streams and high-entropy regions (anf4r static analysis)."""
import sys, struct, zlib, math, collections
data = open(sys.argv[1], 'rb').read()
def ent(b):
    if not b: return 0
    c = collections.Counter(b); n = len(b)
    return -sum(v/n*math.log2(v/n) for v in c.values())
spv = [i for i in range(0, len(data)-4, 4) if data[i:i+4] == b'\x03\x02\x23\x07']
print(f"SPIR-V magic (LE, 4-aligned): {len(spv)}")
for i in spv[:10]: print(f"  0x{i:x} ver=0x{struct.unpack_from('<I',data,i+4)[0]:08x} bound={struct.unpack_from('<I',data,i+12)[0]}")
z = []
for i in range(len(data)-2):
    if data[i] == 0x78 and data[i+1] in (0x01, 0x5e, 0x9c, 0xda) and ((data[i]<<8)|data[i+1]) % 31 == 0:
        d = zlib.decompressobj()
        try:
            out = d.decompress(data[i:i+64*1024*1024], 1<<28)
            if d.eof and len(out) > 64:
                z.append((i, len(data)-len(d.unused_data)-i, len(out), out[:4]))
        except zlib.error: pass
print(f"zlib streams: {len(z)}, total out {sum(x[2] for x in z)} B, total in {sum(x[1] for x in z)} B")
mag = collections.Counter(x[3].hex() for x in z)
print("  leading 4 bytes of decompressed:", mag.most_common(8))
for x in z[:5]: print(f"  0x{x[0]:x} in={x[1]} out={x[2]} head={x[3].hex()}")
print("entropy per 256 KiB window:")
for off in range(0, len(data), 256*1024):
    e = ent(data[off:off+256*1024]); print(f"  0x{off:07x} {e:.2f}", '#'*int(e*4))
