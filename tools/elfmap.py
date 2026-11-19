"""Minimal ELF64 helpers for libanf.so: vaddr<->offset, relocations (RELATIVE + ABS64)."""
import struct
class Elf:
    def __init__(s, path):
        s.d = open(path,'rb').read(); d=s.d
        phoff, = struct.unpack_from('<Q',d,0x20); phn, = struct.unpack_from('<H',d,0x38)
        s.loads=[]
        for i in range(phn):
            t,f,off,va,pa,fs,ms,al = struct.unpack_from('<IIQQQQQQ',d,phoff+56*i)
            if t==1: s.loads.append((va,off,fs,ms))
        shoff, = struct.unpack_from('<Q',d,0x28); shn, = struct.unpack_from('<H',d,0x3c); shstr, = struct.unpack_from('<H',d,0x3e)
        sh=[struct.unpack_from('<IIQQQQIIQQ',d,shoff+64*i) for i in range(shn)]
        so=sh[shstr][4]
        s.sec={d[so+x[0]:d.index(b'\0',so+x[0])].decode():x for x in sh}
        s.rel={}  # vaddr -> target vaddr (R_AARCH64_RELATIVE)
        r=s.sec['.rela.dyn']
        for i in range(r[5]//24):
            o,info,add = struct.unpack_from('<QQq',d,r[4]+24*i)
            if info&0xffffffff==1027: s.rel[o]=add
    def off(s,va):
        for v,o,fs,ms in s.loads:
            if v<=va<v+fs: return o+va-v
        return None
    def va(s,off):
        for v,o,fs,ms in s.loads:
            if o<=off<o+fs: return v+off-o
    def cstr(s,va):
        o=s.off(va); return s.d[o:s.d.index(b'\0',o)].decode('latin1')
