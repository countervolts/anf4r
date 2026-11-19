#!/usr/bin/env python3
"""Reflect a SPIR-V compute module: entry, local size (literal or spec-id), spec constants,
push-constant block size, and descriptor bindings with inferred Vulkan descriptor type.
Emits JSON. Pure binary parse, no external deps."""
import sys, struct, json

def words(b):
    assert b[:4] == b'\x03\x02\x23\x07', "not SPIR-V"
    return struct.unpack('<%dI' % (len(b)//4), b)

def reflect(path):
    w = words(open(path,'rb').read())
    i = 5; ops = []
    while i < len(w):
        wc = w[i] >> 16; op = w[i] & 0xffff
        ops.append((op, w[i+1:i+wc])); i += wc
    names={}; decor={}; member_decor={}; types={}; consts={}; spec={}; entry=None; exec_modes=[]
    ptr_sc={}  # id -> storage class for OpTypePointer
    var_type={} # id -> (type_id, storage_class) for OpVariable
    for op, a in ops:
        if op==5: names[a[0]]=bytes(struct.pack('<%dI'%(len(a)-1), *a[1:])).split(b'\0')[0].decode('latin1')
        elif op==71: decor.setdefault(a[0],{})[a[1]]=a[2:]          # OpDecorate
        elif op==72: member_decor.setdefault((a[0],a[1]),{})[a[2]]=a[3:]  # OpMemberDecorate
        elif op==15: entry=(a[0], bytes(struct.pack('<%dI'%(len(a)-2),*a[2:])).split(b'\0')[0].decode('latin1'), a[0])  # OpEntryPoint (model,name)
        elif op==16: exec_modes.append(a)                          # OpExecutionMode
        elif op==19: types[a[0]]=('void',)
        elif op==21: types[a[0]]=('int',a[1],a[2])                 # width,signed
        elif op==22: types[a[0]]=('float',a[1])
        elif op==32: ptr_sc[a[0]]=a[1]; types[a[0]]=('ptr',a[1],a[2])  # OpTypePointer sc,type
        elif op==25: types[a[0]]=('image',)+tuple(a[1:])           # OpTypeImage
        elif op==27: types[a[0]]=('sampled_image',a[1])
        elif op==28: types[a[0]]=('array',a[1],a[2])
        elif op==29: types[a[0]]=('runtime_array',a[1])
        elif op==30: types[a[0]]=('struct',)+tuple(a[1:])          # OpTypeStruct members...
        elif op==33: types[a[0]]=('fn',)
        elif op==43: consts[a[1]]=a[2] if len(a)>2 else 0          # OpConstant (type,id,val)-> a=[type,id,val]
        elif op==50: spec[a[1]]={'type':a[0],'default':a[2] if len(a)>2 else 0}  # OpSpecConstant
        elif op==48 or op==49: spec[a[1]]={'type':a[0],'default':(op==49)}       # OpSpecConstantTrue/False
        elif op==59: var_type[a[1]]=(a[0],a[2])                    # OpVariable result_type(ptr), storage_class
    # local size: ExecutionMode LocalSize(17) literals, or LocalSizeId(38) via consts
    lsize=None
    for a in exec_modes:
        if len(a)>=5 and a[1]==17: lsize=[a[2],a[3],a[4]]
        if len(a)>=5 and a[1]==38: lsize=['id:%d'%a[2],'id:%d'%a[3],'id:%d'%a[4]]
    # spec constants used for workgroup size via BuiltIn WorkgroupSize decorate
    specs=[]
    for sid,info in spec.items():
        nm=names.get(sid,''); sc_id=decor.get(sid,{}).get(1,[None])[0]  # Decoration SpecId=1
        specs.append({'id':sc_id,'name':nm,'spirv_id':sid})
    # descriptors
    binds=[]
    for vid,(tid,sc) in var_type.items():
        d=decor.get(vid,{})
        if 34 in d or 33 in d:  # DescriptorSet(34)/Binding(33)
            setn=d.get(34,[0])[0]; bind=d.get(33,[None])[0]
            # resolve pointee type
            pt=types.get(tid)
            pointee=types.get(pt[2]) if pt and pt[0]=='ptr' else None
            kind='unknown'; readonly=None
            if sc==0: kind='uniform_constant'   # images/samplers
            if pointee:
                base=pointee
                if base[0]=='struct': kind='buffer'
                elif base[0] in ('image',):
                    kind='storage_image' if base[7]==2 else 'sampled_image'  # Sampled: 1=sampled,2=storage
                elif base[0]=='sampled_image': kind='sampled_image'
            if sc==12: kind='storage_buffer'   # StorageBuffer class
            if sc==2: kind='uniform_buffer'    # Uniform
            if sc==9: kind='push_constant'
            binds.append({'set':setn,'binding':bind,'name':names.get(vid,''),'storage_class':sc,'kind':kind})
    # push constant block size
    pc_size=None
    for vid,(tid,sc) in var_type.items():
        if sc==9:
            pt=types.get(tid); st=types.get(pt[2]) if pt and pt[0]=='ptr' else None
            if st and st[0]=='struct': pc_size=len(st)-1  # member count (bytes computed later if needed)
    return {'path':path,'entry':entry[1] if entry else None,'local_size':lsize,
            'spec_constants':sorted(specs,key=lambda x:(x['id'] if x['id'] is not None else 999)),
            'push_constant_members':pc_size,
            'bindings':sorted([b for b in binds if b['binding'] is not None],key=lambda x:(x['set'],x['binding']))}

if __name__=='__main__':
    for p in sys.argv[1:]:
        print(json.dumps(reflect(p),indent=1))
