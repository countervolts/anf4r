#include <stdlib.h>
// Interpose bionic __system_property_get to log queried keys and return spoofed values (Snapdragon 8 Elite Gen 6).
#include <string.h>
#include <stdio.h>
#include "log.h"
int __system_property_get(const char* name, char* value){
    const char* v = "";
    if(!strcmp(name,"ro.soc.model")) v="SM8850";
    else if(!strcmp(name,"ro.soc.manufacturer")) v="QTI";
    else if(!strcmp(name,"ro.arch")) v="arm64";
    else if(!strcmp(name,"ro.hardware")) v="qcom";
    else if(!strcmp(name,"ro.board.platform")) v="kaanapali";
    else if(!strcmp(name,"ro.product.board")) v="kaanapali";
    else if(strstr(name,"device_id_override")) v="";
    LOG("__system_property_get(%s) -> '%s'", name, v);
    if(value){ strncpy(value, v, 91); value[91]=0; }
    return (int)strlen(v);
}

#include <unistd.h>
#include <pthread.h>
// Keep the process alive when a worker thread calls abort() (via std::terminate): park that thread so the
// main thread can continue far enough to emit the kernel build defines / args we want to capture.
void abort(void){ LOG("[abort intercepted — parking thread]"); for(;;) pause(); }
