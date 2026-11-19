#pragma once
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
static FILE* g_log;
static void LOGINIT(){ if(!g_log){ const char* p=getenv("ANF_TRACE_LOG"); g_log=p?fopen(p,"w"):stderr; if(!g_log)g_log=stderr; } }
static void LOG(const char* fmt,...){ LOGINIT(); va_list a; va_start(a,fmt); vfprintf(g_log,fmt,a); va_end(a); fputc('\n',g_log); fflush(g_log); }
