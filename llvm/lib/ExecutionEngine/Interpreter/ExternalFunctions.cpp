//===-- ExternalFunctions.cpp - Implement External Functions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains both code to deal with invoking "external" functions, but
//  also contains code that implements "exported" external functions.
//
//  There are currently two mechanisms for handling external functions in the
//  Interpreter.  The first is to implement lle_* wrapper functions that are
//  specific to well-known library functions which manually translate the
//  arguments from GenericValues and make the call.  If such a wrapper does
//  not exist, and libffi is available, then the Interpreter will attempt to
//  invoke the function using libffi, after finding its address.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/config.h" // Detect libffi
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_FFI_CALL
#ifdef HAVE_FFI_H
#include <ffi.h>
#define USE_LIBFFI
#elif HAVE_FFI_FFI_H
#include <ffi/ffi.h>
#define USE_LIBFFI
#endif
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
#include <CoreFoundation/CoreFoundation.h>
#include <objc/objc.h>
#include <objc/message.h>
#include "ios_error.h"
#include <fcntl.h>
#include <Python.h>
#define Py_BUILD_CORE 1
#include "pycore_tuple.h"
#include "pycore_abstract.h"
#include "pycore_gc.h"
#include "pycore_initconfig.h"
#include "pycore_object.h"
#endif
#endif

extern "C" {
    void llvm_remove_traverse_from_cython_type(PyObject *type);
}

/* New getargs implementation */

#include <ctype.h>
#include <float.h>

size_t strlen(void *str);


#ifdef __cplusplus
extern "C" {
#endif
int PyArg_Parse(PyObject *, const char *, ...);
int PyArg_ParseTuple(PyObject *, const char *, ...);
int PyArg_VaParse(PyObject *, const char *, va_list);

int PyArg_ParseTupleAndKeywords(PyObject *, PyObject *,
                                const char *, char **, ...);
int PyArg_VaParseTupleAndKeywords(PyObject *, PyObject *,
                                const char *, char **, va_list);

int _PyArg_ParseTupleAndKeywordsFast(PyObject *, PyObject *,
                                            struct _PyArg_Parser *, ...);
int _PyArg_VaParseTupleAndKeywordsFast(PyObject *, PyObject *,
                                            struct _PyArg_Parser *, va_list);

#ifdef HAVE_DECLSPEC_DLL
/* Export functions */
PyAPI_FUNC(int) _PyArg_Parse_SizeT(PyObject *, const char *, ...);
PyAPI_FUNC(int) _PyArg_ParseStack_SizeT(PyObject *const *args, Py_ssize_t nargs,
                                        const char *format, ...);
PyAPI_FUNC(int) _PyArg_ParseStackAndKeywords_SizeT(PyObject *const *args, Py_ssize_t nargs,
                                        PyObject *kwnames,
                                        struct _PyArg_Parser *parser, ...);
PyAPI_FUNC(int) _PyArg_ParseTuple_SizeT(PyObject *, const char *, ...);
PyAPI_FUNC(int) _PyArg_ParseTupleAndKeywords_SizeT(PyObject *, PyObject *,
                                                  const char *, char **, ...);
PyAPI_FUNC(PyObject *) _Py_BuildValue_SizeT(const char *, ...);
PyAPI_FUNC(int) _PyArg_VaParse_SizeT(PyObject *, const char *, va_list);
PyAPI_FUNC(int) _PyArg_VaParseTupleAndKeywords_SizeT(PyObject *, PyObject *,
                                              const char *, char **, va_list);

PyAPI_FUNC(int) _PyArg_ParseTupleAndKeywordsFast_SizeT(PyObject *, PyObject *,
                                            struct _PyArg_Parser *, ...);
PyAPI_FUNC(int) _PyArg_VaParseTupleAndKeywordsFast_SizeT(PyObject *, PyObject *,
                                            struct _PyArg_Parser *, va_list);
#endif

#define FLAG_COMPAT 1
#define FLAG_SIZE_T 2

typedef int (*destr_t)(PyObject *, void *);


/* Keep track of "objects" that have been allocated or initialized and
   which will need to be deallocated or cleaned up somehow if overall
   parsing fails.
*/
typedef struct {
  void *item;
  destr_t destructor;
} freelistentry_t;

typedef struct {
  freelistentry_t *entries;
  int first_available;
  int entries_malloced;
} freelist_t;

#define STATIC_FREELIST_ENTRIES 8

/* Forward */
static int vgetargs1_impl(PyObject *args, PyObject *const *stack, Py_ssize_t nargs,
                          const char *format, va_list *p_va, int flags);
static int vgetargs1(PyObject *, const char *, va_list *, int);
static void seterror(Py_ssize_t, const char *, int *, const char *, const char *);
static const char *convertitem(PyObject *, const char **, va_list *, int, int *,
                               char *, size_t, freelist_t *);
static const char *converttuple(PyObject *, const char **, va_list *, int,
                                int *, char *, size_t, int, freelist_t *);
static const char *convertsimple(PyObject *, const char **, va_list *, int,
                                 char *, size_t, freelist_t *);
static Py_ssize_t convertbuffer(PyObject *, const void **p, const char **);
static int getbuffer(PyObject *, Py_buffer *, const char**);

static int vgetargskeywords(PyObject *, PyObject *,
                            const char *, char **, va_list *, int);
static int vgetargskeywordsfast(PyObject *, PyObject *,
                            struct _PyArg_Parser *, va_list *, int);
static int vgetargskeywordsfast_impl(PyObject *const *args, Py_ssize_t nargs,
                          PyObject *keywords, PyObject *kwnames,
                          struct _PyArg_Parser *parser,
                          va_list *p_va, int flags);
static const char *skipitem(const char **, va_list *, int);

int
PyArg_Parse(PyObject *args, const char *format, ...)
{
    int retval;
    va_list va;

    va_start(va, format);
    retval = vgetargs1(args, format, &va, FLAG_COMPAT);
    va_end(va);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_Parse_SizeT(PyObject *args, const char *format, ...)
{
    int retval;
    va_list va;

    va_start(va, format);
    retval = vgetargs1(args, format, &va, FLAG_COMPAT|FLAG_SIZE_T);
    va_end(va);
    return retval;
}


int
PyArg_ParseTuple(PyObject *args, const char *format, ...)
{
    int retval;
    va_list va;

    va_start(va, format);
    retval = vgetargs1(args, format, &va, 0);
    va_end(va);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_ParseTuple_SizeT(PyObject *args, const char *format, ...)
{
    int retval;
    va_list va;

    va_start(va, format);
    retval = vgetargs1(args, format, &va, FLAG_SIZE_T);
    va_end(va);
    return retval;
}


int
_PyArg_ParseStack(PyObject *const *args, Py_ssize_t nargs, const char *format, ...)
{
    int retval;
    va_list va;

    va_start(va, format);
    retval = vgetargs1_impl(NULL, args, nargs, format, &va, 0);
    va_end(va);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_ParseStack_SizeT(PyObject *const *args, Py_ssize_t nargs, const char *format, ...)
{
    int retval;
    va_list va;

    va_start(va, format);
    retval = vgetargs1_impl(NULL, args, nargs, format, &va, FLAG_SIZE_T);
    va_end(va);
    return retval;
}


int
PyArg_VaParse(PyObject *args, const char *format, va_list va)
{
    va_list lva;
    int retval;

    va_copy(lva, va);

    retval = vgetargs1(args, format, &lva, 0);
    va_end(lva);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_VaParse_SizeT(PyObject *args, const char *format, va_list va)
{
    va_list lva;
    int retval;

    va_copy(lva, va);

    retval = vgetargs1(args, format, &lva, FLAG_SIZE_T);
    va_end(lva);
    return retval;
}


/* Handle cleanup of allocated memory in case of exception */

static int
cleanup_ptr(PyObject *self, void *ptr)
{
    if (ptr) {
        PyMem_Free(ptr);
    }
    return 0;
}

static int
cleanup_buffer(PyObject *self, void *ptr)
{
    Py_buffer *buf = (Py_buffer *)ptr;
    if (buf) {
        PyBuffer_Release(buf);
    }
    return 0;
}

static int
addcleanup(void *ptr, freelist_t *freelist, destr_t destructor)
{
    int index;

    index = freelist->first_available;
    freelist->first_available += 1;

    freelist->entries[index].item = ptr;
    freelist->entries[index].destructor = destructor;

    return 0;
}

static int
cleanreturn(int retval, freelist_t *freelist)
{
    int index;

    if (retval == 0) {
      /* A failure occurred, therefore execute all of the cleanup
         functions.
      */
      for (index = 0; index < freelist->first_available; ++index) {
          freelist->entries[index].destructor(NULL,
                                              freelist->entries[index].item);
      }
    }
    if (freelist->entries_malloced)
        PyMem_Free(freelist->entries);
    return retval;
}


static int
vgetargs1_impl(PyObject *compat_args, PyObject *const *stack, Py_ssize_t nargs, const char *format,
               va_list *p_va, int flags)
{
    char msgbuf[256];
    int levels[32];
    const char *fname = NULL;
    const char *message = NULL;
    int min = -1;
    int max = 0;
    int level = 0;
    int endfmt = 0;
    const char *formatsave = format;
    Py_ssize_t i;
    const char *msg;
    int compat = flags & FLAG_COMPAT;
    freelistentry_t static_entries[STATIC_FREELIST_ENTRIES];
    freelist_t freelist;

    assert(nargs == 0 || stack != NULL);

    freelist.entries = static_entries;
    freelist.first_available = 0;
    freelist.entries_malloced = 0;

    flags = flags & ~FLAG_COMPAT;

    while (endfmt == 0) {
        int c = *format++;
        switch (c) {
        case '(':
            if (level == 0)
                max++;
            level++;
            if (level >= 30)
                Py_FatalError("too many tuple nesting levels "
                              "in argument format string");
            break;
        case ')':
            if (level == 0)
                Py_FatalError("excess ')' in getargs format");
            else
                level--;
            break;
        case '\0':
            endfmt = 1;
            break;
        case ':':
            fname = format;
            endfmt = 1;
            break;
        case ';':
            message = format;
            endfmt = 1;
            break;
        case '|':
            if (level == 0)
                min = max;
            break;
        default:
            if (level == 0) {
                if (Py_ISALPHA(c))
                    if (c != 'e') /* skip encoded */
                        max++;
            }
            break;
        }
    }

    if (level != 0)
        Py_FatalError(/* '(' */ "missing ')' in getargs format");

    if (min < 0)
        min = max;

    format = formatsave;

    if (max > STATIC_FREELIST_ENTRIES) {
        freelist.entries = PyMem_NEW(freelistentry_t, max);
        if (freelist.entries == NULL) {
            PyErr_NoMemory();
            return 0;
        }
        freelist.entries_malloced = 1;
    }

    if (compat) {
        if (max == 0) {
            if (compat_args == NULL)
                return 1;
            PyErr_Format(PyExc_TypeError,
                         "%.200s%s takes no arguments",
                         fname==NULL ? "function" : fname,
                         fname==NULL ? "" : "()");
            return cleanreturn(0, &freelist);
        }
        else if (min == 1 && max == 1) {
            if (compat_args == NULL) {
                PyErr_Format(PyExc_TypeError,
                             "%.200s%s takes at least one argument",
                             fname==NULL ? "function" : fname,
                             fname==NULL ? "" : "()");
                return cleanreturn(0, &freelist);
            }
            msg = convertitem(compat_args, &format, p_va, flags, levels,
                              msgbuf, sizeof(msgbuf), &freelist);
            if (msg == NULL)
                return cleanreturn(1, &freelist);
            seterror(levels[0], msg, levels+1, fname, message);
            return cleanreturn(0, &freelist);
        }
        else {
            PyErr_SetString(PyExc_SystemError,
                "old style getargs format uses new features");
            return cleanreturn(0, &freelist);
        }
    }

    if (nargs < min || max < nargs) {
        if (message == NULL)
            PyErr_Format(PyExc_TypeError,
                         "%.150s%s takes %s %d argument%s (%zd given)",
                         fname==NULL ? "function" : fname,
                         fname==NULL ? "" : "()",
                         min==max ? "exactly"
                         : nargs < min ? "at least" : "at most",
                         nargs < min ? min : max,
                         (nargs < min ? min : max) == 1 ? "" : "s",
                         nargs);
        else
            PyErr_SetString(PyExc_TypeError, message);
        return cleanreturn(0, &freelist);
    }

    for (i = 0; i < nargs; i++) {
        if (*format == '|')
            format++;
        msg = convertitem(stack[i], &format, p_va,
                          flags, levels, msgbuf,
                          sizeof(msgbuf), &freelist);
        if (msg) {
            seterror(i+1, msg, levels, fname, message);
            return cleanreturn(0, &freelist);
        }
    }

    if (*format != '\0' && !Py_ISALPHA(*format) &&
        *format != '(' &&
        *format != '|' && *format != ':' && *format != ';') {
        PyErr_Format(PyExc_SystemError,
                     "bad format string: %.200s", formatsave);
        return cleanreturn(0, &freelist);
    }

    return cleanreturn(1, &freelist);
}

static int
vgetargs1(PyObject *args, const char *format, va_list *p_va, int flags)
{
    PyObject **stack;
    Py_ssize_t nargs;

    if (!(flags & FLAG_COMPAT)) {
        assert(args != NULL);

        if (!PyTuple_Check(args)) {
            PyErr_SetString(PyExc_SystemError,
                "new style getargs format but argument is not a tuple");
            return 0;
        }

        stack = _PyTuple_ITEMS(args);
        nargs = PyTuple_GET_SIZE(args);
    }
    else {
        stack = NULL;
        nargs = 0;
    }

    return vgetargs1_impl(args, stack, nargs, format, p_va, flags);
}


static void
seterror(Py_ssize_t iarg, const char *msg, int *levels, const char *fname,
         const char *message)
{
    char buf[512];
    int i;
    char *p = buf;

    if (PyErr_Occurred())
        return;
    else if (message == NULL) {
        if (fname != NULL) {
            PyOS_snprintf(p, sizeof(buf), "%.200s() ", fname);
            p += strlen(p);
        }
        if (iarg != 0) {
            PyOS_snprintf(p, sizeof(buf) - (p - buf),
                          "argument %zd", iarg);
            i = 0;
            p += strlen(p);
            while (i < 32 && levels[i] > 0 && (int)(p-buf) < 220) {
                PyOS_snprintf(p, sizeof(buf) - (p - buf),
                              ", item %d", levels[i]-1);
                p += strlen(p);
                i++;
            }
        }
        else {
            PyOS_snprintf(p, sizeof(buf) - (p - buf), "argument");
            p += strlen(p);
        }
        PyOS_snprintf(p, sizeof(buf) - (p - buf), " %.256s", msg);
        message = buf;
    }
    if (msg[0] == '(') {
        PyErr_SetString(PyExc_SystemError, message);
    }
    else {
        PyErr_SetString(PyExc_TypeError, message);
    }
}


/* Convert a tuple argument.
   On entry, *p_format points to the character _after_ the opening '('.
   On successful exit, *p_format points to the closing ')'.
   If successful:
      *p_format and *p_va are updated,
      *levels and *msgbuf are untouched,
      and NULL is returned.
   If the argument is invalid:
      *p_format is unchanged,
      *p_va is undefined,
      *levels is a 0-terminated list of item numbers,
      *msgbuf contains an error message, whose format is:
     "must be <typename1>, not <typename2>", where:
        <typename1> is the name of the expected type, and
        <typename2> is the name of the actual type,
      and msgbuf is returned.
*/

static const char *
converttuple(PyObject *arg, const char **p_format, va_list *p_va, int flags,
             int *levels, char *msgbuf, size_t bufsize, int toplevel,
             freelist_t *freelist)
{
    int level = 0;
    int n = 0;
    const char *format = *p_format;
    int i;
    Py_ssize_t len;

    for (;;) {
        int c = *format++;
        if (c == '(') {
            if (level == 0)
                n++;
            level++;
        }
        else if (c == ')') {
            if (level == 0)
                break;
            level--;
        }
        else if (c == ':' || c == ';' || c == '\0')
            break;
        else if (level == 0 && Py_ISALPHA(c))
            n++;
    }

    if (!PySequence_Check(arg) || PyBytes_Check(arg)) {
        levels[0] = 0;
        PyOS_snprintf(msgbuf, bufsize,
                      toplevel ? "expected %d arguments, not %.50s" :
                      "must be %d-item sequence, not %.50s",
                  n,
                  arg == Py_None ? "None" : Py_TYPE(arg)->tp_name);
        return msgbuf;
    }

    len = PySequence_Size(arg);
    if (len != n) {
        levels[0] = 0;
        if (toplevel) {
            PyOS_snprintf(msgbuf, bufsize,
                          "expected %d argument%s, not %zd",
                          n,
                          n == 1 ? "" : "s",
                          len);
        }
        else {
            PyOS_snprintf(msgbuf, bufsize,
                          "must be sequence of length %d, not %zd",
                          n, len);
        }
        return msgbuf;
    }

    format = *p_format;
    for (i = 0; i < n; i++) {
        const char *msg;
        PyObject *item;
        item = PySequence_GetItem(arg, i);
        if (item == NULL) {
            PyErr_Clear();
            levels[0] = i+1;
            levels[1] = 0;
            strncpy(msgbuf, "is not retrievable", bufsize);
            return msgbuf;
        }
        msg = convertitem(item, &format, p_va, flags, levels+1,
                          msgbuf, bufsize, freelist);
        /* PySequence_GetItem calls tp->sq_item, which INCREFs */
        Py_XDECREF(item);
        if (msg != NULL) {
            levels[0] = i+1;
            return msg;
        }
    }

    *p_format = format;
    return NULL;
}


/* Convert a single item. */

static const char *
convertitem(PyObject *arg, const char **p_format, va_list *p_va, int flags,
            int *levels, char *msgbuf, size_t bufsize, freelist_t *freelist)
{
    const char *msg;
    const char *format = *p_format;

    if (*format == '(' /* ')' */) {
        format++;
        msg = converttuple(arg, &format, p_va, flags, levels, msgbuf,
                           bufsize, 0, freelist);
        if (msg == NULL)
            format++;
    }
    else {
        msg = convertsimple(arg, &format, p_va, flags,
                            msgbuf, bufsize, freelist);
        if (msg != NULL)
            levels[0] = 0;
    }
    if (msg == NULL)
        *p_format = format;
    return msg;
}



/* Format an error message generated by convertsimple().
   displayname must be UTF-8 encoded.
*/

void
_PyArg_BadArgument(const char *fname, const char *displayname,
                   const char *expected, PyObject *arg)
{
    PyErr_Format(PyExc_TypeError,
                 "%.200s() %.200s must be %.50s, not %.50s",
                 fname, displayname, expected,
                 arg == Py_None ? "None" : Py_TYPE(arg)->tp_name);
}

static const char *
converterr(const char *expected, PyObject *arg, char *msgbuf, size_t bufsize)
{
    assert(expected != NULL);
    assert(arg != NULL);
    if (expected[0] == '(') {
        PyOS_snprintf(msgbuf, bufsize,
                      "%.100s", expected);
    }
    else {
        PyOS_snprintf(msgbuf, bufsize,
                      "must be %.50s, not %.50s", expected,
                      arg == Py_None ? "None" : Py_TYPE(arg)->tp_name);
    }
    return msgbuf;
}

#define CONV_UNICODE "(unicode conversion error)"

/* Convert a non-tuple argument.  Return NULL if conversion went OK,
   or a string with a message describing the failure.  The message is
   formatted as "must be <desired type>, not <actual type>".
   When failing, an exception may or may not have been raised.
   Don't call if a tuple is expected.

   When you add new format codes, please don't forget poor skipitem() below.
*/

static const char *
convertsimple(PyObject *arg, const char **p_format, va_list *p_va, int flags,
              char *msgbuf, size_t bufsize, freelist_t *freelist)
{
#define RETURN_ERR_OCCURRED return msgbuf
    /* For # codes */
#define REQUIRE_PY_SSIZE_T_CLEAN \
    if (!(flags & FLAG_SIZE_T)) { \
        PyErr_SetString(PyExc_SystemError, \
                        "PY_SSIZE_T_CLEAN macro must be defined for '#' formats"); \
        RETURN_ERR_OCCURRED; \
    }

    const char *format = *p_format;
    char c = *format++;
    const char *sarg;

    switch (c) {

    case 'b': { /* unsigned byte -- very short int */
        char *p = va_arg(*p_va, char *);
        long ival = PyLong_AsLong(arg);
        if (ival == -1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else if (ival < 0) {
            PyErr_SetString(PyExc_OverflowError,
                            "unsigned byte integer is less than minimum");
            RETURN_ERR_OCCURRED;
        }
        else if (ival > UCHAR_MAX) {
            PyErr_SetString(PyExc_OverflowError,
                            "unsigned byte integer is greater than maximum");
            RETURN_ERR_OCCURRED;
        }
        else
            *p = (unsigned char) ival;
        break;
    }

    case 'B': {/* byte sized bitfield - both signed and unsigned
                  values allowed */
        char *p = va_arg(*p_va, char *);
        unsigned long ival = PyLong_AsUnsignedLongMask(arg);
        if (ival == (unsigned long)-1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else
            *p = (unsigned char) ival;
        break;
    }

    case 'h': {/* signed short int */
        short *p = va_arg(*p_va, short *);
        long ival = PyLong_AsLong(arg);
        if (ival == -1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else if (ival < SHRT_MIN) {
            PyErr_SetString(PyExc_OverflowError,
                            "signed short integer is less than minimum");
            RETURN_ERR_OCCURRED;
        }
        else if (ival > SHRT_MAX) {
            PyErr_SetString(PyExc_OverflowError,
                            "signed short integer is greater than maximum");
            RETURN_ERR_OCCURRED;
        }
        else
            *p = (short) ival;
        break;
    }

    case 'H': { /* short int sized bitfield, both signed and
                   unsigned allowed */
        unsigned short *p = va_arg(*p_va, unsigned short *);
        unsigned long ival = PyLong_AsUnsignedLongMask(arg);
        if (ival == (unsigned long)-1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else
            *p = (unsigned short) ival;
        break;
    }

    case 'i': {/* signed int */
        int *p = va_arg(*p_va, int *);
        long ival = PyLong_AsLong(arg);
        if (ival == -1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else if (ival > INT_MAX) {
            PyErr_SetString(PyExc_OverflowError,
                            "signed integer is greater than maximum");
            RETURN_ERR_OCCURRED;
        }
        else if (ival < INT_MIN) {
            PyErr_SetString(PyExc_OverflowError,
                            "signed integer is less than minimum");
            RETURN_ERR_OCCURRED;
        }
        else
            *p = ival;
        break;
    }

    case 'I': { /* int sized bitfield, both signed and
                   unsigned allowed */
        unsigned int *p = va_arg(*p_va, unsigned int *);
        unsigned long ival = PyLong_AsUnsignedLongMask(arg);
        if (ival == (unsigned long)-1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else
            *p = (unsigned int) ival;
        break;
    }

    case 'n': /* Py_ssize_t */
    {
        PyObject *iobj;
        Py_ssize_t *p = va_arg(*p_va, Py_ssize_t *);
        Py_ssize_t ival = -1;
        iobj = _PyNumber_Index(arg);
        if (iobj != NULL) {
            ival = PyLong_AsSsize_t(iobj);
            Py_DECREF(iobj);
        }
        if (ival == -1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        *p = ival;
        break;
    }
    case 'l': {/* long int */
        long *p = va_arg(*p_va, long *);
        long ival = PyLong_AsLong(arg);
        if (ival == -1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else
            *p = ival;
        break;
    }

    case 'k': { /* long sized bitfield */
        unsigned long *p = va_arg(*p_va, unsigned long *);
        unsigned long ival;
        if (PyLong_Check(arg))
            ival = PyLong_AsUnsignedLongMask(arg);
        else
            return converterr("int", arg, msgbuf, bufsize);
        *p = ival;
        break;
    }

    case 'L': {/* long long */
        long long *p = va_arg( *p_va, long long * );
        long long ival = PyLong_AsLongLong(arg);
        if (ival == (long long)-1 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else
            *p = ival;
        break;
    }

    case 'K': { /* long long sized bitfield */
        unsigned long long *p = va_arg(*p_va, unsigned long long *);
        unsigned long long ival;
        if (PyLong_Check(arg))
            ival = PyLong_AsUnsignedLongLongMask(arg);
        else
            return converterr("int", arg, msgbuf, bufsize);
        *p = ival;
        break;
    }

    case 'f': {/* float */
        float *p = va_arg(*p_va, float *);
        double dval = PyFloat_AsDouble(arg);
        if (dval == -1.0 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else
            *p = (float) dval;
        break;
    }

    case 'd': {/* double */
        double *p = va_arg(*p_va, double *);
        double dval = PyFloat_AsDouble(arg);
        if (dval == -1.0 && PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else
            *p = dval;
        break;
    }

    case 'D': {/* complex double */
        Py_complex *p = va_arg(*p_va, Py_complex *);
        Py_complex cval;
        cval = PyComplex_AsCComplex(arg);
        if (PyErr_Occurred())
            RETURN_ERR_OCCURRED;
        else
            *p = cval;
        break;
    }

    case 'c': {/* char */
        char *p = va_arg(*p_va, char *);
        if (PyBytes_Check(arg) && PyBytes_Size(arg) == 1)
            *p = PyBytes_AS_STRING(arg)[0];
        else if (PyByteArray_Check(arg) && PyByteArray_Size(arg) == 1)
            *p = PyByteArray_AS_STRING(arg)[0];
        else
            return converterr("a byte string of length 1", arg, msgbuf, bufsize);
        break;
    }

    case 'C': {/* unicode char */
        int *p = va_arg(*p_va, int *);
        int kind;
        const void *data;

        if (!PyUnicode_Check(arg))
            return converterr("a unicode character", arg, msgbuf, bufsize);

        if (PyUnicode_READY(arg))
            RETURN_ERR_OCCURRED;

        if (PyUnicode_GET_LENGTH(arg) != 1)
            return converterr("a unicode character", arg, msgbuf, bufsize);

        kind = PyUnicode_KIND(arg);
        data = PyUnicode_DATA(arg);
        *p = PyUnicode_READ(kind, data, 0);
        break;
    }

    case 'p': {/* boolean *p*redicate */
        int *p = va_arg(*p_va, int *);
        int val = PyObject_IsTrue(arg);
        if (val > 0)
            *p = 1;
        else if (val == 0)
            *p = 0;
        else
            RETURN_ERR_OCCURRED;
        break;
    }

    /* XXX WAAAAH!  's', 'y', 'z', 'u', 'Z', 'e', 'w' codes all
       need to be cleaned up! */

    case 'y': {/* any bytes-like object */
        void **p = (void **)va_arg(*p_va, char **);
        const char *buf;
        Py_ssize_t count;
        if (*format == '*') {
            if (getbuffer(arg, (Py_buffer*)p, &buf) < 0)
                return converterr(buf, arg, msgbuf, bufsize);
            format++;
            if (addcleanup(p, freelist, cleanup_buffer)) {
                return converterr(
                    "(cleanup problem)",
                    arg, msgbuf, bufsize);
            }
            break;
        }
        count = convertbuffer(arg, (const void **)p, &buf);
        if (count < 0)
            return converterr(buf, arg, msgbuf, bufsize);
        if (*format == '#') {
            REQUIRE_PY_SSIZE_T_CLEAN;
            Py_ssize_t *psize = va_arg(*p_va, Py_ssize_t*);
            *psize = count;
            format++;
        } else {
            if (strlen(*p) != (size_t)count) {
                PyErr_SetString(PyExc_ValueError, "embedded null byte");
                RETURN_ERR_OCCURRED;
            }
        }
        break;
    }

    case 's': /* text string or bytes-like object */
    case 'z': /* text string, bytes-like object or None */
    {
        if (*format == '*') {
            /* "s*" or "z*" */
            Py_buffer *p = (Py_buffer *)va_arg(*p_va, Py_buffer *);

            if (c == 'z' && arg == Py_None)
                PyBuffer_FillInfo(p, NULL, NULL, 0, 1, 0);
            else if (PyUnicode_Check(arg)) {
                Py_ssize_t len;
                sarg = PyUnicode_AsUTF8AndSize(arg, &len);
                if (sarg == NULL)
                    return converterr(CONV_UNICODE,
                                      arg, msgbuf, bufsize);
                PyBuffer_FillInfo(p, arg, (void *)sarg, len, 1, 0);
            }
            else { /* any bytes-like object */
                const char *buf;
                if (getbuffer(arg, p, &buf) < 0)
                    return converterr(buf, arg, msgbuf, bufsize);
            }
            if (addcleanup(p, freelist, cleanup_buffer)) {
                return converterr(
                    "(cleanup problem)",
                    arg, msgbuf, bufsize);
            }
            format++;
        } else if (*format == '#') { /* a string or read-only bytes-like object */
            /* "s#" or "z#" */
            const void **p = (const void **)va_arg(*p_va, const char **);
            REQUIRE_PY_SSIZE_T_CLEAN;
            Py_ssize_t *psize = va_arg(*p_va, Py_ssize_t*);

            if (c == 'z' && arg == Py_None) {
                *p = NULL;
                *psize = 0;
            }
            else if (PyUnicode_Check(arg)) {
                Py_ssize_t len;
                sarg = PyUnicode_AsUTF8AndSize(arg, &len);
                if (sarg == NULL)
                    return converterr(CONV_UNICODE,
                                      arg, msgbuf, bufsize);
                *p = sarg;
                *psize = len;
            }
            else { /* read-only bytes-like object */
                /* XXX Really? */
                const char *buf;
                Py_ssize_t count = convertbuffer(arg, p, &buf);
                if (count < 0)
                    return converterr(buf, arg, msgbuf, bufsize);
                *psize = count;
            }
            format++;
        } else {
            /* "s" or "z" */
            const char **p = va_arg(*p_va, const char **);
            Py_ssize_t len;
            sarg = NULL;

            if (c == 'z' && arg == Py_None)
                *p = NULL;
            else if (PyUnicode_Check(arg)) {
                sarg = PyUnicode_AsUTF8AndSize(arg, &len);
                if (sarg == NULL)
                    return converterr(CONV_UNICODE,
                                      arg, msgbuf, bufsize);
                if (strlen(sarg) != (size_t)len) {
                    PyErr_SetString(PyExc_ValueError, "embedded null character");
                    RETURN_ERR_OCCURRED;
                }
                *p = sarg;
            }
            else
                return converterr(c == 'z' ? "str or None" : "str",
                                  arg, msgbuf, bufsize);
        }
        break;
    }

    case 'u': /* raw unicode buffer (Py_UNICODE *) */
    case 'Z': /* raw unicode buffer or None */
    {
        if (PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
                "getargs: The '%c' format is deprecated. Use 'U' instead.", c)) {
            return NULL;
        }
_Py_COMP_DIAG_PUSH
_Py_COMP_DIAG_IGNORE_DEPR_DECLS
        Py_UNICODE **p = va_arg(*p_va, Py_UNICODE **);

        if (*format == '#') {
            /* "u#" or "Z#" */
            REQUIRE_PY_SSIZE_T_CLEAN;
            Py_ssize_t *psize = va_arg(*p_va, Py_ssize_t*);

            if (c == 'Z' && arg == Py_None) {
                *p = NULL;
                *psize = 0;
            }
            else if (PyUnicode_Check(arg)) {
                Py_ssize_t len;
                *p = PyUnicode_AsUnicodeAndSize(arg, &len);
                if (*p == NULL)
                    RETURN_ERR_OCCURRED;
                *psize = len;
            }
            else
                return converterr(c == 'Z' ? "str or None" : "str",
                                  arg, msgbuf, bufsize);
            format++;
        } else {
            /* "u" or "Z" */
            if (c == 'Z' && arg == Py_None)
                *p = NULL;
            else if (PyUnicode_Check(arg)) {
                Py_ssize_t len;
                *p = PyUnicode_AsUnicodeAndSize(arg, &len);
                if (*p == NULL)
                    RETURN_ERR_OCCURRED;
                if (wcslen(*p) != (size_t)len) {
                    PyErr_SetString(PyExc_ValueError, "embedded null character");
                    RETURN_ERR_OCCURRED;
                }
            } else
                return converterr(c == 'Z' ? "str or None" : "str",
                                  arg, msgbuf, bufsize);
        }
        break;
_Py_COMP_DIAG_POP
    }

    case 'e': {/* encoded string */
        char **buffer;
        const char *encoding;
        PyObject *s;
        int recode_strings;
        Py_ssize_t size;
        const char *ptr;

        /* Get 'e' parameter: the encoding name */
        encoding = (const char *)va_arg(*p_va, const char *);
        if (encoding == NULL)
            encoding = PyUnicode_GetDefaultEncoding();

        /* Get output buffer parameter:
           's' (recode all objects via Unicode) or
           't' (only recode non-string objects)
        */
        if (*format == 's')
            recode_strings = 1;
        else if (*format == 't')
            recode_strings = 0;
        else
            return converterr(
                "(unknown parser marker combination)",
                arg, msgbuf, bufsize);
        buffer = (char **)va_arg(*p_va, char **);
        format++;
        if (buffer == NULL)
            return converterr("(buffer is NULL)",
                              arg, msgbuf, bufsize);

        /* Encode object */
        if (!recode_strings &&
            (PyBytes_Check(arg) || PyByteArray_Check(arg))) {
            s = arg;
            Py_INCREF(s);
            if (PyBytes_Check(arg)) {
                size = PyBytes_GET_SIZE(s);
                ptr = PyBytes_AS_STRING(s);
            }
            else {
                size = PyByteArray_GET_SIZE(s);
                ptr = PyByteArray_AS_STRING(s);
            }
        }
        else if (PyUnicode_Check(arg)) {
            /* Encode object; use default error handling */
            s = PyUnicode_AsEncodedString(arg,
                                          encoding,
                                          NULL);
            if (s == NULL)
                return converterr("(encoding failed)",
                                  arg, msgbuf, bufsize);
            assert(PyBytes_Check(s));
            size = PyBytes_GET_SIZE(s);
            ptr = PyBytes_AS_STRING(s);
            if (ptr == NULL)
                ptr = "";
        }
        else {
            return converterr(
                recode_strings ? "str" : "str, bytes or bytearray",
                arg, msgbuf, bufsize);
        }

        /* Write output; output is guaranteed to be 0-terminated */
        if (*format == '#') {
            /* Using buffer length parameter '#':

               - if *buffer is NULL, a new buffer of the
               needed size is allocated and the data
               copied into it; *buffer is updated to point
               to the new buffer; the caller is
               responsible for PyMem_Free()ing it after
               usage

               - if *buffer is not NULL, the data is
               copied to *buffer; *buffer_len has to be
               set to the size of the buffer on input;
               buffer overflow is signalled with an error;
               buffer has to provide enough room for the
               encoded string plus the trailing 0-byte

               - in both cases, *buffer_len is updated to
               the size of the buffer /excluding/ the
               trailing 0-byte

            */
            REQUIRE_PY_SSIZE_T_CLEAN;
            Py_ssize_t *psize = va_arg(*p_va, Py_ssize_t*);

            format++;
            if (psize == NULL) {
                Py_DECREF(s);
                return converterr(
                    "(buffer_len is NULL)",
                    arg, msgbuf, bufsize);
            }
            if (*buffer == NULL) {
                *buffer = PyMem_NEW(char, size + 1);
                if (*buffer == NULL) {
                    Py_DECREF(s);
                    PyErr_NoMemory();
                    RETURN_ERR_OCCURRED;
                }
                if (addcleanup(*buffer, freelist, cleanup_ptr)) {
                    Py_DECREF(s);
                    return converterr(
                        "(cleanup problem)",
                        arg, msgbuf, bufsize);
                }
            } else {
                if (size + 1 > *psize) {
                    Py_DECREF(s);
                    PyErr_Format(PyExc_ValueError,
                                 "encoded string too long "
                                 "(%zd, maximum length %zd)",
                                 (Py_ssize_t)size, (Py_ssize_t)(*psize - 1));
                    RETURN_ERR_OCCURRED;
                }
            }
            memcpy(*buffer, ptr, size+1);

            *psize = size;
        }
        else {
            /* Using a 0-terminated buffer:

               - the encoded string has to be 0-terminated
               for this variant to work; if it is not, an
               error raised

               - a new buffer of the needed size is
               allocated and the data copied into it;
               *buffer is updated to point to the new
               buffer; the caller is responsible for
               PyMem_Free()ing it after usage

            */
            if ((Py_ssize_t)strlen(ptr) != size) {
                Py_DECREF(s);
                return converterr(
                    "encoded string without null bytes",
                    arg, msgbuf, bufsize);
            }
            *buffer = PyMem_NEW(char, size + 1);
            if (*buffer == NULL) {
                Py_DECREF(s);
                PyErr_NoMemory();
                RETURN_ERR_OCCURRED;
            }
            if (addcleanup(*buffer, freelist, cleanup_ptr)) {
                Py_DECREF(s);
                return converterr("(cleanup problem)",
                                arg, msgbuf, bufsize);
            }
            memcpy(*buffer, ptr, size+1);
        }
        Py_DECREF(s);
        break;
    }

    case 'S': { /* PyBytes object */
        PyObject **p = va_arg(*p_va, PyObject **);
        if (PyBytes_Check(arg))
            *p = arg;
        else
            return converterr("bytes", arg, msgbuf, bufsize);
        break;
    }

    case 'Y': { /* PyByteArray object */
        PyObject **p = va_arg(*p_va, PyObject **);
        if (PyByteArray_Check(arg))
            *p = arg;
        else
            return converterr("bytearray", arg, msgbuf, bufsize);
        break;
    }

    case 'U': { /* PyUnicode object */
        PyObject **p = va_arg(*p_va, PyObject **);
        if (PyUnicode_Check(arg)) {
            if (PyUnicode_READY(arg) == -1)
                RETURN_ERR_OCCURRED;
            *p = arg;
        }
        else
            return converterr("str", arg, msgbuf, bufsize);
        break;
    }

    case 'O': { /* object */
        PyTypeObject *type;
        PyObject **p;
        if (*format == '!') {
            type = va_arg(*p_va, PyTypeObject*);
            p = va_arg(*p_va, PyObject **);
            format++;
            if (PyType_IsSubtype(Py_TYPE(arg), type))
                *p = arg;
            else
                return converterr(type->tp_name, arg, msgbuf, bufsize);

        }
        else if (*format == '&') {
            typedef int (*converter)(PyObject *, void *);
            converter convert = va_arg(*p_va, converter);
            void *addr = va_arg(*p_va, void *);
            int res;
            format++;
            if (! (res = (*convert)(arg, addr)))
                return converterr("(unspecified)",
                                  arg, msgbuf, bufsize);
            if (res == Py_CLEANUP_SUPPORTED &&
                addcleanup(addr, freelist, convert) == -1)
                return converterr("(cleanup problem)",
                                arg, msgbuf, bufsize);
        }
        else {
            p = va_arg(*p_va, PyObject **);
            *p = arg;
        }
        break;
    }


    case 'w': { /* "w*": memory buffer, read-write access */
        void **p = va_arg(*p_va, void **);

        if (*format != '*')
            return converterr(
                "(invalid use of 'w' format character)",
                arg, msgbuf, bufsize);
        format++;

        /* Caller is interested in Py_buffer, and the object
           supports it directly. */
        if (PyObject_GetBuffer(arg, (Py_buffer*)p, PyBUF_WRITABLE) < 0) {
            PyErr_Clear();
            return converterr("read-write bytes-like object",
                              arg, msgbuf, bufsize);
        }
        if (!PyBuffer_IsContiguous((Py_buffer*)p, 'C')) {
            PyBuffer_Release((Py_buffer*)p);
            return converterr("contiguous buffer", arg, msgbuf, bufsize);
        }
        if (addcleanup(p, freelist, cleanup_buffer)) {
            return converterr(
                "(cleanup problem)",
                arg, msgbuf, bufsize);
        }
        break;
    }

    default:
        return converterr("(impossible<bad format char>)", arg, msgbuf, bufsize);

    }

    *p_format = format;
    return NULL;

#undef REQUIRE_PY_SSIZE_T_CLEAN
#undef RETURN_ERR_OCCURRED
}

static Py_ssize_t
convertbuffer(PyObject *arg, const void **p, const char **errmsg)
{
    PyBufferProcs *pb = Py_TYPE(arg)->tp_as_buffer;
    Py_ssize_t count;
    Py_buffer view;

    *errmsg = NULL;
    *p = NULL;
    if (pb != NULL && pb->bf_releasebuffer != NULL) {
        *errmsg = "read-only bytes-like object";
        return -1;
    }

    if (getbuffer(arg, &view, errmsg) < 0)
        return -1;
    count = view.len;
    *p = view.buf;
    PyBuffer_Release(&view);
    return count;
}

static int
getbuffer(PyObject *arg, Py_buffer *view, const char **errmsg)
{
    if (PyObject_GetBuffer(arg, view, PyBUF_SIMPLE) != 0) {
        *errmsg = "bytes-like object";
        return -1;
    }
    if (!PyBuffer_IsContiguous(view, 'C')) {
        PyBuffer_Release(view);
        *errmsg = "contiguous buffer";
        return -1;
    }
    return 0;
}

/* Support for keyword arguments donated by
   Geoff Philbrick <philbric@delphi.hks.com> */

/* Return false (0) for error, else true. */
int
PyArg_ParseTupleAndKeywords(PyObject *args,
                            PyObject *keywords,
                            const char *format,
                            char **kwlist, ...)
{
    int retval;
    va_list va;

    if ((args == NULL || !PyTuple_Check(args)) ||
        (keywords != NULL && !PyDict_Check(keywords)) ||
        format == NULL ||
        kwlist == NULL)
    {
        PyErr_BadInternalCall();
        return 0;
    }

    va_start(va, kwlist);
    retval = vgetargskeywords(args, keywords, format, kwlist, &va, 0);
    va_end(va);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_ParseTupleAndKeywords_SizeT(PyObject *args,
                                  PyObject *keywords,
                                  const char *format,
                                  char **kwlist, ...)
{
    int retval;
    va_list va;

    if ((args == NULL || !PyTuple_Check(args)) ||
        (keywords != NULL && !PyDict_Check(keywords)) ||
        format == NULL ||
        kwlist == NULL)
    {
        PyErr_BadInternalCall();
        return 0;
    }

    va_start(va, kwlist);
    retval = vgetargskeywords(args, keywords, format,
                              kwlist, &va, FLAG_SIZE_T);
    va_end(va);
    return retval;
}


int
PyArg_VaParseTupleAndKeywords(PyObject *args,
                              PyObject *keywords,
                              const char *format,
                              char **kwlist, va_list va)
{
    int retval;
    va_list lva;

    if ((args == NULL || !PyTuple_Check(args)) ||
        (keywords != NULL && !PyDict_Check(keywords)) ||
        format == NULL ||
        kwlist == NULL)
    {
        PyErr_BadInternalCall();
        return 0;
    }

    va_copy(lva, va);

    retval = vgetargskeywords(args, keywords, format, kwlist, &lva, 0);
    va_end(lva);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_VaParseTupleAndKeywords_SizeT(PyObject *args,
                                    PyObject *keywords,
                                    const char *format,
                                    char **kwlist, va_list va)
{
    int retval;
    va_list lva;

    if ((args == NULL || !PyTuple_Check(args)) ||
        (keywords != NULL && !PyDict_Check(keywords)) ||
        format == NULL ||
        kwlist == NULL)
    {
        PyErr_BadInternalCall();
        return 0;
    }

    va_copy(lva, va);

    retval = vgetargskeywords(args, keywords, format,
                              kwlist, &lva, FLAG_SIZE_T);
    va_end(lva);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_ParseTupleAndKeywordsFast(PyObject *args, PyObject *keywords,
                            struct _PyArg_Parser *parser, ...)
{
    int retval;
    va_list va;

    va_start(va, parser);
    retval = vgetargskeywordsfast(args, keywords, parser, &va, 0);
    va_end(va);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_ParseTupleAndKeywordsFast_SizeT(PyObject *args, PyObject *keywords,
                            struct _PyArg_Parser *parser, ...)
{
    int retval;
    va_list va;

    va_start(va, parser);
    retval = vgetargskeywordsfast(args, keywords, parser, &va, FLAG_SIZE_T);
    va_end(va);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_ParseStackAndKeywords(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                  struct _PyArg_Parser *parser, ...)
{
    int retval;
    va_list va;

    va_start(va, parser);
    retval = vgetargskeywordsfast_impl(args, nargs, NULL, kwnames, parser, &va, 0);
    va_end(va);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_ParseStackAndKeywords_SizeT(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        struct _PyArg_Parser *parser, ...)
{
    int retval;
    va_list va;

    va_start(va, parser);
    retval = vgetargskeywordsfast_impl(args, nargs, NULL, kwnames, parser, &va, FLAG_SIZE_T);
    va_end(va);
    return retval;
}


PyAPI_FUNC(int)
_PyArg_VaParseTupleAndKeywordsFast(PyObject *args, PyObject *keywords,
                            struct _PyArg_Parser *parser, va_list va)
{
    int retval;
    va_list lva;

    va_copy(lva, va);

    retval = vgetargskeywordsfast(args, keywords, parser, &lva, 0);
    va_end(lva);
    return retval;
}

PyAPI_FUNC(int)
_PyArg_VaParseTupleAndKeywordsFast_SizeT(PyObject *args, PyObject *keywords,
                            struct _PyArg_Parser *parser, va_list va)
{
    int retval;
    va_list lva;

    va_copy(lva, va);

    retval = vgetargskeywordsfast(args, keywords, parser, &lva, FLAG_SIZE_T);
    va_end(lva);
    return retval;
}

int
PyArg_ValidateKeywordArguments(PyObject *kwargs)
{
    if (!PyDict_Check(kwargs)) {
        PyErr_BadInternalCall();
        return 0;
    }
    if (!_PyDict_HasOnlyStringKeys(kwargs)) {
        PyErr_SetString(PyExc_TypeError,
                        "keywords must be strings");
        return 0;
    }
    return 1;
}

#define IS_END_OF_FORMAT(c) (c == '\0' || c == ';' || c == ':')

static int
vgetargskeywords(PyObject *args, PyObject *kwargs, const char *format,
                 char **kwlist, va_list *p_va, int flags)
{
    char msgbuf[512];
    int levels[32];
    const char *fname, *msg, *custom_msg;
    int min = INT_MAX;
    int max = INT_MAX;
    int i, pos, len;
    int skip = 0;
    Py_ssize_t nargs, nkwargs;
    PyObject *current_arg;
    freelistentry_t static_entries[STATIC_FREELIST_ENTRIES];
    freelist_t freelist;

    freelist.entries = static_entries;
    freelist.first_available = 0;
    freelist.entries_malloced = 0;

    assert(args != NULL && PyTuple_Check(args));
    assert(kwargs == NULL || PyDict_Check(kwargs));
    assert(format != NULL);
    assert(kwlist != NULL);
    assert(p_va != NULL);

    /* grab the function name or custom error msg first (mutually exclusive) */
    fname = strchr(format, ':');
    if (fname) {
        fname++;
        custom_msg = NULL;
    }
    else {
        custom_msg = strchr(format,';');
        if (custom_msg)
            custom_msg++;
    }

    /* scan kwlist and count the number of positional-only parameters */
    for (pos = 0; kwlist[pos] && !*kwlist[pos]; pos++) {
    }
    /* scan kwlist and get greatest possible nbr of args */
    for (len = pos; kwlist[len]; len++) {
        if (!*kwlist[len]) {
            PyErr_SetString(PyExc_SystemError,
                            "Empty keyword parameter name");
            return cleanreturn(0, &freelist);
        }
    }

    if (len > STATIC_FREELIST_ENTRIES) {
        freelist.entries = PyMem_NEW(freelistentry_t, len);
        if (freelist.entries == NULL) {
            PyErr_NoMemory();
            return 0;
        }
        freelist.entries_malloced = 1;
    }

    nargs = PyTuple_GET_SIZE(args);
    nkwargs = (kwargs == NULL) ? 0 : PyDict_GET_SIZE(kwargs);
    if (nargs + nkwargs > len) {
        /* Adding "keyword" (when nargs == 0) prevents producing wrong error
           messages in some special cases (see bpo-31229). */
        PyErr_Format(PyExc_TypeError,
                     "%.200s%s takes at most %d %sargument%s (%zd given)",
                     (fname == NULL) ? "function" : fname,
                     (fname == NULL) ? "" : "()",
                     len,
                     (nargs == 0) ? "keyword " : "",
                     (len == 1) ? "" : "s",
                     nargs + nkwargs);
        return cleanreturn(0, &freelist);
    }

    /* convert tuple args and keyword args in same loop, using kwlist to drive process */
    for (i = 0; i < len; i++) {
        if (*format == '|') {
            if (min != INT_MAX) {
                PyErr_SetString(PyExc_SystemError,
                                "Invalid format string (| specified twice)");
                return cleanreturn(0, &freelist);
            }

            min = i;
            format++;

            if (max != INT_MAX) {
                PyErr_SetString(PyExc_SystemError,
                                "Invalid format string ($ before |)");
                return cleanreturn(0, &freelist);
            }
        }
        if (*format == '$') {
            if (max != INT_MAX) {
                PyErr_SetString(PyExc_SystemError,
                                "Invalid format string ($ specified twice)");
                return cleanreturn(0, &freelist);
            }

            max = i;
            format++;

            if (max < pos) {
                PyErr_SetString(PyExc_SystemError,
                                "Empty parameter name after $");
                return cleanreturn(0, &freelist);
            }
            if (skip) {
                /* Now we know the minimal and the maximal numbers of
                 * positional arguments and can raise an exception with
                 * informative message (see below). */
                break;
            }
            if (max < nargs) {
                if (max == 0) {
                    PyErr_Format(PyExc_TypeError,
                                 "%.200s%s takes no positional arguments",
                                 (fname == NULL) ? "function" : fname,
                                 (fname == NULL) ? "" : "()");
                }
                else {
                    PyErr_Format(PyExc_TypeError,
                                 "%.200s%s takes %s %d positional argument%s"
                                 " (%zd given)",
                                 (fname == NULL) ? "function" : fname,
                                 (fname == NULL) ? "" : "()",
                                 (min != INT_MAX) ? "at most" : "exactly",
                                 max,
                                 max == 1 ? "" : "s",
                                 nargs);
                }
                return cleanreturn(0, &freelist);
            }
        }
        if (IS_END_OF_FORMAT(*format)) {
            PyErr_Format(PyExc_SystemError,
                         "More keyword list entries (%d) than "
                         "format specifiers (%d)", len, i);
            return cleanreturn(0, &freelist);
        }
        if (!skip) {
            if (i < nargs) {
                current_arg = PyTuple_GET_ITEM(args, i);
            }
            else if (nkwargs && i >= pos) {
                current_arg = _PyDict_GetItemStringWithError(kwargs, kwlist[i]);
                if (current_arg) {
                    --nkwargs;
                }
                else if (PyErr_Occurred()) {
                    return cleanreturn(0, &freelist);
                }
            }
            else {
                current_arg = NULL;
            }

            if (current_arg) {
                msg = convertitem(current_arg, &format, p_va, flags,
                    levels, msgbuf, sizeof(msgbuf), &freelist);
                if (msg) {
                    seterror(i+1, msg, levels, fname, custom_msg);
                    return cleanreturn(0, &freelist);
                }
                continue;
            }

            if (i < min) {
                if (i < pos) {
                    assert (min == INT_MAX);
                    assert (max == INT_MAX);
                    skip = 1;
                    /* At that moment we still don't know the minimal and
                     * the maximal numbers of positional arguments.  Raising
                     * an exception is deferred until we encounter | and $
                     * or the end of the format. */
                }
                else {
                    PyErr_Format(PyExc_TypeError,  "%.200s%s missing required "
                                 "argument '%s' (pos %d)",
                                 (fname == NULL) ? "function" : fname,
                                 (fname == NULL) ? "" : "()",
                                 kwlist[i], i+1);
                    return cleanreturn(0, &freelist);
                }
            }
            /* current code reports success when all required args
             * fulfilled and no keyword args left, with no further
             * validation. XXX Maybe skip this in debug build ?
             */
            if (!nkwargs && !skip) {
                return cleanreturn(1, &freelist);
            }
        }

        /* We are into optional args, skip through to any remaining
         * keyword args */
        msg = skipitem(&format, p_va, flags);
        if (msg) {
            PyErr_Format(PyExc_SystemError, "%s: '%s'", msg,
                         format);
            return cleanreturn(0, &freelist);
        }
    }

    if (skip) {
        PyErr_Format(PyExc_TypeError,
                     "%.200s%s takes %s %d positional argument%s"
                     " (%zd given)",
                     (fname == NULL) ? "function" : fname,
                     (fname == NULL) ? "" : "()",
                     (Py_MIN(pos, min) < i) ? "at least" : "exactly",
                     Py_MIN(pos, min),
                     Py_MIN(pos, min) == 1 ? "" : "s",
                     nargs);
        return cleanreturn(0, &freelist);
    }

    if (!IS_END_OF_FORMAT(*format) && (*format != '|') && (*format != '$')) {
        PyErr_Format(PyExc_SystemError,
            "more argument specifiers than keyword list entries "
            "(remaining format:'%s')", format);
        return cleanreturn(0, &freelist);
    }

    if (nkwargs > 0) {
        PyObject *key;
        Py_ssize_t j;
        /* make sure there are no arguments given by name and position */
        for (i = pos; i < nargs; i++) {
            current_arg = _PyDict_GetItemStringWithError(kwargs, kwlist[i]);
            if (current_arg) {
                /* arg present in tuple and in dict */
                PyErr_Format(PyExc_TypeError,
                             "argument for %.200s%s given by name ('%s') "
                             "and position (%d)",
                             (fname == NULL) ? "function" : fname,
                             (fname == NULL) ? "" : "()",
                             kwlist[i], i+1);
                return cleanreturn(0, &freelist);
            }
            else if (PyErr_Occurred()) {
                return cleanreturn(0, &freelist);
            }
        }
        /* make sure there are no extraneous keyword arguments */
        j = 0;
        while (PyDict_Next(kwargs, &j, &key, NULL)) {
            int match = 0;
            if (!PyUnicode_Check(key)) {
                PyErr_SetString(PyExc_TypeError,
                                "keywords must be strings");
                return cleanreturn(0, &freelist);
            }
            for (i = pos; i < len; i++) {
                if (_PyUnicode_EqualToASCIIString(key, kwlist[i])) {
                    match = 1;
                    break;
                }
            }
            if (!match) {
                PyErr_Format(PyExc_TypeError,
                             "'%U' is an invalid keyword "
                             "argument for %.200s%s",
                             key,
                             (fname == NULL) ? "this function" : fname,
                             (fname == NULL) ? "" : "()");
                return cleanreturn(0, &freelist);
            }
        }
    }

    return cleanreturn(1, &freelist);
}


/* List of static parsers. */
static struct _PyArg_Parser *static_arg_parsers = NULL;

static int
parser_init(struct _PyArg_Parser *parser)
{
    const char * const *keywords;
    const char *format, *msg;
    int i, len, min, max, nkw;
    PyObject *kwtuple;

    assert(parser->keywords != NULL);
    if (parser->kwtuple != NULL) {
        return 1;
    }

    keywords = parser->keywords;
    /* scan keywords and count the number of positional-only parameters */
    for (i = 0; keywords[i] && !*keywords[i]; i++) {
    }
    parser->pos = i;
    /* scan keywords and get greatest possible nbr of args */
    for (; keywords[i]; i++) {
        if (!*keywords[i]) {
            PyErr_SetString(PyExc_SystemError,
                            "Empty keyword parameter name");
            return 0;
        }
    }
    len = i;

    format = parser->format;
    if (format) {
        /* grab the function name or custom error msg first (mutually exclusive) */
        parser->fname = strchr(parser->format, ':');
        if (parser->fname) {
            parser->fname++;
            parser->custom_msg = NULL;
        }
        else {
            parser->custom_msg = strchr(parser->format,';');
            if (parser->custom_msg)
                parser->custom_msg++;
        }

        min = max = INT_MAX;
        for (i = 0; i < len; i++) {
            if (*format == '|') {
                if (min != INT_MAX) {
                    PyErr_SetString(PyExc_SystemError,
                                    "Invalid format string (| specified twice)");
                    return 0;
                }
                if (max != INT_MAX) {
                    PyErr_SetString(PyExc_SystemError,
                                    "Invalid format string ($ before |)");
                    return 0;
                }
                min = i;
                format++;
            }
            if (*format == '$') {
                if (max != INT_MAX) {
                    PyErr_SetString(PyExc_SystemError,
                                    "Invalid format string ($ specified twice)");
                    return 0;
                }
                if (i < parser->pos) {
                    PyErr_SetString(PyExc_SystemError,
                                    "Empty parameter name after $");
                    return 0;
                }
                max = i;
                format++;
            }
            if (IS_END_OF_FORMAT(*format)) {
                PyErr_Format(PyExc_SystemError,
                            "More keyword list entries (%d) than "
                            "format specifiers (%d)", len, i);
                return 0;
            }

            msg = skipitem(&format, NULL, 0);
            if (msg) {
                PyErr_Format(PyExc_SystemError, "%s: '%s'", msg,
                            format);
                return 0;
            }
        }
        parser->min = Py_MIN(min, len);
        parser->max = Py_MIN(max, len);

        if (!IS_END_OF_FORMAT(*format) && (*format != '|') && (*format != '$')) {
            PyErr_Format(PyExc_SystemError,
                "more argument specifiers than keyword list entries "
                "(remaining format:'%s')", format);
            return 0;
        }
    }

    nkw = len - parser->pos;
    kwtuple = PyTuple_New(nkw);
    if (kwtuple == NULL) {
        return 0;
    }
    keywords = parser->keywords + parser->pos;
    for (i = 0; i < nkw; i++) {
        PyObject *str = PyUnicode_FromString(keywords[i]);
        if (str == NULL) {
            Py_DECREF(kwtuple);
            return 0;
        }
        PyUnicode_InternInPlace(&str);
        PyTuple_SET_ITEM(kwtuple, i, str);
    }
    parser->kwtuple = kwtuple;

    assert(parser->next == NULL);
    parser->next = static_arg_parsers;
    static_arg_parsers = parser;
    return 1;
}

static void
parser_clear(struct _PyArg_Parser *parser)
{
    Py_CLEAR(parser->kwtuple);
}

static PyObject*
find_keyword(PyObject *kwnames, PyObject *const *kwstack, PyObject *key)
{
    Py_ssize_t i, nkwargs;

    nkwargs = PyTuple_GET_SIZE(kwnames);
    for (i = 0; i < nkwargs; i++) {
        PyObject *kwname = PyTuple_GET_ITEM(kwnames, i);

        /* kwname == key will normally find a match in since keyword keys
           should be interned strings; if not retry below in a new loop. */
        if (kwname == key) {
            return kwstack[i];
        }
    }

    for (i = 0; i < nkwargs; i++) {
        PyObject *kwname = PyTuple_GET_ITEM(kwnames, i);
        assert(PyUnicode_Check(kwname));
        if (_PyUnicode_EQ(kwname, key)) {
            return kwstack[i];
        }
    }
    return NULL;
}

static int
vgetargskeywordsfast_impl(PyObject *const *args, Py_ssize_t nargs,
                          PyObject *kwargs, PyObject *kwnames,
                          struct _PyArg_Parser *parser,
                          va_list *p_va, int flags)
{
    PyObject *kwtuple;
    char msgbuf[512];
    int levels[32];
    const char *format;
    const char *msg;
    PyObject *keyword;
    int i, pos, len;
    Py_ssize_t nkwargs;
    PyObject *current_arg;
    freelistentry_t static_entries[STATIC_FREELIST_ENTRIES];
    freelist_t freelist;
    PyObject *const *kwstack = NULL;

    freelist.entries = static_entries;
    freelist.first_available = 0;
    freelist.entries_malloced = 0;

    assert(kwargs == NULL || PyDict_Check(kwargs));
    assert(kwargs == NULL || kwnames == NULL);
    assert(p_va != NULL);

    if (parser == NULL) {
        PyErr_BadInternalCall();
        return 0;
    }

    if (kwnames != NULL && !PyTuple_Check(kwnames)) {
        PyErr_BadInternalCall();
        return 0;
    }

    if (!parser_init(parser)) {
        return 0;
    }

    kwtuple = parser->kwtuple;
    pos = parser->pos;
    len = pos + (int)PyTuple_GET_SIZE(kwtuple);

    if (len > STATIC_FREELIST_ENTRIES) {
        freelist.entries = PyMem_NEW(freelistentry_t, len);
        if (freelist.entries == NULL) {
            PyErr_NoMemory();
            return 0;
        }
        freelist.entries_malloced = 1;
    }

    if (kwargs != NULL) {
        nkwargs = PyDict_GET_SIZE(kwargs);
    }
    else if (kwnames != NULL) {
        nkwargs = PyTuple_GET_SIZE(kwnames);
        kwstack = args + nargs;
    }
    else {
        nkwargs = 0;
    }
    if (nargs + nkwargs > len) {
        /* Adding "keyword" (when nargs == 0) prevents producing wrong error
           messages in some special cases (see bpo-31229). */
        PyErr_Format(PyExc_TypeError,
                     "%.200s%s takes at most %d %sargument%s (%zd given)",
                     (parser->fname == NULL) ? "function" : parser->fname,
                     (parser->fname == NULL) ? "" : "()",
                     len,
                     (nargs == 0) ? "keyword " : "",
                     (len == 1) ? "" : "s",
                     nargs + nkwargs);
        return cleanreturn(0, &freelist);
    }
    if (parser->max < nargs) {
        if (parser->max == 0) {
            PyErr_Format(PyExc_TypeError,
                         "%.200s%s takes no positional arguments",
                         (parser->fname == NULL) ? "function" : parser->fname,
                         (parser->fname == NULL) ? "" : "()");
        }
        else {
            PyErr_Format(PyExc_TypeError,
                         "%.200s%s takes %s %d positional argument%s (%zd given)",
                         (parser->fname == NULL) ? "function" : parser->fname,
                         (parser->fname == NULL) ? "" : "()",
                         (parser->min < parser->max) ? "at most" : "exactly",
                         parser->max,
                         parser->max == 1 ? "" : "s",
                         nargs);
        }
        return cleanreturn(0, &freelist);
    }

    format = parser->format;
    /* convert tuple args and keyword args in same loop, using kwtuple to drive process */
    for (i = 0; i < len; i++) {
        if (*format == '|') {
            format++;
        }
        if (*format == '$') {
            format++;
        }
        assert(!IS_END_OF_FORMAT(*format));

        if (i < nargs) {
            current_arg = args[i];
        }
        else if (nkwargs && i >= pos) {
            keyword = PyTuple_GET_ITEM(kwtuple, i - pos);
            if (kwargs != NULL) {
                current_arg = PyDict_GetItemWithError(kwargs, keyword);
                if (!current_arg && PyErr_Occurred()) {
                    return cleanreturn(0, &freelist);
                }
            }
            else {
                current_arg = find_keyword(kwnames, kwstack, keyword);
            }
            if (current_arg) {
                --nkwargs;
            }
        }
        else {
            current_arg = NULL;
        }

        if (current_arg) {
            msg = convertitem(current_arg, &format, p_va, flags,
                levels, msgbuf, sizeof(msgbuf), &freelist);
            if (msg) {
                seterror(i+1, msg, levels, parser->fname, parser->custom_msg);
                return cleanreturn(0, &freelist);
            }
            continue;
        }

        if (i < parser->min) {
            /* Less arguments than required */
            if (i < pos) {
                Py_ssize_t min = Py_MIN(pos, parser->min);
                PyErr_Format(PyExc_TypeError,
                             "%.200s%s takes %s %d positional argument%s"
                             " (%zd given)",
                             (parser->fname == NULL) ? "function" : parser->fname,
                             (parser->fname == NULL) ? "" : "()",
                             min < parser->max ? "at least" : "exactly",
                             min,
                             min == 1 ? "" : "s",
                             nargs);
            }
            else {
                keyword = PyTuple_GET_ITEM(kwtuple, i - pos);
                PyErr_Format(PyExc_TypeError,  "%.200s%s missing required "
                             "argument '%U' (pos %d)",
                             (parser->fname == NULL) ? "function" : parser->fname,
                             (parser->fname == NULL) ? "" : "()",
                             keyword, i+1);
            }
            return cleanreturn(0, &freelist);
        }
        /* current code reports success when all required args
         * fulfilled and no keyword args left, with no further
         * validation. XXX Maybe skip this in debug build ?
         */
        if (!nkwargs) {
            return cleanreturn(1, &freelist);
        }

        /* We are into optional args, skip through to any remaining
         * keyword args */
        msg = skipitem(&format, p_va, flags);
        assert(msg == NULL);
    }

    assert(IS_END_OF_FORMAT(*format) || (*format == '|') || (*format == '$'));

    if (nkwargs > 0) {
        Py_ssize_t j;
        /* make sure there are no arguments given by name and position */
        for (i = pos; i < nargs; i++) {
            keyword = PyTuple_GET_ITEM(kwtuple, i - pos);
            if (kwargs != NULL) {
                current_arg = PyDict_GetItemWithError(kwargs, keyword);
                if (!current_arg && PyErr_Occurred()) {
                    return cleanreturn(0, &freelist);
                }
            }
            else {
                current_arg = find_keyword(kwnames, kwstack, keyword);
            }
            if (current_arg) {
                /* arg present in tuple and in dict */
                PyErr_Format(PyExc_TypeError,
                             "argument for %.200s%s given by name ('%U') "
                             "and position (%d)",
                             (parser->fname == NULL) ? "function" : parser->fname,
                             (parser->fname == NULL) ? "" : "()",
                             keyword, i+1);
                return cleanreturn(0, &freelist);
            }
        }
        /* make sure there are no extraneous keyword arguments */
        j = 0;
        while (1) {
            int match;
            if (kwargs != NULL) {
                if (!PyDict_Next(kwargs, &j, &keyword, NULL))
                    break;
            }
            else {
                if (j >= PyTuple_GET_SIZE(kwnames))
                    break;
                keyword = PyTuple_GET_ITEM(kwnames, j);
                j++;
            }

            match = PySequence_Contains(kwtuple, keyword);
            if (match <= 0) {
                if (!match) {
                    PyErr_Format(PyExc_TypeError,
                                 "'%S' is an invalid keyword "
                                 "argument for %.200s%s",
                                 keyword,
                                 (parser->fname == NULL) ? "this function" : parser->fname,
                                 (parser->fname == NULL) ? "" : "()");
                }
                return cleanreturn(0, &freelist);
            }
        }
    }

    return cleanreturn(1, &freelist);
}

static int
vgetargskeywordsfast(PyObject *args, PyObject *keywords,
                     struct _PyArg_Parser *parser, va_list *p_va, int flags)
{
    PyObject **stack;
    Py_ssize_t nargs;

    if (args == NULL
        || !PyTuple_Check(args)
        || (keywords != NULL && !PyDict_Check(keywords)))
    {
        PyErr_BadInternalCall();
        return 0;
    }

    stack = _PyTuple_ITEMS(args);
    nargs = PyTuple_GET_SIZE(args);
    return vgetargskeywordsfast_impl(stack, nargs, keywords, NULL,
                                     parser, p_va, flags);
}


#undef _PyArg_UnpackKeywords

PyObject * const *
_PyArg_UnpackKeywords(PyObject *const *args, Py_ssize_t nargs,
                      PyObject *kwargs, PyObject *kwnames,
                      struct _PyArg_Parser *parser,
                      int minpos, int maxpos, int minkw,
                      PyObject **buf)
{
    PyObject *kwtuple;
    PyObject *keyword;
    int i, posonly, minposonly, maxargs;
    int reqlimit = minkw ? maxpos + minkw : minpos;
    Py_ssize_t nkwargs;
    PyObject *current_arg;
    PyObject * const *kwstack = NULL;

    assert(kwargs == NULL || PyDict_Check(kwargs));
    assert(kwargs == NULL || kwnames == NULL);

    if (parser == NULL) {
        PyErr_BadInternalCall();
        return NULL;
    }

    if (kwnames != NULL && !PyTuple_Check(kwnames)) {
        PyErr_BadInternalCall();
        return NULL;
    }

    if (args == NULL && nargs == 0) {
        args = buf;
    }

    if (!parser_init(parser)) {
        return NULL;
    }

    kwtuple = parser->kwtuple;
    posonly = parser->pos;
    minposonly = Py_MIN(posonly, minpos);
    maxargs = posonly + (int)PyTuple_GET_SIZE(kwtuple);

    if (kwargs != NULL) {
        nkwargs = PyDict_GET_SIZE(kwargs);
    }
    else if (kwnames != NULL) {
        nkwargs = PyTuple_GET_SIZE(kwnames);
        kwstack = args + nargs;
    }
    else {
        nkwargs = 0;
    }
    if (nkwargs == 0 && minkw == 0 && minpos <= nargs && nargs <= maxpos) {
        /* Fast path. */
        return args;
    }
    if (nargs + nkwargs > maxargs) {
        /* Adding "keyword" (when nargs == 0) prevents producing wrong error
           messages in some special cases (see bpo-31229). */
        PyErr_Format(PyExc_TypeError,
                     "%.200s%s takes at most %d %sargument%s (%zd given)",
                     (parser->fname == NULL) ? "function" : parser->fname,
                     (parser->fname == NULL) ? "" : "()",
                     maxargs,
                     (nargs == 0) ? "keyword " : "",
                     (maxargs == 1) ? "" : "s",
                     nargs + nkwargs);
        return NULL;
    }
    if (nargs > maxpos) {
        if (maxpos == 0) {
            PyErr_Format(PyExc_TypeError,
                         "%.200s%s takes no positional arguments",
                         (parser->fname == NULL) ? "function" : parser->fname,
                         (parser->fname == NULL) ? "" : "()");
        }
        else {
            PyErr_Format(PyExc_TypeError,
                         "%.200s%s takes %s %d positional argument%s (%zd given)",
                         (parser->fname == NULL) ? "function" : parser->fname,
                         (parser->fname == NULL) ? "" : "()",
                         (minpos < maxpos) ? "at most" : "exactly",
                         maxpos,
                         (maxpos == 1) ? "" : "s",
                         nargs);
        }
        return NULL;
    }
    if (nargs < minposonly) {
        PyErr_Format(PyExc_TypeError,
                     "%.200s%s takes %s %d positional argument%s"
                     " (%zd given)",
                     (parser->fname == NULL) ? "function" : parser->fname,
                     (parser->fname == NULL) ? "" : "()",
                     minposonly < maxpos ? "at least" : "exactly",
                     minposonly,
                     minposonly == 1 ? "" : "s",
                     nargs);
        return NULL;
    }

    /* copy tuple args */
    for (i = 0; i < nargs; i++) {
        buf[i] = args[i];
    }

    /* copy keyword args using kwtuple to drive process */
    for (i = Py_MAX((int)nargs, posonly); i < maxargs; i++) {
        if (nkwargs) {
            keyword = PyTuple_GET_ITEM(kwtuple, i - posonly);
            if (kwargs != NULL) {
                current_arg = PyDict_GetItemWithError(kwargs, keyword);
                if (!current_arg && PyErr_Occurred()) {
                    return NULL;
                }
            }
            else {
                current_arg = find_keyword(kwnames, kwstack, keyword);
            }
        }
        else if (i >= reqlimit) {
            break;
        }
        else {
            current_arg = NULL;
        }

        buf[i] = current_arg;

        if (current_arg) {
            --nkwargs;
        }
        else if (i < minpos || (maxpos <= i && i < reqlimit)) {
            /* Less arguments than required */
            keyword = PyTuple_GET_ITEM(kwtuple, i - posonly);
            PyErr_Format(PyExc_TypeError,  "%.200s%s missing required "
                         "argument '%U' (pos %d)",
                         (parser->fname == NULL) ? "function" : parser->fname,
                         (parser->fname == NULL) ? "" : "()",
                         keyword, i+1);
            return NULL;
        }
    }

    if (nkwargs > 0) {
        Py_ssize_t j;
        /* make sure there are no arguments given by name and position */
        for (i = posonly; i < nargs; i++) {
            keyword = PyTuple_GET_ITEM(kwtuple, i - posonly);
            if (kwargs != NULL) {
                current_arg = PyDict_GetItemWithError(kwargs, keyword);
                if (!current_arg && PyErr_Occurred()) {
                    return NULL;
                }
            }
            else {
                current_arg = find_keyword(kwnames, kwstack, keyword);
            }
            if (current_arg) {
                /* arg present in tuple and in dict */
                PyErr_Format(PyExc_TypeError,
                             "argument for %.200s%s given by name ('%U') "
                             "and position (%d)",
                             (parser->fname == NULL) ? "function" : parser->fname,
                             (parser->fname == NULL) ? "" : "()",
                             keyword, i+1);
                return NULL;
            }
        }
        /* make sure there are no extraneous keyword arguments */
        j = 0;
        while (1) {
            int match;
            if (kwargs != NULL) {
                if (!PyDict_Next(kwargs, &j, &keyword, NULL))
                    break;
            }
            else {
                if (j >= PyTuple_GET_SIZE(kwnames))
                    break;
                keyword = PyTuple_GET_ITEM(kwnames, j);
                j++;
            }

            match = PySequence_Contains(kwtuple, keyword);
            if (match <= 0) {
                if (!match) {
                    PyErr_Format(PyExc_TypeError,
                                 "'%S' is an invalid keyword "
                                 "argument for %.200s%s",
                                 keyword,
                                 (parser->fname == NULL) ? "this function" : parser->fname,
                                 (parser->fname == NULL) ? "" : "()");
                }
                return NULL;
            }
        }
    }

    return buf;
}


static const char *
skipitem(const char **p_format, va_list *p_va, int flags)
{
    const char *format = *p_format;
    char c = *format++;

    switch (c) {

    /*
     * codes that take a single data pointer as an argument
     * (the type of the pointer is irrelevant)
     */

    case 'b': /* byte -- very short int */
    case 'B': /* byte as bitfield */
    case 'h': /* short int */
    case 'H': /* short int as bitfield */
    case 'i': /* int */
    case 'I': /* int sized bitfield */
    case 'l': /* long int */
    case 'k': /* long int sized bitfield */
    case 'L': /* long long */
    case 'K': /* long long sized bitfield */
    case 'n': /* Py_ssize_t */
    case 'f': /* float */
    case 'd': /* double */
    case 'D': /* complex double */
    case 'c': /* char */
    case 'C': /* unicode char */
    case 'p': /* boolean predicate */
    case 'S': /* string object */
    case 'Y': /* string object */
    case 'U': /* unicode string object */
        {
            if (p_va != NULL) {
                (void) va_arg(*p_va, void *);
            }
            break;
        }

    /* string codes */

    case 'e': /* string with encoding */
        {
            if (p_va != NULL) {
                (void) va_arg(*p_va, const char *);
            }
            if (!(*format == 's' || *format == 't'))
                /* after 'e', only 's' and 't' is allowed */
                goto err;
            format++;
        }
        /* fall through */

    case 's': /* string */
    case 'z': /* string or None */
    case 'y': /* bytes */
    case 'u': /* unicode string */
    case 'Z': /* unicode string or None */
    case 'w': /* buffer, read-write */
        {
            if (p_va != NULL) {
                (void) va_arg(*p_va, char **);
            }
            if (*format == '#') {
                if (p_va != NULL) {
                    if (!(flags & FLAG_SIZE_T)) {
                        PyErr_SetString(PyExc_SystemError,
                                "PY_SSIZE_T_CLEAN macro must be defined for '#' formats");
                        return NULL;
                    }
                    (void) va_arg(*p_va, Py_ssize_t *);
                }
                format++;
            } else if ((c == 's' || c == 'z' || c == 'y' || c == 'w')
                       && *format == '*')
            {
                format++;
            }
            break;
        }

    case 'O': /* object */
        {
            if (*format == '!') {
                format++;
                if (p_va != NULL) {
                    (void) va_arg(*p_va, PyTypeObject*);
                    (void) va_arg(*p_va, PyObject **);
                }
            }
            else if (*format == '&') {
                typedef int (*converter)(PyObject *, void *);
                if (p_va != NULL) {
                    (void) va_arg(*p_va, converter);
                    (void) va_arg(*p_va, void *);
                }
                format++;
            }
            else {
                if (p_va != NULL) {
                    (void) va_arg(*p_va, PyObject **);
                }
            }
            break;
        }

    case '(':           /* bypass tuple, not handled at all previously */
        {
            const char *msg;
            for (;;) {
                if (*format==')')
                    break;
                if (IS_END_OF_FORMAT(*format))
                    return "Unmatched left paren in format "
                           "string";
                msg = skipitem(&format, p_va, flags);
                if (msg)
                    return msg;
            }
            format++;
            break;
        }

    case ')':
        return "Unmatched right paren in format string";

    default:
err:
        return "impossible<bad format char>";

    }

    *p_format = format;
    return NULL;
}


#undef _PyArg_CheckPositional

int
_PyArg_CheckPositional(const char *name, Py_ssize_t nargs,
                       Py_ssize_t min, Py_ssize_t max)
{
    assert(min >= 0);
    assert(min <= max);

    if (nargs < min) {
        if (name != NULL)
            PyErr_Format(
                PyExc_TypeError,
                "%.200s expected %s%zd argument%s, got %zd",
                name, (min == max ? "" : "at least "), min, min == 1 ? "" : "s", nargs);
        else
            PyErr_Format(
                PyExc_TypeError,
                "unpacked tuple should have %s%zd element%s,"
                " but has %zd",
                (min == max ? "" : "at least "), min, min == 1 ? "" : "s", nargs);
        return 0;
    }

    if (nargs == 0) {
        return 1;
    }

    if (nargs > max) {
        if (name != NULL)
            PyErr_Format(
                PyExc_TypeError,
                "%.200s expected %s%zd argument%s, got %zd",
                name, (min == max ? "" : "at most "), max, max == 1 ? "" : "s", nargs);
        else
            PyErr_Format(
                PyExc_TypeError,
                "unpacked tuple should have %s%zd element%s,"
                " but has %zd",
                (min == max ? "" : "at most "), max, max == 1 ? "" : "s", nargs);
        return 0;
    }

    return 1;
}

static int
unpack_stack(PyObject *const *args, Py_ssize_t nargs, const char *name,
             Py_ssize_t min, Py_ssize_t max, va_list vargs)
{
    Py_ssize_t i;
    PyObject **o;

    if (!_PyArg_CheckPositional(name, nargs, min, max)) {
        return 0;
    }

    for (i = 0; i < nargs; i++) {
        o = va_arg(vargs, PyObject **);
        *o = args[i];
    }
    return 1;
}

int
PyArg_UnpackTuple(PyObject *args, const char *name, Py_ssize_t min, Py_ssize_t max, ...)
{
    PyObject **stack;
    Py_ssize_t nargs;
    int retval;
    va_list vargs;

    if (!PyTuple_Check(args)) {
        PyErr_SetString(PyExc_SystemError,
            "PyArg_UnpackTuple() argument list is not a tuple");
        return 0;
    }
    stack = _PyTuple_ITEMS(args);
    nargs = PyTuple_GET_SIZE(args);

#ifdef HAVE_STDARG_PROTOTYPES
    va_start(vargs, max);
#else
    va_start(vargs);
#endif
    retval = unpack_stack(stack, nargs, name, min, max, vargs);
    va_end(vargs);
    return retval;
}

int
_PyArg_UnpackStack(PyObject *const *args, Py_ssize_t nargs, const char *name,
                   Py_ssize_t min, Py_ssize_t max, ...)
{
    int retval;
    va_list vargs;

#ifdef HAVE_STDARG_PROTOTYPES
    va_start(vargs, max);
#else
    va_start(vargs);
#endif
    retval = unpack_stack(args, nargs, name, min, max, vargs);
    va_end(vargs);
    return retval;
}


#undef _PyArg_NoKeywords
#undef _PyArg_NoKwnames
#undef _PyArg_NoPositional

/* For type constructors that don't take keyword args
 *
 * Sets a TypeError and returns 0 if the args/kwargs is
 * not empty, returns 1 otherwise
 */
int
_PyArg_NoKeywords(const char *funcname, PyObject *kwargs)
{
    if (kwargs == NULL) {
        return 1;
    }
    if (!PyDict_CheckExact(kwargs)) {
        PyErr_BadInternalCall();
        return 0;
    }
    if (PyDict_GET_SIZE(kwargs) == 0) {
        return 1;
    }

    PyErr_Format(PyExc_TypeError, "%.200s() takes no keyword arguments",
                    funcname);
    return 0;
}

int
_PyArg_NoPositional(const char *funcname, PyObject *args)
{
    if (args == NULL)
        return 1;
    if (!PyTuple_CheckExact(args)) {
        PyErr_BadInternalCall();
        return 0;
    }
    if (PyTuple_GET_SIZE(args) == 0)
        return 1;

    PyErr_Format(PyExc_TypeError, "%.200s() takes no positional arguments",
                    funcname);
    return 0;
}

int
_PyArg_NoKwnames(const char *funcname, PyObject *kwnames)
{
    if (kwnames == NULL) {
        return 1;
    }

    assert(PyTuple_CheckExact(kwnames));

    if (PyTuple_GET_SIZE(kwnames) == 0) {
        return 1;
    }

    PyErr_Format(PyExc_TypeError, "%s() takes no keyword arguments", funcname);
    return 0;
}

void
_PyArg_Fini(void)
{
    struct _PyArg_Parser *tmp, *s = static_arg_parsers;
    while (s) {
        tmp = s->next;
        s->next = NULL;
        parser_clear(s);
        s = tmp;
    }
    static_arg_parsers = NULL;
}

#ifdef __cplusplus
};
#endif


using namespace llvm;

static ManagedStatic<sys::Mutex> FunctionsLock;

typedef GenericValue (*ExFunc)(FunctionType *, ArrayRef<GenericValue>);
static ManagedStatic<std::map<const Function *, ExFunc> > ExportedFunctions;
static ManagedStatic<std::map<std::string, ExFunc> > FuncNames;

#ifdef USE_LIBFFI
typedef void (*RawFunc)();
static ManagedStatic<std::map<const Function *, RawFunc> > RawFunctions;
#endif

static Interpreter *TheInterpreter;

static char getTypeID(Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::VoidTyID:    return 'V';
  case Type::IntegerTyID:
    switch (cast<IntegerType>(Ty)->getBitWidth()) {
      case 1:  return 'o';
      case 8:  return 'B';
      case 16: return 'S';
      case 32: return 'I';
      case 64: return 'L';
      default: return 'N';
    }
  case Type::FloatTyID:   return 'F';
  case Type::DoubleTyID:  return 'D';
  case Type::PointerTyID: return 'P';
  case Type::FunctionTyID:return 'M';
  case Type::StructTyID:  return 'T';
  case Type::ArrayTyID:   return 'A';
  default: return 'U';
  }
}

// Try to find address of external function given a Function object.
// Please note, that interpreter doesn't know how to assemble a
// real call in general case (this is JIT job), that's why it assumes,
// that all external functions has the same (and pretty "general") signature.
// The typical example of such functions are "lle_X_" ones.
static ExFunc lookupFunction(const Function *F) {
  // Function not found, look it up... start by figuring out what the
  // composite function name should be.
  std::string ExtName = "lle_";
  FunctionType *FT = F->getFunctionType();
  ExtName += getTypeID(FT->getReturnType());
  for (Type *T : FT->params())
    ExtName += getTypeID(T);
  ExtName += ("_" + F->getName()).str();

  sys::ScopedLock Writer(*FunctionsLock);
  ExFunc FnPtr = (*FuncNames)[ExtName];
  if (!FnPtr)
    FnPtr = (*FuncNames)[("lle_X_" + F->getName()).str()];
#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
  if (!FnPtr && (F->getName().str().at(0) == '\x01')) {
	  std::string funcName = std::string(F->getName().substr(1,  F->getName().size() - 1)); 
	  FnPtr = (*FuncNames)[("lle_X_" + funcName)];  
  }
#endif
  if (!FnPtr)  // Try calling a generic function... if it exists...
    FnPtr = (ExFunc)(intptr_t)sys::DynamicLibrary::SearchForAddressOfSymbol(
        ("lle_X_" + F->getName()).str());
  if (FnPtr)
    ExportedFunctions->insert(std::make_pair(F, FnPtr));  // Cache for later
  return FnPtr;
}

#ifdef USE_LIBFFI
static ffi_type *ffiTypeFor(Type *Ty) {
  switch (Ty->getTypeID()) {
    case Type::VoidTyID: return &ffi_type_void;
    case Type::IntegerTyID:
      switch (cast<IntegerType>(Ty)->getBitWidth()) {
        case 8:  return &ffi_type_sint8;
        case 16: return &ffi_type_sint16;
        case 32: return &ffi_type_sint32;
        case 64: return &ffi_type_sint64;
      }
    case Type::FloatTyID:   return &ffi_type_float;
    case Type::DoubleTyID:  return &ffi_type_double;
    case Type::PointerTyID: return &ffi_type_pointer;
    default: break;
  }
  // TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
  // So, for Swift, it's a StructTyID that breaks things. 
#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
  fprintf(thread_stderr, "Type error: type= %d \n", Ty->getTypeID());
#endif
  report_fatal_error("Type could not be mapped for use with libffi.");
  return NULL;
}

static void *ffiValueFor(Type *Ty, const GenericValue &AV,
                         void *ArgDataPtr) {
  switch (Ty->getTypeID()) {
    case Type::IntegerTyID:
      switch (cast<IntegerType>(Ty)->getBitWidth()) {
        case 8: {
          int8_t *I8Ptr = (int8_t *) ArgDataPtr;
          *I8Ptr = (int8_t) AV.IntVal.getZExtValue();
          return ArgDataPtr;
        }
        case 16: {
          int16_t *I16Ptr = (int16_t *) ArgDataPtr;
          *I16Ptr = (int16_t) AV.IntVal.getZExtValue();
          return ArgDataPtr;
        }
        case 32: {
          int32_t *I32Ptr = (int32_t *) ArgDataPtr;
          *I32Ptr = (int32_t) AV.IntVal.getZExtValue();
          return ArgDataPtr;
        }
        case 64: {
          int64_t *I64Ptr = (int64_t *) ArgDataPtr;
          *I64Ptr = (int64_t) AV.IntVal.getZExtValue();
          return ArgDataPtr;
        }
      }
    case Type::FloatTyID: {
      float *FloatPtr = (float *) ArgDataPtr;
      *FloatPtr = AV.FloatVal;
      return ArgDataPtr;
    }
    case Type::DoubleTyID: {
      double *DoublePtr = (double *) ArgDataPtr;
      *DoublePtr = AV.DoubleVal;
      return ArgDataPtr;
    }
    case Type::PointerTyID: {
      void **PtrPtr = (void **) ArgDataPtr;
      *PtrPtr = GVTOP(AV);
      return ArgDataPtr;
    }
    default: break;
  }
  // TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
  fprintf(thread_stderr, "Type value error: type= %d \n", Ty->getTypeID());
#endif
  report_fatal_error("Type value could not be mapped for use with libffi.");
  return NULL;
}

static bool ffiInvoke(RawFunc Fn, Function *F, ArrayRef<GenericValue> ArgVals,
                      const DataLayout &TD, GenericValue &Result) {
  ffi_cif cif;
  FunctionType *FTy = F->getFunctionType();
  const unsigned NumArgs = F->arg_size();

  // TODO: We don't have type information about the remaining arguments, because
  // this information is never passed into ExecutionEngine::runFunction().
  if (ArgVals.size() > NumArgs && F->isVarArg()) {
    report_fatal_error("Calling external var arg function '" + F->getName()
                      + "' is not supported by the Interpreter.");
  }

  unsigned ArgBytes = 0;

  std::vector<ffi_type*> args(NumArgs);
  for (Function::const_arg_iterator A = F->arg_begin(), E = F->arg_end();
       A != E; ++A) {
    const unsigned ArgNo = A->getArgNo();
    Type *ArgTy = FTy->getParamType(ArgNo);
    args[ArgNo] = ffiTypeFor(ArgTy);
    ArgBytes += TD.getTypeStoreSize(ArgTy);
  }

  SmallVector<uint8_t, 128> ArgData;
  ArgData.resize(ArgBytes);
  uint8_t *ArgDataPtr = ArgData.data();
  SmallVector<void*, 16> values(NumArgs);
  for (Function::const_arg_iterator A = F->arg_begin(), E = F->arg_end();
       A != E; ++A) {
    const unsigned ArgNo = A->getArgNo();
    Type *ArgTy = FTy->getParamType(ArgNo);
    values[ArgNo] = ffiValueFor(ArgTy, ArgVals[ArgNo], ArgDataPtr);
    ArgDataPtr += TD.getTypeStoreSize(ArgTy);
  }

  Type *RetTy = FTy->getReturnType();
  ffi_type *rtype = ffiTypeFor(RetTy);

  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, NumArgs, rtype, args.data()) ==
      FFI_OK) {
    SmallVector<uint8_t, 128> ret;
    if (RetTy->getTypeID() != Type::VoidTyID)
      ret.resize(TD.getTypeStoreSize(RetTy));
    ffi_call(&cif, Fn, ret.data(), values.data());
    switch (RetTy->getTypeID()) {
      case Type::IntegerTyID:
        switch (cast<IntegerType>(RetTy)->getBitWidth()) {
          case 8:  Result.IntVal = APInt(8 , *(int8_t *) ret.data()); break;
          case 16: Result.IntVal = APInt(16, *(int16_t*) ret.data()); break;
          case 32: Result.IntVal = APInt(32, *(int32_t*) ret.data()); break;
          case 64: Result.IntVal = APInt(64, *(int64_t*) ret.data()); break;
        }
        break;
      case Type::FloatTyID:   Result.FloatVal   = *(float *) ret.data(); break;
      case Type::DoubleTyID:  Result.DoubleVal  = *(double*) ret.data(); break;
      case Type::PointerTyID: Result.PointerVal = *(void **) ret.data(); break;
      default: break;
    }
    return true;
  }

  return false;
}
#endif // USE_LIBFFI

GenericValue Interpreter::callExternalFunction(Function *F,
                                               ArrayRef<GenericValue> ArgVals) {
  TheInterpreter = this;

  std::unique_lock<sys::Mutex> Guard(*FunctionsLock);

  // Do a lookup to see if the function is in our cache... this should just be a
  // deferred annotation!
  std::map<const Function *, ExFunc>::iterator FI = ExportedFunctions->find(F);
  if (ExFunc Fn = (FI == ExportedFunctions->end()) ? lookupFunction(F)
                                                   : FI->second) {
    Guard.unlock();
    return Fn(F->getFunctionType(), ArgVals);
  }

#ifdef USE_LIBFFI
  std::map<const Function *, RawFunc>::iterator RF = RawFunctions->find(F);
  RawFunc RawFn;
  if (RF == RawFunctions->end()) {
    RawFn = (RawFunc)(intptr_t)
      sys::DynamicLibrary::SearchForAddressOfSymbol(std::string(F->getName()));
    if (!RawFn)
      RawFn = (RawFunc)(intptr_t)getPointerToGlobalIfAvailable(F);
    if (RawFn != 0)
      RawFunctions->insert(std::make_pair(F, RawFn));  // Cache for later
  } else {
    RawFn = RF->second;
  }

  Guard.unlock();

  GenericValue Result;
  if (RawFn != 0 && ffiInvoke(RawFn, F, ArgVals, getDataLayout(), Result))
    return Result;
#endif // USE_LIBFFI

  if (F->getName() == "__main")
    errs() << "Tried to execute an unknown external function: "
      << *F->getType() << " __main\n";
  else
    report_fatal_error("Tried to execute an unknown external function: " +
                       F->getName());
#ifndef USE_LIBFFI
  errs() << "Recompiling LLVM with --enable-libffi might help.\n";
#endif
  return GenericValue();
}

//===----------------------------------------------------------------------===//
//  Functions "exported" to the running application...
//

// void atexit(Function*)
static GenericValue lle_X_atexit(FunctionType *FT,
                                 ArrayRef<GenericValue> Args) {
  assert(Args.size() == 1);
  TheInterpreter->addAtExitHandler((Function*)GVTOP(Args[0]));
  GenericValue GV;
  GV.IntVal = 0;
  return GV;
}

// void exit(int)
static GenericValue lle_X_exit(FunctionType *FT, ArrayRef<GenericValue> Args) {
	// TODO: insert here cleanup functions, if required.
  TheInterpreter->exitCalled(Args[0]);
  return GenericValue();
}

// void abort(void)
static GenericValue lle_X_abort(FunctionType *FT, ArrayRef<GenericValue> Args) {
#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
       report_fatal_error("LLVM interpreter raised SIGABRT");
#else 
  //FIXME: should we report or raise here?
  //report_fatal_error("Interpreted program raised SIGABRT");
  raise (SIGABRT);
#endif
  return GenericValue();
}

// int sprintf(char *, const char *, ...) - a very rough implementation to make
// output useful.
static GenericValue lle_X_sprintf(FunctionType *FT,
                                  ArrayRef<GenericValue> Args) {
  char *OutputBuffer = (char *)GVTOP(Args[0]);
  const char *FmtStr = (const char *)GVTOP(Args[1]);
  unsigned ArgNo = 2;

  // printf should return # chars printed.  This is completely incorrect, but
  // close enough for now.
  GenericValue GV;
  GV.IntVal = APInt(32, strlen(FmtStr));
  while (true) {
    switch (*FmtStr) {
    case 0: return GV;             // Null terminator...
    default:                       // Normal nonspecial character
      sprintf(OutputBuffer++, "%c", *FmtStr++);
      break;
    case '\\': {                   // Handle escape codes
      sprintf(OutputBuffer, "%c%c", *FmtStr, *(FmtStr+1));
      FmtStr += 2; OutputBuffer += 2;
      break;
    }
    case '%': {                    // Handle format specifiers
      char FmtBuf[100] = "", Buffer[1000] = "";
      char *FB = FmtBuf;
      *FB++ = *FmtStr++;
      char Last = *FB++ = *FmtStr++;
      unsigned HowLong = 0;
      while (Last != 'c' && Last != 'd' && Last != 'i' && Last != 'u' &&
             Last != 'o' && Last != 'x' && Last != 'X' && Last != 'e' &&
             Last != 'E' && Last != 'g' && Last != 'G' && Last != 'f' &&
             Last != 'p' && Last != 's' && Last != '%') {
        if (Last == 'l' || Last == 'L') HowLong++;  // Keep track of l's
        Last = *FB++ = *FmtStr++;
      }
      *FB = 0;

      switch (Last) {
      case '%':
        memcpy(Buffer, "%", 2); break;
      case 'c':
        sprintf(Buffer, FmtBuf, uint32_t(Args[ArgNo++].IntVal.getZExtValue()));
        break;
      case 'd': case 'i':
      case 'u': case 'o':
      case 'x': case 'X':
        if (HowLong >= 1) {
          if (HowLong == 1 &&
              TheInterpreter->getDataLayout().getPointerSizeInBits() == 64 &&
              sizeof(long) < sizeof(int64_t)) {
            // Make sure we use %lld with a 64 bit argument because we might be
            // compiling LLI on a 32 bit compiler.
            unsigned Size = strlen(FmtBuf);
            FmtBuf[Size] = FmtBuf[Size-1];
            FmtBuf[Size+1] = 0;
            FmtBuf[Size-1] = 'l';
          }
          sprintf(Buffer, FmtBuf, Args[ArgNo++].IntVal.getZExtValue());
        } else
          sprintf(Buffer, FmtBuf,uint32_t(Args[ArgNo++].IntVal.getZExtValue()));
        break;
      case 'e': case 'E': case 'g': case 'G': case 'f':
        sprintf(Buffer, FmtBuf, Args[ArgNo++].DoubleVal); break;
      case 'p':
        sprintf(Buffer, FmtBuf, (void*)GVTOP(Args[ArgNo++])); break;
      case 's':
        sprintf(Buffer, FmtBuf, (char*)GVTOP(Args[ArgNo++])); break;
      default:
        errs() << "<unknown printf code '" << *FmtStr << "'!>";
        ArgNo++; break;
      }
      size_t Len = strlen(Buffer);
      memcpy(OutputBuffer, Buffer, Len + 1);
      OutputBuffer += Len;
      }
      break;
    }
  }
  return GV;
}

// int printf(const char *, ...) - a very rough implementation to make output
// useful.
static GenericValue lle_X_printf(FunctionType *FT,
                                 ArrayRef<GenericValue> Args) {
  char Buffer[10000];
  std::vector<GenericValue> NewArgs;
  NewArgs.push_back(PTOGV((void*)&Buffer[0]));
  llvm::append_range(NewArgs, Args);
  GenericValue GV = lle_X_sprintf(FT, NewArgs);
  outs() << Buffer;
  return GV;
}

// int sscanf(const char *format, ...);
static GenericValue lle_X_sscanf(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  assert(args.size() < 10 && "Only handle up to 10 args to sscanf right now!");

  char *Args[10];
  for (unsigned i = 0; i < args.size(); ++i)
    Args[i] = (char*)GVTOP(args[i]);

  GenericValue GV;
  GV.IntVal = APInt(32, sscanf(Args[0], Args[1], Args[2], Args[3], Args[4],
                    Args[5], Args[6], Args[7], Args[8], Args[9]));
  return GV;
}

// int scanf(const char *format, ...);
static GenericValue lle_X_scanf(FunctionType *FT, ArrayRef<GenericValue> args) {
  assert(args.size() < 10 && "Only handle up to 10 args to scanf right now!");

  char *Args[10];
  for (unsigned i = 0; i < args.size(); ++i)
    Args[i] = (char*)GVTOP(args[i]);

  GenericValue GV;
#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
  outs().flush();
  fflush(thread_stdout);
  GV.IntVal = APInt(32, fscanf(thread_stdin, Args[0], Args[1], Args[2], Args[3], Args[4],
                  Args[5], Args[6], Args[7], Args[8], Args[9]));
#else
  GV.IntVal = APInt(32, scanf( Args[0], Args[1], Args[2], Args[3], Args[4],
                    Args[5], Args[6], Args[7], Args[8], Args[9]));
#endif
  return GV;
}

// int fprintf(FILE *, const char *, ...) - a very rough implementation to make
// output useful.
static GenericValue lle_X_fprintf(FunctionType *FT,
                                  ArrayRef<GenericValue> Args) {
  assert(Args.size() >= 2);
  char Buffer[10000];
  std::vector<GenericValue> NewArgs;
  NewArgs.push_back(PTOGV(Buffer));
  NewArgs.insert(NewArgs.end(), Args.begin()+1, Args.end());
  GenericValue GV = lle_X_sprintf(FT, NewArgs);

  fputs(Buffer, (FILE *) GVTOP(Args[0]));
  return GV;
}

static GenericValue lle_X_memset(FunctionType *FT,
                                 ArrayRef<GenericValue> Args) {
  int val = (int)Args[1].IntVal.getSExtValue();
  size_t len = (size_t)Args[2].IntVal.getZExtValue();
  memset((void *)GVTOP(Args[0]), val, len);
  // llvm.memset.* returns void, lle_X_* returns GenericValue,
  // so here we return GenericValue with IntVal set to zero
  GenericValue GV;
  GV.IntVal = 0;
  return GV;
}

static GenericValue lle_X_memcpy(FunctionType *FT,
                                 ArrayRef<GenericValue> Args) {
  memcpy(GVTOP(Args[0]), GVTOP(Args[1]),
         (size_t)(Args[2].IntVal.getLimitedValue()));

  // llvm.memcpy* returns void, lle_X_* returns GenericValue,
  // so here we return GenericValue with IntVal set to zero
  GenericValue GV;
  GV.IntVal = 0;
  return GV;
}

#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)

// Required because it's a vararg function
static GenericValue lle_X_open(FunctionType *FT,
                                 ArrayRef<GenericValue> Args) {
  int oflag = (int)Args[1].IntVal.getSExtValue();
  int returnValue;
  
  if (Args.size() == 2) {
         returnValue = ::open((char*)GVTOP(Args[0]), oflag);
  } else {
         int mode = (int)Args[2].IntVal.getSExtValue();
         returnValue = ::open((char*)GVTOP(Args[0]), oflag, mode);
         if (Args.size() > 3) errs() << "called open(" << (char*)GVTOP(Args[0]) << ", ...) with " << Args.size() << " arguments.\n"; 
  }
  GenericValue GV;
  GV.IntVal = APInt(32, returnValue);
  return GV;
}
// 
const char* llvm_ios_progname; 
//   void warn(const char *fmt, ...);
static GenericValue lle_X_warn(FunctionType *FT, ArrayRef<GenericValue> Args) {
       fputs(llvm_ios_progname, thread_stderr);
       const char *fmt = (const char *)GVTOP(Args[0]);
       if (fmt != NULL)
       {
               fputs(": ", thread_stderr);
               char Buffer[10000];
               std::vector<GenericValue> NewArgs;
               NewArgs.push_back(PTOGV((void*)&Buffer[0]));
               NewArgs.insert(NewArgs.end(), Args.begin(), Args.end());
               GenericValue GV = lle_X_sprintf(FT, NewArgs);
               fputs(Buffer, thread_stderr); 
       }
       fputs(": ", thread_stderr);
       fputs(strerror(errno), thread_stderr);
       putc('\n', thread_stderr);
       return GenericValue();
}
//   void warnx(const char *fmt, ...);
static GenericValue lle_X_warnx(FunctionType *FT, ArrayRef<GenericValue> Args) {
       fputs(llvm_ios_progname, thread_stderr);
       const char *fmt = (const char *)GVTOP(Args[0]);
       if (fmt != NULL)
       {
               fputs(": ", thread_stderr);
               char Buffer[10000];
               std::vector<GenericValue> NewArgs;
               NewArgs.push_back(PTOGV((void*)&Buffer[0]));
               NewArgs.insert(NewArgs.end(), Args.begin(), Args.end());
               GenericValue GV = lle_X_sprintf(FT, NewArgs);
               fputs(Buffer, thread_stderr); 
       }
       putc('\n', thread_stderr);
       return GenericValue();
}
// void err(int eval, const char *fmt, ...);
static GenericValue lle_X_err(FunctionType *FT, ArrayRef<GenericValue> Args) {
       std::vector<GenericValue> NewArgs = Args;
       NewArgs.erase(NewArgs.begin());
       lle_X_warn(FT, NewArgs); 
       TheInterpreter->exitCalled(Args[0]);
       return GenericValue();
}
//      void errx(int eval, const char *fmt, ...);
static GenericValue lle_X_errx(FunctionType *FT, ArrayRef<GenericValue> Args) {
       std::vector<GenericValue> NewArgs = Args;
       NewArgs.erase(NewArgs.begin());
       lle_X_warnx(FT, NewArgs); 
       TheInterpreter->exitCalled(Args[0]);
       return GenericValue();
}

#if 0
// Objective-C:
static GenericValue lle_X_objc_msgSend(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
       // Note: that method fails at runtime, with: 
       // selector () for message 'stringByAppendingString:' does not match selector known to Objective C runtime ()-- abort
       // Needs linking with -f Foundation -f CoreFoundation. HOW?

  id self = (objc_object *)GVTOP(args[0]); 
  SEL op = (SEL)GVTOP(args[1]);
    
  GenericValue GV;
  if (args.size() <= 2) 
         GV.PointerVal = *(void **) objc_msgSend(self, op); 
  else {
         objc_object *Args[10];
         for (unsigned i = 2; i < args.size(); ++i)
                 Args[i - 2] = (objc_object *)GVTOP(args[i]);

         GenericValue GV;
         // Add -lobjc to bootstrap.sh before removal
         GV.PointerVal = *(void **) (id)objc_msgSend(self, op, Args[0], Args[1], Args[2], Args[3], Args[4],
                         Args[5], Args[6], Args[7], Args[8], Args[9]);
  }
  return GV;
}
// 
#endif 

// Python

static int
all_name_chars(PyObject *o)
{
    const unsigned char *s, *e;

    if (!PyUnicode_IS_ASCII(o))
        return 0;

    s = PyUnicode_1BYTE_DATA(o);
    e = s + PyUnicode_GET_LENGTH(o);
    for (; s != e; s++) {
        if (!Py_ISALNUM(*s) && *s != '_')
            return 0;
    }
    return 1;
}

static int
intern_string_constants(PyObject *tuple, int *modified)
{
    for (Py_ssize_t i = PyTuple_GET_SIZE(tuple); --i >= 0; ) {
        PyObject *v = PyTuple_GET_ITEM(tuple, i);
        if (PyUnicode_CheckExact(v)) {
            if (PyUnicode_READY(v) == -1) {
                return -1;
            }

            if (all_name_chars(v)) {
                PyObject *w = v;
                PyUnicode_InternInPlace(&v);
                if (w != v) {
                    PyTuple_SET_ITEM(tuple, i, v);
                    if (modified) {
                        *modified = 1;
                    }
                }
            }
        }
        else if (PyTuple_CheckExact(v)) {
            if (intern_string_constants(v, NULL) < 0) {
                return -1;
            }
        }
        else if (PyFrozenSet_CheckExact(v)) {
            PyObject *w = v;
            PyObject *tmp = PySequence_Tuple(v);
            if (tmp == NULL) {
                return -1;
            }
            int tmp_modified = 0;
            if (intern_string_constants(tmp, &tmp_modified) < 0) {
                Py_DECREF(tmp);
                return -1;
            }
            if (tmp_modified) {
                v = PyFrozenSet_New(tmp);
                if (v == NULL) {
                    Py_DECREF(tmp);
                    return -1;
                }

                PyTuple_SET_ITEM(tuple, i, v);
                Py_DECREF(w);
                if (modified) {
                    *modified = 1;
                }
            }
            Py_DECREF(tmp);
        }
    }
    return 0;
}

static int
intern_strings(PyObject *tuple)
{
    Py_ssize_t i;

    for (i = PyTuple_GET_SIZE(tuple); --i >= 0; ) {
        PyObject *v = PyTuple_GET_ITEM(tuple, i);
        if (v == NULL || !PyUnicode_CheckExact(v)) {
            PyErr_SetString(PyExc_SystemError,
                            "non-string found in code slot");
            return -1;
        }
        PyUnicode_InternInPlace(&_PyTuple_ITEMS(tuple)[i]);
    }
    return 0;
}

PyCodeObject *
LLVMCode_NewWithPosOnlyArgs(int argcount, int posonlyargcount, int kwonlyargcount,
                          int nlocals, int stacksize, int flags,
                          PyObject *code, PyObject *consts, PyObject *names,
                          PyObject *varnames, PyObject *freevars, PyObject *cellvars,
                          PyObject *filename, PyObject *name, int firstlineno,
                          PyObject *linetable)
{
    PyCodeObject *co;
    Py_ssize_t *cell2arg = NULL;
    Py_ssize_t i, n_cellvars, n_varnames, total_args;

    /* Check argument types */
    if (argcount < posonlyargcount || posonlyargcount < 0 ||
        kwonlyargcount < 0 || nlocals < 0 ||
        stacksize < 0 || flags < 0 ||
        code == NULL || !PyBytes_Check(code) ||
        consts == NULL || !PyTuple_Check(consts) ||
        names == NULL || !PyTuple_Check(names) ||
        varnames == NULL || !PyTuple_Check(varnames) ||
        freevars == NULL || !PyTuple_Check(freevars) ||
        cellvars == NULL || !PyTuple_Check(cellvars) ||
        name == NULL || !PyUnicode_Check(name) ||
        filename == NULL || !PyUnicode_Check(filename) ||
        linetable == NULL || !PyBytes_Check(linetable)) {
        PyErr_BadInternalCall();
        return NULL;
    }

    /* Ensure that strings are ready Unicode string */
    if (PyUnicode_READY(name) < 0) {
        return NULL;
    }
    if (PyUnicode_READY(filename) < 0) {
        return NULL;
    }

    if (intern_strings(names) < 0) {
        return NULL;
    }
    if (intern_strings(varnames) < 0) {
        return NULL;
    }
    if (intern_strings(freevars) < 0) {
        return NULL;
    }
    if (intern_strings(cellvars) < 0) {
        return NULL;
    }
    if (intern_string_constants(consts, NULL) < 0) {
        return NULL;
    }

    /* Make sure that code is indexable with an int, this is
       a long running assumption in ceval.c and many parts of
       the interpreter. */
    if (PyBytes_GET_SIZE(code) > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "co_code larger than INT_MAX");
        return NULL;
    }

    /* Check for any inner or outer closure references */
    n_cellvars = PyTuple_GET_SIZE(cellvars);
    if (!n_cellvars && !PyTuple_GET_SIZE(freevars)) {
        flags |= CO_NOFREE;
    } else {
        flags &= ~CO_NOFREE;
    }

    n_varnames = PyTuple_GET_SIZE(varnames);
    if (argcount <= n_varnames && kwonlyargcount <= n_varnames) {
        /* Never overflows. */
        total_args = (Py_ssize_t)argcount + (Py_ssize_t)kwonlyargcount +
                      ((flags & CO_VARARGS) != 0) + ((flags & CO_VARKEYWORDS) != 0);
    }
    else {
        total_args = n_varnames + 1;
    }
    if (total_args > n_varnames) {
        PyErr_SetString(PyExc_ValueError, "code: varnames is too small");
        return NULL;
    }

    /* Create mapping between cells and arguments if needed. */
    if (n_cellvars) {
        bool used_cell2arg = false;
        cell2arg = PyMem_NEW(Py_ssize_t, n_cellvars);
        if (cell2arg == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        /* Find cells which are also arguments. */
        for (i = 0; i < n_cellvars; i++) {
            Py_ssize_t j;
            PyObject *cell = PyTuple_GET_ITEM(cellvars, i);
            cell2arg[i] = CO_CELL_NOT_AN_ARG;
            for (j = 0; j < total_args; j++) {
                PyObject *arg = PyTuple_GET_ITEM(varnames, j);
                int cmp = PyUnicode_Compare(cell, arg);
                if (cmp == -1 && PyErr_Occurred()) {
                    PyMem_Free(cell2arg);
                    return NULL;
                }
                if (cmp == 0) {
                    cell2arg[i] = j;
                    used_cell2arg = true;
                    break;
                }
            }
        }
        if (!used_cell2arg) {
            PyMem_Free(cell2arg);
            cell2arg = NULL;
        }
    }
    co = PyObject_New(PyCodeObject, &PyCode_Type);
    if (co == NULL) {
        if (cell2arg)
            PyMem_Free(cell2arg);
        return NULL;
    }
    co->co_argcount = argcount;
    co->co_posonlyargcount = posonlyargcount;
    co->co_kwonlyargcount = kwonlyargcount;
    co->co_nlocals = nlocals;
    co->co_stacksize = stacksize;
    co->co_flags = flags;
    Py_INCREF(code);
    co->co_code = code;
    Py_INCREF(consts);
    co->co_consts = consts;
    Py_INCREF(names);
    co->co_names = names;
    Py_INCREF(varnames);
    co->co_varnames = varnames;
    Py_INCREF(freevars);
    co->co_freevars = freevars;
    Py_INCREF(cellvars);
    co->co_cellvars = cellvars;
    co->co_cell2arg = cell2arg;
    Py_INCREF(filename);
    co->co_filename = filename;
    Py_INCREF(name);
    co->co_name = name;
    co->co_firstlineno = firstlineno;
    Py_INCREF(linetable);
    co->co_linetable = linetable;
    co->co_zombieframe = NULL;
    co->co_weakreflist = NULL;
    co->co_extra = NULL;

    co->co_opcache_map = NULL;
    co->co_opcache = NULL;
    co->co_opcache_flag = 0;
    co->co_opcache_size = 0;
    return co;
}

// int PyArg_ParseTuple(PyObject *args, const char *format, ...)
static GenericValue lle_X_PyArg_ParseTuple(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  PyObject *py_args;
  const char *format;
  std::vector<void *> vargs;
  
  int i = 0;
  for(GenericValue arg : args) {
    
    if (i == 0) {
      py_args = (PyObject *)GVTOP(arg);
    } else if (i == 1) {
      format = (const char *)GVTOP(arg);
    } else {
      vargs.push_back(GVTOP(arg));
    }
    
    i += 1;
  }
  
  va_list vargs_list = reinterpret_cast<va_list>(vargs.data());

  GV.IntVal = PyArg_VaParse(py_args, format, vargs_list);
  return GV;
}

// int PyArg_ParseTupleAndKeywords(PyObject *args, PyObject *kw, const char *format, char *keywords[], ...)
static GenericValue lle_X_PyArg_ParseTupleAndKeywords(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  PyObject *py_args;
  PyObject *kw;
  const char *format;
  char **keywords;
  std::vector<void *> vargs;
  
  int i = 0;
  for(GenericValue arg : args) {
    
    if (i == 0) {
      py_args = (PyObject *)GVTOP(arg);
    } else if (i == 1) {
      kw = (PyObject *)GVTOP(arg);
    } else if (i == 2) {
      format = (const char *)GVTOP(arg);
    } else if (i == 3) {
      keywords = (char **)GVTOP(arg);
    } else {
      vargs.push_back(GVTOP(arg));
    }
    
    i += 1;
  }
  
  va_list vargs_list = reinterpret_cast<va_list>(vargs.data());

  GV.IntVal = PyArg_VaParseTupleAndKeywords(py_args, kw, format, keywords, vargs_list);
  return GV;
}

int PyArg_VaUnpackTuple(PyObject *args, const char *name, Py_ssize_t min, Py_ssize_t max, va_list vargs) {
    PyObject **stack;
    Py_ssize_t nargs;
    int retval;

    if (!PyTuple_Check(args)) {
        PyErr_SetString(PyExc_SystemError,
            "PyArg_UnpackTuple() argument list is not a tuple");
        return 0;
    }
    stack = _PyTuple_ITEMS(args);
    nargs = PyTuple_GET_SIZE(args);
  
    retval = unpack_stack(stack, nargs, name, min, max, vargs);
    return retval;
}

// int PyArg_UnpackTuple(PyObject *args, const char *name, Py_ssize_t min, Py_ssize_t max, ...)
static GenericValue lle_X_PyArg_UnpackTuple(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  PyObject *py_args;
  const char *name;
  Py_ssize_t min;
  Py_ssize_t max;
  std::vector<void *> vargs;
  
  int i = 0;
  for(GenericValue arg : args) {
    
    if (i == 0) {
      py_args = (PyObject *)GVTOP(arg);
    } else if (i == 1) {
      name = (const char *)GVTOP(arg);
    } else if (i == 2) {
      min = (Py_ssize_t)GVTOP(arg);
    } else if (i == 3) {
      max = (Py_ssize_t)GVTOP(arg);
    } else {
      vargs.push_back(GVTOP(arg));
    }
    
    i += 1;
  }
  
  va_list vargs_list = reinterpret_cast<va_list>(vargs.data());

  GV.IntVal = PyArg_VaUnpackTuple(py_args, name, min, max, vargs_list);
  return GV;
}

// PyObject *Py_BuildValue(const char *format, ...)
static GenericValue lle_X_Py_BuildValue(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  const char *format;
  std::vector<void *> vargs;
  
  int i = 0;
  for(GenericValue arg : args) {
    
    if (i == 0) {
      format = (const char *)GVTOP(arg);
    } else {
      vargs.push_back(GVTOP(arg));
    }
    
    i += 1;
  }
  
  va_list vargs_list = reinterpret_cast<va_list>(vargs.data());

  GV.PointerVal = Py_VaBuildValue(format, vargs_list);
  return GV;
}

// int PyOS_snprintf(char *str, size_t size, const char *format, ...)
static GenericValue lle_X_PyOS_snprintf(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  char *str;
  size_t size;
  const char *format;
  std::vector<void *> vargs;
  
  int i = 0;
  for(GenericValue arg : args) {
    
    if (i == 0) {
      str = (char *)GVTOP(arg);
    } else if (i == 1) {
      size = (size_t)GVTOP(arg);
    } else if (i == 2) {
      format = (const char *)GVTOP(arg);
    } else {
      vargs.push_back(GVTOP(arg));
    }
    
    i += 1;
  }
  
  va_list vargs_list = reinterpret_cast<va_list>(vargs.data());

  GV.IntVal = PyOS_vsnprintf(str, size, format, vargs_list);
  return GV;
}

static struct _Py_tuple_state *
get_tuple_state(void)
{
    PyInterpreterState *interp = _PyInterpreterState_GET();
    return &interp->tuple;
}

static PyObject *
tuple_get_empty(void)
{
  struct _Py_tuple_state *state = get_tuple_state();
  PyTupleObject *op = state->free_list[0];
  // tuple_get_empty() must not be called before _PyTuple_Init()
  // or after _PyTuple_Fini()
  assert(op != NULL);

  Py_INCREF(op);
  return (PyObject *) op;
}

static PyTupleObject *
tuple_alloc(Py_ssize_t size)
{
    PyTupleObject *op;
    // If Python is built with the empty tuple singleton,
    // tuple_alloc(0) must not be called.
    assert(size != 0);
  
    if (size < 0) {
        PyErr_BadInternalCall();
        return NULL;
    }

    struct _Py_tuple_state *state = get_tuple_state();

    if (size < PyTuple_MAXSAVESIZE && (op = state->free_list[size]) != NULL) {
        assert(size != 0);
        state->free_list[size] = (PyTupleObject *) op->ob_item[0];
        state->numfree[size]--;
        /* Inlined _PyObject_InitVar() without _PyType_HasFeature() test */
        _Py_NewReference((PyObject *)op);
    }
    else
    {
        /* Check for overflow */
        if ((size_t)size > ((size_t)PY_SSIZE_T_MAX - (sizeof(PyTupleObject) -
                    sizeof(PyObject *))) / sizeof(PyObject *)) {
            return (PyTupleObject *)PyErr_NoMemory();
        }
        op = PyObject_GC_NewVar(PyTupleObject, &PyTuple_Type, size);
        if (op == NULL)
            return NULL;
    }
    return op;
}

static inline void
tuple_gc_track(PyTupleObject *op)
{
    _PyObject_GC_TRACK(op);
}

PyObject *
PyTuple_VaPack(Py_ssize_t n, va_list vargs)
{
    Py_ssize_t i;
    PyObject *o;
    PyObject **items;

    if (n == 0) {
        return tuple_get_empty();
    }

    PyTupleObject *result = tuple_alloc(n);
    if (result == NULL) {
        return NULL;
    }
    items = result->ob_item;
    for (i = 0; i < n; i++) {
        o = va_arg(vargs, PyObject *);
        Py_INCREF(o);
        items[i] = o;
    }
    tuple_gc_track(result);
    return (PyObject *)result;
}

// PyObject *PyTuple_Pack(Py_ssize_t n, ...)
static GenericValue lle_X_PyTuple_Pack(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  Py_ssize_t n;
  std::vector<void *> vargs;
  
  int i = 0;
  for(GenericValue arg : args) {
    
    if (i == 0) {
      n = (Py_ssize_t)GVTOP(arg);
    } else {
      vargs.push_back(GVTOP(arg));
    }
    
    i += 1;
  }
  
  va_list vargs_list = reinterpret_cast<va_list>(vargs.data());

  GV.PointerVal = PyTuple_VaPack(n, vargs_list);
  return GV;
}

// PyCode_NewWithPosOnlyArgs(int argcount, int posonlyargcount, int kwonlyargcount, int nlocals, int stacksize, int flags, PyObject *code, PyObject *consts, PyObject *names, PyObject *varnames, PyObject *freevars, PyObject *cellvars, PyObject *filename, PyObject *name, int firstlineno, PyObject *linetable)
static GenericValue lle_X_PyCode_NewWithPosOnlyArgs(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  int argcount = (int)args[0].IntVal.bitsToDouble();
  int posonlyargcount = (int)args[1].IntVal.bitsToDouble();
  int kwonlyargcount = (int)args[2].IntVal.bitsToDouble();
  int nlocals = (int)args[3].IntVal.bitsToDouble();
  int stacksize = (int)args[4].IntVal.bitsToDouble();
  int flags = (int)args[5].IntVal.bitsToDouble();
  PyObject *code = (PyObject *)GVTOP(args[6]);
  PyObject *consts = (PyObject *)GVTOP(args[7]);
  PyObject *names = (PyObject *)GVTOP(args[8]);
  PyObject *varnames = (PyObject *)GVTOP(args[9]);
  PyObject *freevars = (PyObject *)GVTOP(args[10]);
  PyObject *cellvars = (PyObject *)GVTOP(args[11]);
  PyObject *filename = (PyObject *)GVTOP(args[12]);
  PyObject *name = (PyObject *)GVTOP(args[13]);
  int firstlineno = (int)args[14].IntVal.bitsToDouble();
  PyObject *linetable = (PyObject *)GVTOP(args[15]);
  
  GV.PointerVal = LLVMCode_NewWithPosOnlyArgs(argcount, posonlyargcount, kwonlyargcount, nlocals, stacksize, flags, code, consts, names, varnames, freevars, cellvars, filename, name, firstlineno, linetable);
  return GV;
}

// _PyArg_ParseTupleAndKeywords_SizeT(PyObject *args, PyObject *keywords, const char *format, char **kwlist, ...)
static GenericValue lle_X__PyArg_ParseTupleAndKeywords_SizeT(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  PyObject *pyargs;
  PyObject *keywords;
  const char *format;
  char **kwlist;
  std::vector<void *> vargs;
  
  int i = 0;
  for(GenericValue arg : args) {
    
    if (i == 0) {
      pyargs = (PyObject *)GVTOP(arg);
    } else if (i == 1) {
      keywords = (PyObject *)GVTOP(arg);
    } else if (i == 2) {
      format = (const char *)GVTOP(arg);
    } else if (i == 3) {
      kwlist = (char **)GVTOP(arg);
    } else {
      vargs.push_back(GVTOP(arg));
    }
    
    i += 1;
  }
  
  va_list vargs_list = reinterpret_cast<va_list>(vargs.data());

  GV.IntVal = _PyArg_VaParseTupleAndKeywords_SizeT(pyargs, keywords, format, kwlist, vargs_list);
  return GV;
}

// PyObject *PyUnicode_FromFormat(const char *format, ...)
static GenericValue lle_X_PyUnicode_FromFormat(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  const char *format;
  std::vector<void *> vargs;
  
  int i = 0;
  for(GenericValue arg : args) {
    
    if (i == 0) {
      format = (const char *)GVTOP(arg);
    } else {
      vargs.push_back(GVTOP(arg));
    }
    
    i += 1;
  }
  
  va_list vargs_list = reinterpret_cast<va_list>(vargs.data());

  GV.PointerVal = PyUnicode_FromFormatV(format, vargs_list);
  return GV;
}

int llvm_module_traverse(PyObject *self, visitproc visit, void *arg) {
    return 0;
}

// PyObject* PyModuleDef_Init(struct PyModuleDef* def)
static GenericValue lle_X_PyModuleDef_Init(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  struct PyModuleDef *def = (struct PyModuleDef *)GVTOP(args[0]);
  def->m_traverse = NULL;
  def->m_clear = NULL;
    
  GV.PointerVal = PyModuleDef_Init(def);
  return GV;
}

// PyObject *PyModule_Create2(PyModuleDef *def, int module_api_version)
static GenericValue lle_X_PyModule_Create2(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  struct PyModuleDef *def = (struct PyModuleDef *)GVTOP(args[0]);
  def->m_traverse = NULL;
  def->m_clear = NULL;
  int module_api_version = (int)args[1].IntVal.bitsToDouble();
    
  GV.PointerVal = PyModule_Create2(def, module_api_version);
  return GV;
}

// int PyObject_SetAttrString(PyObject *v, const char *name, PyObject *w)
static GenericValue lle_X_PyObject_SetAttrString(FunctionType *FT,
                                 ArrayRef<GenericValue> args) {
  
  GenericValue GV;

  PyObject *v = (PyObject *)GVTOP(args[0]);
  const char *name = (const char *)GVTOP(args[1]);
  PyObject *w = (PyObject *)GVTOP(args[2]);
  
  llvm_remove_traverse_from_cython_type(w);
  GV.IntVal = PyObject_SetAttrString(v, name, w);
  return GV;
}

// TODO: Define PyObject_GC_UnTrack in cext_glue.c

#endif //  (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)

void Interpreter::initializeExternalFunctions() {
  sys::ScopedLock Writer(*FunctionsLock);
  (*FuncNames)["lle_X_atexit"]       = lle_X_atexit;
  (*FuncNames)["lle_X_exit"]         = lle_X_exit;
  (*FuncNames)["lle_X_abort"]        = lle_X_abort;

  (*FuncNames)["lle_X_printf"]       = lle_X_printf;
  (*FuncNames)["lle_X_sprintf"]      = lle_X_sprintf;
  (*FuncNames)["lle_X_sscanf"]       = lle_X_sscanf;
  (*FuncNames)["lle_X_scanf"]        = lle_X_scanf;
  (*FuncNames)["lle_X_fprintf"]      = lle_X_fprintf;
  (*FuncNames)["lle_X_memset"]       = lle_X_memset;
  (*FuncNames)["lle_X_memcpy"]       = lle_X_memcpy;
#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
  // Variadic argument functions (vararg). 
  (*FuncNames)["lle_X_open"]         = lle_X_open;
  (*FuncNames)["lle_X__open"]        = lle_X_open;
  (*FuncNames)["lle_X_err"]          = lle_X_err;
  (*FuncNames)["lle_X_errx"]         = lle_X_errx;
  (*FuncNames)["lle_X_warn"]         = lle_X_warn;
  (*FuncNames)["lle_X_warnx"]        = lle_X_warnx;
  
  (*FuncNames)["lle_X_PyArg_ParseTuple"]                   = lle_X_PyArg_ParseTuple;
  (*FuncNames)["lle_X_PyArg_ParseTupleAndKeywords"]        = lle_X_PyArg_ParseTupleAndKeywords;
  (*FuncNames)["lle_X_PyArg_UnpackTuple"]                  = lle_X_PyArg_UnpackTuple;
  (*FuncNames)["lle_X_Py_BuildValue"]                      = lle_X_Py_BuildValue;
  (*FuncNames)["lle_X_PyOS_snprintf"]                      = lle_X_PyOS_snprintf;
  (*FuncNames)["lle_X_PyTuple_Pack"]                       = lle_X_PyTuple_Pack;
  (*FuncNames)["lle_X_PyCode_NewWithPosOnlyArgs"]          = lle_X_PyCode_NewWithPosOnlyArgs;
  (*FuncNames)["lle_X__PyArg_ParseTupleAndKeywords_SizeT"] = lle_X__PyArg_ParseTupleAndKeywords_SizeT;
  (*FuncNames)["lle_X_PyUnicode_FromFormat"]               = lle_X_PyUnicode_FromFormat;
  (*FuncNames)["lle_X_PyModuleDef_Init"]                   = lle_X_PyModuleDef_Init;
  (*FuncNames)["lle_X_PyModule_Create2"]                   = lle_X_PyModule_Create2;
  (*FuncNames)["lle_X_PyObject_SetAttrString"]             = lle_X_PyObject_SetAttrString;
    
#endif
}
