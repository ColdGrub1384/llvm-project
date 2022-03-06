//===- lli.cpp - LLVM Interpreter / Dynamic compiler ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility provides a simple wrapper around the LLVM Execution Engines,
// which allow the direct execution of LLVM programs through a Just-In-Time
// compiler, or through an interpreter if no JIT is available for this platform.
//
//===----------------------------------------------------------------------===//

#include "ExecutionUtils.h"
#include "ForwardingMemoryManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCEHFrameRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericRTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include <cerrno>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

#ifdef __CYGWIN__
#include <cygwin/version.h>
#if defined(CYGWIN_VERSION_DLL_MAJOR) && CYGWIN_VERSION_DLL_MAJOR<1007
#define DO_NOTHING_ATEXIT 1
#endif
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
#include "ios_error.h"
#undef write
#include <stdio.h>
#undef exit
#define exit(a) { llvm_shutdown(); ios_exit(a); }
extern "C" {
extern const char* llvm_ios_progname;

void llvm_ios_exit(int a) { llvm::llvm_shutdown(); ios_exit(a); }
void llvm_ios_abort(int a) { llvm::report_fatal_error("LLVM JIT compiled program raised SIGABRT"); }
int llvm_ios_putchar(char c) { return fputc(c, thread_stdout); }
int llvm_ios_getchar(void) { return fgetc(thread_stdin); }
wint_t llvm_ios_getwchar(void) { return fgetwc(thread_stdin); }
int llvm_ios_iswprint(wint_t a) { return 1; }
int llvm_ios_scanf (const char *format, ...) {
    int             count;
    va_list ap;
    
    fflush(thread_stdout);
    va_start (ap, format);
    count = vfscanf (thread_stdin, format, ap);
    va_end (ap);
    return (count);
}
int llvm_ios_fputc(int c, FILE *stream) {
	if (fileno(stream) == STDOUT_FILENO) return fputc(c, thread_stdout);
	if (fileno(stream) == STDERR_FILENO) return fputc(c, thread_stderr);
	return fputc(c, stream);
}
int llvm_ios_putw(int w, FILE *stream) {
	if (fileno(stream) == STDOUT_FILENO) return putw(w, thread_stdout);
	if (fileno(stream) == STDERR_FILENO) return putw(w, thread_stderr);
	return putw(w, stream);
}
}
#endif
#endif

using namespace llvm;

static codegen::RegisterCodeGenFlags CGF;

#define DEBUG_TYPE "lli"

namespace {

  enum class JITKind { MCJIT, Orc, OrcLazy };
  enum class JITLinkerKind { Default, RuntimeDyld, JITLink };

  enum class DumpKind {
    NoDump,
    DumpFuncsToStdOut,
    DumpModsToStdOut,
    DumpModsToDisk
  };

  ExitOnError ExitOnErr;
}

LLVM_ATTRIBUTE_USED void linkComponents() {
  errs() << (void *)&llvm_orc_registerEHFrameSectionWrapper
         << (void *)&llvm_orc_deregisterEHFrameSectionWrapper
         << (void *)&llvm_orc_registerJITLoaderGDBWrapper;
}

//===----------------------------------------------------------------------===//
// Object cache
//
// This object cache implementation writes cached objects to disk to the
// directory specified by CacheDir, using a filename provided in the module
// descriptor. The cache tries to load a saved object using that path if the
// file exists. CacheDir defaults to "", in which case objects are cached
// alongside their originating bitcodes.
//
class LLIObjectCache : public ObjectCache {
public:
  LLIObjectCache(const std::string& CacheDir) : CacheDir(CacheDir) {
    // Add trailing '/' to cache dir if necessary.
    if (!this->CacheDir.empty() &&
        this->CacheDir[this->CacheDir.size() - 1] != '/')
      this->CacheDir += '/';
  }
  ~LLIObjectCache() override {}

  void notifyObjectCompiled(const Module *M, MemoryBufferRef Obj) override {
    const std::string &ModuleID = M->getModuleIdentifier();
    std::string CacheName;
    if (!getCacheFilename(ModuleID, CacheName))
      return;
    if (!CacheDir.empty()) { // Create user-defined cache dir.
      SmallString<128> dir(sys::path::parent_path(CacheName));
      sys::fs::create_directories(Twine(dir));
    }

    std::error_code EC;
    raw_fd_ostream outfile(CacheName, EC, sys::fs::OF_None);
    outfile.write(Obj.getBufferStart(), Obj.getBufferSize());
    outfile.close();
  }

  std::unique_ptr<MemoryBuffer> getObject(const Module* M) override {
    const std::string &ModuleID = M->getModuleIdentifier();
    std::string CacheName;
    if (!getCacheFilename(ModuleID, CacheName))
      return nullptr;
    // Load the object from the cache filename
    ErrorOr<std::unique_ptr<MemoryBuffer>> IRObjectBuffer =
        MemoryBuffer::getFile(CacheName, /*IsText=*/false,
                              /*RequiresNullTerminator=*/false);
    // If the file isn't there, that's OK.
    if (!IRObjectBuffer)
      return nullptr;
    // MCJIT will want to write into this buffer, and we don't want that
    // because the file has probably just been mmapped.  Instead we make
    // a copy.  The filed-based buffer will be released when it goes
    // out of scope.
    return MemoryBuffer::getMemBufferCopy(IRObjectBuffer.get()->getBuffer());
  }

private:
  std::string CacheDir;

  bool getCacheFilename(const std::string &ModID, std::string &CacheName) {
    std::string Prefix("file:");
    size_t PrefixLength = Prefix.length();
    if (ModID.substr(0, PrefixLength) != Prefix)
      return false;

    std::string CacheSubdir = ModID.substr(PrefixLength);
    // Transform "X:\foo" => "/X\foo" for convenience on Windows.
    if (is_style_windows(llvm::sys::path::Style::native) &&
        isalpha(CacheSubdir[0]) && CacheSubdir[1] == ':') {
      CacheSubdir[1] = CacheSubdir[0];
      CacheSubdir[0] = '/';
    }

    CacheName = CacheDir + CacheSubdir;
    size_t pos = CacheName.rfind('.');
    CacheName.replace(pos, CacheName.length() - pos, ".o");
    return true;
  }
};

CodeGenOpt::Level getOptLevel() {
  return CodeGenOpt::None;
  llvm_unreachable("Unrecognized opt level.");
}

[[noreturn]] static void reportError(SMDiagnostic Err, const char *ProgName) {
  Err.print(ProgName, errs());
  exit(1);
}

//Error loadDylibs();
int runOrcJIT(const char *ProgName);
void disallowOrcOptions();
Expected<std::unique_ptr<orc::ExecutorProcessControl>> launchRemote();

extern "C" {
//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char **argv, char * const *envp) {
  InitLLVM X(argc, argv);
  
  if (argc > 1) {
    ExitOnErr.setBanner(std::string(argv[0]) + ": ");
  }
  
  // If we have a native target, initialize it to ensure it is linked in and
  // usable by the JIT.
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  
  disallowOrcOptions();
  
  // Old lli implementation based on ExecutionEngine and MCJIT.
  LLVMContext Context;

  // Load the bitcode...
  SMDiagnostic Err;
  std::string IRFile(argv[1]);
  std::unique_ptr<Module> Owner = parseIRFile(IRFile, Err, Context);
  Module *Mod = Owner.get();
  if (!Mod) {
    reportError(Err, argv[0]);
  }
  
  std::string ErrorMsg;
  EngineBuilder builder(std::move(Owner));
  builder.setMArch(codegen::getMArch());
  builder.setMCPU(codegen::getCPUStr());
  builder.setMAttrs(codegen::getFeatureList());
  if (auto RM = codegen::getExplicitRelocModel())
    builder.setRelocationModel(RM.getValue());
  if (auto CM = codegen::getExplicitCodeModel())
    builder.setCodeModel(CM.getValue());
  builder.setErrorStr(&ErrorMsg);
  builder.setEngineKind(EngineKind::Interpreter);
      
  // Enable MCJIT if desired.
  RTDyldMemoryManager *RTDyldMM = nullptr;
  
  builder.setOptLevel(getOptLevel());
#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
    // For ios_system, add symbols that override the existing ones:
    // This needs to be done *before* the engine creation:
    // This way, we act on both interpreter and JIT:
    sys::DynamicLibrary::AddSymbol("stdin", &thread_stdin);
    sys::DynamicLibrary::AddSymbol("stdout", &thread_stdout);
    sys::DynamicLibrary::AddSymbol("stderr", &thread_stderr);
    sys::DynamicLibrary::AddSymbol("__stdinp", &thread_stdin);
    sys::DynamicLibrary::AddSymbol("__stdoutp", &thread_stdout);
    sys::DynamicLibrary::AddSymbol("__stderrp", &thread_stderr);
    // External functions defined in ios_system:
    sys::DynamicLibrary::AddSymbol("system", (void*)&ios_system);
    sys::DynamicLibrary::AddSymbol("popen", (void*)&ios_popen);
    sys::DynamicLibrary::AddSymbol("pclose", (void*)&fclose);
    sys::DynamicLibrary::AddSymbol("isatty", (void*)&ios_isatty);
    sys::DynamicLibrary::AddSymbol("dup2", (void*)&ios_dup2);
    sys::DynamicLibrary::AddSymbol("execv", (void*)&ios_execv);
    sys::DynamicLibrary::AddSymbol("execvp", (void*)&ios_execv);
    sys::DynamicLibrary::AddSymbol("execve", (void*)&ios_execve);
    // External functions defined locally:
    sys::DynamicLibrary::AddSymbol("exit", (void*)&ios_exit);
    sys::DynamicLibrary::AddSymbol("_exit", (void*)&ios_exit);
    // sys::DynamicLibrary::AddSymbol("abort", (void*)&llvm_ios_abort);
    sys::DynamicLibrary::AddSymbol("putchar", (void*)&llvm_ios_putchar);
    sys::DynamicLibrary::AddSymbol("getchar", (void*)&llvm_ios_getchar);
    sys::DynamicLibrary::AddSymbol("getwchar", (void*)&llvm_ios_getwchar);
    sys::DynamicLibrary::AddSymbol("iswprint", (void*)&llvm_ios_iswprint);
    // scanf, printf, write: redirect to right stream
    // printf, fprintf: already redirected to lle_X_printf in ExternalFunctions.cpp
    sys::DynamicLibrary::AddSymbol("scanf", (void*)&llvm_ios_scanf);
    sys::DynamicLibrary::AddSymbol("write", (void*)&ios_write);
    sys::DynamicLibrary::AddSymbol("puts", (void*)&ios_puts);
    sys::DynamicLibrary::AddSymbol("fputs", (void*)&ios_fputs);
    sys::DynamicLibrary::AddSymbol("fputc", (void*)&ios_fputc);
    sys::DynamicLibrary::AddSymbol("putw", (void*)&ios_putw);
    // fork, waitpid: minimal service here:
    sys::DynamicLibrary::AddSymbol("fork", (void*)&ios_fork);
    sys::DynamicLibrary::AddSymbol("waitpid", (void*)&ios_waitpid);
    // err, errx, warnx, warn:  already redirected to lle_X_printf in ExternalFunctions.cpp
    llvm_ios_progname = argv[1];
#endif
  
  std::unique_ptr<ExecutionEngine> EE(builder.create());
  if (!EE) {
    if (!ErrorMsg.empty())
      WithColor::error(errs(), argv[0])
          << "error creating EE: " << ErrorMsg << "\n";
    else
      WithColor::error(errs(), argv[0]) << "unknown error creating EE!\n";
    exit(1);
  }

  std::unique_ptr<LLIObjectCache> CacheManager;

  // The following functions have no effect if their respective profiling
  // support wasn't enabled in the build configuration.
  EE->RegisterJITEventListener(
                JITEventListener::createOProfileJITEventListener());
  EE->RegisterJITEventListener(
                JITEventListener::createIntelJITEventListener());

  // Add the module's name to the start of the vector of arguments to main().
  //InputArgv.insert(InputArgv.begin(), StringRef(argv[1]));

  // Call the main function from M as if its signature were:
  //   int main (int argc, char **argv, const char **envp)
  // using the contents of Args to determine argc & argv, and the contents of
  // EnvVars to determine envp.
  //
  std::string EntryFunc("main");
  Function *EntryFn = Mod->getFunction(EntryFunc);
  if (!EntryFn) {
    WithColor::error(errs(), argv[0])
        << '\'' << EntryFunc << "\' function not found in module.\n";
    return -1;
  }

  // Reset errno to zero on entry to main.
  errno = 0;

  int Result = -1;

  // If the program doesn't explicitly call exit, we will need the Exit
  // function later on to make an explicit call, so get the function now.

#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
  // on iOS, normally, ForceInterpreter = true, but if your run the JIT you need this:
  FunctionCallee Exit;
  Exit = Mod->getOrInsertFunction(
      "exit", Type::getVoidTy(Context), Type::getInt32Ty(Context));
#else
  FunctionCallee Exit = Mod->getOrInsertFunction(
      "exit", Type::getVoidTy(Context), Type::getInt32Ty(Context));
#endif
  // Run static constructors.
  EE->runStaticConstructorsDestructors(false);

  // Trigger compilation separately so code regions that need to be
  // invalidated will be known.
  (void)EE->getPointerToFunction(EntryFn);
  // Clear instruction cache before code will be executed.
  if (RTDyldMM)
    static_cast<SectionMemoryManager*>(RTDyldMM)->invalidateInstructionCache();

  // Run main.
  Result = EE->runFunctionAsMainWithoutParams(EntryFn, envp);
  
  // Run static destructors.
  EE->runStaticConstructorsDestructors(true);
  
  // If the program didn't call exit explicitly, we should call it now.
  // This ensures that any atexit handlers get called correctly.
  if (Function *ExitF =
          dyn_cast<Function>(Exit.getCallee()->stripPointerCasts())) {
    if (ExitF->getFunctionType() == Exit.getFunctionType()) {
      std::vector<GenericValue> Args;
      GenericValue ResultGV;
      ResultGV.IntVal = APInt(32, Result);
      Args.push_back(ResultGV);
      EE->runFunction(ExitF, Args);
      WithColor::error(errs(), argv[0])
          << "exit(" << Result << ") returned!\n";
      abort();
    }
  }
  WithColor::error(errs(), argv[0]) << "exit defined with wrong prototype!\n";
  abort();
  
  return Result;
}
}

Expected<orc::ThreadSafeModule>
loadModule(StringRef Path, orc::ThreadSafeContext TSCtx) {
  SMDiagnostic Err;
  auto M = parseIRFile(Path, Err, *TSCtx.getContext());
  if (!M) {
    std::string ErrMsg;
    {
      raw_string_ostream ErrMsgStream(ErrMsg);
      Err.print("lli", ErrMsgStream);
    }
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());
  }

  return orc::ThreadSafeModule(std::move(M), std::move(TSCtx));
}

void disallowOrcOptions() {

}
