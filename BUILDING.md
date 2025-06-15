# Building

As I'm writting this I tested this with iOS + Simulator, watchOS + Simulator, tvOS + Simulator and Mac Catalyst.

This is the code you can use to compile with [multibuild](https://github.com/ColdGrub1384/multibuild).

```swift
Project(
    directoryURL: rootURL.appendingPathComponent("llvm-project/llvm"),
    version: .custom("19.0.0"),
    dependencies: [.name("ios_system")],
    builder: CMake(
        products: [
            // libLLVM
            .dynamicLibrary("lib/libLLVM.dylib", includePath: "../../../llvm/include"),

            // libclang-cpp + clang
            .dynamicLibrary(
                objectFiles: readProducts("libclang-cpp")+[
                "tools/clang/tools/driver/CMakeFiles/clang.dir/driver.cpp.o",
                "tools/clang/tools/driver/CMakeFiles/clang.dir/cc1_main.cpp.o",
                "tools/clang/tools/driver/CMakeFiles/clang.dir/cc1as_main.cpp.o",
                "tools/clang/tools/driver/CMakeFiles/clang.dir/cc1gen_reproducer_main.cpp.o",
                "tools/clang/tools/driver/CMakeFiles/clang.dir/clang-driver.cpp.o",
            ],
                binaryName: "clang",
                includePath: "../../../clang/include",
                additionalLinkerFlags: toolFlags),

            // lli
            .dynamicLibrary(
                staticArchives: ["lib/libLLVMOrcDebugging.a"],
                objectFiles: ["tools/lli/CMakeFiles/lli.dir/lli.cpp.o"],
                binaryName: "lli",
                additionalLinkerFlags: toolFlags),

            // opt
            .dynamicLibrary(
                objectFiles: [
                    "tools/opt/CMakeFiles/opt.dir/NewPMDriver.cpp.o",
                    "tools/opt/CMakeFiles/opt.dir/opt.cpp.o"
                ],
                binaryName: "opt",
                additionalLinkerFlags: toolFlags),

            // nm
            .dynamicLibrary(
                objectFiles: ["tools/llvm-nm/CMakeFiles/llvm-nm.dir/llvm-nm.cpp.o"],
                binaryName: "nm",
                additionalLinkerFlags: toolFlags),

            // dis
            .dynamicLibrary(
                objectFiles: ["tools/llvm-dis/CMakeFiles/llvm-dis.dir/llvm-dis.cpp.o"],
                binaryName: "dis",
                additionalLinkerFlags: toolFlags),

            // link
            .dynamicLibrary(
                objectFiles: ["tools/llvm-link/CMakeFiles/llvm-link.dir/llvm-link.cpp.o"],
                binaryName: "link",
                additionalLinkerFlags: toolFlags),

            // llc
            .dynamicLibrary(
                objectFiles: [
                    "tools/llc/CMakeFiles/llc.dir/NewPMDriver.cpp.o",
                    "tools/llc/CMakeFiles/llc.dir/llc.cpp.o"
                ],
                binaryName: "llc",
                additionalLinkerFlags: toolFlags),

            // lld
            .dynamicLibrary(
                staticArchives: [
                    "lib/liblldWasm.a",
                    "lib/liblldCOFF.a",
                    "lib/liblldCommon.a",
                    "lib/liblldMachO.a",
                    "lib/liblldMinGW.a",
                    "lib/liblldELF.a"
                ],
                objectFiles: [
                    "tools/lld/tools/lld/CMakeFiles/lld.dir/lld.cpp.o",
                ],
                binaryName: "lld",
                additionalLinkerFlags: toolFlags),

            // ar
            .dynamicLibrary(
                objectFiles: ["tools/llvm-ar/CMakeFiles/llvm-ar.dir/llvm-ar.cpp.o"],
                binaryName: "ar",
                additionalLinkerFlags: toolFlags),
        ],
        generator: .ninja,
        options: { target in
            let osxBuildDir = rootURL.appendingPathComponent("llvm-project/llvm/build/macosx")
            let ios_system = self.build(for: "ios_system")!.buildDirectoryURL(for: target)!
            let ios_systemFlags = "-F'\(ios_system.path)' -framework ios_system"
            let targetArch: String
            switch target.architectures.first! {
                case .arm64:
                    targetArch = "AArch64"
                case .x86_64:
                    targetArch = "X86"
                default:
                    targetArch = target.architectures.first!.rawValue
            }

            var opts =  [
                "LLVM_ENABLE_ZSTD": "OFF",
                "LLVM_LINK_LLVM_DYLIB": "ON",
                "LLVM_ENABLE_PROJECTS": "clang;lld;compiler-rt",
                "LLVM_TARGET_ARCH": targetArch,
                "LLVM_TARGETS_TO_BUILD": "AArch64;X86;WebAssembly",
                "LLVM_DEFAULT_TARGET_TRIPLE": "\(target.architectures.first!.rawValue)-apple-darwin",
                "CMAKE_BUILD_TYPE": "Release",
                "LLVM_OPTIMIZED_TABLEGEN": "ON",
                "LLVM_ENABLE_THREADS": "OFF",
                "LLVM_ENABLE_TERMINFO": "OFF",
                "LLVM_ENABLE_BACKTRACES": "OFF",
                "LLVM_ENABLE_LIBCXX": "OFF",
                "LLVM_ENABLE_LIBEDIT": "OFF",
                "CMAKE_CROSSCOMPILING": "ON",
                "LLVM_INCLUDE_BENCHMARKS": "OFF",
                "LLVM_USE_HOST_TOOLS": "ON",
                "LLVM_NATIVE_TOOL_DIR": osxBuildDir.appendingPathComponent("bin").path,
                "LLVM_TABLEGEN": osxBuildDir.appendingPathComponent("bin/llvm-tblgen").path,
                "CLANG_TABLEGEN": osxBuildDir.appendingPathComponent("bin/clang-tblgen").path,
                "CMAKE_OSX_SYSROOT": target.sdkURL?.path ?? "",
                "CMAKE_C_COMPILER": osxBuildDir.appendingPathComponent("bin/clang").path,
                "CMAKE_CXX_COMPILER": osxBuildDir.appendingPathComponent("bin/clang++").path,
                "CMAKE_LIBRARY_PATH": osxBuildDir.appendingPathComponent("lib").path,
                "CMAKE_INCLUDE_PATH": osxBuildDir.appendingPathComponent("include").path,
                "COMPILER_RT_DEFAULT_TARGET_TRIPLE": "\(target.architectures.first!.rawValue)-apple-darwin",
                "COMPILER_RT_BUILD_BUILTINS": "ON",
                "COMPILER_RT_BUILD_LIBFUZZER": "OFF",
                "COMPILER_RT_BUILD_MEMPROF": "OFF",
                "COMPILER_RT_BUILD_PROFILE": "OFF",
                "COMPILER_RT_BUILD_SANITIZERS": "OFF",
                "COMPILER_RT_BUILD_XRAY": "OFF",
                "LIBOMP_LDFLAGS": "-L lib/clang/14.0.0/lib/darwin -lclang_rt.cc_kext_ios",
                "CMAKE_C_FLAGS": "\(ios_systemFlags) -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS  -I\"\(osxBuildDir.appendingPathComponent("include").path)\"  -I\"\(osxBuildDir.appendingPathComponent("include/c++/v1").path)\" -I\"\(rootURL.appendingPathComponent("llvm-project").path)\"",
                "CMAKE_CXX_FLAGS": "\(ios_systemFlags) -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS -I\"\(osxBuildDir.appendingPathComponent("include").path)\"  -I\"\(rootURL.appendingPathComponent("llvm-project").path)\"",
                "CMAKE_MODULE_LINKER_FLAGS": "-nostdlib \(ios_systemFlags) -lobjc -lc -lc++",
                "CMAKE_SHARED_LINKER_FLAGS": "-nostdlib \(ios_systemFlags) -lobjc -lc -lc++",
                "CMAKE_EXE_LINKER_FLAGS": "-nostdlib \(ios_systemFlags) -lobjc -lc -lc++",
            ]

            if target.systemName == .watchos || target.systemName == .watchsimulator || target.systemName == .appletvos || target.systemName == .appletvsimulator {
                opts["LLVM_ENABLE_CRASH_OVERRIDES"] = "OFF"
            }

            return opts
        })
).willBuild { target in
    //
    // You must build clang for your mac first here, under build/macosx
    //
    // You can use https://github.com/holzschu/llvm-project/blob/main/bootstrap.sh as a reference
    ...
}.didBuild { target, _ in
    // This is like literally translated from https://github.com/holzschu/llvm-project/blob/main/bootstrap.sh

    guard let buildDir = self.build(for: "llvm")?.buildDirectoryURL(for: target) else {
        return
    }
    try? FileManager.default.removeItem(at: buildDir.appendingPathComponent("lib/liblli.a"))
    try? FileManager.default.removeItem(at: buildDir.appendingPathComponent("lib/libllc.a"))
    try? FileManager.default.removeItem(at: buildDir.appendingPathComponent("lib/libclang_tool.a"))
    try? FileManager.default.removeItem(at: buildDir.appendingPathComponent("lib/libopt.a"))

    for args in [
        "-r lib/libclang_tool.a tools/clang/tools/driver/CMakeFiles/clang.dir/driver.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1as_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1gen_reproducer_main.cpp.o".components(separatedBy: " "),
        "-r lib/libopt.a tools/opt/CMakeFiles/opt.dir/NewPMDriver.cpp.o tools/opt/CMakeFiles/opt.dir/opt.cpp.o".components(separatedBy: " ")
    ] {
        let ar = Process()
        ar.executableURL = URL(fileURLWithPath: "/usr/bin/ar")
        ar.arguments = args
        ar.launch()
        ar.waitUntilExit()
    }
}

```
