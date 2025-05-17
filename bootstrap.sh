#! /bin/bash
# With contributions from Ian McDowell: https://github.com/IMcD23

# Bail out on error
set -e

LLVM_SRCDIR=$(pwd)
OSX_BUILDDIR=$(pwd)/build_osx
IOS_BUILDDIR=$(pwd)/build-iphoneos-arm64
SIM_BUILDDIR=$(pwd)/build-iphonesimulator-x86_64
SIM_BUILDDIR_ARM64=$(pwd)/build-iphonesimulator-arm64
MACCATALYST_BUILDDIR=$(pwd)/build-maccatalyst-x86_64
MACCATALYST_BUILDDIR_ARM64=$(pwd)/build-maccatalyst-arm64

echo "Downloading ios_system Framework:"
IOS_SYSTEM_VER="v3.0.2"
HHROOT="https://github.com/holzschu"

echo "Downloading header file:"
curl -OL $HHROOT/ios_system/releases/download/$IOS_SYSTEM_VER/ios_error.h 

echo "Downloading ios_system Framework:"
rm -rf ios_system.xcframework
curl -OL $HHROOT/ios_system/releases/download/$IOS_SYSTEM_VER/ios_system.xcframework.zip
unzip ios_system.xcframework.zip
rm ios_system.xcframework.zip

OSX_SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
IOS_SDKROOT=$(xcrun --sdk iphoneos --show-sdk-path)
SIM_SDKROOT=$(xcrun --sdk iphonesimulator --show-sdk-path)

# Parse arguments
for i in "$@"
do
case $i in
  -c|--clean)
    CLEAN=YES
  shift
  ;;
  *)
    # unknown option
  ;;
esac
done

# compile for OSX (about 1h, 1GB of disk space)
echo "Compiling for OSX:"
if [ $CLEAN ]; then
  rm -rf $OSX_BUILDDIR
fi
if [ ! -d $OSX_BUILDDIR ]; then
  mkdir $OSX_BUILDDIR
fi
# building with -DLLVM_LINK_LLVM_DYLIB (= single big shared lib)
# Easier to make a framework with
# libc; impossible to configure
pushd $OSX_BUILDDIR
cmake -G Ninja \
-DLLVM_TARGETS_TO_BUILD="AArch64;X86;WebAssembly" \
-DLLVM_ENABLE_PROJECTS='clang;compiler-rt;lld;openmp' \
-DLLVM_LINK_LLVM_DYLIB=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_OSX_SYSROOT=${OSX_SDKROOT} \
-DCMAKE_C_COMPILER=$(xcrun --sdk macosx -f clang) \
-DCMAKE_CXX_COMPILER=$(xcrun --sdk macosx -f clang++) \
-DCMAKE_ASM_COMPILER=$(xcrun --sdk macosx -f cc) \
-DCMAKE_LIBRARY_PATH=${OSX_SDKROOT}/lib/ \
-DCMAKE_INCLUDE_PATH=${OSX_SDKROOT}/include/ \
../llvm
ninja
popd
# libtool: where? /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/libtool

# Now, compile for iOS using the previous build:
# About 1h, 12 GB of disk space
# -DLLVM_ENABLE_THREADS=OFF is necessary to run commands multiple times
# -I${OSX_BUILDDIR}/include/c++/v1/
# Try to reduce inlining (doesn't work at compile time)
#  -D_LIBCPP_INLINE_VISIBILITY=\"\" -D_LIBCPP_ALWAYS_INLINE=\"\" -D_LIBCPP_EXTERN_TEMPLATE_INLINE_VISIBILITY=\"\"
echo "Compiling for iOS:"
if [ $CLEAN ]; then
  rm -rf $IOS_BUILDDIR
fi
if [ ! -d $IOS_BUILDDIR ]; then
  mkdir $IOS_BUILDDIR
fi
# libc; impossible to configure
# compiler-rt; tries to set macosx-version-min
# openmp: requires compiler-rt and?
# 2023/03/09: try with compiler-rt, not openmp. ;openmp
# 2024/05: fails with openmp. Try without.
# flang: issue with mlir-tblgen, also not cross-compiling.
pushd $IOS_BUILDDIR
cmake -G Ninja \
-DLLVM_ENABLE_ZSTD=OFF \
-DLLVM_LINK_LLVM_DYLIB=ON \
-DLLVM_TARGET_ARCH=AArch64 \
-DLLVM_TARGETS_TO_BUILD="AArch64;X86;WebAssembly" \
-DLLVM_ENABLE_PROJECTS='clang;lld;compiler-rt' \
-DLLVM_DEFAULT_TARGET_TRIPLE=arm64-apple-darwin \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_THREADS=OFF \
-DLLVM_ENABLE_TERMINFO=OFF \
-DLLVM_ENABLE_BACKTRACES=OFF \
-DLLVM_ENABLE_LIBCXX=OFF \
-DLLVM_ENABLE_LIBEDIT=OFF \
-DCMAKE_CROSSCOMPILING=TRUE \
-DLLVM_TABLEGEN=${OSX_BUILDDIR}/bin/llvm-tblgen \
-DCLANG_TABLEGEN=${OSX_BUILDDIR}/bin/clang-tblgen \
-DCMAKE_OSX_SYSROOT=${IOS_SDKROOT} \
-DCMAKE_C_COMPILER=${OSX_BUILDDIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${OSX_BUILDDIR}/bin/clang++ \
-DCMAKE_LIBRARY_PATH=${OSX_BUILDDIR}/lib/ \
-DCMAKE_INCLUDE_PATH=${OSX_BUILDDIR}/include/ \
-DCOMPILER_RT_BUILD_BUILTINS=ON \
-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
-DCOMPILER_RT_BUILD_MEMPROF=OFF \
-DCOMPILER_RT_BUILD_PROFILE=OFF \
-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
-DCOMPILER_RT_BUILD_XRAY=OFF \
-DLIBOMP_LDFLAGS="-L lib/clang/14.0.0/lib/darwin -lclang_rt.cc_kext_ios" \
-DCMAKE_C_FLAGS="-arch arm64 -target arm64-apple-darwin19.6.0 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS  -I${OSX_BUILDDIR}/include/ -I${OSX_BUILDDIR}/include/c++/v1/ -I${LLVM_SRCDIR} -miphoneos-version-min=14  " \
-DCMAKE_CXX_FLAGS="-arch arm64 -target arm64-apple-darwin19.6.0 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS -I${OSX_BUILDDIR}/include/  -I${LLVM_SRCDIR} -miphoneos-version-min=14 " \
-DCMAKE_MODULE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64 -O2 -framework ios_system -lobjc -lc -lc++ -miphoneos-version-min=14 " \
-DCMAKE_SHARED_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64 -O2 -framework ios_system -lobjc -lc -lc++ -miphoneos-version-min=14 " \
-DCMAKE_EXE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64 -O2 -framework ios_system -lobjc -lc -lc++ -miphoneos-version-min=14 " \
../llvm
ninja
# We could add X86 to target architectures, but that increases the app size too much
# Now build the static libraries for the executables:
# -stdlib=libc++: not required with OSX > Mavericks
# -nostdlib: so ios_system is linked *before* libc and libc++
# try with: -fvisibility=hidden -fvisibility-inlines-hidden in CFLAGS for the warning
# -L lib = crashes every time (self-reference).
# lli crashes, but only lli. When creating main() (before the first line)
rm -f lib/liblli.a
rm -f lib/libllc.a
# Xcode gets confused if a static and a dynamic library share the same name:
rm -f lib/libclang_tool.a
rm -f lib/libopt.a
ar -r lib/libclang_tool.a tools/clang/tools/driver/CMakeFiles/clang.dir/driver.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1as_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1gen_reproducer_main.cpp.o
ar -r lib/libopt.a tools/opt/CMakeFiles/opt.dir/NewPMDriver.cpp.o  tools/opt/CMakeFiles/opt.dir/opt.cpp.o
# No need to make static libraries for these:
# lli: tools/lli/CMakeFiles/lli.dir/lli.cpp.o
# llvm-link: tools/llvm-link/CMakeFiles/llvm-link.dir/llvm-link.cpp.o
# llvm-nm:  tools/llvm-nm/CMakeFiles/llvm-nm.dir/llvm-nm.cpp.o
# llvm-ar:  tools/llvm-ar/CMakeFiles/llvm-ar.dir/llvm-ar.cpp.o
# llvm-dis:  tools/llvm-dis/CMakeFiles/llvm-dis.dir/llvm-dis.cpp.o
# llc: tools/llc/CMakeFiles/llc.dir/llc.cpp.o
# lld, wasm-ld, etc: done in Xcode.
rm -rf frameworks.xcodeproj
cp -r ../frameworks/frameworks.xcodeproj .
# And then build the frameworks from these static libraries:
# Somehow, -alltargets does not build all targets.
xcodebuild -project frameworks.xcodeproj -target libLLVM -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target ar -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target clang -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target opt -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target nm -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target dis -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target link -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target lld -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target lli -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target llc -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target clang-c -sdk iphoneos -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target llvm-c -sdk iphoneos -configuration Release -quiet
popd

# Now, build for the simulator:
echo "Compiling for the simulator (x86_64):"
if [ $CLEAN ]; then
  rm -rf $SIM_BUILDDIR
fi
if [ ! -d $SIM_BUILDDIR ]; then
  mkdir $SIM_BUILDDIR
fi
pushd $SIM_BUILDDIR
cmake -G Ninja \
-DLLVM_ENABLE_ZSTD=OFF \
-DLLVM_LINK_LLVM_DYLIB=ON \
-DLLVM_TARGET_ARCH=X86 \
-DLLVM_TARGETS_TO_BUILD="AArch64;X86;WebAssembly" \
-DLLVM_ENABLE_PROJECTS='clang;lld;compiler-rt' \
-DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-apple-darwin19.6.0 \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_THREADS=OFF \
-DLLVM_ENABLE_TERMINFO=OFF \
-DLLVM_ENABLE_BACKTRACES=OFF \
-DLLVM_ENABLE_LIBCXX=OFF \
-DLLVM_ENABLE_LIBEDIT=OFF \
-DCMAKE_CROSSCOMPILING=TRUE \
-DLLVM_TABLEGEN=${OSX_BUILDDIR}/bin/llvm-tblgen \
-DCLANG_TABLEGEN=${OSX_BUILDDIR}/bin/clang-tblgen \
-DCMAKE_OSX_SYSROOT=${SIM_SDKROOT} \
-DCMAKE_C_COMPILER=${OSX_BUILDDIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${OSX_BUILDDIR}/bin/clang++ \
-DCMAKE_LIBRARY_PATH=${OSX_BUILDDIR}/lib/ \
-DCMAKE_INCLUDE_PATH=${OSX_BUILDDIR}/include/ \
-DCOMPILER_RT_BUILD_BUILTINS=ON \
-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
-DCOMPILER_RT_BUILD_MEMPROF=OFF \
-DCOMPILER_RT_BUILD_PROFILE=OFF \
-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
-DCOMPILER_RT_BUILD_XRAY=OFF \
-DLIBOMP_LDFLAGS="-L lib/clang/14.0.0/lib/darwin -lclang_rt.cc_kext_ios" \
-DCMAKE_C_FLAGS="-target x86_64-apple-darwin19.6.0 -arch x86_64 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS  -I${OSX_BUILDDIR}/include/ -I${OSX_BUILDDIR}/include/c++/v1/ -I${LLVM_SRCDIR} -mios-simulator-version-min=14.0  " \
-DCMAKE_CXX_FLAGS="-target x86_64-apple-darwin19.6.0 -arch x86_64 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS -I${OSX_BUILDDIR}/include/  -I${LLVM_SRCDIR} -mios-simulator-version-min=14.0 " \
-DCMAKE_MODULE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-simulator -O2 -framework ios_system -lobjc -lc -lc++ -mios-simulator-version-min=14.0 " \
-DCMAKE_SHARED_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-simulator -O2 -framework ios_system -lobjc -lc -lc++ -mios-simulator-version-min=14.0 " \
-DCMAKE_EXE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-simulator -O2 -framework ios_system -lobjc -lc -lc++ -mios-simulator-version-min=14.0 " \
../llvm
ninja

# My ARM Mac generates fat binaries even with -arch x86_64, so we strip the unnecessary architectures
# It would be useful if we could right away compile for both architectures from any Mac but I don't know the behaviour of the compiler on Intel Macs
(find . -name "*.o" -exec lipo "{}" -remove arm64 -output "{}" \; || true) 2>/dev/null

# Now build the static libraries for the executables:
# -stdlib=libc++: not required with OSX > Mavericks
# -nostdlib: so ios_system is linked *before* libc and libc++
# try with: -fvisibility=hidden -fvisibility-inlines-hidden in CFLAGS for the warning
# -L lib = crashes every time (self-reference).
# lli crashes, but only lli. When creating main() (before the first line)
rm -f lib/liblli.a
rm -f lib/libllc.a
## Xcode gets confused if a static and a dynamic library share the same name:
rm -f lib/libclang_tool.a
rm -f lib/libopt.a
ar -r lib/libclang_tool.a tools/clang/tools/driver/CMakeFiles/clang.dir/driver.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1as_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1gen_reproducer_main.cpp.o
ar -r lib/libopt.a tools/opt/CMakeFiles/opt.dir/NewPMDriver.cpp.o tools/opt/CMakeFiles/opt.dir/opt.cpp.o
# No need to make static libraries for these:
# lli: tools/lli/CMakeFiles/lli.dir/lli.cpp.o
# llvm-link: tools/llvm-link/CMakeFiles/llvm-link.dir/llvm-link.cpp.o
# llvm-nm:  tools/llvm-nm/CMakeFiles/llvm-nm.dir/llvm-nm.cpp.o
# llvm-ar:  tools/llvm-ar/CMakeFiles/llvm-ar.dir/llvm-ar.cpp.o
# llvm-dis:  tools/llvm-dis/CMakeFiles/llvm-dis.dir/llvm-dis.cpp.o
# llc: tools/llc/CMakeFiles/llc.dir/llc.cpp.o
# lld, wasm-ld, etc: done in Xcode.
rm -rf frameworks.xcodeproj
cp -r ../frameworks/frameworks.xcodeproj .
# And then build the frameworks from these static libraries:
# Somehow, -alltargets does not build all targets.
xcodebuild -project frameworks.xcodeproj -target libLLVM -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target ar -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target clang -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target opt -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target nm -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target dis -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target link -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target lld -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target lli -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target llc -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target clang-c -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target llvm-c -sdk iphonesimulator -arch x86_64 -configuration Release -quiet
popd

echo "Compiling for the simulator (arm64):"
if [ $CLEAN ]; then
  rm -rf $SIM_BUILDDIR_ARM64
fi
if [ ! -d $SIM_BUILDDIR_ARM64 ]; then
  mkdir $SIM_BUILDDIR_ARM64
fi
pushd $SIM_BUILDDIR_ARM64
cmake -G Ninja \
-DLLVM_ENABLE_ZSTD=OFF \
-DLLVM_LINK_LLVM_DYLIB=ON \
-DLLVM_TARGET_ARCH=ARM64 \
-DLLVM_TARGETS_TO_BUILD="AArch64;X86;WebAssembly" \
-DLLVM_ENABLE_PROJECTS='clang;lld;compiler-rt' \
-DLLVM_DEFAULT_TARGET_TRIPLE=arm64-apple-darwin19.6.0 \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_THREADS=OFF \
-DLLVM_ENABLE_TERMINFO=OFF \
-DLLVM_ENABLE_BACKTRACES=OFF \
-DLLVM_ENABLE_LIBCXX=OFF \
-DLLVM_ENABLE_LIBEDIT=OFF \
-DCMAKE_CROSSCOMPILING=TRUE \
-DLLVM_TABLEGEN=${OSX_BUILDDIR}/bin/llvm-tblgen \
-DCLANG_TABLEGEN=${OSX_BUILDDIR}/bin/clang-tblgen \
-DCMAKE_OSX_SYSROOT=${SIM_SDKROOT} \
-DCMAKE_C_COMPILER=${OSX_BUILDDIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${OSX_BUILDDIR}/bin/clang++ \
-DCMAKE_LIBRARY_PATH=${OSX_BUILDDIR}/lib/ \
-DCMAKE_INCLUDE_PATH=${OSX_BUILDDIR}/include/ \
-DCOMPILER_RT_BUILD_BUILTINS=ON \
-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
-DCOMPILER_RT_BUILD_MEMPROF=OFF \
-DCOMPILER_RT_BUILD_PROFILE=OFF \
-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
-DCOMPILER_RT_BUILD_XRAY=OFF \
-DLIBOMP_LDFLAGS="-L lib/clang/14.0.0/lib/darwin -lclang_rt.cc_kext_ios" \
-DCMAKE_C_FLAGS="-target arm64-apple-darwin19.6.0 -arch arm64 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS  -I${OSX_BUILDDIR}/include/ -I${OSX_BUILDDIR}/include/c++/v1/ -I${LLVM_SRCDIR} -mios-simulator-version-min=14.0  " \
-DCMAKE_CXX_FLAGS="-target arm64-apple-darwin19.6.0 -arch arm64 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS -I${OSX_BUILDDIR}/include/  -I${LLVM_SRCDIR} -mios-simulator-version-min=14.0 " \
-DCMAKE_MODULE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-simulator -O2 -framework ios_system -lobjc -lc -lc++ -mios-simulator-version-min=14.0 " \
-DCMAKE_SHARED_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-simulator -O2 -framework ios_system -lobjc -lc -lc++ -mios-simulator-version-min=14.0 " \
-DCMAKE_EXE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-simulator -O2 -framework ios_system -lobjc -lc -lc++ -mios-simulator-version-min=14.0 " \
../llvm
ninja

# Also removing x86_64 here just in case
(find . -name "*.o" -exec lipo "{}" -remove x86_64 -output "{}" \; || true) 2>/dev/null

rm -f lib/liblli.a
rm -f lib/libllc.a

rm -f lib/libclang_tool.a
rm -f lib/libopt.a
ar -r lib/libclang_tool.a tools/clang/tools/driver/CMakeFiles/clang.dir/driver.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1as_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1gen_reproducer_main.cpp.o
ar -r lib/libopt.a tools/opt/CMakeFiles/opt.dir/NewPMDriver.cpp.o tools/opt/CMakeFiles/opt.dir/opt.cpp.o

rm -rf frameworks.xcodeproj
cp -r ../frameworks/frameworks.xcodeproj .

xcodebuild -project frameworks.xcodeproj -target libLLVM -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target ar -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target clang -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target opt -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target nm -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target dis -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target link -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target lld -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target lli -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target llc -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target clang-c -sdk iphonesimulator -arch arm64 -configuration Release -quiet
xcodebuild -project frameworks.xcodeproj -target llvm-c -sdk iphonesimulator -arch arm64 -configuration Release -quiet
popd

# mac catalyst arm64
echo "Compiling for mac catalyst (arm64):"
if [ $CLEAN ]; then
  rm -rf $MACCATALYST_BUILDDIR_ARM64
fi
if [ ! -d $MACCATALYST_BUILDDIR_ARM64 ]; then
  mkdir $MACCATALYST_BUILDDIR_ARM64
fi
pushd $MACCATALYST_BUILDDIR_ARM64
cmake -G Ninja \
-DLLVM_ENABLE_ZSTD=OFF \
-DLLVM_LINK_LLVM_DYLIB=ON \
-DLLVM_TARGET_ARCH=ARM64 \
-DLLVM_TARGETS_TO_BUILD="AArch64;X86;WebAssembly" \
-DLLVM_ENABLE_PROJECTS='clang;lld;compiler-rt' \
-DLLVM_DEFAULT_TARGET_TRIPLE=arm64-apple-ios14.0-macabi \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_THREADS=OFF \
-DLLVM_ENABLE_TERMINFO=OFF \
-DLLVM_ENABLE_BACKTRACES=OFF \
-DLLVM_ENABLE_LIBCXX=OFF \
-DLLVM_ENABLE_LIBEDIT=OFF \
-DCMAKE_CROSSCOMPILING=TRUE \
-DLLVM_TABLEGEN=${OSX_BUILDDIR}/bin/llvm-tblgen \
-DCLANG_TABLEGEN=${OSX_BUILDDIR}/bin/clang-tblgen \
-DCMAKE_OSX_SYSROOT=${OSX_SDKROOT} \
-DCMAKE_C_COMPILER=${OSX_BUILDDIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${OSX_BUILDDIR}/bin/clang++ \
-DCMAKE_LIBRARY_PATH=${OSX_BUILDDIR}/lib/ \
-DCMAKE_INCLUDE_PATH=${OSX_BUILDDIR}/include/ \
-DCOMPILER_RT_BUILD_BUILTINS=ON \
-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
-DCOMPILER_RT_BUILD_MEMPROF=OFF \
-DCOMPILER_RT_BUILD_PROFILE=OFF \
-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
-DCOMPILER_RT_BUILD_XRAY=OFF \
-DLIBOMP_LDFLAGS="-L lib/clang/14.0.0/lib/darwin -lclang_rt.cc_kext_ios" \
-DCMAKE_C_FLAGS="-target arm64-apple-ios14.0-macabi -arch arm64 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS  -I${OSX_BUILDDIR}/include/ -I${OSX_BUILDDIR}/include/c++/v1/ -I${LLVM_SRCDIR}  " \
-DCMAKE_CXX_FLAGS="-target arm64-apple-ios14.0-macabi -arch arm64 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS -I${OSX_BUILDDIR}/include/  -I${LLVM_SRCDIR} " \
-DCMAKE_MODULE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-maccatalyst -O2 -framework ios_system -lobjc -lc -lc++ " \
-DCMAKE_SHARED_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-maccatalyst -O2 -framework ios_system -lobjc -lc -lc++ " \
-DCMAKE_EXE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-maccatalyst -O2 -framework ios_system -lobjc -lc -lc++ " \
../llvm
ninja

# Also removing x86_64 here just in case
(find . -name "*.o" -exec lipo "{}" -remove x86_64 -output "{}" \; || true) 2>/dev/null

rm -f lib/liblli.a
rm -f lib/libllc.a

rm -f lib/libclang_tool.a
rm -f lib/libopt.a
ar -r lib/libclang_tool.a tools/clang/tools/driver/CMakeFiles/clang.dir/driver.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1as_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1gen_reproducer_main.cpp.o
ar -r lib/libopt.a tools/opt/CMakeFiles/opt.dir/NewPMDriver.cpp.o tools/opt/CMakeFiles/opt.dir/opt.cpp.o

rm -rf frameworks.xcodeproj
cp -r ../frameworks/frameworks.xcodeproj .

xcodebuild -project frameworks.xcodeproj -scheme libLLVM -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme ar -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme clang -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme opt -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme nm -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme dis -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme link -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme lld -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme lli -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme llc -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme clang-c -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme llvm-c -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64' ARCHS=arm64 SYMROOT=build
popd

# mac catalyst x86_64
echo "Compiling for mac catalyst x86_64:"
if [ $CLEAN ]; then
  rm -rf $MACCATALYST_BUILDDIR
fi
if [ ! -d $MACCATALYST_BUILDDIR ]; then
  mkdir $MACCATALYST_BUILDDIR
fi
pushd $MACCATALYST_BUILDDIR
cmake -G Ninja \
-DLLVM_ENABLE_ZSTD=OFF \
-DLLVM_LINK_LLVM_DYLIB=ON \
-DLLVM_TARGET_ARCH=X86_64 \
-DLLVM_TARGETS_TO_BUILD="AArch64;X86;WebAssembly" \
-DLLVM_ENABLE_PROJECTS='clang;lld;compiler-rt' \
-DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-apple-ios14.0-macabi \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_THREADS=OFF \
-DLLVM_ENABLE_TERMINFO=OFF \
-DLLVM_ENABLE_BACKTRACES=OFF \
-DLLVM_ENABLE_LIBCXX=OFF \
-DLLVM_ENABLE_LIBEDIT=OFF \
-DCMAKE_CROSSCOMPILING=TRUE \
-DLLVM_TABLEGEN=${OSX_BUILDDIR}/bin/llvm-tblgen \
-DCLANG_TABLEGEN=${OSX_BUILDDIR}/bin/clang-tblgen \
-DCMAKE_OSX_SYSROOT=${OSX_SDKROOT} \
-DCMAKE_C_COMPILER=${OSX_BUILDDIR}/bin/clang \
-DCMAKE_CXX_COMPILER=${OSX_BUILDDIR}/bin/clang++ \
-DCMAKE_LIBRARY_PATH=${OSX_BUILDDIR}/lib/ \
-DCMAKE_INCLUDE_PATH=${OSX_BUILDDIR}/include/ \
-DCOMPILER_RT_BUILD_BUILTINS=ON \
-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
-DCOMPILER_RT_BUILD_MEMPROF=OFF \
-DCOMPILER_RT_BUILD_PROFILE=OFF \
-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
-DCOMPILER_RT_BUILD_XRAY=OFF \
-DLIBOMP_LDFLAGS="-L lib/clang/14.0.0/lib/darwin -lclang_rt.cc_kext_ios" \
-DCMAKE_C_FLAGS="-target x86_64-apple-ios14.0-macabi -arch x86_64 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS  -I${OSX_BUILDDIR}/include/ -I${OSX_BUILDDIR}/include/c++/v1/ -I${LLVM_SRCDIR}  " \
-DCMAKE_CXX_FLAGS="-target x86_64-apple-ios14.0-macabi -arch x86_64 -O2 -D_LIBCPP_STRING_H_HAS_CONST_OVERLOADS -I${OSX_BUILDDIR}/include/  -I${LLVM_SRCDIR} " \
-DCMAKE_MODULE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-maccatalyst -O2 -framework ios_system -lobjc -lc -lc++ " \
-DCMAKE_SHARED_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-maccatalyst -O2 -framework ios_system -lobjc -lc -lc++ " \
-DCMAKE_EXE_LINKER_FLAGS="-nostdlib -F${LLVM_SRCDIR}/ios_system.xcframework/ios-arm64_x86_64-maccatalyst -O2 -framework ios_system -lobjc -lc -lc++ " \
../llvm
ninja

# Also removing arm64 here just in case
(find . -name "*.o" -exec lipo "{}" -remove arm64 -output "{}" \; || true) 2>/dev/null

rm -f lib/liblli.a
rm -f lib/libllc.a

rm -f lib/libclang_tool.a
rm -f lib/libopt.a
ar -r lib/libclang_tool.a tools/clang/tools/driver/CMakeFiles/clang.dir/driver.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1as_main.cpp.o tools/clang/tools/driver/CMakeFiles/clang.dir/cc1gen_reproducer_main.cpp.o
ar -r lib/libopt.a tools/opt/CMakeFiles/opt.dir/NewPMDriver.cpp.o tools/opt/CMakeFiles/opt.dir/opt.cpp.o

rm -rf frameworks.xcodeproj
cp -r ../frameworks/frameworks.xcodeproj .

xcodebuild -project frameworks.xcodeproj -scheme libLLVM -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme ar -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme clang -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme opt -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme nm -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme dis -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme link -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme lld -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme lli -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme llc -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme clang-c -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
xcodebuild -project frameworks.xcodeproj -scheme llvm-c -configuration Release -quiet -destination 'platform=macOS,variant=Mac Catalyst,arch=x86_64' ARCHS=x86_64 SYMROOT=build
popd

# 6)
echo "Merging into xcframeworks:"

universal_simulator_framework_path=build-iphonesimulator/build/Release-iphonesimulator
universal_catalyst_framework_path=build-maccatalyst/build/Release-maccatalyst


for framework in ar lld llc clang dis libLLVM link lli nm opt clang-c llvm-c
do

   # merge simulator builds
   mkdir -p $universal_simulator_framework_path
   pushd $universal_simulator_framework_path
   cp -r $SIM_BUILDDIR/build/Release-iphonesimulator/$framework.framework .
   rm $framework.framework/$framework
   lipo -create $SIM_BUILDDIR/build/Release-iphonesimulator/$framework.framework/$framework $SIM_BUILDDIR_ARM64/build/Release-iphonesimulator/$framework.framework/$framework -output $framework.framework/$framework
   popd
   
   # merge mac catalyst builds
   mkdir -p $universal_catalyst_framework_path
   pushd $universal_catalyst_framework_path
   cp -r $MACCATALYST_BUILDDIR/build/Release-maccatalyst/$framework.framework .
   rm $framework.framework/$framework
   lipo -create $MACCATALYST_BUILDDIR/build/Release-maccatalyst/$framework.framework/$framework $MACCATALYST_BUILDDIR_ARM64/build/Release-maccatalyst/$framework.framework/$framework -output $framework.framework/$framework
   popd

   # merge all platforms
   rm -rf $framework.xcframework
   xcodebuild -create-xcframework -framework build-iphoneos-arm64/build/Release-iphoneos/$framework.framework -framework build-iphonesimulator/build/Release-iphonesimulator/$framework.framework -framework build-maccatalyst/build/Release-maccatalyst/$framework.framework -output $framework.xcframework

   # while we're at it, let's compute the checksum:
   rm -f $framework.xcframework.zip
   zip -rq $framework.xcframework.zip $framework.xcframework
   swift package compute-checksum $framework.xcframework.zip
done
