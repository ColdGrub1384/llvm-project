//
//  llvm_c.h
//  llvm-c
//
//  Created by Emma on 17-05-25.
//  Copyright © 2025 Nicolas Holzschuch. All rights reserved.
//

#import <Foundation/Foundation.h>

//! Project version number for llvm_c.
FOUNDATION_EXPORT double llvm_cVersionNumber;

//! Project version string for llvm_c.
FOUNDATION_EXPORT const unsigned char llvm_cVersionString[];

// In this header, you should import all the public headers of your framework using statements like #import <llvm-c/PublicHeader.h>

#import "Analysis.h"
#import "BitReader.h"
#import "BitWriter.h"
#import "blake3.h"
#import "Comdat.h"
#import "Core.h"
#import "DataTypes.h"
#import "DebugInfo.h"
#import "Deprecated.h"
#import "Disassembler.h"
#import "DisassemblerTypes.h"
#import "Error.h"
#import "ErrorHandling.h"
#import "ExecutionEngine.h"
#import "ExternC.h"
#import "IRReader.h"
#import "Linker.h"
#import "LLJIT.h"
#import "LLJITUtils.h"
#import "lto.h"
#import "Object.h"
#import "Orc.h"
#import "OrcEE.h"
#import "Remarks.h"
#import "Support.h"
#import "Target.h"
#import "TargetMachine.h"
#import "Types.h"
#import "Transforms/PassBuilder.h"
