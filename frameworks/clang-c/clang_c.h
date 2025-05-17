//
//  clang_c.h
//  clang-c
//
//  Created by Emma on 17-05-25.
//  Copyright © 2025 Nicolas Holzschuch. All rights reserved.
//

#import <Foundation/Foundation.h>

//! Project version number for clang_c.
FOUNDATION_EXPORT double clang_cVersionNumber;

//! Project version string for clang_c.
FOUNDATION_EXPORT const unsigned char clang_cVersionString[];

// In this header, you should import all the public headers of your framework using statements like #import <clang-c/PublicHeader.h>

#import "Index.h"
#import "Documentation.h"
#import "CXDiagnostic.h"
#import "CXSourceLocation.h"
#import "BuildSystem.h"
#import "CXCompilationDatabase.h"
#import "CXFile.h"
#import "Rewrite.h"
#import "CXString.h"
#import "ExternC.h"
#import "CXErrorCode.h"
#import "Platform.h"
#import "FatalErrorHandler.h"
