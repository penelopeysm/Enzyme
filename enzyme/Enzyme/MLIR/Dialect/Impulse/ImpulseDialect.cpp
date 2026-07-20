#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::impulse;

#include "Dialect/Impulse/ImpulseEnums.cpp.inc"
#include "Dialect/Impulse/ImpulseOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Impulse/ImpulseOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Impulse/ImpulseAttributes.cpp.inc"

void SymbolAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getPath(), printer);
  printer << ">";
}

Attribute SymbolAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};
  SmallVector<uint64_t> path;
  uint64_t val;
  if (failed(parser.parseInteger(val)))
    return {};
  path.push_back(val);
  while (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseInteger(val)))
      return {};
    path.push_back(val);
  }
  if (parser.parseGreater())
    return {};
  return get(parser.getContext(), path);
}

void mlir::impulse::ImpulseDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Impulse/ImpulseOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Impulse/ImpulseAttributes.cpp.inc"
      >();
}
