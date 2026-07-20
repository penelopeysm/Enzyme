#include "Analysis/SampleDependenceAnalysis.h"
#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_INLINEMCMCINTOREGIONPASS
#define GEN_PASS_DEF_OUTLINEMCMCFROMREGIONPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

static impulse::SymbolAttr composeSymbols(impulse::SymbolAttr outer,
                                         impulse::SymbolAttr inner,
                                         MLIRContext *ctx) {
  SmallVector<uint64_t> composed(outer.getPath());
  composed.append(inner.getPath().begin(), inner.getPath().end());
  return impulse::SymbolAttr::get(ctx, composed);
}

static ArrayAttr flattenAddressesForSymbol(ArrayAttr addresses,
                                           impulse::SymbolAttr outerSymbol,
                                           MLIRContext *ctx) {
  SmallVector<Attribute> newAddresses;
  for (auto addr : addresses) {
    auto address = cast<ArrayAttr>(addr);
    if (address.size() >= 2 && address[0] == outerSymbol) {
      auto inner = cast<impulse::SymbolAttr>(address[1]);
      auto composite = composeSymbols(outerSymbol, inner, ctx);
      SmallVector<Attribute> newAddr;
      newAddr.push_back(composite);
      for (unsigned i = 2; i < address.size(); ++i)
        newAddr.push_back(address[i]);
      newAddresses.push_back(ArrayAttr::get(ctx, newAddr));
    } else {
      newAddresses.push_back(addr);
    }
  }
  return ArrayAttr::get(ctx, newAddresses);
}

static bool inlineSubmodelSampleRegions(impulse::MCMCRegionOp regionOp) {
  bool anyChanged = false;

  SmallVector<impulse::SampleRegionOp> sampleOps;
  regionOp.getSampler().walk(
      [&](impulse::SampleRegionOp op) { sampleOps.push_back(op); });

  for (impulse::SampleRegionOp sampleOp : sampleOps) {
    Region &logpdf = sampleOp.getLogpdf();
    if (!logpdf.empty())
      continue;
    if (sampleOp.getLogpdfFnAttr())
      continue;

    Region &sampler = sampleOp.getSampler();
    if (sampler.empty() || !sampler.hasOneBlock())
      continue;

    auto outerSymbol = sampleOp.getSymbolAttr();
    if (!outerSymbol)
      continue;

    Block &samplerEntry = sampler.front();
    auto *ctx = regionOp.getContext();

    OpBuilder builder(sampleOp);
    IRMapping mapper;
    auto inputs = sampleOp.getInputs();
    for (unsigned i = 0, e = samplerEntry.getNumArguments(); i < e; ++i) {
      if (i < inputs.size())
        mapper.map(samplerEntry.getArgument(i), inputs[i]);
    }

    for (Operation &op : samplerEntry.without_terminator()) {
      Operation *cloned = builder.clone(op, mapper);
      if (auto innerSample = dyn_cast<impulse::SampleRegionOp>(cloned)) {
        if (auto innerSymbol = innerSample.getSymbolAttr())
          innerSample.setSymbolAttr(
              composeSymbols(outerSymbol, innerSymbol, ctx));
      } else if (auto innerSampleOp = dyn_cast<impulse::SampleOp>(cloned)) {
        if (auto innerSymbol = innerSampleOp.getSymbolAttr())
          innerSampleOp.setSymbolAttr(
              composeSymbols(outerSymbol, innerSymbol, ctx));
      }
    }

    auto *yield = samplerEntry.getTerminator();
    for (auto [oldResult, yieldOperand] :
         llvm::zip(sampleOp.getResults(), yield->getOperands()))
      oldResult.replaceAllUsesWith(mapper.lookupOrDefault(yieldOperand));

    sampleOp.erase();

    if (auto allAddrs = regionOp.getAllAddressesAttr())
      regionOp.setAllAddressesAttr(
          flattenAddressesForSymbol(allAddrs, outerSymbol, ctx));
    if (auto sel = regionOp.getSelectionAttr())
      regionOp.setSelectionAttr(
          flattenAddressesForSymbol(sel, outerSymbol, ctx));

    anyChanged = true;
  }

  return anyChanged;
}

static bool extractUnselectedSampleValues(impulse::MCMCRegionOp regionOp) {
  auto selection = regionOp.getSelectionAttr();
  if (!selection)
    return false;

  DenseSet<Attribute> selectedSymbols;
  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (!address.empty())
      selectedSymbols.insert(address[0]);
  }

  Block &entry = regionOp.getSampler().front();
  auto inputs = regionOp.getInputs();

  IRMapping blockToOuter;
  for (auto [idx, blockArg] : llvm::enumerate(entry.getArguments())) {
    if (idx < inputs.size())
      blockToOuter.map(blockArg, inputs[idx]);
  }

  SmallVector<impulse::SampleRegionOp> toExtract;
  for (auto &op : entry.without_terminator()) {
    auto sampleOp = dyn_cast<impulse::SampleRegionOp>(&op);
    if (!sampleOp)
      continue;
    auto symbol = sampleOp.getSymbolAttr();
    if (!symbol || selectedSymbols.contains(symbol))
      continue;

    bool canExtract = true;
    for (Value operand : sampleOp->getOperands()) {
      if (!blockToOuter.contains(operand) &&
          !operand.getDefiningOp<arith::ConstantOp>()) {
        canExtract = false;
        break;
      }
    }
    if (!canExtract)
      continue;

    SetVector<Value> nestedValues;
    for (Region &nestedRegion : sampleOp->getRegions())
      getUsedValuesDefinedAbove(nestedRegion, nestedValues);
    for (Value v : nestedValues) {
      if (!blockToOuter.contains(v) && !v.getDefiningOp<arith::ConstantOp>()) {
        canExtract = false;
        break;
      }
    }
    if (!canExtract)
      continue;

    toExtract.push_back(sampleOp);
  }

  if (toExtract.empty())
    return false;

  unsigned numInputs = inputs.size();
  bool anyChanged = false;

  for (impulse::SampleRegionOp sampleOp : toExtract) {
    OpBuilder builder(regionOp);
    IRMapping cloneMapper(blockToOuter);
    Operation *cloned = builder.clone(*sampleOp, cloneMapper);

    for (auto [original, clonedResult] :
         llvm::zip(sampleOp->getResults(), cloned->getResults())) {
      regionOp->insertOperands(numInputs, {clonedResult});
      auto segSizes =
          regionOp->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
      SmallVector<int32_t> newSizes(segSizes.asArrayRef());
      newSizes[0]++;
      regionOp->setAttr("operandSegmentSizes",
                        builder.getDenseI32ArrayAttr(newSizes));
      numInputs++;

      Value newBlockArg =
          entry.addArgument(original.getType(), sampleOp.getLoc());
      original.replaceAllUsesWith(newBlockArg);
      blockToOuter.map(newBlockArg, clonedResult);
    }

    sampleOp.erase();
    anyChanged = true;
  }

  return anyChanged;
}

static Value resolveValueForLogpdf(OpBuilder &builder, Location loc,
                                   Value value, IRMapping &mapping,
                                   impulse::MCMCRegionOp regionOp) {
  if (mapping.contains(value))
    return mapping.lookup(value);

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == &regionOp.getSampler().front()) {
      unsigned idx = blockArg.getArgNumber();
      auto inputs = regionOp.getInputs();
      if (idx < inputs.size()) {
        mapping.map(value, inputs[idx]);
        return inputs[idx];
      }
    }
    return value;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return value;

  if (defOp->getParentRegion() != &regionOp.getSampler())
    return value;

  if (isa<impulse::SampleRegionOp>(defOp)) {
    Block *logpdfBlock = &regionOp.getLogpdf().front();
    Value newArg = logpdfBlock->addArgument(value.getType(), defOp->getLoc());
    mapping.map(value, newArg);
    return newArg;
  }

  for (Value operand : defOp->getOperands())
    resolveValueForLogpdf(builder, loc, operand, mapping, regionOp);

  Operation *cloned = builder.clone(*defOp, mapping);
  for (auto [orig, clonedRes] :
       llvm::zip(defOp->getResults(), cloned->getResults()))
    mapping.map(orig, clonedRes);

  return mapping.lookup(value);
}

bool enzyme::constructUnifiedLogpdf(impulse::MCMCRegionOp regionOp) {
  Region &samplerRegion = regionOp.getSampler();
  Region &logpdfRegion = regionOp.getLogpdf();
  auto selection = regionOp.getSelectionAttr();
  if (!selection || selection.empty())
    return false;

  SmallVector<impulse::SampleRegionOp> allSampleOps;
  samplerRegion.walk(
      [&](impulse::SampleRegionOp op) { allSampleOps.push_back(op); });

  DenseMap<Attribute, impulse::SampleRegionOp> symbolToSampleOp;
  for (auto sampleOp : allSampleOps) {
    if (auto sym = sampleOp.getSymbolAttr())
      symbolToSampleOp[sym] = sampleOp;
  }

  DenseSet<Attribute> selectedSymbols;
  SmallVector<Attribute> selectionOrder;
  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (!address.empty()) {
      selectedSymbols.insert(address[0]);
      selectionOrder.push_back(address[0]);
    }
  }

  Block *logpdfBlock = new Block();
  logpdfRegion.push_back(logpdfBlock);

  Location loc = regionOp.getLoc();
  IRMapping positionMapping;
  SmallVector<impulse::SupportAttr> supportsVec;
  int64_t totalPositionSize = 0;

  for (auto symbol : selectionOrder) {
    auto it = symbolToSampleOp.find(symbol);
    if (it == symbolToSampleOp.end())
      continue;
    impulse::SampleRegionOp sampleOp = it->second;

    for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
      Value sampleResult = sampleOp.getResult(i);
      auto resultType = sampleResult.getType();
      Value blockArg = logpdfBlock->addArgument(resultType, loc);
      positionMapping.map(sampleResult, blockArg);

      if (auto tensorType = dyn_cast<RankedTensorType>(resultType))
        totalPositionSize += tensorType.getNumElements();
      else
        totalPositionSize += 1;
    }

    if (auto support = sampleOp.getSupportAttr())
      supportsVec.push_back(support);
    else
      supportsVec.push_back(impulse::SupportAttr::get(
          regionOp.getContext(), impulse::SupportKind::REAL, nullptr, nullptr));
  }

  int64_t numPositionArgs = logpdfBlock->getNumArguments();

  OpBuilder logpdfBuilder(logpdfBlock, logpdfBlock->end());
  Value totalLogpdf;
  auto scalarF64 = RankedTensorType::get({}, logpdfBuilder.getF64Type());

  for (auto sampleOp : allSampleOps) {
    Region &siteLogpdf = sampleOp.getLogpdf();
    if (siteLogpdf.empty() || !siteLogpdf.hasOneBlock())
      continue;

    Block &siteEntry = siteLogpdf.front();
    if (siteEntry.getNumArguments() == 0)
      continue;

    unsigned numSampleOutputs = sampleOp.getNumResults() - 1;
    auto inputs = sampleOp.getInputs();

    IRMapping siteMapping(positionMapping);

    for (unsigned i = 0; i < numSampleOutputs; ++i) {
      if (i >= siteEntry.getNumArguments())
        break;
      Value sampleResult = sampleOp.getResult(i + 1);
      Value resolved = resolveValueForLogpdf(logpdfBuilder, loc, sampleResult,
                                             siteMapping, regionOp);
      siteMapping.map(siteEntry.getArgument(i), resolved);
    }

    for (unsigned i = numSampleOutputs, e = siteEntry.getNumArguments(); i < e;
         ++i) {
      unsigned inputIdx = i - numSampleOutputs + 1;
      if (inputIdx < inputs.size()) {
        Value contextVal = inputs[inputIdx];
        Value resolved = resolveValueForLogpdf(logpdfBuilder, loc, contextVal,
                                               siteMapping, regionOp);
        siteMapping.map(siteEntry.getArgument(i), resolved);
      }
    }

    for (Operation &op : siteEntry.without_terminator())
      logpdfBuilder.clone(op, siteMapping);

    auto *yield = siteEntry.getTerminator();
    assert(yield->getNumOperands() > 0);
    Value siteResult = siteMapping.lookupOrDefault(yield->getOperand(0));

    if (!totalLogpdf)
      totalLogpdf = siteResult;
    else
      totalLogpdf =
          arith::AddFOp::create(logpdfBuilder, loc, totalLogpdf, siteResult);
  }

  if (totalLogpdf) {
    impulse::YieldOp::create(logpdfBuilder, loc, {totalLogpdf});
  } else {
    auto zeroConst = arith::ConstantOp::create(
        logpdfBuilder, loc, scalarF64,
        DenseElementsAttr::get(scalarF64, logpdfBuilder.getF64FloatAttr(0.0)));
    impulse::YieldOp::create(logpdfBuilder, loc, {zeroConst});
  }

  logpdfBuilder.getContext();
  regionOp.setNumPositionArgsAttr(
      logpdfBuilder.getI64IntegerAttr(numPositionArgs));
  regionOp.setPositionSizeAttr(
      logpdfBuilder.getI64IntegerAttr(totalPositionSize));
  if (!supportsVec.empty()) {
    regionOp.setSupportsAttr(ArrayAttr::get(
        regionOp.getContext(),
        SmallVector<Attribute>(supportsVec.begin(), supportsVec.end())));
  }

  for (auto sampleOp : allSampleOps) {
    Region &siteLogpdf = sampleOp.getLogpdf();
    if (siteLogpdf.empty())
      continue;
    Block &entry = siteLogpdf.front();
    SmallVector<Operation *> opsToErase;
    for (auto &op : entry)
      opsToErase.push_back(&op);
    for (auto *op : llvm::reverse(opsToErase)) {
      op->dropAllUses();
      op->erase();
    }
    while (entry.getNumArguments() > 0) {
      entry.getArgument(0).dropAllUses();
      entry.eraseArgument(0);
    }
    entry.erase();
  }

  return true;
}

namespace {

static void inlineFunctionIntoRegion(OpBuilder &builder, FunctionOpInterface fn,
                                     Region &targetRegion, Location loc) {
  Block *block = builder.createBlock(&targetRegion);
  Block &fnEntry = fn.getFunctionBody().front();

  for (auto arg : fnEntry.getArguments()) {
    block->addArgument(arg.getType(), arg.getLoc());
  }

  builder.setInsertionPointToStart(block);

  IRMapping fnMapper;
  for (auto [fnArg, blockArg] :
       llvm::zip(fnEntry.getArguments(), block->getArguments())) {
    fnMapper.map(fnArg, blockArg);
  }

  for (Operation &op : fnEntry.getOperations()) {
    if (op.hasTrait<OpTrait::ReturnLike>()) {
      SmallVector<Value> yieldOperands;
      for (Value operand : op.getOperands()) {
        yieldOperands.push_back(fnMapper.lookupOrDefault(operand));
      }
      impulse::YieldOp::create(builder, op.getLoc(), yieldOperands);
      continue;
    }
    builder.clone(op, fnMapper);
  }
}

static LogicalResult
convertSampleToSampleRegion(OpBuilder &builder, impulse::SampleOp sampleOp,
                            IRMapping &mapper,
                            SymbolTableCollection &symbolTable) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  Location loc = sampleOp.getLoc();

  SmallVector<Value> mappedInputs;
  for (Value input : sampleOp.getInputs()) {
    mappedInputs.push_back(mapper.lookupOrDefault(input));
  }

  StringAttr fnStrAttr =
      StringAttr::get(sampleOp.getContext(), sampleOp.getFn());
  StringAttr logpdfStrAttr =
      sampleOp.getLogpdfAttr()
          ? StringAttr::get(sampleOp.getContext(), *sampleOp.getLogpdf())
          : StringAttr{};

  auto sampleRegionOp = impulse::SampleRegionOp::create(
      builder, loc, sampleOp.getResultTypes(), mappedInputs, fnStrAttr,
      logpdfStrAttr, sampleOp.getSymbolAttr(), sampleOp.getSupportAttr(),
      sampleOp.getNameAttr());

  if (sampleOp.getFnAttr()) {
    auto samplerFn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp, sampleOp.getFnAttr()));
    if (samplerFn && !samplerFn.getFunctionBody().empty()) {
      inlineFunctionIntoRegion(builder, samplerFn, sampleRegionOp.getSampler(),
                               loc);
    }
  }

  if (sampleOp.getLogpdfAttr()) {
    auto logpdfFn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp,
                                            sampleOp.getLogpdfAttr()));
    if (logpdfFn && !logpdfFn.getFunctionBody().empty()) {
      inlineFunctionIntoRegion(builder, logpdfFn, sampleRegionOp.getLogpdf(),
                               loc);
    }
  }

  for (auto [oldResult, newResult] :
       llvm::zip(sampleOp.getResults(), sampleRegionOp.getResults())) {
    mapper.map(oldResult, newResult);
  }

  return success();
}

struct InlineMCMCOp : public OpRewritePattern<impulse::InferOp> {
  using OpRewritePattern<impulse::InferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::InferOp mcmcOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    Location loc = mcmcOp.getLoc();

    bool isLogpdfMode = static_cast<bool>(mcmcOp.getLogpdfFnAttr());

    FunctionOpInterface targetFn;
    if (isLogpdfMode) {
      targetFn = dyn_cast_or_null<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mcmcOp,
                                              mcmcOp.getLogpdfFnAttr()));
    } else {
      targetFn = dyn_cast_or_null<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mcmcOp, mcmcOp.getFnAttr()));
    }

    if (!targetFn || targetFn.getFunctionBody().empty()) {
      return mcmcOp.emitError("Cannot inline empty model/logpdf function");
    }

    auto fnStrAttr = isLogpdfMode
                         ? StringAttr{}
                         : StringAttr::get(mcmcOp.getContext(),
                                           mcmcOp.getFnAttr().getValue());

    SmallVector<Type> regionResultTypes;
    regionResultTypes.push_back(mcmcOp.getTrace().getType());
    regionResultTypes.push_back(mcmcOp.getDiagnostics().getType());
    regionResultTypes.push_back(mcmcOp.getLogDensities().getType());
    regionResultTypes.push_back(mcmcOp.getOutputRngState().getType());
    regionResultTypes.push_back(mcmcOp.getFinalPosition().getType());
    regionResultTypes.push_back(mcmcOp.getFinalGradient().getType());
    regionResultTypes.push_back(mcmcOp.getFinalPotentialEnergy().getType());
    regionResultTypes.push_back(mcmcOp.getFinalStepSize().getType());
    regionResultTypes.push_back(
        mcmcOp.getFinalInverseMassMatrix().getType());

    auto mcmcRegionOp = impulse::MCMCRegionOp::create(
        rewriter, loc, regionResultTypes, mcmcOp.getInputs(),
        mcmcOp.getOriginalTrace(), mcmcOp.getSelectionAttr(),
        mcmcOp.getAllAddressesAttr(), mcmcOp.getNumWarmupAttr(),
        mcmcOp.getNumSamplesAttr(), mcmcOp.getThinningAttr(),
        mcmcOp.getInverseMassMatrix(), mcmcOp.getStepSize(),
        mcmcOp.getHmcConfigAttr(), mcmcOp.getNutsConfigAttr(),
        mcmcOp.getLogpdfFnAttr(), mcmcOp.getInitialPosition(),
        mcmcOp.getInitialGradient(), mcmcOp.getInitialPotentialEnergy(),
        fnStrAttr, mcmcOp.getNameAttr(),
        /*num_position_args=*/rewriter.getI64IntegerAttr(0),
        /*position_size=*/rewriter.getI64IntegerAttr(0),
        /*supports=*/ArrayAttr());

    Block *bodyBlock = rewriter.createBlock(&mcmcRegionOp.getSampler());

    Block &fnEntry = targetFn.getFunctionBody().front();
    for (auto arg : fnEntry.getArguments()) {
      bodyBlock->addArgument(arg.getType(), arg.getLoc());
    }

    rewriter.setInsertionPointToStart(bodyBlock);

    IRMapping mapper;
    for (auto [fnArg, blockArg] :
         llvm::zip(fnEntry.getArguments(), bodyBlock->getArguments())) {
      mapper.map(fnArg, blockArg);
    }

    for (Operation &op : fnEntry.getOperations()) {
      if (op.hasTrait<OpTrait::ReturnLike>()) {
        SmallVector<Value> yieldOperands;
        for (Value operand : op.getOperands()) {
          yieldOperands.push_back(mapper.lookupOrDefault(operand));
        }
        impulse::YieldOp::create(rewriter, op.getLoc(), yieldOperands);
        continue;
      }

      if (!isLogpdfMode) {
        if (auto sampleOp = dyn_cast<impulse::SampleOp>(&op)) {
          if (failed(convertSampleToSampleRegion(rewriter, sampleOp, mapper,
                                                 symbolTable))) {
            return failure();
          }
          continue;
        }
      }

      rewriter.clone(op, mapper);
    }

    rewriter.replaceOp(mcmcOp, mcmcRegionOp.getResults());
    return success();
  }
};

static func::FuncOp outlineSampleSubRegion(OpBuilder &moduleBuilder,
                                           Region &region, StringRef funcName) {
  assert(!region.empty() && region.hasOneBlock());
  Block &entry = region.front();

  SmallVector<Type> argTypes(entry.getArgumentTypes());
  SmallVector<Location> argLocs;
  for (auto arg : entry.getArguments())
    argLocs.push_back(arg.getLoc());

  auto *yield = entry.getTerminator();
  SmallVector<Type> resultTypes(yield->getOperandTypes());

  auto fnType = moduleBuilder.getFunctionType(argTypes, resultTypes);
  auto func =
      func::FuncOp::create(moduleBuilder, region.getLoc(), funcName, fnType);
  func.setPrivate();

  OpBuilder bodyBuilder(func.getContext());
  Block *newEntry = bodyBuilder.createBlock(
      &func.getBody(), func.getBody().begin(), argTypes, argLocs);
  bodyBuilder.setInsertionPointToEnd(newEntry);

  IRMapping map;
  for (auto [oldArg, newArg] :
       llvm::zip(entry.getArguments(), newEntry->getArguments()))
    map.map(oldArg, newArg);

  for (Operation &op : entry.getOperations()) {
    if (isa<impulse::YieldOp>(&op)) {
      SmallVector<Value> returnOperands;
      for (Value operand : op.getOperands())
        returnOperands.push_back(map.lookupOrDefault(operand));
      func::ReturnOp::create(bodyBuilder, op.getLoc(), returnOperands);
      continue;
    }
    bodyBuilder.clone(op, map);
  }

  return func;
}

static void convertSampleRegionToSample(func::FuncOp outlinedFunc) {
  SmallVector<impulse::SampleRegionOp> toConvert;
  outlinedFunc.walk(
      [&](impulse::SampleRegionOp op) { toConvert.push_back(op); });

  auto *parentOp = outlinedFunc->getParentOp();
  OpBuilder moduleBuilder(&parentOp->getRegion(0));
  moduleBuilder.setInsertionPointAfter(outlinedFunc);

  unsigned counter = 0;
  for (auto sampleRegionOp : toConvert) {
    OpBuilder builder(sampleRegionOp);
    auto *ctx = builder.getContext();

    FlatSymbolRefAttr fnAttr;
    Region &samplerRegion = sampleRegionOp.getSampler();
    if (!samplerRegion.empty() && samplerRegion.hasOneBlock()) {
      std::string samplerName =
          (Twine(outlinedFunc.getName()) + "_sampler_" + Twine(counter)).str();
      auto samplerFunc =
          outlineSampleSubRegion(moduleBuilder, samplerRegion, samplerName);
      fnAttr = FlatSymbolRefAttr::get(ctx, samplerFunc.getName());
    } else if (auto fnStrAttr = sampleRegionOp.getFnAttr()) {
      fnAttr = FlatSymbolRefAttr::get(ctx, fnStrAttr);
    }

    FlatSymbolRefAttr logpdfAttr;
    Region &logpdfRegion = sampleRegionOp.getLogpdf();
    if (!logpdfRegion.empty() && logpdfRegion.hasOneBlock()) {
      std::string logpdfName =
          (Twine(outlinedFunc.getName()) + "_logpdf_" + Twine(counter)).str();
      auto logpdfFunc =
          outlineSampleSubRegion(moduleBuilder, logpdfRegion, logpdfName);
      logpdfAttr = FlatSymbolRefAttr::get(ctx, logpdfFunc.getName());
    } else if (auto logpdfStrAttr = sampleRegionOp.getLogpdfFnAttr()) {
      logpdfAttr = FlatSymbolRefAttr::get(ctx, logpdfStrAttr);
    }

    auto sampleOp = impulse::SampleOp::create(
        builder, sampleRegionOp.getLoc(), sampleRegionOp.getResultTypes(),
        fnAttr, sampleRegionOp.getInputs(), logpdfAttr,
        sampleRegionOp.getSymbolAttr(), sampleRegionOp.getSupportAttr(),
        sampleRegionOp.getNameAttr());

    sampleRegionOp.replaceAllUsesWith(sampleOp.getResults());
    sampleRegionOp.erase();
    ++counter;
  }
}

static FailureOr<func::FuncOp> outlineRegionToFunction(Region &region,
                                                       StringRef funcName,
                                                       OpBuilder &builder) {
  if (region.empty())
    return failure();

  Block &entryBlock = region.front();

  auto *terminator = entryBlock.getTerminator();
  auto yieldOp = cast<impulse::YieldOp>(terminator);
  SmallVector<Type> resultTypes(yieldOp.getOperandTypes());

  llvm::SetVector<Value> freeValues;
  getUsedValuesDefinedAbove(region, freeValues);

  SmallVector<Type> argTypes(entryBlock.getArgumentTypes());
  SmallVector<Location> argLocs(entryBlock.getNumArguments(), region.getLoc());

  for (Value freeVal : freeValues) {
    argTypes.push_back(freeVal.getType());
    argLocs.push_back(freeVal.getLoc());
  }

  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  auto outlinedFunc =
      func::FuncOp::create(builder, region.getLoc(), funcName, fnType);
  outlinedFunc.setPrivate();
  Region &outlinedBody = outlinedFunc.getBody();

  IRMapping map;
  Block *newEntryBlock = builder.createBlock(
      &outlinedBody, outlinedBody.begin(), argTypes, argLocs);

  unsigned originalArgCount = entryBlock.getNumArguments();
  for (auto arg : entryBlock.getArguments())
    map.map(arg, newEntryBlock->getArgument(arg.getArgNumber()));
  for (auto [idx, freeVal] : llvm::enumerate(freeValues))
    map.map(freeVal, newEntryBlock->getArgument(originalArgCount + idx));

  region.cloneInto(&outlinedBody, map);

  for (Block &block : outlinedBody) {
    if (!block.mightHaveTerminator())
      continue;
    auto terminator = dyn_cast<impulse::YieldOp>(block.getTerminator());
    if (!terminator)
      continue;
    OpBuilder replacer(terminator);
    func::ReturnOp::create(replacer, terminator->getLoc(),
                           terminator->getOperands());
    terminator->erase();
  }

  Block *clonedEntry = map.lookup(&entryBlock);
  newEntryBlock->getOperations().splice(newEntryBlock->getOperations().end(),
                                        clonedEntry->getOperations());
  clonedEntry->erase();

  // TODO: Ponder over this.
  for (auto [idx, freeVal] : llvm::enumerate(freeValues)) {
    Operation *defOp = freeVal.getDefiningOp();
    if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>())
      continue;

    unsigned argIdx = originalArgCount + idx;
    Value funcArg = newEntryBlock->getArgument(argIdx);

    for (OpOperand &use : llvm::make_early_inc_range(funcArg.getUses())) {
      Operation *user = use.getOwner();
      if (user->getParentRegion() == &outlinedBody)
        continue;
      OpBuilder bodyBuilder(user);
      Operation *clonedConst = bodyBuilder.clone(*defOp);
      use.set(
          clonedConst->getResult(cast<OpResult>(freeVal).getResultNumber()));
    }
  }

  convertSampleRegionToSample(outlinedFunc);

  return outlinedFunc;
}

static bool canOutlineLogpdf(impulse::MCMCRegionOp regionOp) {
  Region &logpdf = regionOp.getLogpdf();
  if (logpdf.empty())
    return false;

  int64_t numPosArgs = regionOp.getNumPositionArgs();
  if (numPosArgs == 0)
    return false;

  Block &entry = logpdf.front();
  if (static_cast<int64_t>(entry.getNumArguments()) != numPosArgs)
    return false;

  if (!regionOp.getOriginalTrace())
    return false;

  return true;
}

static LogicalResult outlineLogpdfToFunction(impulse::MCMCRegionOp regionOp,
                                             StringRef logpdfFuncName,
                                             OpBuilder &builder) {
  Location loc = regionOp.getLoc();
  auto elemType = builder.getF64Type();
  int64_t positionSize = regionOp.getPositionSize();
  int64_t numPosArgs = regionOp.getNumPositionArgs();
  auto scalarType = RankedTensorType::get({}, elemType);
  auto positionType = RankedTensorType::get({1, positionSize}, elemType);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  Region &logpdfRegion = regionOp.getLogpdf();
  Block &logpdfEntry = logpdfRegion.front();

  llvm::SetVector<Value> logpdfFreeValues;
  getUsedValuesDefinedAbove(logpdfRegion, logpdfFreeValues);

  SmallVector<Type> wrapperArgTypes;
  wrapperArgTypes.push_back(positionType);
  for (Value freeVal : logpdfFreeValues)
    wrapperArgTypes.push_back(freeVal.getType());

  auto wrapperFnType = builder.getFunctionType(wrapperArgTypes, {scalarType});
  auto wrapperFunc =
      func::FuncOp::create(builder, loc, logpdfFuncName, wrapperFnType);
  wrapperFunc.setPrivate();

  Block *wrapperBlock = builder.createBlock(
      &wrapperFunc.getBody(), wrapperFunc.getBody().begin(), wrapperArgTypes,
      SmallVector<Location>(wrapperArgTypes.size(), loc));
  builder.setInsertionPointToStart(wrapperBlock);

  Value flatPosition = wrapperBlock->getArgument(0);

  IRMapping logpdfMapping;
  unsigned wrapperArgIdx = 1;
  for (Value freeVal : logpdfFreeValues)
    logpdfMapping.map(freeVal, wrapperBlock->getArgument(wrapperArgIdx++));

  auto c0 = arith::ConstantOp::create(
      builder, loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));
  int64_t curOffset = 0;
  for (unsigned i = 0; i < static_cast<unsigned>(numPosArgs); ++i) {
    Type posArgType = logpdfEntry.getArgument(i).getType();
    auto tensorType = cast<RankedTensorType>(posArgType);
    int64_t numElements = tensorType.getNumElements();
    auto sliceType = RankedTensorType::get({1, numElements}, elemType);

    auto offsetConst = arith::ConstantOp::create(
        builder, loc, i64TensorType,
        DenseElementsAttr::get(i64TensorType,
                               builder.getI64IntegerAttr(curOffset)));
    auto slice = impulse::DynamicSliceOp::create(
        builder, loc, sliceType, flatPosition, ValueRange{c0, offsetConst},
        builder.getDenseI64ArrayAttr({1, numElements}));
    auto component = impulse::ReshapeOp::create(builder, loc, posArgType, slice);

    logpdfMapping.map(logpdfEntry.getArgument(i), component);
    curOffset += numElements;
  }

  for (Operation &op : logpdfEntry.without_terminator())
    builder.clone(op, logpdfMapping);

  auto *yield = logpdfEntry.getTerminator();
  Value logpdfResult = logpdfMapping.lookupOrDefault(yield->getOperand(0));
  func::ReturnOp::create(builder, loc, {logpdfResult});

  return success();
}

static Value computeInitialPositionFromTrace(impulse::MCMCRegionOp regionOp,
                                             OpBuilder &builder) {
  Location loc = regionOp.getLoc();
  auto elemType = builder.getF64Type();
  int64_t positionSize = regionOp.getPositionSize();
  auto positionType = RankedTensorType::get({1, positionSize}, elemType);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  Value trace = regionOp.getOriginalTrace();

  DenseMap<Attribute, std::pair<int64_t, int64_t>> traceOffsets;
  {
    DenseMap<Attribute, impulse::SampleRegionOp> symbolToOp;
    regionOp.getSampler().walk([&](impulse::SampleRegionOp sampleOp) {
      if (auto sym = sampleOp.getSymbolAttr())
        symbolToOp[sym] = sampleOp;
    });

    int64_t offset = 0;
    for (auto addr : regionOp.getAllAddressesAttr()) {
      auto addressArray = cast<ArrayAttr>(addr);
      if (addressArray.empty())
        continue;
      auto symbol = addressArray[0];
      auto it = symbolToOp.find(symbol);
      if (it == symbolToOp.end())
        continue;

      auto sampleOp = it->second;
      int64_t size = 0;
      for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
        auto resultType = sampleOp.getResult(i).getType();
        if (auto tensorType = dyn_cast<RankedTensorType>(resultType))
          size += tensorType.getNumElements();
        else
          size += 1;
      }
      traceOffsets[symbol] = {offset, size};
      offset += size;
    }
  }

  auto c0 = arith::ConstantOp::create(
      builder, loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));
  auto zeroPos = arith::ConstantOp::create(
      builder, loc, positionType,
      DenseElementsAttr::get(positionType,
                             builder.getFloatAttr(elemType, 0.0)));
  Value result = zeroPos;

  int64_t posOffset = 0;
  for (auto addr : regionOp.getSelectionAttr()) {
    auto addressArray = cast<ArrayAttr>(addr);
    if (addressArray.empty())
      continue;
    auto symbol = addressArray[0];
    auto it = traceOffsets.find(symbol);
    if (it == traceOffsets.end())
      continue;

    int64_t traceOff = it->second.first;
    int64_t size = it->second.second;
    auto sliceType = RankedTensorType::get({1, size}, elemType);

    auto traceOffConst = arith::ConstantOp::create(
        builder, loc, i64TensorType,
        DenseElementsAttr::get(i64TensorType,
                               builder.getI64IntegerAttr(traceOff)));
    auto posOffConst = arith::ConstantOp::create(
        builder, loc, i64TensorType,
        DenseElementsAttr::get(i64TensorType,
                               builder.getI64IntegerAttr(posOffset)));

    auto traceSlice = impulse::DynamicSliceOp::create(
        builder, loc, sliceType, trace, ValueRange{c0, traceOffConst},
        builder.getDenseI64ArrayAttr({1, size}));
    result = impulse::DynamicUpdateSliceOp::create(builder, loc, positionType,
                                                  result, traceSlice,
                                                  ValueRange{c0, posOffConst});

    posOffset += size;
  }

  return result;
}

LogicalResult outlineMCMCRegion(impulse::MCMCRegionOp regionOp,
                                StringRef funcName, OpBuilder &builder) {
  OpBuilder::InsertionGuard insertionGuard(builder);

  if (canOutlineLogpdf(regionOp)) {
    builder.setInsertionPointAfter(
        regionOp->getParentOfType<SymbolOpInterface>());

    std::string logpdfFuncName = (Twine(funcName) + "_logpdf").str();
    if (failed(outlineLogpdfToFunction(regionOp, logpdfFuncName, builder)))
      return failure();

    auto logpdfSymRef =
        FlatSymbolRefAttr::get(builder.getContext(), logpdfFuncName);

    llvm::SetVector<Value> logpdfFreeValues;
    getUsedValuesDefinedAbove(regionOp.getLogpdf(), logpdfFreeValues);

    builder.setInsertionPoint(regionOp);
    Value initialPosition = computeInitialPositionFromTrace(regionOp, builder);

    SmallVector<Value> mcmcInputs;
    mcmcInputs.push_back(regionOp.getInputs()[0]); // rng
    mcmcInputs.append(logpdfFreeValues.begin(), logpdfFreeValues.end());

    auto newOp = impulse::InferOp::create(
        builder, regionOp.getLoc(), regionOp.getResultTypes(),
        /*fn=*/FlatSymbolRefAttr{}, mcmcInputs,
        /*adaptation_state_in=*/ValueRange{},
        /*original_trace=*/Value(), regionOp.getSelectionAttr(),
        regionOp.getAllAddressesAttr(), regionOp.getNumWarmupAttr(),
        regionOp.getNumSamplesAttr(), regionOp.getThinningAttr(),
        /*total_warmup=*/IntegerAttr{},
        regionOp.getInverseMassMatrix(), regionOp.getStepSize(),
        regionOp.getHmcConfigAttr(), regionOp.getNutsConfigAttr(), logpdfSymRef,
        initialPosition, regionOp.getInitialGradient(),
        regionOp.getInitialPotentialEnergy(),
        /*warmup_offset=*/Value{},
        /*autodiff_attrs=*/DictionaryAttr{},
        regionOp.getNameAttr());

    regionOp.replaceAllUsesWith(newOp.getResults());
    regionOp.erase();
    return success();
  }

  builder.setInsertionPointAfter(
      regionOp->getParentOfType<SymbolOpInterface>());

  llvm::SetVector<Value> freeValues;
  getUsedValuesDefinedAbove(regionOp.getSampler(), freeValues);

  FailureOr<func::FuncOp> outlinedFunc =
      outlineRegionToFunction(regionOp.getSampler(), funcName, builder);
  if (failed(outlinedFunc))
    return failure();

  SmallVector<Value> allInputs(regionOp.getInputs());
  allInputs.append(freeValues.begin(), freeValues.end());

  builder.setInsertionPoint(regionOp);
  auto outlinedSymRef =
      FlatSymbolRefAttr::get(builder.getContext(), outlinedFunc->getName());

  bool isLogpdfMode = static_cast<bool>(regionOp.getLogpdfFnAttr());
  FlatSymbolRefAttr fnAttr =
      isLogpdfMode ? FlatSymbolRefAttr{} : outlinedSymRef;
  FlatSymbolRefAttr logpdfAttr =
      isLogpdfMode ? outlinedSymRef : regionOp.getLogpdfFnAttr();

  auto newOp = impulse::InferOp::create(
      builder, regionOp.getLoc(), regionOp.getResultTypes(), fnAttr, allInputs,
      /*adaptation_state_in=*/ValueRange{},
      regionOp.getOriginalTrace(), regionOp.getSelectionAttr(),
      regionOp.getAllAddressesAttr(), regionOp.getNumWarmupAttr(),
      regionOp.getNumSamplesAttr(), regionOp.getThinningAttr(),
      /*total_warmup=*/IntegerAttr{},
      regionOp.getInverseMassMatrix(), regionOp.getStepSize(),
      regionOp.getHmcConfigAttr(), regionOp.getNutsConfigAttr(), logpdfAttr,
      regionOp.getInitialPosition(), regionOp.getInitialGradient(),
      regionOp.getInitialPotentialEnergy(),
      /*warmup_offset=*/Value{},
      /*autodiff_attrs=*/DictionaryAttr{},
      regionOp.getNameAttr());

  regionOp.replaceAllUsesWith(newOp.getResults());
  regionOp.erase();
  return success();
}

struct InlineMCMCIntoRegion
    : public enzyme::impl::InlineMCMCIntoRegionPassBase<InlineMCMCIntoRegion> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InlineMCMCOp>(&getContext());

    GreedyRewriteConfig config;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);

    SmallVector<impulse::MCMCRegionOp> regionOps;
    getOperation()->walk(
        [&](impulse::MCMCRegionOp op) { regionOps.push_back(op); });

    for (auto regionOp : regionOps) {
      bool submodelChanged = true;
      while (submodelChanged)
        submodelChanged = inlineSubmodelSampleRegions(regionOp);

      extractUnselectedSampleValues(regionOp);
    }
  }
};

struct OutlineMCMCFromRegion
    : public enzyme::impl::OutlineMCMCFromRegionPassBase<
          OutlineMCMCFromRegion> {
  void runOnOperation() override {
    SmallVector<impulse::MCMCRegionOp> toOutline;
    getOperation()->walk(
        [&](impulse::MCMCRegionOp op) { toOutline.push_back(op); });

    OpBuilder builder(getOperation());
    unsigned increment = 0;
    for (auto regionOp : toOutline) {
      auto symbol = regionOp->getParentOfType<SymbolOpInterface>();
      std::string defaultName =
          (Twine(symbol.getName()) + "_mcmc_model" + Twine(increment)).str();
      if (failed(outlineMCMCRegion(regionOp, defaultName, builder)))
        return signalPassFailure();
      ++increment;
    }
  }
};

} // namespace
