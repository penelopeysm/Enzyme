#ifndef ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {

enum class AnalysisTarget {
  Sampler,
  Logpdf,
};

class SampleDependenceAnalysis {
public:
  explicit SampleDependenceAnalysis(impulse::MCMCRegionOp regionOp);

  SampleDependenceAnalysis(impulse::MCMCRegionOp regionOp,
                           AnalysisTarget target);

  bool isSampleDependent(Value value) const;
  bool isSampleDependent(Operation *op) const;
  bool canHoist(Operation *op) const;

  ArrayRef<impulse::SampleRegionOp> getSampleOps() const { return sampleOps; }

  impulse::MCMCRegionOp getRegionOp() const { return regionOp; }
  AnalysisTarget getTarget() const { return target; }

  bool isInTargetRegion(Operation *op);

  Region &getTargetRegion();

private:
  impulse::MCMCRegionOp regionOp;
  AnalysisTarget target;
  DenseSet<Value> sampleDependentValues;
  SmallVector<impulse::SampleRegionOp> sampleOps;

  void runSamplerAnalysis();
  void runLogpdfAnalysis();
  void markSampleDependent(Value value);
  void propagateDependence(Region &region);
};

bool hoistSampleInvariantOps(impulse::MCMCRegionOp regionOp);

bool hoistSampleInvariantOps(impulse::MCMCRegionOp regionOp,
                             AnalysisTarget target);

bool constructUnifiedLogpdf(impulse::MCMCRegionOp regionOp);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
