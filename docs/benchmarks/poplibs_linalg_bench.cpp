#include <poplar/ArrayRef.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include "poplar/CycleCount.hpp"
#include <poplar/Graph.hpp>
#include <poplar/SyncType.hpp>
#include <poplar/TargetType.hpp>
#include <poplar/Tensor.hpp>
#include <poplin/ConvPreplan.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>
#include <stdlib.h>
#include <vector>

#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Loop.hpp>
#include <popops/Operation.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poplin/experimental/QRFactorization.hpp>
#include <chrono>
#include <random>

using namespace std;
using namespace poplar;
using namespace poplin;
using namespace poplin::experimental;
using namespace popops;
using namespace poplar::program;

// to restore return to (n+m) instead of vsize in sliceSet
// testing granulity = 6 for householder vectors

unsigned int m = 128;
unsigned int n = 128;
vector<vector<float>> input = {
    {12, -51, 4}, {6, 167, -68}, {-4, 24, -41}, {3, 4, 5}};
vector<vector<float>> I = {
    {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};

// vector<vector<float>> input = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
// vector<vector<float>> I = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

void generateMatrix() {
  input.resize(m);
  I.resize(m);
  for (int i = 0; i < m; i++) {
    input[i].resize(n);
    for (int j = 0; j < n; j++)
      input[i][j] = rand() % 1000;
    I[i].resize(m);
    for (int j = 0; j < m; j++) {
      I[i][j] = (i == j);
    }
  }
}

void loadValues(Sequence &load, Graph &graph, Tensor &A, Tensor &Q) {
  auto inputStream = graph.addHostToDeviceFIFO("input", FLOAT, m * n);
  auto QStream = graph.addHostToDeviceFIFO("Q", FLOAT, m * m);
  load.add(Copy(inputStream, A));
  load.add(Copy(QStream, Q));
}

void connectStreams(Engine &engine) {
  auto inputCallback = [](void *ptr) {
    const int rowSize = n * sizeof(float);
    for (int i = 0; i < m; i++) {
      const int offset = i * rowSize;
      std::memcpy(ptr + offset, input[i].data(), rowSize);
    }
  };
  auto QCallback = [](void *ptr) {
    const int rowSize = m * sizeof(float);
    for (int i = 0; i < m; i++) {
      const int offset = i * rowSize;
      std::memcpy(ptr + offset, I[i].data(), rowSize);
    }
  };
  engine.connectStreamToCallback("input", inputCallback);
  engine.connectStreamToCallback("Q", QCallback);
}

void addChecksAndRefs(Graph &graph, Sequence &printAndCheck, Tensor &A,
                      Tensor &Q, Tensor &ACopy, Tensor &QCopy, bool print) {
  if (print) {
    printAndCheck.add(PrintTensor("Q", Q));
    printAndCheck.add(PrintTensor("A", A));
    printAndCheck.add(PrintTensor("ACopy", ACopy));
  }
  matMulWithOutput(graph, Q, A, A, printAndCheck);
  matMulWithOutput(graph, Q.transpose(), Q, Q, printAndCheck);
  if (print) {
    printAndCheck.add(PrintTensor("Q", Q));
    printAndCheck.add(PrintTensor("A", A));
  }

  subWithOutput(graph, A, ACopy, A, printAndCheck);
  subWithOutput(graph, Q, QCopy, Q, printAndCheck);
  // mapInPlace(graph, expr::UnaryOpType::ABSOLUTE, A, printAndCheck);
  // mapInPlace(graph, expr::UnaryOpType::ABSOLUTE, Q, printAndCheck);
  // Tensor errA = reduce(graph, A, FLOAT, {0, 1}, Operation::ADD,
  //                              printAndCheck);
  // Tensor errAMax = reduce(graph, A, FLOAT, {0, 1}, Operation::MAX,
  //                              printAndCheck);
  // Tensor errQ = reduce(graph, Q, FLOAT, {0, 1}, Operation::ADD,
  //                              printAndCheck);
  // Tensor errQMax = reduce(graph, Q, FLOAT, {0, 1}, Operation::MAX,
  //                              printAndCheck);

  Tensor norm = mul(graph, A, A, printAndCheck);
  Tensor normCopy = mul(graph, ACopy, ACopy, printAndCheck);
  Tensor normQ = mul(graph, Q, Q, printAndCheck);
  norm = reduce(graph, norm, FLOAT, {0, 1}, Operation::ADD, printAndCheck);
  normCopy = reduce(graph, normCopy, FLOAT, {0, 1}, Operation::ADD, printAndCheck);
  normQ = reduce(graph, normQ, FLOAT, {0, 1}, Operation::ADD, printAndCheck);
  norm = div(graph, norm, normCopy, printAndCheck);
  printAndCheck.add(PrintTensor("Error A norm:", norm));
  printAndCheck.add(PrintTensor("Error Q norm:", normQ));

  // printAndCheck.add(PrintTensor("Error sum A:", errA));
  // printAndCheck.add(PrintTensor("Error max A:", errAMax));
  // printAndCheck.add(PrintTensor("Error sum Q:", errQ));
  // printAndCheck.add(PrintTensor("Error max Q:", errQMax));
}

int main(int argc, char **argv) {
  srand(1234);

  const int ipus = stoi(argv[1]);
  const bool genMatrix = !!stoi(argv[2]);
  m = stoi(argv[3]);
  n = stoi(argv[4]);
  const bool checkResults = !!stoi(argv[5]);
  const bool printResults = !!stoi(argv[6]);

  if (genMatrix)
    generateMatrix();

  DeviceManager dm = DeviceManager();
  auto devices = dm.getDevices(TargetType::IPU, ipus);
  auto device = std::move(devices[0]);
  Target target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  Tensor ACopy, QCopy;
  // if (checkResults) {
    ACopy = graph.addVariable(FLOAT, {m, n}, "ACopy");
    QCopy = graph.addVariable(FLOAT, {m, m}, "QCopy");
  // }

  auto AQ = createQRFactorizationMatrices(graph, FLOAT, m, n, {});
  Tensor &A = AQ[0];
  Tensor &Q = AQ[1];
  // Tensor Q = createQRFactorizationMatrixQ(graph, FLOAT, m, n, {});


  //if (checkResults) {
    poputil::mapTensorLinearly(graph, ACopy);
    poputil::mapTensorLinearly(graph, QCopy);
  // }

  Sequence load;
  loadValues(load, graph, A, Q);
  //if (checkResults) {
    load.add(Copy(A, ACopy));
    load.add(Copy(Q, QCopy));
  //}

  Sequence loop;
  OptionFlags options;
  options.set("rowsPerIteration", std::to_string(32));
  options.set("allocProperlyMappedTensors", "true");
  auto startStamp = cycleStamp(graph, loop, 0, SyncType::INTERNAL); // allocProperlyMappedTensors
  QRFactorization(graph, A, Q, loop, {}, options);
  auto endStamp = cycleStamp(graph, loop, 0, SyncType::INTERNAL);


  Sequence printAndCheck;
  // if (checkResults) {
  //   addChecksAndRefs(graph, printAndCheck, A, Q, ACopy, QCopy, printResults);
  // }

  uint64_t hStartStamp, hEndStamp;
  graph.createHostRead("startStamp", startStamp);
  graph.createHostRead("endStamp", endStamp);

  printf("Compilation\n");
  // timer.start();
  double timeMesByCycles = 0;
  long long timeNs = 0;
  Engine engine(graph, {load, loop /*loopMes*/, printAndCheck});
  connectStreams(engine);
  engine.load(device);
  // timer.stop();
  printf("Compilation done\n");
  engine.run(0);
  const int warmUp = checkResults ? 0 : 3; // todo remove cycles
  for (int it = 0; it < warmUp; it++)
    engine.run(1);
  const int iterations = checkResults ? 1 : 10;
  for (int it = 0; it < iterations; it++) {
    auto begin = chrono::steady_clock::now();
    engine.run(1);
    auto end = chrono::steady_clock::now();

    auto time = end - begin;
    timeNs += (time / std::chrono::nanoseconds(1)) / iterations;

    engine.readTensor("startStamp", &hStartStamp, &hStartStamp + 1);
    engine.readTensor("endStamp", &hEndStamp, &hEndStamp + 1);
    uint64_t cyclesDiff = hEndStamp - hStartStamp;
    // timeMesByCycles += (double)(cyclesDiff / 1325000000.f);
    timeMesByCycles += (double)(cyclesDiff / 1325000000.f);
  }

  if (checkResults)
    engine.run(2);
  timeMesByCycles /= iterations;

  int complexity = 2 * n;
  double flopsMesByTime = (double)complexity / timeNs;
  flopsMesByTime *= m * n;
  printf("FLOPS measured by time: %f GFLOPS\n", flopsMesByTime);
  printf("Time: %lld\n", timeNs);

  double flopsMesByCycles =
      ((double)complexity / timeMesByCycles) / 1000000000.f;
  flopsMesByCycles *= m * n;

  printf("CYCLES measured by cycles: %f FLOPS: %f GFLOPS\n", timeMesByCycles,
         flopsMesByCycles);
  return 0;
}
