// Copyright 2025 Mooncake Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "cuda_alike.h"
#include "transfer_engine.h"
#include "transfer_metadata.h"  // P2PHANDSHAKE
#include "transport/transport.h"

extern char** environ;

using namespace mooncake;

// startAsyncTransfer() switches the active device to the source GPU; if it does
// not restore it, the calling thread (which also launches the engine's compute
// kernels) is left on the wrong GPU and the next kernel fails with
// hipErrorInvalidDevice. This pins the caller to one GPU, transfers from a
// buffer on a different GPU, and asserts the active device is unchanged.

namespace {
constexpr size_t kLen = 64 * 1024;

void* allocOnDevice(size_t size, int device) {
    EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
    void* ptr = nullptr;
    EXPECT_EQ(cudaMalloc(&ptr, size), cudaSuccess);
    return ptr;
}
}  // namespace

TEST(HipTransportTest, RestoresActiveDeviceAfterTransfer) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "Needs >= 2 GPUs: the transfer source must live on a "
                        "different device than the calling thread's device.";
    }

    const int kCallerDevice = 0;  // device the engine thread runs on
    const int kSourceDevice = 1;  // KV source lives on a different GPU

    // P2PHANDSHAKE: self-contained loopback, no external metadata server.
    auto engine = std::make_unique<TransferEngine>(false);
    const std::string server_name = "127.0.0.1:17813";
    if (engine->init(P2PHANDSHAKE, server_name, "127.0.0.1", 17813) != 0) {
        GTEST_SKIP() << "TransferEngine init failed in this environment.";
    }

    Transport* transport = engine->installTransport("hip", nullptr);
    if (transport == nullptr) {
        GTEST_SKIP()
            << "HIP transport unavailable (built without -DUSE_HIP=ON?).";
    }

    // Both buffers on kSourceDevice so the source GPU differs from the caller.
    void* src = allocOnDevice(kLen, kSourceDevice);
    void* dst = allocOnDevice(kLen, kSourceDevice);
    ASSERT_EQ(engine->registerLocalMemory(
                  src, kLen, GPU_PREFIX + std::to_string(kSourceDevice)),
              0);
    ASSERT_EQ(engine->registerLocalMemory(
                  dst, kLen, GPU_PREFIX + std::to_string(kSourceDevice)),
              0);

    // P2P handshake binds a free port, so open the address it actually uses.
    auto segment_id = engine->openSegment(engine->getLocalIpAndPort());
    ASSERT_GE(segment_id, 0);

    ASSERT_EQ(cudaSetDevice(kSourceDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(src, 0xAB, kLen), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Pin the caller to a different device than the source, as the engine does.
    ASSERT_EQ(cudaSetDevice(kCallerDevice), cudaSuccess);

    auto batch_id = engine->allocateBatchID(1);
    TransferRequest entry;
    entry.opcode = TransferRequest::WRITE;
    entry.length = kLen;
    entry.source = src;
    entry.target_id = segment_id;
    entry.target_offset = reinterpret_cast<uint64_t>(dst);
    Status s = engine->submitTransfer(batch_id, {entry});
    ASSERT_TRUE(s.ok());

    // Invariant: the caller's device is unchanged (== kSourceDevice before
    // fix).
    int active_after_submit = -1;
    ASSERT_EQ(cudaGetDevice(&active_after_submit), cudaSuccess);
    EXPECT_EQ(active_after_submit, kCallerDevice)
        << "startAsyncTransfer() must restore the caller's active device. "
           "Leaving it on the source GPU corrupts the engine thread's HIP "
           "context, so its next kernel launch fails with "
           "hipErrorInvalidDevice.";

    // Drain the transfer (also a basic functional check).
    TransferStatus status;
    do {
        ASSERT_TRUE(engine->getTransferStatus(batch_id, 0, status).ok());
    } while (status.s == TransferStatusEnum::WAITING);
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);

    int active_after_wait = -1;
    ASSERT_EQ(cudaGetDevice(&active_after_wait), cudaSuccess);
    EXPECT_EQ(active_after_wait, kCallerDevice);

    engine->freeBatchID(batch_id);
    engine->unregisterLocalMemory(src);
    engine->unregisterLocalMemory(dst);
    (void)cudaSetDevice(kSourceDevice);
    (void)cudaFree(src);
    (void)cudaFree(dst);
}

// A registered buffer that does not start its hipMalloc allocation (as with
// any caching allocator) must receive a peer's IPC write at the buffer, not at
// the allocation base. hipIpcOpenMemHandle maps the whole allocation, so the
// transport has to carry the buffer's offset inside it.
//
// The destination runs in a second process (this binary, re-executed with
// --hip_ipc_dst_port) because a same-process target never takes the IPC path.
// It registers a slice in the middle of a zeroed allocation, prints the slice
// address, waits for the parent's write, then checks the slice holds the
// pattern and every other byte of the allocation is still zero.

DEFINE_int32(hip_ipc_dst_port, 0,
             "Internal: run as the IPC destination process on this port.");

namespace {
constexpr size_t kAllocLen = 4 * 1024 * 1024;
constexpr size_t kSliceOffset = 1024 * 1024 + 4096;  // inside the allocation
constexpr size_t kSliceLen = 64 * 1024;
constexpr unsigned char kPattern = 0xAB;
enum DstExit {
    kDstOk = 0,
    kDstSliceWrong = 1,
    kDstAllocationCorrupted = 2,
    kDstSkip = 77
};

int runIpcDestination(int port) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
        return kDstSkip;
    const int device = device_count > 1 ? 1 : 0;
    auto engine = std::make_unique<TransferEngine>(false);
    const std::string name = "127.0.0.1:" + std::to_string(port);
    if (engine->init(P2PHANDSHAKE, name, "127.0.0.1", port) != 0)
        return kDstSkip;
    if (engine->installTransport("hip", nullptr) == nullptr) return kDstSkip;

    if (cudaSetDevice(device) != cudaSuccess) return kDstSkip;
    char* alloc = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&alloc), kAllocLen) !=
            cudaSuccess ||
        cudaMemset(alloc, 0, kAllocLen) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
        return kDstSkip;
    char* slice = alloc + kSliceOffset;
    if (engine->registerLocalMemory(slice, kSliceLen,
                                    GPU_PREFIX + std::to_string(device)) != 0)
        return kDstSkip;

    printf("%llu\n",
           static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(slice)));
    fflush(stdout);
    char go = 0;
    if (read(STDIN_FILENO, &go, 1) != 1) return kDstSkip;

    std::vector<unsigned char> host(kAllocLen);
    if (cudaMemcpy(host.data(), alloc, kAllocLen, cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        return kDstSkip;
    int rc = kDstOk;
    for (size_t i = 0; i < kAllocLen; ++i) {
        const bool in_slice = i >= kSliceOffset && i < kSliceOffset + kSliceLen;
        if (in_slice && host[i] != kPattern) {
            rc = kDstSliceWrong;
            break;
        }
        if (!in_slice && host[i] != 0) {
            rc = kDstAllocationCorrupted;
            break;
        }
    }
    engine->unregisterLocalMemory(slice);
    (void)cudaFree(alloc);
    return rc;
}
}  // namespace

TEST(HipTransportTest, IpcWriteLandsInBufferInsideLargerAllocation) {
    const int dst_port = 17815;
    int to_child[2], from_child[2];
    ASSERT_EQ(pipe(to_child), 0);
    ASSERT_EQ(pipe(from_child), 0);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, to_child[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, from_child[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, to_child[1]);
    posix_spawn_file_actions_addclose(&actions, from_child[0]);
    std::string port_flag = "--hip_ipc_dst_port=" + std::to_string(dst_port);
    char exe[] = "/proc/self/exe";
    char* child_argv[] = {exe, port_flag.data(), nullptr};
    pid_t pid = 0;
    ASSERT_EQ(posix_spawn(&pid, exe, &actions, nullptr, child_argv, environ),
              0);
    posix_spawn_file_actions_destroy(&actions);
    close(to_child[0]);
    close(from_child[1]);

    auto finish_child = [&]() {
        char go = 1;
        (void)!write(to_child[1], &go, 1);
        close(to_child[1]);
        int status = 0;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    };

    // The child prints its handshake address and the slice address once the
    // slice is registered.
    std::string line;
    char c;
    while (read(from_child[0], &c, 1) == 1 && c != '\n') line.push_back(c);
    close(from_child[0]);
    if (line.empty()) {
        const int rc = finish_child();
        if (rc == kDstSkip) GTEST_SKIP() << "IPC destination unavailable.";
        FAIL() << "IPC destination exited with " << rc;
    }
    const auto space = line.find(' ');
    ASSERT_NE(space, std::string::npos) << "unexpected child output: " << line;
    const std::string dst_name = line.substr(0, space);
    const uint64_t dst_addr = std::stoull(line.substr(space + 1));

    auto engine = std::make_unique<TransferEngine>(false);
    const std::string server_name = "127.0.0.1:17816";
    if (engine->init(P2PHANDSHAKE, server_name, "127.0.0.1", 17816) != 0 ||
        engine->installTransport("hip", nullptr) == nullptr) {
        finish_child();
        GTEST_SKIP() << "HIP transport unavailable in this environment.";
    }
    void* src = allocOnDevice(kSliceLen, 0);
    ASSERT_EQ(cudaMemset(src, kPattern, kSliceLen), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(engine->registerLocalMemory(src, kSliceLen, GPU_PREFIX + "0"), 0);

    auto segment_id = engine->openSegment(dst_name);
    ASSERT_GE(segment_id, 0);
    auto batch_id = engine->allocateBatchID(1);
    TransferRequest entry;
    entry.opcode = TransferRequest::WRITE;
    entry.length = kSliceLen;
    entry.source = src;
    entry.target_id = segment_id;
    entry.target_offset = dst_addr;
    ASSERT_TRUE(engine->submitTransfer(batch_id, {entry}).ok());
    TransferStatus status;
    do {
        ASSERT_TRUE(engine->getTransferStatus(batch_id, 0, status).ok());
    } while (status.s == TransferStatusEnum::WAITING);
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);

    const int rc = finish_child();
    EXPECT_NE(rc, kDstSliceWrong) << "the write did not reach the registered "
                                     "buffer";
    EXPECT_NE(rc, kDstAllocationCorrupted)
        << "the write landed elsewhere in the destination allocation (the "
           "allocation base instead of the registered buffer)";
    EXPECT_EQ(rc, kDstOk);

    engine->freeBatchID(batch_id);
    engine->unregisterLocalMemory(src);
    (void)cudaFree(src);
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_hip_ipc_dst_port != 0) {
        return runIpcDestination(FLAGS_hip_ipc_dst_port);
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
