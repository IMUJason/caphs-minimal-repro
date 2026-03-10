import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

enum ResourceType: String, Codable, CaseIterable {
    case cpu
    case gpu
}

enum PolicyName: String, Codable {
    case fcfs
    case least_loaded
    case fixed_priority
    case work_stealing
    case caphs
}

enum ThermalStateName: String, Codable {
    case nominal
    case fair
    case serious
    case critical
}

struct TaskSpec: Codable {
    let task_id: String
    let size: Int
    let total_chunks: Int
    let priority: Int
    let arrival_ms: Double
    let parents: [String]
    let criticality: Double
    let family: String
    let bytes_moved: Double
}

struct WorkloadSpec: Codable {
    let workload_id: String
    let family: String
    let seed: Int
    let notes: String
    let task_specs: [TaskSpec]
}

struct ProfileEntry: Codable {
    let size: Int
    let cpu_ms: Double
    let gpu_ms: Double
}

struct ProfileStore: Codable {
    let device_name: String
    let created_at: String
    let reps: Int
    let entries: [ProfileEntry]

    func durationMs(size: Int, resource: ResourceType) -> Double {
        if let match = entries.first(where: { $0.size == size }) {
            return resource == .cpu ? match.cpu_ms : match.gpu_ms
        }
        guard let lower = entries.filter({ $0.size <= size }).max(by: { $0.size < $1.size }),
              let upper = entries.filter({ $0.size >= size }).min(by: { $0.size < $1.size }) else {
            fatalError("Missing profile entry for size \(size)")
        }
        if lower.size == upper.size {
            return resource == .cpu ? lower.cpu_ms : lower.gpu_ms
        }
        let alpha = Double(size - lower.size) / Double(upper.size - lower.size)
        let lowerValue = resource == .cpu ? lower.cpu_ms : lower.gpu_ms
        let upperValue = resource == .cpu ? upper.cpu_ms : upper.gpu_ms
        return lowerValue + alpha * (upperValue - lowerValue)
    }
}

struct Config: Codable {
    let cpu_workers: Int
    let eta: Double
    let queue_weight: Double
    let bandwidth_weight: Double
    let thermal_weight: Double
    let migration_weight: Double
    let priority_weight: Double
    let criticality_weight: Double
    let migration_gain_threshold: Double
    let cooldown_chunks: Int
}

struct ChunkEvent: Codable {
    let task_id: String
    let chunk_index: Int
    let resource_id: String
    let resource_type: ResourceType
    let queue_wait_ms: Double
    let exec_ms: Double
    let chunk_start_ms: Double
    let chunk_end_ms: Double
    let thermal_state: ThermalStateName
    let migrated: Bool
}

struct TaskMetric: Codable {
    let task_id: String
    let family: String
    let arrival_ms: Double
    let completion_ms: Double
    let latency_ms: Double
    let priority: Int
    let criticality: Double
    let size: Int
    let total_chunks: Int
    let cpu_chunks: Int
    let gpu_chunks: Int
    let migrations: Int
}

struct RunSummary: Codable {
    let policy: PolicyName
    let workload_id: String
    let family: String
    let seed: Int
    let cpu_workers: Int
    let tasks_total: Int
    let chunks_total: Int
    let makespan_ms: Double
    let throughput_tasks_per_s: Double
    let mean_latency_ms: Double
    let p95_latency_ms: Double
    let mean_queue_wait_ms: Double
    let p95_queue_wait_ms: Double
    let mean_exec_ms: Double
    let scheduler_mean_us: Double
    let scheduler_p95_us: Double
    let migrations_total: Int
    let cpu_chunks: Int
    let gpu_chunks: Int
    let mean_predicted_cpu_ms: Double
    let mean_predicted_gpu_ms: Double
    let thermal_nominal_fraction: Double
    let thermal_non_nominal_events: Int
    let explicit_copy_bytes_after_init: Int
}

struct RunResult: Codable {
    let summary: RunSummary
    let task_metrics: [TaskMetric]
    let chunk_events: [ChunkEvent]
    let config: Config
}

final class MatrixBuffers {
    let size: Int
    let elementCount: Int
    let byteCount: Int
    let a: MTLBuffer
    let b: MTLBuffer
    let c: MTLBuffer

    init(device: MTLDevice, size: Int, rng: inout SeededRNG) {
        self.size = size
        self.elementCount = size * size
        self.byteCount = elementCount * MemoryLayout<Float>.stride
        guard let a = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let b = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let c = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            fatalError("Unable to allocate shared buffers")
        }
        self.a = a
        self.b = b
        self.c = c
        let aPtr = a.contents().bindMemory(to: Float.self, capacity: elementCount)
        let bPtr = b.contents().bindMemory(to: Float.self, capacity: elementCount)
        let cPtr = c.contents().bindMemory(to: Float.self, capacity: elementCount)
        for index in 0..<elementCount {
            aPtr[index] = Float.random(in: -1...1, using: &rng)
            bPtr[index] = Float.random(in: -1...1, using: &rng)
            cPtr[index] = 0
        }
    }
}

final class TaskState {
    let spec: TaskSpec
    let buffers: MatrixBuffers
    var chunksDone: Int = 0
    var isRunning = false
    var isQueued = false
    var completionMs: Double?
    var lastResourceType: ResourceType?
    var lastResourceID: String?
    var cooldownRemaining = 0
    var cpuChunks = 0
    var gpuChunks = 0
    var migrations = 0
    var preference: [ResourceType: Double]
    var queueEnterMs: Double?

    init(spec: TaskSpec, buffers: MatrixBuffers, profile: ProfileStore) {
        self.spec = spec
        self.buffers = buffers
        self.preference = [
            .cpu: -profile.durationMs(size: spec.size, resource: .cpu),
            .gpu: -profile.durationMs(size: spec.size, resource: .gpu),
        ]
    }

    var isCompleted: Bool {
        chunksDone >= spec.total_chunks
    }
}

final class ResourceState {
    let resourceID: String
    let resourceType: ResourceType
    let queue = DispatchQueue(label: UUID().uuidString)
    var waitingTaskIDs: [String] = []
    var runningTaskID: String?

    init(resourceID: String, resourceType: ResourceType) {
        self.resourceID = resourceID
        self.resourceType = resourceType
    }
}

final class SharedExecutors {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private var kernels: [Int: MPSMatrixMultiplication] = [:]

    init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            fatalError("Unable to initialize Metal")
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    func cpuChunk(task: TaskState) -> Double {
        let n = task.spec.size
        let start = DispatchTime.now().uptimeNanoseconds
        let a = task.buffers.a.contents().bindMemory(to: Float.self, capacity: task.buffers.elementCount)
        let b = task.buffers.b.contents().bindMemory(to: Float.self, capacity: task.buffers.elementCount)
        let c = task.buffers.c.contents().bindMemory(to: Float.self, capacity: task.buffers.elementCount)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(n), Int32(n), Int32(n), 1.0, a, Int32(n), b, Int32(n), 0.0, c, Int32(n))
        let end = DispatchTime.now().uptimeNanoseconds
        return Double(end - start) / 1_000_000.0
    }

    func gpuChunk(task: TaskState) -> Double {
        let n = task.spec.size
        let descriptor = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * MemoryLayout<Float>.stride, dataType: .float32)
        let matrixA = MPSMatrix(buffer: task.buffers.a, descriptor: descriptor)
        let matrixB = MPSMatrix(buffer: task.buffers.b, descriptor: descriptor)
        let matrixC = MPSMatrix(buffer: task.buffers.c, descriptor: descriptor)
        let kernel: MPSMatrixMultiplication
        if let cached = kernels[n] {
            kernel = cached
        } else {
            kernel = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: n, resultColumns: n, interiorColumns: n, alpha: 1.0, beta: 0.0)
            kernels[n] = kernel
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Unable to create Metal command buffer")
        }
        let start = DispatchTime.now().uptimeNanoseconds
        kernel.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let end = DispatchTime.now().uptimeNanoseconds
        return Double(end - start) / 1_000_000.0
    }
}

struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x4d595df4d0f33173 : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let position = min(max(Int(Double(sorted.count - 1) * p), 0), sorted.count - 1)
    return sorted[position]
}

func nowISO() -> String {
    ISO8601DateFormatter().string(from: Date())
}

func thermalStateName() -> ThermalStateName {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal:
        return .nominal
    case .fair:
        return .fair
    case .serious:
        return .serious
    case .critical:
        return .critical
    @unknown default:
        return .critical
    }
}

func thermalLevel(_ state: ThermalStateName) -> Double {
    switch state {
    case .nominal: return 0.0
    case .fair: return 1.0
    case .serious: return 2.0
    case .critical: return 3.0
    }
}

func loadWorkload(path: String) throws -> WorkloadSpec {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(WorkloadSpec.self, from: data)
}

func loadProfile(path: String) throws -> ProfileStore {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONDecoder().decode(ProfileStore.self, from: data)
}

func writeJSON<T: Encodable>(_ value: T, path: String) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: URL(fileURLWithPath: path))
}

final class BenchmarkRunner {
    let policy: PolicyName
    let workload: WorkloadSpec
    let profile: ProfileStore
    let config: Config
    let executors: SharedExecutors
    let stateQueue = DispatchQueue(label: "m4.bench.state")
    let completionSemaphore = DispatchSemaphore(value: 0)

    var resources: [ResourceState] = []
    var tasks: [String: TaskState] = [:]
    var chunkEvents: [ChunkEvent] = []
    var schedulerDurationsUs: [Double] = []
    var nextArrivalTimerArmed = false
    var startNs: UInt64 = 0
    var thermalSamples: [ThermalStateName] = []
    var roundRobinIndex = 0

    init(policy: PolicyName, workload: WorkloadSpec, profile: ProfileStore, config: Config) {
        self.policy = policy
        self.workload = workload
        self.profile = profile
        self.config = config
        self.executors = SharedExecutors()
        for index in 0..<config.cpu_workers {
            resources.append(ResourceState(resourceID: "cpu\(index)", resourceType: .cpu))
        }
        resources.append(ResourceState(resourceID: "gpu0", resourceType: .gpu))
        var rng = SeededRNG(seed: UInt64(workload.seed) &+ 101)
        for spec in workload.task_specs {
            let buffers = MatrixBuffers(device: executors.device, size: spec.size, rng: &rng)
            tasks[spec.task_id] = TaskState(spec: spec, buffers: buffers, profile: profile)
        }
    }

    func elapsedMs() -> Double {
        let now = DispatchTime.now().uptimeNanoseconds
        return Double(now - startNs) / 1_000_000.0
    }

    func run() -> RunResult {
        startNs = DispatchTime.now().uptimeNanoseconds
        stateQueue.async {
            self.scheduleIfNeeded(trigger: "start")
        }
        completionSemaphore.wait()
        let taskMetrics = tasks.values.compactMap { task -> TaskMetric? in
            guard let completionMs = task.completionMs else { return nil }
            return TaskMetric(
                task_id: task.spec.task_id,
                family: task.spec.family,
                arrival_ms: task.spec.arrival_ms,
                completion_ms: completionMs,
                latency_ms: completionMs - task.spec.arrival_ms,
                priority: task.spec.priority,
                criticality: task.spec.criticality,
                size: task.spec.size,
                total_chunks: task.spec.total_chunks,
                cpu_chunks: task.cpuChunks,
                gpu_chunks: task.gpuChunks,
                migrations: task.migrations
            )
        }
        let latencies = taskMetrics.map(\.latency_ms)
        let queueWaits = chunkEvents.map(\.queue_wait_ms)
        let execs = chunkEvents.map(\.exec_ms)
        let nominalCount = thermalSamples.filter { $0 == .nominal }.count
        let predictedCPU = workload.task_specs.map { profile.durationMs(size: $0.size, resource: .cpu) }
        let predictedGPU = workload.task_specs.map { profile.durationMs(size: $0.size, resource: .gpu) }
        let summary = RunSummary(
            policy: policy,
            workload_id: workload.workload_id,
            family: workload.family,
            seed: workload.seed,
            cpu_workers: config.cpu_workers,
            tasks_total: taskMetrics.count,
            chunks_total: chunkEvents.count,
            makespan_ms: taskMetrics.map(\.completion_ms).max() ?? 0,
            throughput_tasks_per_s: taskMetrics.isEmpty ? 0 : Double(taskMetrics.count) / ((taskMetrics.map(\.completion_ms).max() ?? 1) / 1000.0),
            mean_latency_ms: latencies.reduce(0, +) / Double(max(latencies.count, 1)),
            p95_latency_ms: percentile(latencies, 0.95),
            mean_queue_wait_ms: queueWaits.reduce(0, +) / Double(max(queueWaits.count, 1)),
            p95_queue_wait_ms: percentile(queueWaits, 0.95),
            mean_exec_ms: execs.reduce(0, +) / Double(max(execs.count, 1)),
            scheduler_mean_us: schedulerDurationsUs.reduce(0, +) / Double(max(schedulerDurationsUs.count, 1)),
            scheduler_p95_us: percentile(schedulerDurationsUs, 0.95),
            migrations_total: taskMetrics.map(\.migrations).reduce(0, +),
            cpu_chunks: chunkEvents.filter { $0.resource_type == .cpu }.count,
            gpu_chunks: chunkEvents.filter { $0.resource_type == .gpu }.count,
            mean_predicted_cpu_ms: predictedCPU.reduce(0, +) / Double(max(predictedCPU.count, 1)),
            mean_predicted_gpu_ms: predictedGPU.reduce(0, +) / Double(max(predictedGPU.count, 1)),
            thermal_nominal_fraction: thermalSamples.isEmpty ? 1.0 : Double(nominalCount) / Double(thermalSamples.count),
            thermal_non_nominal_events: thermalSamples.filter { $0 != .nominal }.count,
            explicit_copy_bytes_after_init: 0
        )
        return RunResult(summary: summary, task_metrics: taskMetrics.sorted(by: { $0.task_id < $1.task_id }), chunk_events: chunkEvents, config: config)
    }

    func scheduleIfNeeded(trigger: String) {
        let begin = DispatchTime.now().uptimeNanoseconds
        let now = elapsedMs()
        thermalSamples.append(thermalStateName())
        enqueueNewlyReady(now: now)
        if policy == .caphs {
            rebalanceWaitingTasks(now: now)
        }
        for resource in resources where resource.runningTaskID == nil {
            if policy == .work_stealing && resource.waitingTaskIDs.isEmpty {
                stealIfPossible(for: resource)
            }
            if let nextTask = popNextTask(for: resource) {
                startChunk(taskID: nextTask, resource: resource, now: now)
            }
        }
        armArrivalTimerIfNeeded(now: now)
        if tasks.values.allSatisfy(\.isCompleted) {
            completionSemaphore.signal()
        }
        let end = DispatchTime.now().uptimeNanoseconds
        schedulerDurationsUs.append(Double(end - begin) / 1_000.0)
    }

    func enqueueNewlyReady(now: Double) {
        let candidates = tasks.values.filter { task in
            !task.isCompleted &&
            !task.isRunning &&
            !task.isQueued &&
            task.spec.arrival_ms <= now &&
            task.spec.parents.allSatisfy { tasks[$0]?.isCompleted == true }
        }
        let ordered: [TaskState]
        switch policy {
        case .fixed_priority:
            ordered = candidates.sorted {
                if $0.spec.priority != $1.spec.priority { return $0.spec.priority > $1.spec.priority }
                if $0.spec.criticality != $1.spec.criticality { return $0.spec.criticality > $1.spec.criticality }
                if $0.spec.arrival_ms != $1.spec.arrival_ms { return $0.spec.arrival_ms < $1.spec.arrival_ms }
                return $0.spec.task_id < $1.spec.task_id
            }
        default:
            ordered = candidates.sorted {
                if $0.spec.arrival_ms != $1.spec.arrival_ms { return $0.spec.arrival_ms < $1.spec.arrival_ms }
                return $0.spec.task_id < $1.spec.task_id
            }
        }
        for task in ordered {
            if let resource = selectResourceForEnqueue(task: task, now: now) {
                resource.waitingTaskIDs.append(task.spec.task_id)
                task.isQueued = true
                task.queueEnterMs = now
            }
        }
    }

    func popNextTask(for resource: ResourceState) -> String? {
        guard !resource.waitingTaskIDs.isEmpty else { return nil }
        let taskID = resource.waitingTaskIDs.removeFirst()
        tasks[taskID]?.isQueued = false
        return taskID
    }

    func startChunk(taskID: String, resource: ResourceState, now: Double) {
        guard let task = tasks[taskID] else { return }
        task.isRunning = true
        resource.runningTaskID = taskID
        let queueWait = max(0, now - (task.queueEnterMs ?? now))
        task.queueEnterMs = nil
        let migrated = task.lastResourceType != nil && task.lastResourceType != resource.resourceType
        resource.queue.async {
            let startMs = self.elapsedMs()
            let execMs = resource.resourceType == .cpu ? self.executors.cpuChunk(task: task) : self.executors.gpuChunk(task: task)
            let endMs = self.elapsedMs()
            self.stateQueue.async {
                self.finishChunk(taskID: taskID, resource: resource, queueWait: queueWait, execMs: execMs, startMs: startMs, endMs: endMs, migrated: migrated)
            }
        }
    }

    func finishChunk(taskID: String, resource: ResourceState, queueWait: Double, execMs: Double, startMs: Double, endMs: Double, migrated: Bool) {
        guard let task = tasks[taskID] else { return }
        resource.runningTaskID = nil
        task.isRunning = false
        task.chunksDone += 1
        if resource.resourceType == .cpu {
            task.cpuChunks += 1
        } else {
            task.gpuChunks += 1
        }
        if migrated {
            task.migrations += 1
        }
        updatePreference(task: task, resource: resource.resourceType, observedMs: execMs)
        task.lastResourceType = resource.resourceType
        task.lastResourceID = resource.resourceID
        if task.cooldownRemaining > 0 {
            task.cooldownRemaining -= 1
        }
        chunkEvents.append(
            ChunkEvent(
                task_id: task.spec.task_id,
                chunk_index: task.chunksDone,
                resource_id: resource.resourceID,
                resource_type: resource.resourceType,
                queue_wait_ms: queueWait,
                exec_ms: execMs,
                chunk_start_ms: startMs,
                chunk_end_ms: endMs,
                thermal_state: thermalStateName(),
                migrated: migrated
            )
        )
        if task.isCompleted {
            task.completionMs = endMs
        } else {
            task.isQueued = false
            if migrated {
                task.cooldownRemaining = config.cooldown_chunks
            }
        }
        scheduleIfNeeded(trigger: "completion")
    }

    func updatePreference(task: TaskState, resource: ResourceType, observedMs: Double) {
        let eta = config.eta
        let observedReward = -observedMs
        for type in ResourceType.allCases {
            let targetReward = type == resource ? observedReward : -profile.durationMs(size: task.spec.size, resource: type)
            let current = task.preference[type] ?? targetReward
            task.preference[type] = (1.0 - eta) * current + eta * targetReward
        }
    }

    func armArrivalTimerIfNeeded(now: Double) {
        if nextArrivalTimerArmed { return }
        let futureTimes = tasks.values.compactMap { task -> Double? in
            guard !task.isCompleted && !task.isQueued && !task.isRunning else { return nil }
            guard task.spec.arrival_ms > now else { return nil }
            return task.spec.arrival_ms
        }
        guard let nextTime = futureTimes.min() else { return }
        nextArrivalTimerArmed = true
        let delayNs = UInt64(max(0, (nextTime - now) * 1_000_000.0))
        DispatchQueue.global().asyncAfter(deadline: .now() + .nanoseconds(Int(delayNs))) {
            self.stateQueue.async {
                self.nextArrivalTimerArmed = false
                self.scheduleIfNeeded(trigger: "arrival_timer")
            }
        }
    }

    func stealIfPossible(for resource: ResourceState) {
        guard let donor = resources
            .filter({ $0.resourceID != resource.resourceID && !$0.waitingTaskIDs.isEmpty })
            .max(by: { estimatedBacklogMs(for: $0) < estimatedBacklogMs(for: $1) }),
              estimatedBacklogMs(for: donor) > estimatedBacklogMs(for: resource) else { return }
        guard let stolen = donor.waitingTaskIDs.popLast() else { return }
        resource.waitingTaskIDs.append(stolen)
        tasks[stolen]?.isQueued = true
        tasks[stolen]?.queueEnterMs = elapsedMs()
    }

    func selectResourceForEnqueue(task: TaskState, now: Double) -> ResourceState? {
        switch policy {
        case .fcfs:
            return resources.min(by: { resourceSortKey($0) < resourceSortKey($1) })
        case .least_loaded:
            return resources.min(by: { estimatedBacklogMs(for: $0) < estimatedBacklogMs(for: $1) })
        case .fixed_priority:
            return selectFixedPriorityResource(task: task)
        case .work_stealing:
            let index = roundRobinIndex % resources.count
            roundRobinIndex += 1
            return resources[index]
        case .caphs:
            return selectCAPHSResource(task: task, now: now)
        }
    }

    func selectFixedPriorityResource(task: TaskState) -> ResourceState? {
        let cpuMs = profile.durationMs(size: task.spec.size, resource: .cpu)
        let gpuMs = profile.durationMs(size: task.spec.size, resource: .gpu)
        let preferredType: ResourceType = cpuMs <= gpuMs ? .cpu : .gpu
        let candidates = resources.filter { $0.resourceType == preferredType }
        if let best = candidates.min(by: { estimatedBacklogMs(for: $0) < estimatedBacklogMs(for: $1) }) {
            return best
        }
        return resources.min(by: { estimatedBacklogMs(for: $0) < estimatedBacklogMs(for: $1) })
    }

    func selectCAPHSResource(task: TaskState, now: Double) -> ResourceState? {
        let currentType = task.lastResourceType
        let currentScore = currentType.flatMap { type in
            resources.filter { $0.resourceType == type }.map { score(task: task, resource: $0, now: now) }.max()
        } ?? -Double.infinity
        let scored = resources.map { resource in
            (resource, score(task: task, resource: resource, now: now))
        }.sorted { lhs, rhs in
            if lhs.1 != rhs.1 { return lhs.1 > rhs.1 }
            return resourceSortKey(lhs.0) < resourceSortKey(rhs.0)
        }
        guard let best = scored.first else { return nil }
        if let currentType, currentType != best.0.resourceType {
            let gain = best.1 - currentScore
            if task.cooldownRemaining > 0 || gain < config.migration_gain_threshold {
                let sameType = scored.first { $0.0.resourceType == currentType }
                return sameType?.0 ?? best.0
            }
        }
        return best.0
    }

    func rebalanceWaitingTasks(now: Double) {
        let waitingTasks = tasks.values.filter { $0.isQueued && !$0.isRunning && !$0.isCompleted }
        for task in waitingTasks {
            guard let currentID = resources.first(where: { $0.waitingTaskIDs.contains(task.spec.task_id) })?.resourceID else { continue }
            guard let better = selectCAPHSResource(task: task, now: now), better.resourceID != currentID else { continue }
            guard let currentResource = resources.first(where: { $0.resourceID == currentID }) else { continue }
            let currentScore = score(task: task, resource: currentResource, now: now)
            let betterScore = score(task: task, resource: better, now: now)
            if task.cooldownRemaining == 0 && betterScore - currentScore >= config.migration_gain_threshold {
                currentResource.waitingTaskIDs.removeAll { $0 == task.spec.task_id }
                better.waitingTaskIDs.append(task.spec.task_id)
                task.queueEnterMs = now
            }
        }
    }

    func score(task: TaskState, resource: ResourceState, now: Double) -> Double {
        let pref = task.preference[resource.resourceType] ?? -profile.durationMs(size: task.spec.size, resource: resource.resourceType)
        let backlog = estimatedBacklogMs(for: resource)
        let bandwidth = estimatedBandwidthPressure(for: resource)
        let thermalPenalty = thermalLevel(thermalStateName()) * (resource.resourceType == .gpu ? 1.2 : 0.5)
        let migrationPenalty = (task.lastResourceType != nil && task.lastResourceType != resource.resourceType) ? 1.0 : 0.0
        return pref
            - config.queue_weight * log(1.0 + backlog)
            - config.bandwidth_weight * bandwidth
            - config.thermal_weight * thermalPenalty
            - config.migration_weight * migrationPenalty
            + config.priority_weight * Double(task.spec.priority)
            + config.criticality_weight * task.spec.criticality
    }

    func estimatedBacklogMs(for resource: ResourceState) -> Double {
        var total = 0.0
        if let runningID = resource.runningTaskID, let task = tasks[runningID] {
            total += profile.durationMs(size: task.spec.size, resource: resource.resourceType)
        }
        for taskID in resource.waitingTaskIDs {
            if let task = tasks[taskID] {
                total += profile.durationMs(size: task.spec.size, resource: resource.resourceType)
            }
        }
        return total
    }

    func estimatedBandwidthPressure(for resource: ResourceState) -> Double {
        var bytes = 0.0
        if let runningID = resource.runningTaskID, let task = tasks[runningID] {
            bytes += task.spec.bytes_moved
        }
        for taskID in resource.waitingTaskIDs {
            if let task = tasks[taskID] {
                bytes += task.spec.bytes_moved
            }
        }
        return bytes / 1_000_000_000.0
    }

    func resourceSortKey(_ resource: ResourceState) -> String {
        if resource.resourceType == .cpu {
            return "0_\(resource.resourceID)"
        }
        return "1_\(resource.resourceID)"
    }
}

func runProfile(sizes: [Int], reps: Int, outputPath: String) throws {
    let executors = SharedExecutors()
    var rng = SeededRNG(seed: 20260310)
    var entries: [ProfileEntry] = []
    for size in sizes {
        var cpuTimes: [Double] = []
        var gpuTimes: [Double] = []
        let spec = TaskSpec(
            task_id: "profile_\(size)",
            size: size,
            total_chunks: 1,
            priority: 1,
            arrival_ms: 0,
            parents: [],
            criticality: 0,
            family: "profile",
            bytes_moved: Double(3 * size * size * MemoryLayout<Float>.stride)
        )
        let buffers = MatrixBuffers(device: executors.device, size: size, rng: &rng)
        let profileStore = ProfileStore(
            device_name: executors.device.name,
            created_at: nowISO(),
            reps: reps,
            entries: [ProfileEntry(size: size, cpu_ms: 1.0, gpu_ms: 1.0)]
        )
        let task = TaskState(spec: spec, buffers: buffers, profile: profileStore)
        _ = executors.cpuChunk(task: task)
        _ = executors.gpuChunk(task: task)
        for _ in 0..<reps {
            cpuTimes.append(executors.cpuChunk(task: task))
            gpuTimes.append(executors.gpuChunk(task: task))
        }
        entries.append(
            ProfileEntry(
                size: size,
                cpu_ms: percentile(cpuTimes, 0.5),
                gpu_ms: percentile(gpuTimes, 0.5)
            )
        )
    }
    let store = ProfileStore(device_name: executors.device.name, created_at: nowISO(), reps: reps, entries: entries.sorted(by: { $0.size < $1.size }))
    try writeJSON(store, path: outputPath)
}

func loadDefaultConfig(cpuWorkers: Int) -> Config {
    Config(
        cpu_workers: cpuWorkers,
        eta: 0.68,
        queue_weight: 0.18,
        bandwidth_weight: 0.22,
        thermal_weight: 0.16,
        migration_weight: 0.18,
        priority_weight: 0.07,
        criticality_weight: 0.12,
        migration_gain_threshold: 0.18,
        cooldown_chunks: 1
    )
}

func parseArgument(named name: String) -> String? {
    guard let index = CommandLine.arguments.firstIndex(of: name), index + 1 < CommandLine.arguments.count else {
        return nil
    }
    return CommandLine.arguments[index + 1]
}

func parseFlag(_ flag: String) -> Bool {
    CommandLine.arguments.contains(flag)
}

do {
    guard let mode = parseArgument(named: "--mode") else {
        fputs("Missing --mode\n", stderr)
        exit(1)
    }
    if mode == "profile" {
        let sizes = (parseArgument(named: "--sizes") ?? "768,1024,1536,2048")
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let reps = Int(parseArgument(named: "--reps") ?? "5") ?? 5
        guard let output = parseArgument(named: "--output") else {
            fputs("Missing --output\n", stderr)
            exit(1)
        }
        try runProfile(sizes: sizes, reps: reps, outputPath: output)
    } else if mode == "run" {
        guard let workloadPath = parseArgument(named: "--workload"),
              let profilePath = parseArgument(named: "--profile"),
              let outputPath = parseArgument(named: "--output"),
              let policyRaw = parseArgument(named: "--policy"),
              let policy = PolicyName(rawValue: policyRaw) else {
            fputs("Missing required run arguments\n", stderr)
            exit(1)
        }
        let cpuWorkers = Int(parseArgument(named: "--cpu-workers") ?? "8") ?? 8
        let workload = try loadWorkload(path: workloadPath)
        let profile = try loadProfile(path: profilePath)
        let config = loadDefaultConfig(cpuWorkers: cpuWorkers)
        let runner = BenchmarkRunner(policy: policy, workload: workload, profile: profile, config: config)
        let result = runner.run()
        try writeJSON(result, path: outputPath)
    } else if mode == "probe" {
        let device = SharedExecutors().device
        let payload = [
            "device_name": device.name,
            "supports_unified_memory": true.description,
            "has_unified_memory": String(device.hasUnifiedMemory),
            "recommended_max_working_set_mb": String(device.recommendedMaxWorkingSetSize / 1024 / 1024),
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        FileHandle.standardOutput.write(data)
    } else {
        fputs("Unsupported mode \(mode)\n", stderr)
        exit(1)
    }
} catch {
    fputs("ERROR: \(error)\n", stderr)
    exit(1)
}
