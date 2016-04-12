/* cds_job.h -- Lock-free job queue in C++11
 *              No warranty implied; use at your own risk.
 *
 * Do this:
 *   #define CDS_JOB_IMPLEMENTATION
 * before including this file in *one* C/C++ file to provide the function
 * implementations.
 *
 * For a unit test on g++/Clang:
 *   cc -Wall -pthread -std=c++11 -D_POSIX_C_SOURCE=199309L -g -x c -DCDS_JOB_TEST -o test_cds_job.exe cds_job.h -lstdc++ -lpthread
 * Clang users may also pass -fsanitize=thread to enable Clang's
 * ThreadSanitizer feature.
 *
 * For a unit test on Visual C++:
 *   "%VS120COMNTOOLS%\..\..\VC\vcvarsall.bat"
 *   cl -W4 -MT -nologo -EHsc -TP -DCDS_JOB_TEST /Fetest_cds_job.exe cds_job.h
 * Debug-mode:
 *   cl -W4 -Od -Z7 -FC -MTd -nologo -EHsc -TP -DCDS_JOB_TEST /Fetest_cds_job.exe cds_job.h
 *
 * LICENSE:
 * This software is in the public domain. Where that dedication is not
 * recognized, you are granted a perpetual, irrevocable license to
 * copy, distribute, and modify this file as you see fit.
 */

#ifndef CDS_JOB_H
#define CDS_JOB_H

#include <stdint.h>

namespace cds {
    namespace job {
        struct Job;
        class Context;
        typedef void (*JobFunction)(struct Job*, const void*);

        // Called by main thread to create the shared job context for a pool of worker threads.
        Context *createContext(int numWorkers, int maxJobsPerWorker);

        // Called by each worker thread.
        int initWorker(Context *ctx);

        // Called by worker threads to create a new job to execute. This function does *not* enqueue the new job for execution.
        Job *createJob(JobFunction function, Job *parent, const void *embeddedData, size_t embeddedDataBytes);

        // Called by worker threads to enqueue a job for execution. This gives the next available thread permission to execute this
        // job. All prior dependencies must be complete before a job is enqueued.
        int enqueueJob(Job *job);

        // Fetch and run any available queued jobs until the specified job is complete.
        void waitForJob(const Job *job);

        // Return the worker ID of the calling thread. If initWorker()
        // was called by this thread, the worker ID will be an index
        // from [0..numWorkers-1]. Otherwise, the worker ID is undefined.
        int workerId(void);

        template <typename T, typename S>
        struct ParallelForJobData {
            typedef T DataType;
            typedef S SplitterType;
            typedef void (*FunctionType)(DataType*, unsigned int, void*);

            ParallelForJobData(DataType* data, unsigned int count, void *userData, FunctionType function, const SplitterType& splitter)
                :    data(data)
                ,    userData(userData)
                ,    function(function)
                ,    splitter(splitter)
                ,    count(count)
                {
                }

            DataType* data;
            void *userData;
            FunctionType function;
            SplitterType splitter;
            unsigned int count;
        };

        template <typename T, typename S>
        Job* createParallelForJob(T* data, unsigned int count, void *userData, void (*function)(T*, unsigned int, void*),
            const S& splitter, Job *parent = nullptr)
        {
            typedef ParallelForJobData<T, S> JobData;
            const JobData jobData(data, count, userData, function, splitter);

            return createJob(parallelForJobFunc<JobData>, parent, &jobData, sizeof(jobData));
        }

        template <typename JobData>
        void parallelForJobFunc(struct Job* job, const void* jobData) {
            const JobData* data = static_cast<const JobData*>(jobData);
            const JobData::SplitterType& splitter = data->splitter;
            if (splitter.split<JobData::DataType>(data->count)) {
                // split in two
                const unsigned int leftCount = data->count / 2U;
                const JobData leftData(data->data + 0, leftCount, data->userData, data->function, splitter);
                Job *leftJob = createJob(parallelForJobFunc<JobData>, job, &leftData, sizeof(leftData));
                enqueueJob( leftJob );

                const unsigned int rightCount = data->count - leftCount;
                const JobData rightData(data->data + leftCount, rightCount, data->userData, data->function, splitter);
                Job *rightJob = createJob(parallelForJobFunc<JobData>, job, &rightData, sizeof(rightData));
                enqueueJob( rightJob );
            } else {
                // execute the function on the range of data
                (data->function)(data->data, data->count, data->userData);
            }
        }

        class CountSplitter {
        public:
            explicit CountSplitter(unsigned int count) : m_count(count) {}
            template <typename T> inline bool split(unsigned int count) const { return (count > m_count); }
        private:
            unsigned int m_count;
        };

        class DataSizeSplitter {
        public:
            explicit DataSizeSplitter(unsigned int size) : m_size(size) {}
            template <typename T> inline bool split(unsigned int count) const { return (count*sizeof(T) > m_size); }
        private:
            unsigned int m_size;
        };
    }
}

#endif ////////////////////////////////////// end header file

#if defined(CDS_JOB_TEST)
#   if !defined(CDS_JOB_IMPLEMENTATION)
#       define CDS_JOB_IMPLEMENTATION
#   endif
#endif

#ifdef CDS_JOB_IMPLEMENTATION

#if   defined(_MSC_VER)
#   if _MSC_VER < 1900
#       define CDS_JOB_THREADLOCAL __declspec(thread)
#   else
#       define CDS_JOB_THREADLOCAL thread_local
#   endif
#elif defined(__GNUC__)
#   define CDS_JOB_THREADLOCAL __thread
#elif defined(__clang__)
#   if defined(__APPLE__) || defined(__MACH__)
#       define CDS_JOB_THREADLOCAL __thread
#   else
#       define CDS_JOB_THREADLOCAL thread_local
#   endif
#endif

#ifdef _MSC_VER
#   include <windows.h>
#   define JOB_YIELD() YieldProcessor()
#   define JOB_COMPILER_BARRIER _ReadWriteBarrier()
#   define JOB_MEMORY_BARRIER std::atomic_thread_fence(std::memory_order_seq_cst);
#else
#   include <emmintrin.h>
#   define JOB_YIELD() _mm_pause()
#   define JOB_COMPILER_BARRIER asm volatile("" ::: "memory")
#   define JOB_MEMORY_BARRIER asm volatile("mfence" ::: "memory")
#endif

#include <assert.h>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace cds::job;

namespace {
    class WorkStealingQueue {
    public:
        static size_t BufferSize(int capacity) {
            return capacity*sizeof(Job*);
        }

        int Init(int capacity, void *buffer, size_t bufferSize);
        int Push(Job *job);
        Job *Pop();
        Job *Steal();

    private:
        Job **m_entries;
        std::atomic<uint64_t> m_top;
        uint64_t m_bottom;
        int m_capacity;
    };
}

int WorkStealingQueue::Init(int capacity, void *buffer, size_t bufferSize) {
    if ( (capacity & (capacity-1)) != 0) {
        return -2; // capacity must be a power of 2
    }
    size_t minBufferSize = BufferSize(capacity);
    if (bufferSize < minBufferSize) {
        return -1; // inadequate buffer size
    }
    uint8_t *bufferNext = (uint8_t*)buffer;
    m_entries = (Job**)bufferNext;
    bufferNext += capacity*sizeof(Job*);
    assert( bufferNext - (uint8_t*)buffer == (intptr_t)minBufferSize );

    for(int iEntry=0; iEntry<capacity; iEntry+=1) {
        m_entries[iEntry] = nullptr;
    }

    m_top = 0;
    m_bottom = 0;
    m_capacity = capacity;

    return 0;
}

int WorkStealingQueue::Push(Job *job) {
    // TODO: assert that this is only ever called by the owning thread
    uint64_t jobIndex = m_bottom;
    m_entries[jobIndex & (m_capacity-1)] = job;

    // Ensure the job is written before the m_bottom increment is published.
    // A StoreStore memory barrier would also be necessary on platforms with a weak memory model.
    JOB_COMPILER_BARRIER;

    m_bottom = jobIndex+1;
    return 0;
}
Job *WorkStealingQueue::Pop() {
    // TODO: assert that this is only ever called by the owning thread
    uint64_t bottom = m_bottom-1;
    m_bottom = bottom;

    // Make sure m_bottom is published before reading top.
    // Requires a full StoreLoad memory barrier, even on x86/64.
    JOB_MEMORY_BARRIER;

    uint64_t top = m_top;
    if (top <= bottom) {
        Job *job = m_entries[bottom & (m_capacity-1)];
        if (top != bottom) {
            // still >0 jobs left in the queue
            return job;
        } else {
            // popping the last element in the queue
            if (!std::atomic_compare_exchange_strong(&m_top, &top, top)) {
                // failed race against Steal()
                job = nullptr;
            }
            m_bottom = top+1;
            return job;
        }
    } else {
        // queue already empty
        m_bottom = top;
        return nullptr;
    }
}
Job *WorkStealingQueue::Steal() {
    // TODO: assert that this is never called by the owning thread
    uint64_t top    = m_top;

    // Ensure top is always read before bottom.
    // A LoadLoad memory barrier would also be necessary on platforms with a weak memory model.
    JOB_COMPILER_BARRIER;

    uint64_t bottom = m_bottom;
    if (top < bottom) {
        Job *job = m_entries[top & (m_capacity-1)];
        // CAS serves as a compiler barrier as-is.
        if (!std::atomic_compare_exchange_strong(&m_top, &top, top+1)) {
            // concurrent Steal()/Pop() got this entry first.
            return nullptr;
        }
        m_entries[top & (m_capacity-1)] = nullptr;
        return job;
    } else {
        return nullptr; // queue empty
    }
}

///////////////////

#define kCdsJobCacheLineBytes 64
#define kCdsJobPaddingBytes ( (kCdsJobCacheLineBytes) - (sizeof(JobFunction) + sizeof(struct Job*) + sizeof(void*) + sizeof(std::atomic_int_fast32_t)) )

#ifdef _MSC_VER
#   define JOB_ATTR_ALIGN(alignment) __declspec(align(alignment))
#else
#   define JOB_ATTR_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif

namespace cds {
    namespace job {
        typedef JOB_ATTR_ALIGN(kCdsJobCacheLineBytes) struct Job {
            JobFunction function;
            struct Job *parent;
            void *data;
            std::atomic_int_fast32_t unfinishedJobs;
            char padding[kCdsJobPaddingBytes];
        } Job;

        class Context {
        public:
            Context() = delete;
            Context(const Context &ctx) = delete;
            Context(int numWorkerThreads, int maxJobsPerThread);
            ~Context();

            WorkStealingQueue **m_workerJobQueues;
            void *m_jobPoolBuffer;
            void *m_queueEntryBuffer;
            std::atomic<int> m_nextWorkerId;
            int m_numWorkerThreads;
            int m_maxJobsPerThread;
        };
    }
}

static_assert((sizeof(struct Job) % kCdsJobCacheLineBytes) == 0, "Job struct is not cache-line-aligned!");

static CDS_JOB_THREADLOCAL Context *tls_jobContext = nullptr;
static CDS_JOB_THREADLOCAL uint64_t tls_jobCount = 0;
static CDS_JOB_THREADLOCAL int tls_workerId = -1;
static CDS_JOB_THREADLOCAL Job *tls_jobPool = nullptr;

static inline uint32_t nextPowerOfTwo(uint32_t x)
{
    x = x-1;
    x = x | (x>> 1);
    x = x | (x>> 2);
    x = x | (x>> 4);
    x = x | (x>> 8);
    x = x | (x>>16);
    return x+1;
}

Context::Context(int numWorkerThreads, int maxJobsPerThread)
    :   m_workerJobQueues(nullptr)
    ,   m_nextWorkerId(0)
    ,   m_numWorkerThreads(numWorkerThreads)
{
    maxJobsPerThread = nextPowerOfTwo(maxJobsPerThread);
    m_maxJobsPerThread = maxJobsPerThread;

    m_workerJobQueues = new WorkStealingQueue*[numWorkerThreads];
    const size_t jobPoolBufferSize = numWorkerThreads*maxJobsPerThread*sizeof(Job) + kCdsJobCacheLineBytes - 1;
    m_jobPoolBuffer = malloc(jobPoolBufferSize);
    size_t queueBufferSize = WorkStealingQueue::BufferSize(maxJobsPerThread);
    m_queueEntryBuffer = malloc(queueBufferSize * numWorkerThreads);
    for(int iWorker=0; iWorker<numWorkerThreads; ++iWorker)
    {
        m_workerJobQueues[iWorker] = new WorkStealingQueue();
        int initError = m_workerJobQueues[iWorker]->Init(
            maxJobsPerThread,
            (void*)( intptr_t(m_queueEntryBuffer) + iWorker*queueBufferSize ),
            queueBufferSize);
        (void)initError;
        assert(initError == 0);
    }

}
Context::~Context()
{
    for(int iWorker=0; iWorker<m_numWorkerThreads; ++iWorker)
    {
        delete m_workerJobQueues[iWorker];
    }
    delete [] m_workerJobQueues;
    free(m_queueEntryBuffer);
    free(m_jobPoolBuffer);
}

static inline Job *AllocateJob() {
    // TODO(cort): no protection against over-allocation
    uint64_t index = tls_jobCount++;
    return &tls_jobPool[index & (tls_jobContext->m_maxJobsPerThread-1)];
}

static inline bool IsJobComplete(const Job *job) {
    return (job->unfinishedJobs == 0);
}

static void FinishJob(Job *job) {
    const int32_t unfinishedJobs = --(job->unfinishedJobs);
    assert(unfinishedJobs >= 0);
    if (unfinishedJobs == 0 && job->parent) {
        FinishJob(job->parent);
    }
}

static Job *GetJob(void) {
    WorkStealingQueue *myQueue = tls_jobContext->m_workerJobQueues[tls_workerId];
    Job *job = myQueue->Pop();
    if (!job) {
        // this worker's queue is empty; try to steal a job from another thread
        int victimOffset = 1 + (rand() % tls_jobContext->m_numWorkerThreads-1);
        int victimIndex = (tls_workerId + victimOffset) % tls_jobContext->m_numWorkerThreads;
        WorkStealingQueue *victimQueue = tls_jobContext->m_workerJobQueues[victimIndex];
        job = victimQueue->Steal();
        if (!job) { // nothing to steal
            JOB_YIELD(); // TODO(cort): busy-wait bad, right? But there might be a job to steal in ANOTHER queue, so we should try again shortly.
            return nullptr;
        }
    }
    return job;
}

static inline void ExecuteJob(Job *job) {
    (job->function)(job, job->data);
    FinishJob(job);
}

Context *cds::job::createContext(int numWorkers, int maxJobsPerWorker)
{
    return new Context(numWorkers, maxJobsPerWorker);
}

int cds::job::initWorker(Context *ctx)
{
    tls_jobContext = ctx;
    tls_jobCount = 0;
    tls_workerId = ctx->m_nextWorkerId++;
    assert(tls_workerId < ctx->m_numWorkerThreads);
    void *jobPoolBufferAligned = (void*)( (uintptr_t(ctx->m_jobPoolBuffer) + kCdsJobCacheLineBytes-1) & ~(kCdsJobCacheLineBytes-1) );
    assert( (uintptr_t(jobPoolBufferAligned) % kCdsJobCacheLineBytes) == 0 );
    tls_jobPool = (Job*)(jobPoolBufferAligned) + tls_workerId*ctx->m_maxJobsPerThread;
    return tls_workerId;
}

Job *cds::job::createJob(JobFunction function, Job *parent, const void *embeddedData, size_t embeddedDataBytes) {
    if (embeddedData != nullptr && embeddedDataBytes > kCdsJobPaddingBytes) {
        assert(0);
        return NULL;
    }
    if (parent) {
        parent->unfinishedJobs++;
    }
    Job *job = AllocateJob();
    job->function = function;
    job->parent = parent;
    job->unfinishedJobs = 1;
    if (embeddedData) {
        memcpy(job->padding, embeddedData, embeddedDataBytes);
        job->data = job->padding;
    } else {
        job->data = nullptr;
    }
    return job;
}

// Enqueues a job for eventual execution
int cds::job::enqueueJob(Job *job) {
    int pushError = tls_jobContext->m_workerJobQueues[tls_workerId]->Push(job);
    return pushError;
}

// Fetch and run queued jobs until the specified job is complete
void cds::job::waitForJob(const Job *job) {
    while(!IsJobComplete(job)) {
        Job *nextJob = GetJob();
        if (nextJob) {
            ExecuteJob(nextJob);
        }
    }
}

int cds::job::workerId(void) {
    return tls_workerId;
}

#endif // defined(CDS_JOB_IMPLEMENTATION)

#ifdef CDS_JOB_TEST ////////////////////////////// test code

#define kNumWorkers 16
#define kTotalJobCount (64*1024)
static const int kMaxJobsPerThread = (kTotalJobCount / kNumWorkers);
static std::atomic_int_fast32_t g_finishedJobCount(0);

static void empty_job(Job *job, const void*data) {
    (void)job;
    (void)data;
    g_finishedJobCount++;
    //int *jobId = (int*)data;
    //printf("worker %2d, job 0x%08X\n", tls_workerId, *jobId);
}

static void emptyWorkerTest(Context *jobCtx) {
    int workerId = cds::job::initWorker(jobCtx);

    const int jobCount = jobCtx->m_maxJobsPerThread;
    int jobId = (workerId<<16) | 0;
    Job *root = createJob(empty_job, nullptr, &jobId, sizeof(int));
    enqueueJob(root);
    for(int iJob=1; iJob<jobCount; iJob+=1) {
        int jobId = (workerId<<16) | iJob;
        Job *job = createJob(empty_job, root, &jobId, sizeof(int));
        int addError = enqueueJob(job);
        assert(!addError);
    }
    waitForJob(root);
}

static void squareInts(uint64_t *data, unsigned int count, void *userData) {
    (void)userData;
    for(unsigned int i=0; i<count; ++i) {
        data[i] *= data[i];
    }
}

static void parallelForTest(Context *jobCtx, Job *rootJob) {
    cds::job::initWorker(jobCtx);
    waitForJob(rootJob);
}

#include <chrono>
#include <thread>

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    {
        cds::job::Context *jobCtx = cds::job::createContext(kNumWorkers, kMaxJobsPerThread);

        auto startTime = std::chrono::high_resolution_clock::now();
        std::thread workers[kNumWorkers];
        for(int iThread=0; iThread<kNumWorkers; iThread+=1) {
            workers[iThread] = std::thread(emptyWorkerTest, jobCtx);
        }

        for(int iThread=0; iThread<kNumWorkers; iThread+=1) {
            workers[iThread].join();
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime-startTime).count();
        printf("%d jobs complete in %.3fms\n", (int)g_finishedJobCount.load(), (double)elapsedNanos/1e6);
        delete jobCtx;
    }

    {
        const int kNumSquares = 1*1024*1024;
        uint64_t *squares = new uint64_t[kNumSquares];
        for(uint64_t i=0; i<kNumSquares; ++i) {
            squares[i] = i;
        }

        cds::job::Context *jobCtx = cds::job::createContext(kNumWorkers, kNumSquares/(32*1024/sizeof(uint64_t))); // TODO(cort): touchy touchy!

        auto startTime = std::chrono::high_resolution_clock::now();

        // in this test, the main thread is a worker.
        initWorker(jobCtx);
        Job *rootJob = createParallelForJob(squares, kNumSquares, nullptr, squareInts, DataSizeSplitter(32*1024), nullptr);
        enqueueJob(rootJob);

#if 0
        waitForJob(rootJob);
#else
        std::thread workers[kNumWorkers-1];
        for(int iThread=0; iThread<kNumWorkers-1; iThread+=1) {
            workers[iThread] = std::thread(parallelForTest, jobCtx, rootJob);
        }
        waitForJob(rootJob);
        for(int iThread=0; iThread<kNumWorkers-1; iThread+=1) {
            workers[iThread].join();
        }
#endif

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime-startTime).count();
        printf("%d jobs complete in %.3fms\n", kNumSquares, (double)elapsedNanos/1e6);
        for(uint64_t i=0; i<kNumSquares; ++i) {
            if (squares[i] != i*i) {
                printf("Error: squares[%lld] = %lld (expected %lld)\n", i, squares[i], i*i);
            }
        }
        printf("%d squares computed successfully\n", kNumSquares);
        free(squares);
        delete jobCtx;
    }
    return 0;
}
#endif // CDS_JOB_TEST
