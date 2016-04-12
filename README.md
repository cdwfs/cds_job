cds_job
========

This
[single header file](https://github.com/nothings/stb/blob/master/docs/other_libs.md)
C++11 library provides a lock-free, work-stealing job queue system. It is based on
an implementation described extensively by [Stefan Reinalter](http://www.molecular-matters.com/) on
his [blog](https://blog.molecular-matters.com/tag/job-system/).

*This library is a work in progress, and should not yet be used in production code.*

No documentation yet, but here's a list of things to keep in mind:
-	Each job's data is stored (as a copy) in the leftover space of the Job structure itself.
	-	If you pass more data to a job than will fit in this space, you fail.
	-	This should be handled with a different createJob() variant.
-	Each job can have a "parent" job specified at creation time, but this has nothing to do with job dependencies.
	It's merely a way to say "when you wait on a job that's a parent, you wait on all its children (recursively) as well."
-	Waiting on a job with waitForJob() is a read-only operation, completely orthogonal to job execution.
	-	waitForJob() does not attempt to steal or execute the specified job; instead, it causes the thread to process
		jobs from its own queue (and/or by stealing from other queues) until the specified job is finished.
	-	The job you're waiting on may very well be executed while you're waiting for it to complete. This is fine.
		The wait will not terminate until both the parent and all its children have been executed. Note that parent/child
		execution order is NOT guaranteed.
	-	Waiting on a dummy root job seems common enough that there should be a shortcut.
-	For efficiency, Jobs are allocated out of pools stored in TLS. Each worker thread has a maximum number of jobs it can have "in-flight"
	simultaneously.
	-	There is currently no check to enforce the maxJobsPerThread limit; accidentally exceeding this limit is by far the most common source
		of nasty bugs I've encountered so far.
	-	This scheme is best suited to cases where the worker threads are generating their own work (in roughly similar quantities),
		rather than one master thread generating an entire workload and letting the workers divvy it up. Although, see parallel_for()
		for an example of the latter strategy in action.
-	Currently, threads that can't find work to do will call YieldProcessor(). This is effectively a busy-wait, and is totally
	inappropriate in production code. The original author suggests various approaches to put workers to sleep when there's no work left,
	and wake them again when more is ready.

Key Features / Design Goals
---------------------------
- **Identical API on all supported platforms**. The following
  platforms are tested regularly:
  - Microsoft Windows 7
    - Visual Studio 2010
    - Visual Studio 2012
    - Visual Studio 2013
  - Linux Mint
    - LLVM/Clang 3.5
    - gcc 4.8.4
  - Apple OSX
    - Apple LLVM/Clang 6.1.0
- **No (mandatory) external dependencies**. Only C++11 standard library
  functions are used.
- **Dirt-simple integration**. Just a single header file to include in
your project.
- **Public domain license terms**. 

Acknowledgements
----------------
- [Sean Barrett](http://nothings.org/): master of single-header C libraries.
- [Stefan Reinalter](https://blog.molecular-matters.com/tag/job-system/): author of the Molecule Engine.
