(.venv) alex_yang@HarryYang2369:/mnt/d/IoT_Models/model_harry/pipeline_v4$ ./run.sh --folds 5,6,7,8,9 --gpu --workers 4
Python: ../../.venv/bin/python
2026-11-13 12:07:29 ===== pipeline v4 -> /mnt/d/IoT_Models/model_harry/pipeline_v4/results; recipe jan; seeds 0; folds [5, 6, 7, 8, 9]; candidates all; GPU
2026-11-13 12:07:31 35 jobs in the plan, 0 already finished, 35 to run (4 workers x 8 threads)
I0000 00:00:1794571677.326490    8646 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5560 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9
I0000 00:00:1794571677.345288    8641 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5560 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9
I0000 00:00:1794571677.788141    8645 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5560 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9
I0000 00:00:1794571677.805341    8644 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5560 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9
I0000 00:00:1794571679.098943    8843 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
I0000 00:00:1794571679.098965    8972 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
I0000 00:00:1794571679.161528    8978 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
I0000 00:00:1794571679.197849    8848 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
F0000 00:00:1794571747.562800    8841 device_event_mgr.cc:226] Unexpected Event status: 1
*** Check failure stack trace: ***
    @     0x74d49caa2fe4  absl::lts_20250127::log_internal::LogMessage::SendToLog()
    @     0x74d49caa2976  absl::lts_20250127::log_internal::LogMessage::Flush()
    @     0x74d49caa3539  absl::lts_20250127::log_internal::LogMessageFatal::~LogMessageFatal()
    @     0x74d49ac4b5ca  tensorflow::EventMgr::PollEvents()::$_0::operator()<>()
    @     0x74d49ac4adab  tensorflow::EventMgr::PollEvents()
    @     0x74d49ac4aaa0  tensorflow::EventMgr::PollLoop()
    @     0x74d49c76cec4  Eigen::ThreadPoolTempl<>::WorkerLoop()
    @     0x74d49c76cd41  std::__invoke_impl<>()
    @     0x74d49c758def  tsl::(anonymous namespace)::PThread::ThreadFn()
    @     0x74d4f029cb64  (unknown)
    @     0x74d4f0329e8c  (unknown)
Traceback (most recent call last):
  File "/mnt/d/IoT_Models/model_harry/pipeline_v4/run_v4.py", line 224, in <module>
    main()
  File "/mnt/d/IoT_Models/model_harry/pipeline_v4/run_v4.py", line 218, in main
    jobs.run_all(p, a.workers, threads, a.smoke, log)
  File "/mnt/d/IoT_Models/model_harry/pipeline_v4/fullrun4/jobs.py", line 342, in run_all
    jid, status, sec = f.result()
                       ^^^^^^^^^^
  File "/usr/lib/python3.12/concurrent/futures/_base.py", line 449, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.
(.venv) alex_yang@HarryYang2369:/mnt/d/IoT_Models/model_harry/pipeline_v4$
