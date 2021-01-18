 #include "tritonserver.h"
#include "http_server.h"


std::vector<Option> options_
{
  {OPTION_HELP, "help", Option::ArgNone, "Print usage"},
#ifdef TRITON_ENABLE_LOGGING
      {OPTION_LOG_VERBOSE, "log-verbose", Option::ArgInt,
       "Set verbose logging level. Zero (0) disables verbose logging and "
       "values >= 1 enable verbose logging."},
      {OPTION_LOG_INFO, "log-info", Option::ArgBool,
       "Enable/disable info-level logging."},
      {OPTION_LOG_WARNING, "log-warning", Option::ArgBool,
       "Enable/disable warning-level logging."},
      {OPTION_LOG_ERROR, "log-error", Option::ArgBool,
       "Enable/disable error-level logging."},
#endif  // TRITON_ENABLE_LOGGING
      {OPTION_ID, "id", Option::ArgStr, "Identifier for this server."},
      {OPTION_MODEL_REPOSITORY, "model-store", Option::ArgStr,
       "Equivalent to --model-repository."},
      {OPTION_MODEL_REPOSITORY, "model-repository", Option::ArgStr,
       "Path to model repository directory. It may be specified multiple times "
       "to add multiple model repositories. Note that if a model is not unique "
       "across all model repositories at any time, the model will not be "
       "available."},
      {OPTION_EXIT_ON_ERROR, "exit-on-error", Option::ArgBool,
       "Exit the inference server if an error occurs during initialization."},
      {OPTION_STRICT_MODEL_CONFIG, "strict-model-config", Option::ArgBool,
       "If true model configuration files must be provided and all required "
       "configuration settings must be specified. If false the model "
       "configuration may be absent or only partially specified and the "
       "server will attempt to derive the missing required configuration."},
      {OPTION_STRICT_READINESS, "strict-readiness", Option::ArgBool,
       "If true /v2/health/ready endpoint indicates ready if the server "
       "is responsive and all models are available. If false "
       "/v2/health/ready endpoint indicates ready if server is responsive "
       "even if some/all models are unavailable."},
#if defined(TRITON_ENABLE_HTTP)
      {OPTION_ALLOW_HTTP, "allow-http", Option::ArgBool,
       "Allow the server to listen for HTTP requests."},
      {OPTION_HTTP_PORT, "http-port", Option::ArgInt,
       "The port for the server to listen on for HTTP requests."},
      {OPTION_HTTP_THREAD_COUNT, "http-thread-count", Option::ArgInt,
       "Number of threads handling HTTP requests."},
#endif  // TRITON_ENABLE_HTTP
#if defined(TRITON_ENABLE_GRPC)
      {OPTION_ALLOW_GRPC, "allow-grpc", Option::ArgBool,
       "Allow the server to listen for GRPC requests."},
      {OPTION_GRPC_PORT, "grpc-port", Option::ArgInt,
       "The port for the server to listen on for GRPC requests."},
      {OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE,
       "grpc-infer-allocation-pool-size", Option::ArgInt,
       "The maximum number of inference request/response objects that remain "
       "allocated for reuse. As long as the number of in-flight requests "
       "doesn't exceed this value there will be no allocation/deallocation of "
       "request/response objects."},
      {OPTION_GRPC_USE_SSL, "grpc-use-ssl", Option::ArgBool,
       "Use SSL authentication for GRPC requests. Default is false."},
      {OPTION_GRPC_SERVER_CERT, "grpc-server-cert", Option::ArgStr,
       "File holding PEM-encoded server certificate. Ignored unless "
       "--grpc-use-ssl is true."},
      {OPTION_GRPC_SERVER_KEY, "grpc-server-key", Option::ArgStr,
       "File holding PEM-encoded server key. Ignored unless "
       "--grpc-use-ssl is true."},
      {OPTION_GRPC_ROOT_CERT, "grpc-root-cert", Option::ArgStr,
       "File holding PEM-encoded root certificate. Ignore unless "
       "--grpc-use-ssl is false."},
#endif  // TRITON_ENABLE_GRPC
#ifdef TRITON_ENABLE_METRICS
      {OPTION_ALLOW_METRICS, "allow-metrics", Option::ArgBool,
       "Allow the server to provide prometheus metrics."},
      {OPTION_ALLOW_GPU_METRICS, "allow-gpu-metrics", Option::ArgBool,
       "Allow the server to provide GPU metrics. Ignored unless "
       "--allow-metrics is true."},
      {OPTION_METRICS_PORT, "metrics-port", Option::ArgInt,
       "The port reporting prometheus metrics."},
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_TRACING
      {OPTION_TRACE_FILEPATH, "trace-file", Option::ArgStr,
       "Set the file where trace output will be saved."},
      {OPTION_TRACE_LEVEL, "trace-level", Option::ArgStr,
       "Set the trace level. OFF to disable tracing, MIN for minimal tracing, "
       "MAX for maximal tracing. Default is OFF."},
      {OPTION_TRACE_RATE, "trace-rate", Option::ArgInt,
       "Set the trace sampling rate. Default is 1000."},
#endif  // TRITON_ENABLE_TRACING
      {OPTION_MODEL_CONTROL_MODE, "model-control-mode", Option::ArgStr,
       "Specify the mode for model management. Options are \"none\", \"poll\" "
       "and \"explicit\". The default is \"none\". "
       "For \"none\", the server will load all models in the model "
       "repository(s) at startup and will not make any changes to the load "
       "models after that. For \"poll\", the server will poll the model "
       "repository(s) to detect changes and will load/unload models based on "
       "those changes. The poll rate is controlled by 'repository-poll-secs'. "
       "For \"explicit\", model load and unload is initiated by using the "
       "model control APIs, and only models specified with --load-model will "
       "be loaded at startup."},
      {OPTION_POLL_REPO_SECS, "repository-poll-secs", Option::ArgInt,
       "Interval in seconds between each poll of the model repository to check "
       "for changes. Valid only when --model-control-mode=poll is "
       "specified."},
      {OPTION_STARTUP_MODEL, "load-model", Option::ArgStr,
       "Name of the model to be loaded on server startup. It may be specified "
       "multiple times to add multiple models. Note that this option will only "
       "take affect if --model-control-mode=explicit is true."},
      {OPTION_PINNED_MEMORY_POOL_BYTE_SIZE, "pinned-memory-pool-byte-size",
       Option::ArgInt,
       "The total byte size that can be allocated as pinned system memory. "
       "If GPU support is enabled, the server will allocate pinned system "
       "memory to accelerate data transfer between host and devices until it "
       "exceeds the specified byte size. This option will not affect the "
       "allocation conducted by the backend frameworks. Default is 256 MB."},
      {OPTION_CUDA_MEMORY_POOL_BYTE_SIZE, "cuda-memory-pool-byte-size",
       "<integer>:<integer>",
       "The total byte size that can be allocated as CUDA memory for the GPU "
       "device. If GPU support is enabled, the server will allocate CUDA "
       "memory to minimize data transfer between host and devices until it "
       "exceeds the specified byte size. This option will not affect the "
       "allocation conducted by the backend frameworks. The argument should be "
       "2 integers separated by colons in the format "
       "<GPU device ID>:<pool byte size>. This option can be used multiple "
       "times, but only once per GPU device. Subsequent uses will overwrite "
       "previous uses for the same GPU device. Default is 64 MB."},
      {OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY,
       "min-supported-compute-capability", Option::ArgFloat,
       "The minimum supported CUDA compute capability. GPUs that don't support "
       "this compute capability will not be used by the server."},
      {OPTION_EXIT_TIMEOUT_SECS, "exit-timeout-secs", Option::ArgInt,
       "Timeout (in seconds) when exiting to wait for in-flight inferences to "
       "finish. After the timeout expires the server exits even if inferences "
       "are still in flight."},
      {OPTION_BACKEND_DIR, "backend-directory", Option::ArgStr,
       "The global directory searched for backend shared libraries. Default is "
       "'/opt/tritonserver/backends'."},
      {OPTION_BUFFER_MANAGER_THREAD_COUNT, "buffer-manager-thread-count",
       Option::ArgInt,
       "The number of threads used to accelerate copies and other operations "
       "required to manage input and output tensor contents. Default is 0."},
  {
    OPTION_BACKEND_CONFIG, "backend-config", "<string>,<string>=<string>",
        "Specify a backend-specific configuration setting. The format of this "
        "flag is --backend-config=<backend_name>,<setting>=<value>. Where "
        "<backend_name> is the name of the backend, such as 'tensorrt'."
  }
};


bool
Parse(TRITONSERVER_ServerOptions** server_options, int argc, char** argv)
{
    double min_supported_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;

#if defined(TRITON_ENABLE_HTTP)
    std::unique_ptr<nvidia::inferenceserver::HTTPServer> http_service_;
    bool allow_http_ = true;
    int32_t http_port_ = 8000;
    int32_t http_port = http_port_;
    int32_t http_thread_cnt = http_thread_cnt_;
#endif  // TRITON_ENABLE_HTTP



#ifdef TRITON_ENABLE_TRACING
    std::string trace_filepath = trace_filepath_;
    TRITONSERVER_InferenceTraceLevel trace_level = trace_level_;
    int32_t trace_rate = trace_rate_;
#endif  // TRITON_ENABLE_TRACING

    TRITONSERVER_ModelControlMode control_mode = TRITONSERVER_MODEL_CONTROL_NONE;
    std::set<std::string> startup_models_;

#ifdef TRITON_ENABLE_LOGGING
    bool log_info = true;
    bool log_warn = true;
    bool log_error = true;
    int32_t log_verbose = 0;
#endif  // TRITON_ENABLE_LOGGING

    std::vector<struct option> long_options;
    for (const auto& o : options_) {
        long_options.push_back(o.GetLongOption());
    }
    long_options.push_back({ nullptr, 0, nullptr, 0 });

    int flag;
//    while ((flag = getopt_long(argc, argv, "", &long_options[0], NULL)) != -1) {
//        switch (flag) {
//        case OPTION_HELP:
//        case '?':
//            std::cerr << Usage() << std::endl;
//            return false;
//#ifdef TRITON_ENABLE_LOGGING
//        case OPTION_LOG_VERBOSE:
//            log_verbose = ParseIntBoolOption(optarg);
//            break;
//        case OPTION_LOG_INFO:
//            log_info = ParseBoolOption(optarg);
//            break;
//        case OPTION_LOG_WARNING:
//            log_warn = ParseBoolOption(optarg);
//            break;
//        case OPTION_LOG_ERROR:
//            log_error = ParseBoolOption(optarg);
//            break;
//#endif  // TRITON_ENABLE_LOGGING
//
//        case OPTION_ID:
//            server_id = optarg;
//            break;
//        case OPTION_MODEL_REPOSITORY:
//            model_repository_paths.insert(optarg);
//            break;
//
//        case OPTION_EXIT_ON_ERROR:
//            exit_on_error = ParseBoolOption(optarg);
//            break;
//        case OPTION_STRICT_MODEL_CONFIG:
//            strict_model_config = ParseBoolOption(optarg);
//            break;
//        case OPTION_STRICT_READINESS:
//            strict_readiness = ParseBoolOption(optarg);
//            break;
//
//#if defined(TRITON_ENABLE_HTTP)
//        case OPTION_ALLOW_HTTP:
//            allow_http_ = ParseBoolOption(optarg);
//            break;
//        case OPTION_HTTP_PORT:
//            http_port = ParseIntOption(optarg);
//            break;
//        case OPTION_HTTP_THREAD_COUNT:
//            http_thread_cnt = ParseIntOption(optarg);
//            break;
//#endif  // TRITON_ENABLE_HTTP
//
//#if defined(TRITON_ENABLE_GRPC)
//        case OPTION_ALLOW_GRPC:
//            allow_grpc_ = ParseBoolOption(optarg);
//            break;
//        case OPTION_GRPC_PORT:
//            grpc_port = ParseIntOption(optarg);
//            break;
//        case OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE:
//            grpc_infer_allocation_pool_size = ParseIntOption(optarg);
//            break;
//        case OPTION_GRPC_USE_SSL:
//            grpc_use_ssl = ParseBoolOption(optarg);
//            break;
//        case OPTION_GRPC_SERVER_CERT:
//            grpc_ssl_options_.server_cert = optarg;
//            break;
//        case OPTION_GRPC_SERVER_KEY:
//            grpc_ssl_options_.server_key = optarg;
//            break;
//        case OPTION_GRPC_ROOT_CERT:
//            grpc_ssl_options_.root_cert = optarg;
//            break;
//#endif  // TRITON_ENABLE_GRPC
//
//#ifdef TRITON_ENABLE_METRICS
//        case OPTION_ALLOW_METRICS:
//            allow_metrics_ = ParseBoolOption(optarg);
//            break;
//        case OPTION_ALLOW_GPU_METRICS:
//            allow_gpu_metrics = ParseBoolOption(optarg);
//            break;
//        case OPTION_METRICS_PORT:
//            metrics_port = ParseIntOption(optarg);
//            break;
//#endif  // TRITON_ENABLE_METRICS
//
//#ifdef TRITON_ENABLE_TRACING
//        case OPTION_TRACE_FILEPATH:
//            trace_filepath = optarg;
//            break;
//        case OPTION_TRACE_LEVEL:
//            trace_level = ParseTraceLevelOption(optarg);
//            break;
//        case OPTION_TRACE_RATE:
//            trace_rate = ParseIntOption(optarg);
//            break;
//#endif  // TRITON_ENABLE_TRACING
//
//        case OPTION_POLL_REPO_SECS:
//            repository_poll_secs = ParseIntOption(optarg);
//            break;
//        case OPTION_STARTUP_MODEL:
//            startup_models_.insert(optarg);
//            break;
//        case OPTION_MODEL_CONTROL_MODE: {
//            std::string mode_str(optarg);
//            std::transform(
//                mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
//            if (mode_str == "none") {
//                control_mode = TRITONSERVER_MODEL_CONTROL_NONE;
//            }
//            else if (mode_str == "poll") {
//                control_mode = TRITONSERVER_MODEL_CONTROL_POLL;
//            }
//            else if (mode_str == "explicit") {
//                control_mode = TRITONSERVER_MODEL_CONTROL_EXPLICIT;
//            }
//            else {
//                std::cerr << "invalid argument for --model-control-mode" << std::endl;
//                std::cerr << Usage() << std::endl;
//                return false;
//            }
//            break;
//        }
//        case OPTION_PINNED_MEMORY_POOL_BYTE_SIZE:
//            pinned_memory_pool_byte_size = ParseLongLongOption(optarg);
//            break;
//        case OPTION_CUDA_MEMORY_POOL_BYTE_SIZE:
//            cuda_pools.push_back(ParsePairOption(optarg));
//            break;
//        case OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY:
//            min_supported_compute_capability = ParseDoubleOption(optarg);
//            break;
//        case OPTION_EXIT_TIMEOUT_SECS:
//            exit_timeout_secs = ParseIntOption(optarg);
//            break;
//        case OPTION_BACKEND_DIR:
//            backend_dir = optarg;
//            break;
//        case OPTION_BUFFER_MANAGER_THREAD_COUNT:
//            buffer_manager_thread_count = ParseIntOption(optarg);
//            break;
//        case OPTION_BACKEND_CONFIG:
//            backend_config_settings.push_back(ParseBackendConfigOption(optarg));
//            break;
//        }
//    }
//
//    if (optind < argc) {
//        std::cerr << "Unexpected argument: " << argv[optind] << std::endl;
//        std::cerr << Usage() << std::endl;
//        return false;
//    }
//
//#ifdef TRITON_ENABLE_LOGGING
//    // Initialize our own logging instance since it is used by GRPC and
//    // HTTP endpoints. This logging instance is separate from the one in
//    // libtritonserver so we must initialize explicitly.
//    LOG_ENABLE_INFO(log_info);
//    LOG_ENABLE_WARNING(log_warn);
//    LOG_ENABLE_ERROR(log_error);
//    LOG_SET_VERBOSE(log_verbose);
//#endif  // TRITON_ENABLE_LOGGING
//
//    repository_poll_secs_ = 0;
//    if (control_mode == TRITONSERVER_MODEL_CONTROL_POLL) {
//        repository_poll_secs_ = std::max(0, repository_poll_secs);
//    }
//
//#if defined(TRITON_ENABLE_HTTP)
//    http_port_ = http_port;
//    http_thread_cnt_ = http_thread_cnt;
//#endif  // TRITON_ENABLE_HTTP
//
//#if defined(TRITON_ENABLE_GRPC)
//    grpc_port_ = grpc_port;
//    grpc_infer_allocation_pool_size_ = grpc_infer_allocation_pool_size;
//    grpc_use_ssl_ = grpc_use_ssl;
//#endif  // TRITON_ENABLE_GRPC
//
//#ifdef TRITON_ENABLE_METRICS
//    metrics_port_ = allow_metrics_ ? metrics_port : -1;
//    allow_gpu_metrics = allow_metrics_ ? allow_gpu_metrics : false;
//#endif  // TRITON_ENABLE_METRICS
//
//#ifdef TRITON_ENABLE_TRACING
//    trace_filepath_ = trace_filepath;
//    trace_level_ = trace_level;
//    trace_rate_ = trace_rate;
//#endif  // TRITON_ENABLE_TRACING
//
//    // Check if HTTP, GRPC and metrics port clash
//    if (CheckPortCollision()) {
//        return false;
//    }
//
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsNew(server_options), "creating server options");
//    auto loptions = *server_options;
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetServerId(loptions, server_id.c_str()),
//        "setting server ID");
//    for (const auto& model_repository_path : model_repository_paths) {
//        FAIL_IF_ERR(
//            TRITONSERVER_ServerOptionsSetModelRepositoryPath(
//                loptions, model_repository_path.c_str()),
//            "setting model repository path");
//    }
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetModelControlMode(loptions, control_mode),
//        "setting model control mode");
//    for (const auto& model : startup_models_) {
//        FAIL_IF_ERR(
//            TRITONSERVER_ServerOptionsSetStartupModel(loptions, model.c_str()),
//            "setting startup model");
//    }
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
//            loptions, pinned_memory_pool_byte_size),
//        "setting total pinned memory byte size");
//    for (const auto& cuda_pool : cuda_pools) {
//        FAIL_IF_ERR(
//            TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
//                loptions, cuda_pool.first, cuda_pool.second),
//            "setting total CUDA memory byte size");
//    }
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
//            loptions, min_supported_compute_capability),
//        "setting minimum supported CUDA compute capability");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetExitOnError(loptions, exit_on_error),
//        "setting exit on error");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetStrictModelConfig(
//            loptions, strict_model_config),
//        "setting strict model configuration");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetStrictReadiness(loptions, strict_readiness),
//        "setting strict readiness");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetExitTimeout(
//            loptions, std::max(0, exit_timeout_secs)),
//        "setting exit timeout");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
//            loptions, std::max(0, buffer_manager_thread_count)),
//        "setting buffer manager thread count");
//
//#ifdef TRITON_ENABLE_LOGGING
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetLogInfo(loptions, log_info),
//        "setting log info enable");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetLogWarn(loptions, log_warn),
//        "setting log warn enable");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetLogError(loptions, log_error),
//        "setting log error enable");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetLogVerbose(loptions, log_verbose),
//        "setting log verbose level");
//#endif  // TRITON_ENABLE_LOGGING
//
//#ifdef TRITON_ENABLE_METRICS
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetMetrics(loptions, allow_metrics_),
//        "setting metrics enable");
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetGpuMetrics(loptions, allow_gpu_metrics),
//        "setting GPU metrics enable");
//#endif  // TRITON_ENABLE_METRICS
//
//    FAIL_IF_ERR(
//        TRITONSERVER_ServerOptionsSetBackendDirectory(
//            loptions, backend_dir.c_str()),
//        "setting backend directory");
//    for (const auto& bcs : backend_config_settings) {
//        FAIL_IF_ERR(
//            TRITONSERVER_ServerOptionsSetBackendConfig(
//                loptions, std::get<0>(bcs).c_str(), std::get<1>(bcs).c_str(),
//                std::get<2>(bcs).c_str()),
//            "setting backend configurtion");
//    }

    return true;
}

int main(int argc,char** argv)
{
    
    
    TRITONSERVER_ServerOptions* server_options = nullptr;
    if (!Parse(&server_options, argc, argv)) {
        exit(1);
    }
    std::cout << server_options << std::endl;
    return 0;
}