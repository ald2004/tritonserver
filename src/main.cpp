#include <csignal>

#include "tritonserver.h"
#include "options.h"

int main(int argc,char** argv)
{
    
    
    TRITONSERVER_ServerOptions* server_options = nullptr;
    if (!Parse(&server_options, argc, argv)) {
        std::cerr << "parse argument err!" << std::endl;
        exit(1);
    }
    std::cout << server_options << std::endl;
    // Trace manager.
    nvidia::inferenceserver::TraceManager* trace_manager;


    // Manager for shared memory blocks.
    auto shm_manager = std::make_shared<nvidia::inferenceserver::SharedMemoryManager>();

    // Create the server...
    TRITONSERVER_Server* server_ptr = nullptr;
    FAIL_IF_ERR(
        TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server");
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsDelete(server_options),
        "deleting server options");

    std::shared_ptr<TRITONSERVER_Server> server(
        server_ptr, TRITONSERVER_ServerDelete);

    // Configure and start tracing if specified on the command line.
    if (!StartTracing(&trace_manager)) {
        exit(1);
    }

    // Start the HTTP, GRPC, and metrics endpoints.
    if (!StartEndpoints(server, trace_manager, shm_manager)) {
        exit(1);
    }

    // Trap SIGINT and SIGTERM to allow server to exit gracefully
    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);
    // Wait until a signal terminates the server...
    while (!exiting_) {
        // If enabled, poll the model repository to see if there have been
        // any changes.
        if (repository_poll_secs_ > 0) {
            LOG_TRITONSERVER_ERROR(
                TRITONSERVER_ServerPollModelRepository(server_ptr),
                "failed to poll model repository");
        }

        // Wait for the polling interval (or a long time if polling is not
        // enabled). Will be woken if the server is exiting.
        std::unique_lock<std::mutex> lock(exit_mu_);
        std::chrono::seconds wait_timeout(
            (repository_poll_secs_ == 0) ? 3600 : repository_poll_secs_);
        exit_cv_.wait_for(lock, wait_timeout);
    }

    TRITONSERVER_Error* stop_err = TRITONSERVER_ServerStop(server_ptr);

    // If unable to gracefully stop the server then Triton threads and
    // state are potentially in an invalid state, so just exit
    // immediately.
    if (stop_err != nullptr) {
        LOG_TRITONSERVER_ERROR(stop_err, "failed to stop server");
        exit(1);
    }

    // Stop tracing and the HTTP, GRPC, and metrics endpoints.
    StopEndpoints();
    StopTracing(&trace_manager);

#ifdef TRITON_ENABLE_ASAN
    // Can invoke ASAN before exit though this is typically not very
    // useful since there are many objects that are not yet destructed.
    //  __lsan_do_leak_check();
#endif  // TRITON_ENABLE_ASAN

    return 0;
}